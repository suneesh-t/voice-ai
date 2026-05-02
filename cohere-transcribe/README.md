# Cohere Transcribe — Self-Hosted Speech-to-Text on EKS

A production-ready, GPU-accelerated **Automatic Speech Recognition (ASR)** service built on [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026). Deployed to Amazon EKS via **Ray Serve** with **vLLM** as the inference engine, auto-scaled by **Karpenter** GPU node pools.

---

## Architecture

```
Client (audio) ──▶ Ray Serve (CPU head) ──▶ vLLM subprocess (GPU worker, port 8100)
                         ▲                         │
                   KubeRay operator          OpenAI-compatible API
                         │                         │
                   Karpenter GPU            ◀── transcription ──▶ Client
                   NodePool (g6/g5)
```

| Component | Role |
|---|---|
| **Ray Serve** | Request routing, autoscaling (1–4 replicas), health checks |
| **vLLM** | Model inference engine with `trust_remote_code=True` |
| **Karpenter** | On-demand GPU node provisioning (g6e/g6/g5 instances) |
| **KubeRay** | Kubernetes-native Ray cluster lifecycle management |

---

## Model Details

| Property | Value |
|---|---|
| Model | `CohereLabs/cohere-transcribe-03-2026` |
| Revision | `76b8b23e8607f35f0265a23d481b338fb0e26aea` |
| Task | Speech-to-Text (ASR) |
| Inference | vLLM 0.19 as subprocess (proxied via `httpx.AsyncClient`) |
| GPU | NVIDIA L40S / L4 / A10G |
| Max audio size | 50 MB |

---

## API Reference

### Transcribe Audio

**Endpoint:** `POST /`

Supports two input methods:

#### 1. Multipart Form Upload

```bash
curl -X POST http://localhost:8000/ \
  -F "file=@audio.wav"
```

#### 2. JSON with Base64 Audio

```bash
curl -X POST http://localhost:8000/ \
  -H "Content-Type: application/json" \
  -d '{"audio_base64": "<base64-encoded-audio>"}'
```

#### Streaming Response

Append `?stream=true` for Server-Sent Events (SSE) streaming:

```bash
curl -X POST "http://localhost:8000/?stream=true" \
  -F "file=@audio.wav"
```

SSE format:
```
data: {"choices":[{"delta":{"content":"Hello "}}]}
data: {"choices":[{"delta":{"content":"world"}}]}
data: [DONE]
```

#### Non-Streaming Response

```json
{
  "text": "Hello world, this is a transcription."
}
```

---

## File Structure

```
cohere-transcribe/
├── serve_app.py                    # Ray Serve application (inference handler)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image (rayproject/ray:2.52.0-py311-cu124)
├── deploy.sh                       # End-to-end deployment script
├── ray-service-transcribe.yaml     # RayService + ServiceAccount K8s manifests
└── karpenter-gpu-nodepool.yaml     # Karpenter NodePool for GPU provisioning
```

---

## Prerequisites

- **AWS CLI** configured with appropriate IAM permissions
- **kubectl** pointing at the `model-management` EKS cluster in `us-east-1`
- **Docker** running locally
- **HuggingFace token** (`HF_TOKEN`) with access to the Cohere model
- **KubeRay operator** installed on the cluster
- **Karpenter** installed and configured on the cluster

---

## Deployment

### Quick Start

```bash
cd cohere-transcribe
export HF_TOKEN=hf_xxxxx
bash deploy.sh
```

### What `deploy.sh` Does (6 Steps)

1. **ECR Repository** — Creates `cohere-transcribe` repo in ECR (with image scanning enabled)
2. **Docker Build & Push** — Builds image tagged `<git-short-sha>-<timestamp>`, pushes to ECR
3. **Kubeconfig** — Updates kubectl context for `model-management` cluster
4. **Karpenter NodePool** — Applies GPU node pool config (`gpu-g6-transcribe`)
5. **HF Token Secret** — Creates/updates `hf-token` Kubernetes secret in `default` namespace
6. **RayService Deploy** — Substitutes `<AWS_ACCOUNT_ID>` and `<IMAGE_TAG>` placeholders, applies manifest

---

## Infrastructure Details

### Head Node (CPU-only)

| Resource | Value |
|---|---|
| CPU | 4 |
| Memory | 8Gi |
| GPU | None |
| Ports | 8000 (serve), 8080 (metrics), 6379 (GCS), 8265 (dashboard), 10001 (client) |
| Probes | TCP on GCS port (readiness: 15s delay; liveness: 120s delay) |

### GPU Worker Nodes

| Resource | Value |
|---|---|
| CPU | 3 |
| Memory | 14Gi |
| GPU | 1x NVIDIA |
| Ephemeral Storage | 50Gi |
| Replicas | 1–4 (autoscaling) |
| Probes | `ray health-check` (readiness: 30s delay; liveness: 600s delay) |

### Karpenter GPU NodePool (`gpu-g6-transcribe`)

| Property | Value |
|---|---|
| Instance Types | `g6e.xlarge`, `g6e.2xlarge`, `g6.xlarge`, `g5.xlarge` |
| Capacity Type | On-Demand |
| Architecture | amd64 |
| Cluster Limits | 32 CPU, 128Gi memory, 4 GPUs |
| Node TTL | 336 hours (14 days) |
| Consolidation | When empty or underutilized (300s delay) |

---

## Monitoring & Testing

### Check Deployment Status

```bash
kubectl get rayservice cohere-transcribe-service
kubectl get pods -l ray.io/cluster
```

### View Logs

```bash
# Head node
kubectl logs -l ray.io/node-type=head -c ray-head --tail=100 -f

# Worker node
kubectl logs -l ray.io/node-type=worker -c ray-worker --tail=100 -f
```

### Local Testing via Port Forward

```bash
kubectl port-forward svc/cohere-transcribe-service-serve-svc 8000:8000
```

Then test:

```bash
# Non-streaming
curl -X POST http://localhost:8000/ -F "file=@test_audio.wav"

# Streaming
curl -X POST "http://localhost:8000/?stream=true" -F "file=@test_audio.wav"
```

### Ray Dashboard

```bash
kubectl port-forward svc/cohere-transcribe-service-head-svc 8265:8265
# Open http://localhost:8265
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `vllm[audio]` | ~=0.19.0 | Inference engine with audio support |
| `httpx` | ~=0.27.0 | Async HTTP client for vLLM subprocess proxy |
| `librosa` | ~=0.10.0 | Audio processing |
| `soundfile` | ~=0.12.0 | Audio I/O |
| `pandas` | >=2.0.0 | Required for numpy 2.x ABI compatibility with Ray |

---

## Security

- Container runs as **non-root** user (UID 1000)
- All Linux capabilities **dropped**
- No privilege escalation allowed
- HuggingFace token stored as Kubernetes Secret (not baked into image)

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `MetricsHead` 30s timeout crash | Ray 2.52 dashboard subprocess startup | Set `RAY_DASHBOARD_SUBPROCESS_MODULE_WAIT_READY_TIMEOUT=300` |
| `numpy.dtype size changed` | numpy 2.x / pandas 1.x ABI mismatch | Pin `pandas>=2.0.0` |
| Stale pending clusters not rolling | KubeRay doesn't recreate pending clusters on spec change | `kubectl delete raycluster <name>` to force re-creation |
