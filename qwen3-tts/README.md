# Qwen3 TTS — Self-Hosted Text-to-Speech on EKS

A production-ready, GPU-accelerated **Text-to-Speech (TTS)** service built on [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice). Deployed to Amazon EKS via **Ray Serve** with **vllm-omni** as the inference engine, auto-scaled by **Karpenter** GPU node pools. Supports streaming audio output, multiple audio formats, and custom voice cloning.

---

## Architecture

```
Client (text) ──▶ Ray Serve (CPU head) ──▶ vllm-omni subprocess (GPU worker, port 8100)
                        ▲                         │
                  KubeRay operator          OpenAI-compatible API
                        │                         │
                  Karpenter GPU            ◀── audio stream ──▶ Client
                  NodePool (g6e/g6/g5)
```

| Component | Role |
|---|---|
| **Ray Serve** | Request routing, autoscaling (1–4 replicas), health checks |
| **vllm-omni** | TTS model inference with `--omni` flag for audio generation |
| **Karpenter** | On-demand GPU node provisioning (g6e/g6/g5 instances) |
| **KubeRay** | Kubernetes-native Ray cluster lifecycle management |

---

## Model Details

| Property | Value |
|---|---|
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
| Task | Text-to-Speech (TTS) with voice cloning |
| Parameters | 1.7B |
| Inference | vllm-omni as subprocess (proxied via httpx) |
| GPU | NVIDIA L40S / L4 / A10G |
| GPU Memory Utilization | 0.9 |
| Output Sample Rate | 24,000 Hz (mono, 16-bit signed PCM) |

---

## API Reference

### 1. Generate Speech

**Endpoint:** `POST /v1/audio/speech`

#### Streaming PCM (recommended for real-time)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you today?",
    "voice": "vivian",
    "language": "Auto",
    "stream": true,
    "response_format": "pcm"
  }' --output audio.pcm
```

Returns raw **16-bit signed PCM** at 24kHz mono, streamed as chunked HTTP response.

#### Non-Streaming (WAV, MP3, FLAC, AAC, Opus)

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, how are you today?",
    "voice": "vivian",
    "response_format": "wav"
  }' --output output.wav
```

Supported formats: `wav`, `mp3`, `flac`, `pcm`, `aac`, `opus`

#### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | required | Text to synthesize |
| `voice` | string | `"vivian"` | Speaker name or uploaded voice ID |
| `language` | string | `"Auto"` | Language hint (Auto-detected if not specified) |
| `stream` | boolean | `false` | Enable chunked streaming response |
| `response_format` | string | `"wav"` | Output audio format |
| `instructions` | string | `""` | Optional style/emotion instructions |

### 2. List Voices

**Endpoint:** `GET /v1/audio/voices`

```bash
curl http://localhost:8000/v1/audio/voices
```

### 3. Upload Custom Voice

**Endpoint:** `POST /v1/audio/voices`

Upload a reference audio sample for voice cloning:

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "file=@reference_voice.wav" \
  -F "name=my_custom_voice"
```

### 4. WebSocket Streaming

**Endpoint:** `WS /v1/audio/speech/stream`

For real-time bidirectional streaming TTS via WebSocket.

---

## File Structure

```
qwen3-tts/
├── serve_app.py                # Ray Serve application (inference handler + vllm-omni proxy)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container image (rayproject/ray:2.52.0-py311-cu124)
├── deploy.sh                   # End-to-end deployment script
├── ray-service-tts.yaml        # RayService + ServiceAccount K8s manifests
└── karpenter-gpu-nodepool.yaml # Karpenter NodePool for GPU provisioning
```

---

## Prerequisites

- **AWS CLI** configured with appropriate IAM permissions
- **kubectl** pointing at the `model-management` EKS cluster in `us-east-1`
- **Docker** running locally
- **HuggingFace token** (`HF_TOKEN`) with access to the Qwen model
- **KubeRay operator** installed on the cluster
- **Karpenter** installed and configured on the cluster

---

## Deployment

### Quick Start

```bash
cd qwen3-tts
export HF_TOKEN=hf_xxxxx
bash deploy.sh
```

### What `deploy.sh` Does (6 Steps)

1. **ECR Repository** — Creates `qwen3-tts` repo in ECR (with image scanning enabled)
2. **Docker Build & Push** — Builds image tagged `<git-short-sha>-<timestamp>`, pushes to ECR
3. **Kubeconfig** — Updates kubectl context for `model-management` cluster
4. **Karpenter NodePool** — Applies GPU node pool config (`gpu-g6e-tts`)
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

### Karpenter GPU NodePool (`gpu-g6e-tts`)

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
kubectl get rayservice qwen3-tts-service
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
kubectl port-forward svc/qwen3-tts-service-serve-svc 8001:8000
```

Then test:

```bash
# Streaming PCM
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"vivian","stream":true,"response_format":"pcm"}' \
  --output test.pcm

# Non-streaming WAV
curl -X POST http://localhost:8001/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello world","voice":"vivian","response_format":"wav"}' \
  --output test.wav

# Play PCM (macOS)
play -t raw -r 24000 -e signed -b 16 -c 1 test.pcm

# List available voices
curl http://localhost:8001/v1/audio/voices
```

### Ray Dashboard

```bash
kubectl port-forward svc/qwen3-tts-service-head-svc 8265:8265
# Open http://localhost:8265
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `vllm-omni` | ~=0.18.0 | TTS inference engine (omni-modal vLLM fork) |
| `vllm` | ~=0.18.0 | Core vLLM (must match vllm-omni version) |
| `pandas` | >=2.0.0 | Required for numpy 2.x ABI compatibility with Ray |
| `httpx` | ~=0.27.0 | Async HTTP client for vllm-omni subprocess proxy |
| `websockets` | ~=13.0 | WebSocket proxy for streaming TTS |

> **Important:** `vllm-omni` and `vllm` versions must match (both ~=0.18.0). vllm-omni 0.18.x is incompatible with vllm 0.19.x due to internal API changes.

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
| `ModuleNotFoundError: vllm` | vllm-omni excludes vllm to avoid entrypoint conflicts | Add `vllm~=0.18.0` explicitly |
| `ModuleNotFoundError: vllm.inputs.data` | vllm-omni 0.18 + vllm 0.19 mismatch | Pin both to `~=0.18.0` |
| Stale pending clusters not rolling | KubeRay doesn't recreate pending clusters on spec change | `kubectl delete raycluster <name>` to force re-creation |
