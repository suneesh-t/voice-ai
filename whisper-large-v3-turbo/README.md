# Whisper Large-v3-Turbo — Fast Multilingual ASR on EKS

A GPU-accelerated **Automatic Speech Recognition (ASR)** service built on [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo). A pruned, speed-optimized variant of Whisper Large-v3 with 809M parameters — nearly half the size of the original while maintaining comparable accuracy. Deployed to Amazon EKS via **Ray Serve** with a **vLLM subprocess** for inference, auto-scaled by **Karpenter** GPU node pools.

---

## Model Details

| Property | Value |
|---|---|
| Model | `openai/whisper-large-v3-turbo` |
| Architecture | Transformer encoder-decoder (seq2seq) |
| Parameters | 809M (pruned from 1550M) |
| Decoder Layers | 4 (reduced from 32) |
| Languages | 99 |
| Tasks | Transcription, translation to English |
| License | MIT |
| RTFX | ~200x real-time |
| Mean WER | 7.83 |

---

## Architecture

```
Client (audio) ──▶ Ray Serve (CPU head) ──▶ WhisperLargeV3Turbo deployment (GPU worker)
                          ▲                         │
                    KubeRay operator          vLLM subprocess (port 8100)
                          │                    /v1/audio/transcriptions
                    Karpenter GPU                   │
                    NodePool (g6/g5)          ◀── JSON/SSE ──▶ Client
```

The Ray Serve deployment spawns **vLLM** as a subprocess running `vllm.entrypoints.openai.api_server` on port 8100 with the following flags:

| Flag | Value | Purpose |
|---|---|---|
| `--max-model-len` | 448 | Whisper's max output tokens |
| `--max-num-seqs` | 400 | High concurrent request batching |
| `--gpu-memory-utilization` | 0.85 | GPU memory allocation |
| `--limit-mm-per-prompt` | `audio=1` | One audio file per request |
| `--kv-cache-dtype` | `fp8` | FP8 KV cache for memory efficiency |

Requests are proxied from Ray Serve to the vLLM subprocess via `httpx.AsyncClient`. This is the same pattern used by `cohere-transcribe`.

---

## API Reference

### Transcribe Audio

**Endpoint:** `POST /`

#### Multipart Form Upload

```bash
curl -X POST http://localhost:8003/ -F "file=@audio.wav"
```

#### JSON with Base64 Audio

```bash
curl -X POST http://localhost:8003/ \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64-encoded-audio>"}'
```

#### Response

```json
{
  "text": "Hello world, this is a transcription."
}
```

#### Streaming Response (SSE)

Append `?stream=true` — returns Server-Sent Events streamed directly from vLLM:

```bash
curl -X POST "http://localhost:8003/?stream=true" -F "file=@audio.wav"
```

```
data: {"choices":[{"delta":{"content":"Hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

#### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` / `audio` | file/base64 | required | Audio input |
| `language` | string | `"en"` | Language code (e.g., `"en"`, `"fr"`, `"es"`) |
| `task` | string | `"transcribe"` | `"transcribe"` or `"translate"` (translate to English) |
| `stream` | query param | `false` | `?stream=true` for SSE streaming |

### Health Check

Ray Serve checks the vLLM subprocess is still running via `check_health()`. If the subprocess exits, the replica is marked unhealthy and restarted.

---

## File Structure

```
whisper-large-v3-turbo/
├── serve_app.py                    # Ray Serve app (vLLM subprocess proxy)
├── requirements.txt                # Python dependencies (vllm[audio], httpx)
├── Dockerfile                      # Container image (rayproject/ray:2.52.0-py311-cu124)
├── deploy.sh                       # End-to-end deployment script
├── ray-service-whisper.yaml        # RayService + ServiceAccount K8s manifests
└── karpenter-gpu-nodepool.yaml     # Karpenter NodePool for GPU provisioning
```

---

## Prerequisites

- **AWS CLI** configured with appropriate IAM permissions
- **kubectl** pointing at the `model-management` EKS cluster in `us-east-1`
- **Docker** running locally
- **HuggingFace token** (`HF_TOKEN`) — model is MIT licensed but token needed for gated download
- **KubeRay operator** and **Karpenter** installed on the cluster (see [infrastructure/](../infrastructure/))

---

## Deployment

```bash
cd whisper-large-v3-turbo
export HF_TOKEN=hf_xxxxx
bash deploy.sh
```

### What `deploy.sh` Does (6 Steps)

1. **ECR Repository** — Creates `whisper-v3-turbo-serve` repo in ECR
2. **Docker Build & Push** — Builds image tagged `<git-short-sha>-<timestamp>`
3. **Kubeconfig** — Updates kubectl context for `model-management` cluster
4. **Karpenter NodePool** — Applies GPU node pool config (`gpu-g6-whisper`)
5. **HF Token Secret** — Creates/updates `hf-token` Kubernetes secret
6. **RayService Deploy** — Substitutes placeholders and applies manifest

---

## Infrastructure

### Head Node (CPU-only)

| Resource | Value |
|---|---|
| CPU | 4 |
| Memory | 8Gi |
| GPU | None |

### GPU Worker Nodes

| Resource | Value |
|---|---|
| CPU | 3 |
| Memory | 14Gi |
| GPU | 1x NVIDIA |
| Ephemeral Storage | 50Gi |
| Replicas | 1–4 (autoscaling) |

### Karpenter GPU NodePool (`gpu-g6-whisper`)

| Property | Value |
|---|---|
| Instance Types | `g6e.xlarge`, `g6e.2xlarge`, `g6.xlarge`, `g5.xlarge` |
| Capacity Type | On-Demand |
| Cluster Limits | 32 CPU, 128Gi memory, 4 GPUs |
| Consolidation | When empty or underutilized (5 min delay) |

---

## Monitoring & Testing

```bash
# Check status
kubectl get rayservice whisper-v3-turbo-service
kubectl get pods -l ray.io/cluster

# Port forward
kubectl port-forward svc/whisper-v3-turbo-service-serve-svc 8003:8000

# Transcribe
curl -X POST http://localhost:8003/ -F "file=@test_audio.wav"

# Translate to English
curl -X POST http://localhost:8003/ \
  -F "file=@french_audio.wav" \
  -F "task=translate"

# Streaming
curl -X POST "http://localhost:8003/?stream=true" -F "file=@audio.wav"

# JSON with base64
curl -X POST http://localhost:8003/ \
  -H "Content-Type: application/json" \
  -d '{"audio": "'$(base64 < audio.wav)'", "language": "en"}'
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `vllm[audio]` | ~0.19.0 | vLLM inference engine with audio support |
| `httpx` | ~0.27.0 | Async HTTP client for proxying to vLLM subprocess |
| `pandas` | >= 2.0.0 | numpy 2.x ABI compatibility with Ray |

---

## Security

- Container runs as **non-root** (UID 1000)
- All Linux capabilities **dropped**
- No privilege escalation allowed
- HuggingFace token stored as Kubernetes Secret
- Model license: **MIT** (open source, no usage restrictions)

---

## Comparison with Cohere Transcribe

| Feature | Whisper Large-v3-Turbo | Cohere Transcribe |
|---|---|---|
| Parameters | 809M | — |
| Languages | 99 | Multi |
| Inference | vLLM subprocess | vLLM subprocess |
| Translation | Yes (to English) | No |
| Streaming | SSE (native vLLM streaming) | SSE (native vLLM streaming) |
| Speed | RTFX ~200x | Depends on audio length |
| License | MIT | Model-specific |

---

## Known Issues

- `MODEL_REVISION` is set to `"main"` — pin to an exact commit SHA before production
- Audio size limit is 50 MB (configurable via `MAX_AUDIO_BYTES` in `serve_app.py`)
- vLLM subprocess startup takes 2–5 minutes on first deploy while the model downloads
