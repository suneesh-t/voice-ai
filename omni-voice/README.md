# OmniVoice — Multilingual Zero-Shot TTS on EKS

A GPU-accelerated **Text-to-Speech** service built on [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice). Supports **600+ languages**, zero-shot voice cloning, and voice design via speaker attributes. Deployed to Amazon EKS via **Ray Serve**, auto-scaled by **Karpenter** GPU node pools.

---

## Model Details

| Property | Value |
|---|---|
| Model | `k2-fsa/OmniVoice` |
| Base | Qwen3-0.6B (Diffusion Language Model) |
| Task | Text-to-Speech, voice cloning, voice design |
| Languages | 600+ (zero-shot) |
| Output | 24kHz, 16-bit signed PCM, mono |
| Inference Speed | RTF ~0.025 (40x real-time) |
| GPU | NVIDIA L4 / A10G / L40S |

---

## Architecture

```
Client (text + optional ref audio) ──▶ Ray Serve (CPU head) ──▶ OmniVoice (GPU worker)
                                              ▲                         │
                                        KubeRay operator          audio waveform
                                              │                         │
                                        Karpenter GPU            ◀── PCM/WAV ──▶ Client
                                        NodePool (g6e/g6/g5)
```

Unlike `qwen3-tts`, OmniVoice loads directly as a Python library (no subprocess proxy). The model runs in-process on the GPU worker via `OmniVoice.from_pretrained()`.

---

## API Reference

### Generate Speech

**Endpoint:** `POST /v1/audio/speech`

#### JSON Request

```bash
# Basic TTS (WAV output)
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test of OmniVoice.",
    "response_format": "wav"
  }' --output output.wav

# Streaming PCM
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "response_format": "pcm",
    "stream": true
  }' --output output.pcm

# Voice cloning (reference audio path on server)
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Clone this voice please.",
    "ref_audio": "/path/to/ref.wav",
    "ref_text": "Transcription of reference audio."
  }' --output cloned.wav

# Voice design (no reference audio needed)
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Design a voice with these attributes.",
    "instruct": "female, low pitch, British accent"
  }' --output designed.wav
```

#### Multipart Form (with reference audio upload)

```bash
curl -X POST http://localhost:8002/v1/audio/speech \
  -F "text=Clone this voice please." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=Transcription of reference audio." \
  -F "response_format=wav" \
  --output cloned.wav
```

#### Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` / `text` | string | required | Text to synthesize |
| `ref_audio` | string/file | — | Reference audio for voice cloning |
| `ref_text` | string | — | Transcription of reference audio |
| `instruct` | string | — | Speaker attributes for voice design (e.g., `"male, British accent"`) |
| `speed` | float | `1.0` | Speed factor (>1.0 = faster) |
| `num_step` | int | `32` | Diffusion steps (lower = faster, 16 for speed) |
| `response_format` | string | `"wav"` | Output format: `wav` or `pcm` |
| `stream` | boolean | `false` | Stream PCM in chunks (only with `response_format=pcm`). Note: generates full audio first, then chunks — not true incremental streaming |

### Health Check

```bash
curl http://localhost:8002/health
```

---

## Three Generation Modes

### 1. Voice Cloning (Zero-Shot)

Provide reference audio + its transcription. The model clones the speaker's voice to synthesize new text:

```json
{
  "input": "New text to speak.",
  "ref_audio": "reference.wav",
  "ref_text": "What the reference audio says."
}
```

### 2. Voice Design

Describe the desired voice attributes — no reference audio needed:

```json
{
  "input": "Speak with designed attributes.",
  "instruct": "female, young, high pitch, whisper"
}
```

Supported attributes: gender, age, pitch, style (whisper), English accents (British, American, Australian, Indian), Chinese dialects.

### 3. Auto Voice

Send text only — the model picks a voice automatically:

```json
{
  "input": "Just say this with a default voice."
}
```

---

## File Structure

```
omni-voice/
├── serve_app.py                    # Ray Serve app (in-process OmniVoice inference)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image (rayproject/ray:2.52.0-py311-cu124)
├── deploy.sh                       # End-to-end deployment script
├── ray-service-omnivoice.yaml      # RayService + ServiceAccount K8s manifests
└── karpenter-gpu-nodepool.yaml     # Karpenter NodePool for GPU provisioning
```

---

## Prerequisites

- **AWS CLI** configured with appropriate IAM permissions
- **kubectl** pointing at the `model-management` EKS cluster in `us-east-1`
- **Docker** running locally
- **HuggingFace token** (`HF_TOKEN`) with access to the model
- **KubeRay operator** and **Karpenter** installed on the cluster (see [infrastructure/](../infrastructure/))

---

## Deployment

```bash
cd omni-voice
export HF_TOKEN=hf_xxxxx
bash deploy.sh
```

### What `deploy.sh` Does (6 Steps)

1. **ECR Repository** — Creates `omni-voice-serve` repo in ECR
2. **Docker Build & Push** — Builds image tagged `<git-short-sha>-<timestamp>`
3. **Kubeconfig** — Updates kubectl context for `model-management` cluster
4. **Karpenter NodePool** — Applies GPU node pool config (`gpu-g6e-omnivoice`)
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
| Ports | 8000 (serve), 8080 (metrics), 6379 (GCS), 8265 (dashboard), 10001 (client) |

### GPU Worker Nodes

| Resource | Value |
|---|---|
| CPU | 3 |
| Memory | 14Gi |
| GPU | 1x NVIDIA |
| Ephemeral Storage | 50Gi |
| Replicas | 1–4 (autoscaling) |

### Karpenter GPU NodePool (`gpu-g6e-omnivoice`)

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
kubectl get rayservice omni-voice-service
kubectl get pods -l ray.io/cluster

# Port forward
kubectl port-forward svc/omni-voice-service-serve-svc 8002:8000

# Test
curl -X POST http://localhost:8002/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from OmniVoice","response_format":"wav"}' \
  --output test.wav

# Play (macOS)
afplay test.wav
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `omnivoice` | >= 0.1.0 | OmniVoice TTS library |
| `torch` | 2.8.0 | PyTorch (CUDA 12.4) |
| `torchaudio` | 2.8.0 | Audio processing |
| `numpy` | >= 1.26.0 | Numerical operations |
| `soundfile` | >= 0.12.0 | Audio I/O |
| `pandas` | >= 2.0.0 | numpy 2.x ABI compatibility with Ray |

---

## Security

- Container runs as **non-root** (UID 1000)
- All Linux capabilities **dropped**
- No privilege escalation allowed
- HuggingFace token stored as Kubernetes Secret

---

## Comparison with Qwen3 TTS

| Feature | OmniVoice | Qwen3 TTS |
|---|---|---|
| Languages | 600+ | Multi (fewer) |
| Parameters | 0.6B (Qwen3 base) | 1.7B |
| Voice Cloning | Zero-shot (ref audio) | Custom voice upload |
| Voice Design | Attribute-based | N/A |
| Inference | In-process Python | vllm-omni subprocess |
| Streaming | PCM chunking (post-generation) | Native PCM streaming |
| Inference Speed | RTF ~0.025 | Depends on text length |
