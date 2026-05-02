# Voice AI

End-to-end **voice-to-voice AI assistant** platform with self-hosted speech models on AWS EKS and a real-time conversational pipeline.

Users speak into a browser, audio flows through a cascaded pipeline — Voice Activity Detection, Speech-to-Text, LLM reasoning, and Text-to-Speech — and spoken responses stream back in real time via WebRTC.

---

## Architecture

```
                         ┌──────────────────────────────┐
                         │     Browser (WebRTC UI)       │
                         └──────────┬───────────────────┘
                                    │ audio ↕ audio
                         ┌──────────▼───────────────────┐
                         │      voice-pipeline           │
                         │      (Pipecat + Bedrock)      │
                         │                               │
                         │  Mic → VAD → STT → LLM → TTS │
                         └───────┬──────────────┬───────┘
                                 │              │
                    HTTP/SSE     │              │  HTTP/PCM stream
                                 ▼              ▼
               ┌─────────────────────┐  ┌─────────────────────┐
               │  cohere-transcribe  │  │     qwen3-tts       │
               │  (Speech-to-Text)   │  │  (Text-to-Speech)   │
               │                     │  │                     │
               │  Ray Serve + vLLM   │  │  Ray Serve + vllm-  │
               │  GPU: L4/A10G/L40S  │  │  omni               │
               └─────────┬──────────┘  │  GPU: L40S/L4/A10G  │
                         │              └──────────┬──────────┘
                         │                         │
               ┌─────────▼─────────────────────────▼──────────┐
               │          EKS Cluster: model-management       │
               │                                               │
               │  ┌─────────────────────┐ ┌─────────────────┐ │
               │  │  whisper-large-v3-  │ │   omni-voice    │ │
               │  │  turbo (ASR)        │ │   (TTS)         │ │
               │  │  Ray Serve + vLLM   │ │   Ray Serve +   │ │
               │  │  GPU: L4/A10G/L40S  │ │   OmniVoice     │ │
               │  └─────────────────────┘ └─────────────────┘ │
               │                                               │
               │  Karpenter GPU NodePools (on-demand g6e/g6/g5)│
               │  KubeRay Operator │ NVIDIA Device Plugin      │
               │  CloudWatch Container Insights                │
               └───────────────────────────────────────────────┘
```

---

## Pipeline Flow

The voice pipeline processes audio through five stages in a streaming, low-latency cascade. STT and TTS backends are swappable via environment variables:

```
User speaks → Silero VAD (speech boundary detection)
            → STT  (cohere-transcribe | whisper-large-v3-turbo, streaming SSE)
            → Amazon Bedrock (LLM, streaming)
            → TTS  (qwen3-tts | omni-voice, streaming 24 kHz PCM)
            → User hears response via WebRTC
```

| Stage | Service | Latency | Runs On |
|---|---|---|---|
| VAD | Silero VAD | ~10 ms | Local (voice-pipeline) |
| STT | Cohere Transcribe / Whisper Large-v3-Turbo | ~1–2 s | EKS GPU worker |
| LLM | Amazon Bedrock | ~1–1.5 s TTFB | AWS managed |
| TTS | Qwen3 TTS / OmniVoice | ~0.9–2.8 s | EKS GPU worker |

Audio input is 16 kHz mono; audio output is 24 kHz 16-bit signed PCM mono. The pipeline uses Pipecat for orchestration with context aggregation, metrics, and interruption handling. Silero VAD is attached to the LLM user aggregator (not the transport), which drives segmentation for the `SegmentedSTTService` wrappers.

---

## Components

### [infrastructure/](infrastructure/)

Complete IaC to provision the `model-management` EKS cluster from scratch. Includes eksctl cluster config (EKS 1.32, system node group), Karpenter for GPU node auto-provisioning, KubeRay operator for Ray Serve lifecycle, NVIDIA device plugin for GPU scheduling, and CloudWatch Container Insights for observability. One-command setup and teardown scripts.

### [cohere-transcribe/](cohere-transcribe/)

Self-hosted **Speech-to-Text** service running [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (pinned model revision) on Ray Serve. vLLM 0.19 runs as a subprocess on port 8100 and is proxied via `httpx.AsyncClient`. Accepts audio via multipart form upload or base64 JSON. Supports both streaming (SSE) and non-streaming transcription. 50 MB upload cap. Deployed as a RayService with 1–4 autoscaling GPU workers on Karpenter-provisioned g6e/g6/g5 instances (L40S/L4/A10G GPUs).

### [whisper-large-v3-turbo/](whisper-large-v3-turbo/)

Self-hosted **Speech-to-Text** service running [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) (809M parameters) on Ray Serve with vLLM as a subprocess proxy. Supports 99 languages, transcription and translation to English, and true token-level streaming via SSE. ~200x real-time factor with ~1s time-to-first-token. Deployed as a RayService with 1–4 autoscaling GPU workers.

### [qwen3-tts/](qwen3-tts/)

Self-hosted **Text-to-Speech** service running [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) (1.7B parameters) on Ray Serve with vllm-omni 0.18 as a subprocess proxy. Supports streaming PCM output, multiple audio formats (WAV, MP3, FLAC, AAC, Opus), WebSocket streaming, and custom voice cloning via reference audio upload. Deployed as a RayService with 1–4 autoscaling GPU workers on Karpenter-provisioned g6e/g6/g5 instances (L40S/L4/A10G GPUs).

### [omni-voice/](omni-voice/)

Self-hosted **Text-to-Speech** service running [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) — a multilingual zero-shot TTS model supporting 600+ languages. Runs in-process via the OmniVoice Python library (no subprocess). Supports three generation modes: zero-shot voice cloning (reference audio), voice design (attribute-based), and auto voice. Outputs WAV or PCM at 24kHz. Deployed as a RayService with 1–4 autoscaling GPU workers.

### [kokoro-tts/](kokoro-tts/)

Self-hosted **Text-to-Speech** service running [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — an 82-million-parameter StyleTTS2/ISTFTNet model with Apache-2.0 license. Runs in-process via the `kokoro` PyTorch library (no subprocess, no vLLM). 9 languages, 54 preset voices, 24 kHz PCM output. True sentence-level PCM streaming via `KPipeline`'s generator (advanced via `run_in_executor` to keep the event loop free). Each replica requests `num_gpus=0.5`, so two replicas share a single L4/A10G — lowest $/char of any TTS in this repo. Autoscales 1–6 replicas on `g6/g5` instances.

### [voice-pipeline/](voice-pipeline/)

Local **real-time voice pipeline** built on [Pipecat](https://github.com/pipecat-ai/pipecat) (>= 1.0.0). Connects the browser via SmallWebRTC transport, runs Silero VAD on the LLM user aggregator, calls the self-hosted STT and TTS services over HTTP, and uses Amazon Bedrock for conversational LLM responses. Ships Pipecat wrappers for all five backends — [cohere_stt.py](voice-pipeline/services/cohere_stt.py) and [whisper_stt.py](voice-pipeline/services/whisper_stt.py) (both subclass `SegmentedSTTService`), plus [qwen3_tts.py](voice-pipeline/services/qwen3_tts.py), [omnivoice_tts.py](voice-pipeline/services/omnivoice_tts.py), and [kokoro_tts.py](voice-pipeline/services/kokoro_tts.py) (all subclass `TTSService`). Provider is selected via `STT_PROVIDER` and `TTS_PROVIDER` env vars.

---

## Models

| Model | Task | Parameters | Inference Engine | GPU Types |
|---|---|---|---|---|
| [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) | Speech-to-Text | — | vLLM 0.19 (subprocess) | L40S / L4 / A10G |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | Speech-to-Text | 809M | vLLM 0.19 (subprocess) | L4 / A10G / L40S |
| [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | Text-to-Speech | 1.7B | vllm-omni 0.18 (subprocess) | L40S / L4 / A10G |
| [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) | Text-to-Speech | 0.6B | In-process (OmniVoice library) | L40S / L4 / A10G |
| [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Text-to-Speech | 82M | In-process (`kokoro` + PyTorch) | L4 / A10G |
| Amazon Bedrock | Conversational LLM | — | Bedrock (managed) | — |

---

## Infrastructure Overview

The platform runs on a single EKS cluster (`model-management`, `us-east-1`) with a separation between system workloads and GPU model serving:

| Layer | Technology | Purpose |
|---|---|---|
| **Cluster** | EKS 1.32 via eksctl | Managed Kubernetes control plane |
| **System Nodes** | m5.xlarge (2–4) | CoreDNS, kube-proxy, Karpenter, KubeRay operator |
| **GPU Nodes** | g6e/g6/g5 (on-demand) | Model inference workers, provisioned by Karpenter |
| **Node Scaling** | Karpenter 1.3.0 | Provisions GPU nodes on demand, consolidates when idle |
| **Model Serving** | Ray Serve 2.52.0 + KubeRay 1.3.0 | Autoscaling model replicas (1–4 per service) |
| **GPU Scheduling** | NVIDIA Device Plugin | Exposes `nvidia.com/gpu` as a K8s resource |
| **Observability** | CloudWatch Container Insights | Cluster metrics, application logs, performance data |
| **Container Registry** | ECR | Docker images for model services |

GPU nodes are tainted (`nvidia.com/gpu=true:NoSchedule`) so only model workers with matching tolerations are scheduled on them. Karpenter consolidates idle GPU nodes after 5 minutes, minimizing cost when no inference workload is running.

---

## Security

- All containers run as **non-root** (UID 1000) with all Linux capabilities dropped
- No privilege escalation allowed in any pod
- HuggingFace tokens stored as **Kubernetes Secrets**, never baked into images
- AWS credentials use IAM Roles for Service Accounts (IRSA), not static keys
- ECR image scanning enabled on all repositories
- Each service has a dedicated ServiceAccount (least-privilege)

---

## Project Structure

```
voice-ai/
├── README.md
├── CLAUDE.md
├── .gitignore
│
├── infrastructure/                     # EKS cluster & platform setup
│   ├── setup-cluster.sh                #   One-command full provisioning
│   ├── teardown-cluster.sh             #   One-command full teardown
│   ├── eks/cluster-config.yaml         #   eksctl cluster definition
│   ├── karpenter/                      #   Karpenter IAM, Helm, EC2NodeClass
│   ├── kuberay/                        #   KubeRay operator Helm install
│   ├── nvidia/                         #   NVIDIA device plugin
│   └── monitoring/                     #   CloudWatch + Prometheus
│
├── cohere-transcribe/                  # Speech-to-Text service (EKS)
│   ├── serve_app.py                    #   Ray Serve inference handler
│   ├── deploy.sh                       #   ECR + Docker + K8s deployment
│   ├── ray-service-transcribe.yaml     #   RayService manifest
│   ├── karpenter-gpu-nodepool.yaml     #   GPU NodePool definition
│   ├── Dockerfile
│   └── requirements.txt
│
├── whisper-large-v3-turbo/             # Speech-to-Text service (EKS)
│   ├── serve_app.py                    #   Ray Serve + vLLM subprocess proxy
│   ├── deploy.sh                       #   ECR + Docker + K8s deployment
│   ├── ray-service-whisper.yaml        #   RayService manifest
│   ├── karpenter-gpu-nodepool.yaml     #   GPU NodePool definition
│   ├── Dockerfile
│   └── requirements.txt
│
├── qwen3-tts/                          # Text-to-Speech service (EKS)
│   ├── serve_app.py                    #   Ray Serve inference handler
│   ├── deploy.sh                       #   ECR + Docker + K8s deployment
│   ├── ray-service-tts.yaml            #   RayService manifest
│   ├── karpenter-gpu-nodepool.yaml     #   GPU NodePool definition
│   ├── Dockerfile
│   └── requirements.txt
│
├── omni-voice/                         # Text-to-Speech service (EKS)
│   ├── serve_app.py                    #   Ray Serve + in-process OmniVoice
│   ├── deploy.sh                       #   ECR + Docker + K8s deployment
│   ├── ray-service-omnivoice.yaml      #   RayService manifest
│   ├── karpenter-gpu-nodepool.yaml     #   GPU NodePool definition
│   ├── Dockerfile
│   └── requirements.txt
│
├── kokoro-tts/                         # Text-to-Speech service (EKS)
│   ├── serve_app.py                    #   Ray Serve + in-process Kokoro-82M
│   ├── deploy.sh                       #   ECR + Docker + K8s deployment
│   ├── ray-service-kokoro.yaml         #   RayService manifest (num_gpus=0.5)
│   ├── karpenter-gpu-nodepool.yaml     #   GPU NodePool definition (L4/A10G only)
│   ├── Dockerfile                      #   Includes espeak-ng system dep
│   └── requirements.txt
│
└── voice-pipeline/                     # Real-time voice pipeline (local)
    ├── bot.py                          #   Pipecat pipeline entry point
    ├── pyproject.toml                  #   Python dependencies
    ├── .env.example                    #   Environment config template
    └── services/
        ├── cohere_stt.py               #   Pipecat STT wrapper (Cohere Transcribe)
        ├── whisper_stt.py              #   Pipecat STT wrapper (Whisper)
        ├── qwen3_tts.py                #   Pipecat TTS wrapper (Qwen3)
        ├── omnivoice_tts.py            #   Pipecat TTS wrapper (OmniVoice)
        └── kokoro_tts.py               #   Pipecat TTS wrapper (Kokoro-82M)
```

## Local Service Ports

The voice pipeline expects each model service to be reachable on localhost. Typical setup: `kubectl port-forward` each enabled backend once, then run `python bot.py`.

| Service | Default local port | Env var |
|---|---|---|
| cohere-transcribe | 8000 | `COHERE_STT_URL` |
| qwen3-tts | 8001 | `QWEN3_TTS_URL` |
| omni-voice | 8002 | `OMNIVOICE_TTS_URL` |
| whisper-large-v3-turbo | 8003 | `WHISPER_STT_URL` |
| kokoro-tts | 8004 | `KOKORO_TTS_URL` |

---

## Prerequisites

| Tool | Purpose |
|---|---|
| AWS CLI v2 | AWS resource management |
| eksctl >= 0.200.0 | EKS cluster provisioning |
| kubectl >= 1.32 | Kubernetes operations |
| helm >= 3.16 | Karpenter, KubeRay, NVIDIA plugin installation |
| docker | Container image builds |
| Python >= 3.11 | Voice pipeline (local) |
| uv | Python dependency management (recommended) |

**AWS services used:** EKS, EC2 (GPU instances), ECR, IAM, Bedrock, CloudWatch

**HuggingFace access:** A valid `HF_TOKEN` with access to the model repositories is required for deployment.

---

## License

Private repository.
