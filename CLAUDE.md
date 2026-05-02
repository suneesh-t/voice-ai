# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repo contains independently deployable ML model services for voice AI workloads, each deployed to an EKS cluster (`model-management` in `us-east-1`) via Ray Serve with GPU-backed Karpenter node pools. It also includes a real-time voice pipeline that orchestrates these services into a conversational AI assistant.

## Service Architecture

Each service directory is a self-contained deployment unit with the same structure:

- `serve_app.py` — Ray Serve application (the inference handler)
- `requirements.txt` — Python dependencies
- `Dockerfile` — Based on `rayproject/ray:2.52.0-py311-cu124`
- `ray-service-*.yaml` — RayService + ServiceAccount manifests (image placeholders `<AWS_ACCOUNT_ID>` and `<IMAGE_TAG>` are substituted at deploy time)
- `karpenter-gpu-nodepool.yaml` — Karpenter NodePool for GPU node provisioning
- `deploy.sh` — End-to-end deploy script (ECR repo → Docker build/push → kubeconfig → Karpenter NodePool → HF token secret → RayService apply)

## Key Differences Between Services

| | cohere-transcribe | whisper-large-v3-turbo | qwen3-tts | omni-voice | kokoro-tts |
|---|---|---|---|---|---|
| Model | `CohereLabs/cohere-transcribe-03-2026` | `openai/whisper-large-v3-turbo` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `k2-fsa/OmniVoice` | `hexgrad/Kokoro-82M` |
| Task | Speech-to-Text | Speech-to-Text | Text-to-Speech | Text-to-Speech | Text-to-Speech |
| Inference engine | vLLM as **subprocess** (proxied via httpx) | vLLM as **subprocess** (proxied via httpx) | vllm-omni as **subprocess** (proxied via httpx) | OmniVoice **in-process** (Python library) | Kokoro/StyleTTS2 **in-process** (PyTorch) |
| GPU instances | g6e/g6/g5 (L40S/L4/A10G) | g6/g5 (L4/A10G) | g6e/g6/g5 (L40S/L4/A10G) | g6e/g6/g5 (L40S/L4/A10G) | g6/g5 (L4/A10G) |
| Streaming | SSE (native vLLM) | SSE (native vLLM, true token-level) | PCM streaming (native) + WebSocket | PCM chunking (post-generation, 100 ms chunks) | PCM streaming (sentence-chunked via `split_pattern`) |
| GPU share per replica | 1 | 1 | 1 | 1 | **0.5** (82 M params) |
| Local port (pipeline) | 8000 | 8003 | 8001 | 8002 | 8004 |

## Deployment

Each service is deployed independently via its `deploy.sh`. Requires:
- `aws` CLI configured, `kubectl` pointing at `model-management` cluster, Docker running
- `HF_TOKEN` env var set (used to create a shared `hf-token` Kubernetes secret)
- Docker builds must use `--platform linux/amd64` (deploy scripts handle this)

```bash
cd <service-dir>
export HF_TOKEN=hf_xxxxx
bash deploy.sh
```

## Conventions

- All services use Ray Serve autoscaling (1–4 replicas) with GPU workers and a CPU-only head node.
- Docker images are tagged with `<git-short-sha>-<timestamp>`.
- All containers run as non-root (UID 1000) with dropped capabilities.
- The HF token Kubernetes secret is shared across services (same name `hf-token` in `default` namespace), mounted as `HUGGING_FACE_HUB_TOKEN` on workers only.
- All Karpenter NodePools require a `nodeClassRef` pointing to the `default` EC2NodeClass (defined in `infrastructure/karpenter/ec2-nodeclass.yaml`).
- vLLM subprocess pattern (used by cohere-transcribe, whisper, and qwen3-tts): spawns vLLM / vllm-omni on port 8100, proxies via `httpx.AsyncClient`. `check_health` re-raises if the subprocess died so Ray restarts the replica.
- Ray Serve deployment defaults: 1 GPU / 2 CPU per replica, `max_ongoing_requests=10`, `target_ongoing_requests=5`. OmniVoice uses tighter limits (8 / 3) since it runs the model in-process. Kokoro uses `num_gpus=0.5` to pack 2 replicas per physical GPU — the 82 M model leaves plenty of HBM headroom.
- Kokoro's blocking `KPipeline` generator is advanced via `run_in_executor(None, next)` so the event loop stays free — this gives true sentence-level PCM streaming instead of post-generation chunking.
- 50 MB audio upload cap on STT services; both `multipart/form-data` and `application/json` (base64 audio) request bodies are accepted.

## Voice Pipeline ([voice-pipeline/](voice-pipeline/))

Pipecat-based (>= 1.0) real-time pipeline with SmallWebRTC transport. Audio in 16 kHz, audio out 24 kHz.

- Pipeline graph: `transport.input() → stt → context_pair.user() → llm → tts → transport.output() → context_pair.assistant()`
- VAD: `SileroVADAnalyzer(sample_rate=16000)` attached to the **user aggregator** (not the transport) — it drives segmentation for `SegmentedSTTService`.
- STT provider is selected via `STT_PROVIDER` env (`cohere` default, `whisper`). Both wrappers accumulate SSE `choices[0].delta.content` into a single `TranscriptionFrame`.
- TTS provider is selected via `TTS_PROVIDER` env (`qwen3` default, `omnivoice`, `kokoro`). All three wrappers handle the odd-byte boundary (`leftover = raw[usable:]`) so no half-sample PCM is emitted.
- LLM: `AWSBedrockLLMService` using `BEDROCK_MODEL_ID` env. Fixed "1–3 sentences, conversational" system prompt.
- Greeting: on `on_client_ready`, an `LLMMessagesAppendFrame` is queued so the assistant speaks first.
- Dev loop: `kubectl port-forward` each enabled backend locally (cohere → 8000, qwen3-tts → 8001, omni-voice → 8002, whisper → 8003, kokoro-tts → 8004), then `python bot.py`.

## Infrastructure ([infrastructure/](infrastructure/))

`setup-cluster.sh` bootstraps everything in order: `eksctl` cluster (EKS 1.32, m5.xlarge system nodegroup 2–4) → NVIDIA device plugin → Karpenter v1.3.0 (CloudFormation IAM, subnet/SG tagging, IRSA, instance profile, Helm install) → `EC2NodeClass` + cohere + qwen3 NodePools → KubeRay operator v1.3.0 → CloudWatch Container Insights.

Caveats to be aware of when editing infra:
- Whisper, omni-voice, and kokoro-tts NodePools are **not** applied by `setup-cluster.sh` — they are applied by each service's own `deploy.sh`.
- GPU nodes carry the taint `nvidia.com/gpu=true:NoSchedule` and label `ray.io/node-type: gpu-worker`; workers tolerate the taint and nodeSelect the label.
- Karpenter consolidates idle GPU nodes after 300 s (`WhenEmptyOrUnderutilized`) with a 10 % disruption budget; `expireAfter: 336h`.

## Known documentation drift

- Old docs sometimes describe cohere-transcribe as running vLLM **in-process** (`LLM` class). The actual code uses the subprocess + httpx proxy pattern like whisper. Keep docs and code in sync if you touch either.
