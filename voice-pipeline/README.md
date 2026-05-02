# Voice Pipeline — Real-Time Voice AI with Pipecat

A real-time **voice-to-voice AI assistant** that connects self-hosted STT and TTS models with Amazon Bedrock via [Pipecat](https://github.com/pipecat-ai/pipecat). Runs locally with a WebRTC browser UI — speak into your mic, get spoken responses back.

Two STT backends and three TTS backends are swappable via environment variables.

---

## Architecture

```
Browser (WebRTC) ──▶ Pipecat Pipeline ──▶ Browser (WebRTC)

Pipeline flow:
  Mic ─▶ Transport ─▶ VAD (Silero) ─▶ STT ─▶ User Agg ─▶ Bedrock LLM ─▶ TTS ─▶ Transport ─▶ Speaker
                                                                                  │
                                                                      Asst Agg ◀──┘
```

| Stage | What it is | Configurable via |
|---|---|---|
| **Transport** | SmallWebRTC — browser ↔ server over WebRTC, includes prebuilt UI | — |
| **VAD** | `VADProcessor(SileroVADAnalyzer(sample_rate=16000))` — emits `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame` so the segmented STT knows when to package audio | — |
| **STT** | Cohere Transcribe or Whisper Large-v3-Turbo | `STT_PROVIDER` env |
| **LLM** | Amazon Bedrock (model id via env) | `BEDROCK_MODEL_ID` env |
| **TTS** | Qwen3 TTS, OmniVoice, or Kokoro-82M | `TTS_PROVIDER` env |
| **Orchestration** | Pipecat ≥ 1.0 `Pipeline` + `LLMContextAggregatorPair` for conversation state | — |

Audio in is **16 kHz mono**, audio out is **24 kHz 16-bit mono PCM**.

---

## How it works

1. **User speaks** → browser captures mic and streams via WebRTC to the server.
2. **VADProcessor** (first stage after `transport.input()`) runs Silero VAD on the PCM frames and emits speech-start / speech-stop frames into the pipeline.
3. **SegmentedSTTService** (Cohere or Whisper wrapper) buffers audio between the VAD events, packages it as WAV, POSTs it to its backend with `?stream=true`, and parses the SSE chunks into a single `TranscriptionFrame`.
4. **LLMContextAggregatorPair.user()** accumulates the transcription into `LLMContext` and pushes it to the LLM.
5. **AWSBedrockLLMService** streams a conversational response.
6. **TTSService** (Qwen3 / OmniVoice / Kokoro wrapper) turns each response chunk into 24 kHz PCM and yields `TTSAudioRawFrame`s.
7. **transport.output()** plays PCM back to the browser over WebRTC; **LLMContextAggregatorPair.assistant()** records the bot's utterance into the context for the next turn.

On first connect, the assistant automatically speaks a greeting (`on_client_ready` handler queues an `LLMMessagesAppendFrame` with `run_llm=True`).

---

## File structure

```
voice-pipeline/
├── bot.py                      # Pipeline wiring + provider dispatch + runner entrypoint
├── pyproject.toml              # uv / pip project metadata
├── .env.example                # Environment variables template
├── .env                        # Your local config (git-ignored)
└── services/
    ├── __init__.py
    ├── cohere_stt.py           # Pipecat SegmentedSTTService — cohere-transcribe backend
    ├── whisper_stt.py          # Pipecat SegmentedSTTService — whisper-large-v3-turbo backend
    ├── qwen3_tts.py            # Pipecat TTSService — qwen3-tts backend (PCM streaming)
    ├── omnivoice_tts.py        # Pipecat TTSService — omni-voice backend (PCM chunking)
    └── kokoro_tts.py           # Pipecat TTSService — kokoro-tts backend (sentence-level PCM)
```

---

## Prerequisites

- **Python** ≥ 3.11 (3.12 tested)
- **uv** recommended (`pip install uv`)
- **AWS credentials** with Bedrock access in `AWS_REGION` (default `us-east-1`)
- At least one STT backend and one TTS backend reachable on localhost (either a local dev container or a `kubectl port-forward` from the EKS cluster)
- **Chrome / Edge / Firefox** with mic permission

---

## Setup

### 1. Install dependencies

```bash
cd voice-pipeline
uv sync                     # recommended; creates .venv and installs everything
# or: python -m venv .venv && source .venv/bin/activate && pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — the dispatch is driven entirely by `STT_PROVIDER` and `TTS_PROVIDER`:

```env
# Provider selection
STT_PROVIDER=cohere          # cohere | whisper
TTS_PROVIDER=qwen3           # qwen3 | omnivoice | kokoro

# Per-backend URLs
COHERE_STT_URL=http://localhost:8000
WHISPER_STT_URL=http://localhost:8003
QWEN3_TTS_URL=http://localhost:8001
OMNIVOICE_TTS_URL=http://localhost:8002
KOKORO_TTS_URL=http://localhost:8004

# LLM
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=<your-bedrock-model-id>
```

### 3. Port-forward the backends you've chosen

Each EKS RayService exposes its ClusterIP Service at port 8000 inside the cluster. Use a **distinct local port** per backend so they can coexist:

| Backend | `kubectl port-forward` |
|---|---|
| cohere-transcribe | `kubectl port-forward svc/cohere-transcribe-serve-svc 8000:8000` |
| qwen3-tts | `kubectl port-forward svc/qwen3-tts-serve-svc 8001:8000` |
| omni-voice | `kubectl port-forward svc/omni-voice-service-serve-svc 8002:8000` |
| whisper | `kubectl port-forward svc/whisper-v3-turbo-service-serve-svc 8003:8000` |
| kokoro-tts | `kubectl port-forward svc/kokoro-tts-serve-svc 8004:8000` |

Only the providers you actually selected need to be forwarded.

### 4. Run the pipeline

```bash
uv run python bot.py
# or (with .venv activated): python bot.py
```

Server binds to **http://localhost:7860**. Open `http://localhost:7860/client` in Chrome, grant mic access, click Connect.

On first connection you should see:

```
__main__:create_stt - Using Cohere Transcribe STT     # or Whisper
__main__:create_tts - Using Qwen3 TTS                 # or OmniVoice / Kokoro-82M
pipecat.processors.audio.vad_processor - User started speaking
services.{cohere,whisper}_stt:run_stt - ...
services.{qwen3,omnivoice,kokoro}_tts:run_tts - Generating TTS [...]
```

---

## Backend dispatch

### STT — `create_stt()` in [bot.py](bot.py)

| `STT_PROVIDER` | Wrapper | Backend service | Default URL |
|---|---|---|---|
| `cohere` (default) | `CohereTranscribeSTTService` | `cohere-transcribe/` | `http://localhost:8000` |
| `whisper` | `WhisperSTTService` | `whisper-large-v3-turbo/` | `http://localhost:8003` |

Both wrappers subclass `SegmentedSTTService` and accumulate `choices[0].delta.content` from the SSE stream into a single `TranscriptionFrame`.

### TTS — `create_tts()` in [bot.py](bot.py)

| `TTS_PROVIDER` | Wrapper | Backend service | Default URL |
|---|---|---|---|
| `qwen3` (default) | `Qwen3TTSService` | `qwen3-tts/` | `http://localhost:8001` |
| `omnivoice` | `OmniVoiceTTSService` | `omni-voice/` | `http://localhost:8002` |
| `kokoro` | `KokoroTTSService` | `kokoro-tts/` | `http://localhost:8004` |

All three TTS wrappers stream PCM and handle odd-byte boundaries with a `leftover` buffer so no half-sample audio is emitted.

---

## Configuration reference

### Audio

| Setting | Value | Configured in |
|---|---|---|
| Input sample rate | 16,000 Hz | `PipelineParams.audio_in_sample_rate` |
| Output sample rate | 24,000 Hz | `PipelineParams.audio_out_sample_rate` |
| VAD | Silero VAD (16 kHz) | `VADProcessor` stage in `Pipeline` |

### LLM

| Setting | Value | Env |
|---|---|---|
| Provider | Amazon Bedrock | `AWS_REGION` |
| Model | — | `BEDROCK_MODEL_ID` |
| Max tokens | 300 | (hardcoded) |
| Temperature | 0.7 | (hardcoded) |
| Top-p | 0.9 | (hardcoded) |

### TTS env vars (by provider)

**Qwen3 TTS**
| Var | Default |
|---|---|
| `QWEN3_TTS_URL` | `http://localhost:8001` |
| `TTS_VOICE` | `vivian` |
| `TTS_LANGUAGE` | `Auto` |

**OmniVoice**
| Var | Default |
|---|---|
| `OMNIVOICE_TTS_URL` | `http://localhost:8002` |
| `OMNIVOICE_INSTRUCT` | `""` |
| `OMNIVOICE_SPEED` | `1.0` |

**Kokoro-82M**
| Var | Default | Notes |
|---|---|---|
| `KOKORO_TTS_URL` | `http://localhost:8004` | |
| `KOKORO_VOICE` | `af_heart` | Any of the 54 preset voices |
| `KOKORO_LANG_CODE` | `""` (inferred) | Falls back to voice prefix (`a`=en-US, `b`=en-GB, `j`=ja, `z`=zh, `e`=es, `f`=fr, `h`=hi, `i`=it, `p`=pt-BR) |
| `KOKORO_SPEED` | `1.0` | 0.5–2.0 |

---

## Pipeline internals

### Why `VADProcessor` as a pipeline stage (not on the transport)

Pipecat 1.0's `TransportParams` does **not** accept a `vad_analyzer` field — passing it silently no-ops. The canonical pattern is:

```python
vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))

pipeline = Pipeline([
    transport.input(),
    vad,              # emits VADUserStartedSpeakingFrame / VADUserStoppedSpeakingFrame
    stt,              # SegmentedSTTService buffers audio between those frames
    context_pair.user(),
    llm,
    tts,
    transport.output(),
    context_pair.assistant(),
])
```

Without the `VADProcessor` stage in front of STT, `SegmentedSTTService` never receives speech-boundary frames and never calls `run_stt`, so you'd get no transcription even though audio reaches the server.

### Custom service wrappers

All five are thin HTTP clients over `httpx.AsyncClient`:

- **STT wrappers** (subclass `SegmentedSTTService`) — `run_stt(audio: bytes)` POSTs a WAV to `/?stream=true`, parses SSE `choices[0].delta.content`, yields one `TranscriptionFrame`.
- **TTS wrappers** (subclass `TTSService`) — `run_tts(text)` POSTs JSON to `/v1/audio/speech` with `stream=true, response_format=pcm`, yields `TTSAudioRawFrame` chunks at 24 kHz mono.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pipecat-ai[aws,silero,webrtc,runner]` | ≥ 1.0.0 | Pipeline framework + Bedrock + Silero VAD + WebRTC + runner |
| `pipecat-ai-small-webrtc-prebuilt` | ≥ 2.0.0 | Prebuilt browser UI served from `/client` |
| `httpx` | ~= 0.27.0 | Async HTTP client for STT/TTS backend calls |
| `python-dotenv` | latest | `.env` loading |
| `loguru` | latest | Structured logging |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| UI shows "Agent: connecting" forever | Pipeline crashed during init | Check terminal for Python traceback |
| Greeting plays but mic input never triggers STT | `VADProcessor` missing from pipeline | Confirm `bot.py` has `vad` stage between `transport.input()` and `stt` |
| `OmniVoice TTS error: All connection attempts failed` | Selected provider's port-forward isn't running | Check `TTS_PROVIDER` in `.env` and start the matching `kubectl port-forward` |
| Bot loads wrong provider after editing `.env` | Stale `bot.py` process or stale `__pycache__` | `pkill -f "python.*bot.py"`, delete `__pycache__`, relaunch |
| Multiple `.venv` directories on disk cause wrong interpreter | `VIRTUAL_ENV` leaked from another project | Run bot with an explicit absolute Python path: `/abs/path/.venv/bin/python bot.py` |
| `STTSettings/TTSSettings: fields are NOT_GIVEN` | Warning only — these wrappers don't use Pipecat's built-in settings | Safe to ignore |
| `AVFFrameReceiver`/`AVFAudioReceiver` duplicate class warning | `av` and `cv2` bundle different `libavdevice` | Cosmetic, no functional impact |
| AWS Bedrock `AccessDeniedException` | Missing IAM permission or wrong region | Confirm IAM user/role has `bedrock:InvokeModel` and `AWS_REGION` matches model availability |
