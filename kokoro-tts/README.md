# kokoro-tts

Self-hosted **Text-to-Speech** service running [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on Ray Serve.

Kokoro is an 82M-parameter StyleTTS2 + ISTFTNet model released under Apache-2.0. It is much smaller than the other TTS models in this repo (Qwen3-TTS 1.7B, OmniVoice 0.6B) while scoring near the top of TTS Arena, which lets us run multiple replicas on a single GPU and serve traffic at a significantly lower $/char.

---

## Why not vLLM?

vLLM is an inference engine for autoregressive transformer LMs. Kokoro is a non-autoregressive StyleTTS2 model with an ISTFTNet GAN vocoder — paged attention and KV caching don't apply. The model runs **in-process** via the official `kokoro` PyTorch library, same pattern as [omni-voice/](../omni-voice/).

---

## Key properties

| | Value |
|---|---|
| Model | `hexgrad/Kokoro-82M` |
| Parameters | 82M |
| License | Apache-2.0 |
| Architecture | StyleTTS2 (decoder-only) + ISTFTNet vocoder |
| Sample rate | 24 kHz, 16-bit mono |
| Languages | 9 (American / British English, Japanese, Mandarin, Spanish, French, Hindi, Italian, Brazilian Portuguese) |
| Voices | 54 preset voices (see [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)) |
| GPU footprint (fp16) | < 1 GB |
| Inference throughput | ~35–100× realtime on L4/A10G |
| Voice cloning | ❌ (preset voices only; voices can be blended by averaging their tensors) |

---

## Deployment

```bash
export HF_TOKEN=hf_xxxxx       # reuses the shared hf-token secret
cd kokoro-tts
bash deploy.sh
```

`deploy.sh` does the usual: create ECR repo → `docker build --platform linux/amd64` → push → update kubeconfig → apply Karpenter `NodePool` → ensure `hf-token` secret → `kubectl apply` the substituted RayService.

Monitor:

```bash
kubectl get rayservice kokoro-tts -w
kubectl get pods -l ray.io/cluster -w
kubectl port-forward svc/kokoro-tts-serve-svc 8004:8000
```

---

## API

All endpoints speak OpenAI-compatible JSON. The service mounts at `/`.

### `POST /v1/audio/speech`

Request (JSON):

```json
{
  "input": "Hello from Kokoro.",
  "voice": "af_heart",
  "speed": 1.0,
  "lang_code": "a",
  "response_format": "wav",
  "stream": false,
  "split_pattern": "\\n+"
}
```

- `input` (aliased as `text`) — required, ≤ 5000 chars by default (`KOKORO_MAX_CHARS`).
- `voice` — any of the 56 preset voice IDs (e.g. `af_heart`, `bm_george`, `jf_alpha`).
- `lang_code` — optional; if omitted, inferred from the voice prefix (`a` = American English, `b` = British, `j` = Japanese, `z` = Mandarin, `e` = Spanish, `f` = French, `h` = Hindi, `i` = Italian, `p` = Brazilian Portuguese).
- `speed` — 0.5–2.0.
- `response_format` — `wav` | `pcm` | `flac` | `ogg`.
- `stream` — if `true` **and** `response_format=pcm`, the body streams raw PCM as sentences finish synthesizing. Otherwise the full encoded audio is returned in the response body.
- `split_pattern` — regex on which to chunk input. Defaults to `\n+`. Used to bound per-chunk latency in streaming mode.

### `GET /v1/audio/voices`

Returns `{ "voices": [ { "id": "af_heart", "lang_code": "a", "gender": "female" }, ... ] }`.

### `GET /health`

Returns `{ "status": "ok", "model": "hexgrad/Kokoro-82M", "device": "cuda", "loaded_langs": ["a"] }`.

### Examples

```bash
# Full-file WAV download
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Kokoro.","voice":"af_heart"}' \
  --output test.wav

# True sentence-level PCM streaming
curl -N -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Kokoro. This is a streaming test.","voice":"af_heart","stream":true,"response_format":"pcm"}' \
  --output test.pcm

# Japanese
curl -X POST http://localhost:8004/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"こんにちは","voice":"jf_alpha"}' \
  --output ja.wav
```

Convert raw PCM to WAV for inspection:

```bash
ffmpeg -f s16le -ar 24000 -ac 1 -i test.pcm test.wav
```

---

## Ray Serve shape

From [serve_app.py](serve_app.py):

- `ray_actor_options={"num_gpus": 0.5, "num_cpus": 2}` — **two replicas per physical GPU**, since the model is < 1 GB.
- `max_ongoing_requests=8`, autoscale 1–6 replicas, `target_ongoing_requests=4`.
- One `KPipeline` per requested language, lazily created on first use; preload list controlled by `KOKORO_PRELOAD_LANGS` (default `a`).
- Warmup synthesis at `__init__` so the first user request doesn't eat cuDNN autotune time.
- Blocking `KPipeline` generator is advanced via `loop.run_in_executor(None, next)` so streaming yields each sentence's PCM the moment it's ready without blocking the event loop.

---

## Karpenter NodePool

From [karpenter-gpu-nodepool.yaml](karpenter-gpu-nodepool.yaml):

- Instance types: `g6.xlarge`, `g6.2xlarge`, `g5.xlarge` (L4 / A10G only — L40S is wasted on an 82M model).
- On-demand, amd64, tainted `nvidia.com/gpu=true:NoSchedule`.
- NodePool cap: 24 vCPU / 96 Gi / **3 GPUs**.
- Consolidates after 300 s when empty/underutilized.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Voice used when request omits `voice`. |
| `KOKORO_DEFAULT_LANG` | `a` | Lang code fallback when neither `lang_code` nor `voice` implies one. |
| `KOKORO_PRELOAD_LANGS` | `a` | Comma-separated lang codes to instantiate at replica boot. |
| `KOKORO_MAX_CHARS` | `5000` | Reject requests with longer input. |
| `HF_HOME` | `/home/ray/.cache/huggingface` | HF cache root for voice tensor downloads. |

---

## Integration with the voice pipeline

The [voice-pipeline](../voice-pipeline) already has a Pipecat wrapper ([kokoro_tts.py](../voice-pipeline/services/kokoro_tts.py)). Switch the whole pipeline to Kokoro with:

```bash
# .env
TTS_PROVIDER=kokoro
KOKORO_TTS_URL=http://localhost:8004
KOKORO_VOICE=af_heart
KOKORO_SPEED=1.0
```

Then port-forward and run:

```bash
kubectl port-forward svc/kokoro-tts-serve-svc 8004:8000 &
cd voice-pipeline
python bot.py
```

---

## Notes and limitations

- **No voice cloning.** If you need custom-voice support, use [qwen3-tts/](../qwen3-tts/) or [omni-voice/](../omni-voice/) instead.
- **Preset voice quality varies.** Only `af_heart` is grade-A on the official voice card; many voices are C/D and have noticeable prosody artifacts on long text. Blending two voices (client-side tensor average) often smooths this.
- **G2P for non-English languages** uses `espeak-ng` via `misaki`, which can mispronounce loan words and numerals. Production use on those languages may benefit from a domain dictionary pass.
- **CC-BY voices** (e.g. `jf_gongitsune`, `ff_siwis`) require attribution if shipped.
- **Scam lookalikes.** The upstream card warns that `kokorottsai.com` and `kokorotts.net` are **not** affiliated with hexgrad.
