"""Ray Serve deployment for openai/whisper-large-v3-turbo — fast multilingual ASR via vLLM."""

import logging
import subprocess
import time

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from ray import serve

logger = logging.getLogger("ray.serve")

MODEL_ID = "openai/whisper-large-v3-turbo"
VLLM_PORT = 8100
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"

MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MB


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 2},
    max_ongoing_requests=10,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 5,
    },
)
class WhisperLargeV3Turbo:
    def __init__(self):
        logger.info(f"Starting vLLM serve for {MODEL_ID} on port {VLLM_PORT}...")

        self.process = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", MODEL_ID,
                "--dtype", "auto",
                "--max-model-len", "448",
                "--max-num-seqs", "400",
                "--gpu-memory-utilization", "0.85",
                "--port", str(VLLM_PORT),
                "--limit-mm-per-prompt", '{"audio": 1}',
                "--kv-cache-dtype", "fp8",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._wait_for_server()
        self.client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=120.0)
        logger.info("vLLM OpenAI API server ready")

    def _wait_for_server(self, timeout: int = 300):
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                output = self.process.stdout.read().decode() if self.process.stdout else ""
                raise RuntimeError(
                    f"vLLM process exited early (code {self.process.returncode}): {output[-2000:]}"
                )
            try:
                resp = urllib.request.urlopen(f"{VLLM_BASE_URL}/health", timeout=2)
                if resp.status == 200:
                    return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(f"vLLM server did not start within {timeout}s")

    def check_health(self):
        if self.process.poll() is not None:
            raise RuntimeError("vLLM subprocess is not running")

    async def __call__(self, request: Request):
        content_type = request.headers.get("content-type", "")

        try:
            if "multipart/form-data" in content_type:
                form = await request.form()
                audio_file = form.get("file")
                if audio_file is None:
                    return JSONResponse(
                        {"error": "No 'file' field in form data"}, status_code=400
                    )
                audio_bytes = await audio_file.read()
                filename = getattr(audio_file, "filename", "audio.wav") or "audio.wav"
                language = form.get("language", "en")
                task = form.get("task")
            elif "application/json" in content_type:
                import base64
                body = await request.json()
                if "audio" not in body:
                    return JSONResponse(
                        {"error": "Missing 'audio' field (base64-encoded)"},
                        status_code=400,
                    )
                try:
                    audio_bytes = base64.b64decode(body["audio"])
                except Exception:
                    return JSONResponse(
                        {"error": "Invalid base64 in 'audio' field"}, status_code=400
                    )
                filename = body.get("filename", "audio.wav")
                language = body.get("language", "en")
                task = body.get("task")
            else:
                return JSONResponse(
                    {"error": "Content-Type must be multipart/form-data or application/json"},
                    status_code=400,
                )

            if len(audio_bytes) > MAX_AUDIO_BYTES:
                return JSONResponse(
                    {"error": f"Audio exceeds {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit"},
                    status_code=413,
                )

            stream = request.query_params.get("stream", "").lower() == "true"

            data = {
                "model": MODEL_ID,
                "language": language,
                "response_format": "json",
            }
            if task == "translate":
                data["task"] = "translate"

            if stream:
                data["stream"] = "true"
                return StreamingResponse(
                    self._stream_transcribe(audio_bytes, filename, data),
                    media_type="text/event-stream",
                )
            else:
                resp = await self.client.post(
                    "/v1/audio/transcriptions",
                    files={"file": (filename, audio_bytes)},
                    data=data,
                )
                if resp.status_code != 200:
                    logger.error(f"vLLM returned {resp.status_code}: {resp.text}")
                    return JSONResponse(
                        {"error": "Transcription failed"}, status_code=502
                    )
                return JSONResponse(resp.json())

        except Exception:
            logger.exception("Transcription request failed")
            return JSONResponse(
                {"error": "Internal transcription error"}, status_code=500
            )

    async def _stream_transcribe(self, audio_bytes: bytes, filename: str, data: dict):
        async with self.client.stream(
            "POST",
            "/v1/audio/transcriptions",
            files={"file": (filename, audio_bytes)},
            data=data,
        ) as resp:
            if resp.status_code != 200:
                yield f"data: {{\"error\": \"vLLM returned {resp.status_code}\"}}\n\n"
                return
            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"
        yield "data: [DONE]\n\n"

    def __del__(self):
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()


app = WhisperLargeV3Turbo.bind()
