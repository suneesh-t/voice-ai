import logging
import subprocess
import time

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

from ray import serve

logger = logging.getLogger("ray.serve")

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
VLLM_PORT = 8100
VLLM_BASE_URL = f"http://127.0.0.1:{VLLM_PORT}"


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
class Qwen3TTSVLLM:
    def __init__(self):
        logger.info(f"Starting vllm-omni serve for {MODEL_ID} on port {VLLM_PORT}...")

        self.process = subprocess.Popen(
            [
                "vllm-omni", "serve", MODEL_ID,
                "--omni",
                "--host", "127.0.0.1",
                "--port", str(VLLM_PORT),
                "--gpu-memory-utilization", "0.9",
                "--trust-remote-code",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self._wait_for_server()
        self.client = httpx.AsyncClient(base_url=VLLM_BASE_URL, timeout=300.0)
        logger.info("vllm-omni OpenAI API server ready")

    def _wait_for_server(self, timeout: int = 600):
        import urllib.request
        import urllib.error

        start = time.time()
        while time.time() - start < timeout:
            if self.process.poll() is not None:
                output = self.process.stdout.read().decode() if self.process.stdout else ""
                raise RuntimeError(
                    f"vllm-omni process exited early (code {self.process.returncode}): {output[-2000:]}"
                )
            try:
                resp = urllib.request.urlopen(f"{VLLM_BASE_URL}/health", timeout=2)
                if resp.status == 200:
                    return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(2)
        raise TimeoutError(f"vllm-omni server did not start within {timeout}s")

    def check_health(self):
        if self.process.poll() is not None:
            raise RuntimeError("vllm-omni subprocess is not running")

    async def __call__(self, request: Request):
        path = request.url.path.rstrip("/")
        method = request.method

        try:
            # GET /v1/audio/voices — list available speakers
            if method == "GET" and path == "/v1/audio/voices":
                resp = await self.client.get("/v1/audio/voices")
                return JSONResponse(resp.json(), status_code=resp.status_code)

            # POST /v1/audio/voices — upload custom voice for cloning
            if method == "POST" and path == "/v1/audio/voices":
                form = await request.form()
                files = {}
                data = {}
                audio = form.get("audio_sample")
                if audio:
                    files["audio_sample"] = (
                        audio.filename,
                        await audio.read(),
                        audio.content_type,
                    )
                for key in ("consent", "name", "ref_text"):
                    val = form.get(key)
                    if val:
                        data[key] = val
                resp = await self.client.post("/v1/audio/voices", files=files, data=data)
                return JSONResponse(resp.json(), status_code=resp.status_code)

            # POST /v1/audio/speech — generate speech (streaming or non-streaming)
            if method == "POST" and path == "/v1/audio/speech":
                body = await request.json()
                stream = body.get("stream", False)

                if stream:
                    body.setdefault("response_format", "pcm")
                    return StreamingResponse(
                        self._stream_speech(body),
                        media_type="audio/pcm",
                        headers={
                            "X-Audio-Sample-Rate": "24000",
                            "X-Audio-Channels": "1",
                            "X-Audio-Bit-Depth": "16",
                        },
                    )

                resp = await self.client.post("/v1/audio/speech", json=body)
                if resp.status_code != 200:
                    logger.error(f"vllm-omni returned {resp.status_code}: {resp.text[:500]}")
                    return JSONResponse(
                        {"error": "TTS generation failed"}, status_code=502
                    )

                fmt = body.get("response_format", "wav")
                content_types = {
                    "wav": "audio/wav",
                    "mp3": "audio/mpeg",
                    "flac": "audio/flac",
                    "pcm": "audio/pcm",
                    "aac": "audio/aac",
                    "opus": "audio/opus",
                }
                return Response(
                    content=resp.content,
                    media_type=content_types.get(fmt, "application/octet-stream"),
                )

            return JSONResponse(
                {
                    "error": (
                        f"Unknown endpoint: {method} {path}. "
                        "Use /v1/audio/speech, /v1/audio/voices, "
                        "or ws:///v1/audio/speech/stream"
                    )
                },
                status_code=404,
            )

        except Exception:
            logger.exception("TTS request failed")
            return JSONResponse({"error": "Internal TTS error"}, status_code=500)

    async def _stream_speech(self, body: dict):
        async with self.client.stream(
            "POST",
            "/v1/audio/speech",
            json=body,
        ) as resp:
            if resp.status_code != 200:
                return
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk

    async def handle_websocket(self, ws: WebSocket):
        """Proxy WebSocket streaming TTS to vllm-omni's /v1/audio/speech/stream."""
        import websockets

        await ws.accept()
        backend_url = f"ws://127.0.0.1:{VLLM_PORT}/v1/audio/speech/stream"

        try:
            async with websockets.connect(backend_url) as backend_ws:

                async def client_to_backend():
                    try:
                        while True:
                            data = await ws.receive_text()
                            await backend_ws.send(data)
                    except WebSocketDisconnect:
                        await backend_ws.close()

                async def backend_to_client():
                    try:
                        async for message in backend_ws:
                            if isinstance(message, bytes):
                                await ws.send_bytes(message)
                            else:
                                await ws.send_text(message)
                    except Exception:
                        pass

                import asyncio

                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(client_to_backend()),
                        asyncio.create_task(backend_to_client()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()

        except Exception:
            logger.exception("WebSocket TTS proxy failed")
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    def __del__(self):
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()


app = Qwen3TTSVLLM.bind()
