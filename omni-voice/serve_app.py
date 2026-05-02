"""Ray Serve deployment for k2-fsa/OmniVoice — multilingual zero-shot TTS."""

import io
from typing import Optional

import numpy as np
import ray
import soundfile as sf
import torch
from omnivoice import OmniVoice
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

MODEL_ID = "k2-fsa/OmniVoice"
SAMPLE_RATE = 24000


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 2},
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 4,
        "target_ongoing_requests": 3,
    },
    max_ongoing_requests=8,
)
class OmniVoiceTTS:
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = OmniVoice.from_pretrained(
            MODEL_ID,
            device_map=device,
            dtype=torch.float16,
        )

    def _generate(
        self,
        text: str,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        speed: float = 1.0,
        num_step: int = 32,
    ) -> np.ndarray:
        """Generate audio from text with optional voice cloning or design."""
        kwargs = {"text": text, "speed": speed, "num_step": num_step}
        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
        elif instruct:
            kwargs["instruct"] = instruct

        audio = self.model.generate(**kwargs)
        return audio[0]

    def _wav_bytes(self, waveform: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def _pcm_bytes(self, waveform: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, waveform, SAMPLE_RATE, format="RAW", subtype="PCM_16")
        return buf.getvalue()

    async def _handle_speech(self, request: Request) -> Response:
        """POST /v1/audio/speech — generate speech from text."""
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            form = await request.form()
            text = form.get("text", "")
            ref_audio_file = form.get("ref_audio")
            ref_text = form.get("ref_text", "")
            instruct = form.get("instruct", "")
            speed = float(form.get("speed", "1.0"))
            num_step = int(form.get("num_step", "32"))
            response_format = form.get("response_format", "wav")
            stream = form.get("stream", "false").lower() == "true"

            ref_audio_path = None
            if ref_audio_file is not None:
                import tempfile
                audio_bytes = await ref_audio_file.read()
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                ref_audio_path = tmp.name
        else:
            body = await request.json()
            text = body.get("input", body.get("text", ""))
            ref_text = body.get("ref_text", "")
            instruct = body.get("instruct", "")
            speed = float(body.get("speed", 1.0))
            num_step = int(body.get("num_step", 32))
            response_format = body.get("response_format", "wav")
            stream = body.get("stream", False)

            ref_audio_path = None
            ref_audio_b64 = body.get("ref_audio")
            if ref_audio_b64:
                import base64
                import tempfile
                try:
                    audio_bytes = base64.b64decode(ref_audio_b64)
                except Exception:
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid base64 in 'ref_audio' field"},
                    )
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                ref_audio_path = tmp.name

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'text' or 'input' field"},
            )

        waveform = self._generate(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text or None,
            instruct=instruct or None,
            speed=speed,
            num_step=num_step,
        )

        if response_format == "pcm" and stream:
            pcm = self._pcm_bytes(waveform)
            chunk_size = 4800  # 100ms chunks at 24kHz 16-bit mono

            async def stream_pcm():
                for i in range(0, len(pcm), chunk_size):
                    yield pcm[i : i + chunk_size]

            return StreamingResponse(
                stream_pcm(),
                media_type="application/octet-stream",
                headers={"X-Sample-Rate": str(SAMPLE_RATE)},
            )
        elif response_format == "pcm":
            pcm = self._pcm_bytes(waveform)
            return Response(
                content=pcm,
                media_type="application/octet-stream",
                headers={"X-Sample-Rate": str(SAMPLE_RATE)},
            )
        else:
            wav = self._wav_bytes(waveform)
            return Response(content=wav, media_type="audio/wav")

    async def _handle_health(self, request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "model": MODEL_ID})

    async def __call__(self, request: Request) -> Response:
        path = request.url.path.rstrip("/")
        method = request.method

        if method == "GET" and path in ("", "/health", "/healthz"):
            return await self._handle_health(request)

        if method == "POST" and path in ("/v1/audio/speech", ""):
            return await self._handle_speech(request)

        return JSONResponse(
            status_code=404,
            content={"error": f"Not found: {method} {path}"},
        )


app = OmniVoiceTTS.bind()
