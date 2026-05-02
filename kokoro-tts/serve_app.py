"""Ray Serve deployment for hexgrad/Kokoro-82M — lightweight 82M-param StyleTTS2 TTS."""

import asyncio
import io
import logging
import os
import re
import time
from typing import AsyncIterator, Optional

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

logger = logging.getLogger("ray.serve")

SAMPLE_RATE = 24000
DEFAULT_VOICE = os.getenv("KOKORO_DEFAULT_VOICE", "af_heart")
DEFAULT_LANG = os.getenv("KOKORO_DEFAULT_LANG", "a")
PRELOAD_LANGS = [
    code.strip() for code in os.getenv("KOKORO_PRELOAD_LANGS", "a").split(",") if code.strip()
]
MAX_CHARS = int(os.getenv("KOKORO_MAX_CHARS", "5000"))
SUPPORTED_LANGS = {"a", "b", "e", "f", "h", "i", "j", "p", "z"}

CONTENT_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
}


@serve.deployment(
    ray_actor_options={"num_gpus": 0.5, "num_cpus": 2},
    max_ongoing_requests=8,
    autoscaling_config={
        "min_replicas": 1,
        "initial_replicas": 1,
        "max_replicas": 6,
        "target_ongoing_requests": 4,
    },
    health_check_period_s=30,
    health_check_timeout_s=30,
)
class KokoroTTS:
    def __init__(self):
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Kokoro-82M on {self._device}")

        self._pipelines: dict[str, KPipeline] = {}
        for lang in PRELOAD_LANGS:
            self._get_pipeline(lang)

        self._warmup()
        logger.info("Kokoro-82M ready")

    def _get_pipeline(self, lang_code: str) -> KPipeline:
        if lang_code not in SUPPORTED_LANGS:
            raise ValueError(
                f"Unsupported lang_code '{lang_code}'. "
                f"Supported: {sorted(SUPPORTED_LANGS)}"
            )
        if lang_code not in self._pipelines:
            logger.info(f"Loading pipeline for lang_code='{lang_code}'")
            self._pipelines[lang_code] = KPipeline(lang_code=lang_code, device=self._device)
        return self._pipelines[lang_code]

    def _warmup(self):
        """One synthesis pass to JIT/cudnn-autotune so the first user request is fast."""
        start = time.time()
        pipe = self._get_pipeline(DEFAULT_LANG)
        for _ in pipe("Warmup.", voice=DEFAULT_VOICE):
            pass
        logger.info(f"Warmup completed in {time.time() - start:.2f}s")

    def check_health(self):
        if not self._pipelines:
            raise RuntimeError("No pipelines loaded")

    async def __call__(self, request: Request) -> Response:
        path = request.url.path.rstrip("/")
        method = request.method

        if method == "GET" and path in ("", "/health", "/healthz"):
            return JSONResponse({
                "status": "ok",
                "model": "hexgrad/Kokoro-82M",
                "device": self._device,
                "loaded_langs": sorted(self._pipelines.keys()),
            })

        if method == "GET" and path == "/v1/audio/voices":
            return JSONResponse({"voices": _VOICES})

        if method == "POST" and path in ("/v1/audio/speech", ""):
            return await self._handle_speech(request)

        return JSONResponse(
            status_code=404,
            content={"error": f"Not found: {method} {path}"},
        )

    async def _handle_speech(self, request: Request) -> Response:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid JSON body"},
            )

        text = body.get("input") or body.get("text") or ""
        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing 'input' (or 'text') field"},
            )
        if len(text) > MAX_CHARS:
            return JSONResponse(
                status_code=413,
                content={"error": f"Text exceeds {MAX_CHARS} character limit"},
            )

        voice = body.get("voice", DEFAULT_VOICE)
        lang_code = body.get("lang_code") or _infer_lang(voice)
        speed = float(body.get("speed", 1.0))
        if not (0.5 <= speed <= 2.0):
            return JSONResponse(
                status_code=400,
                content={"error": "'speed' must be between 0.5 and 2.0"},
            )
        response_format = body.get("response_format", "wav").lower()
        if response_format not in CONTENT_TYPES:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unsupported response_format: {response_format}"},
            )
        stream = bool(body.get("stream", False))
        split_pattern = body.get("split_pattern", r"\n+")
        if len(split_pattern) > 64:
            return JSONResponse(
                status_code=400,
                content={"error": "'split_pattern' is limited to 64 characters"},
            )
        try:
            re.compile(split_pattern)
        except re.error as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid split_pattern regex: {e}"},
            )

        try:
            pipeline = self._get_pipeline(lang_code)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

        if stream and response_format == "pcm":
            return StreamingResponse(
                self._stream_pcm(pipeline, text, voice, speed, split_pattern),
                media_type=CONTENT_TYPES["pcm"],
                headers={
                    "X-Audio-Sample-Rate": str(SAMPLE_RATE),
                    "X-Audio-Channels": "1",
                    "X-Audio-Bit-Depth": "16",
                    "Cache-Control": "no-cache",
                },
            )

        audio = await asyncio.get_running_loop().run_in_executor(
            None,
            self._synthesize_full,
            pipeline, text, voice, speed, split_pattern,
        )

        if response_format == "pcm":
            return Response(
                content=_pcm_bytes(audio),
                media_type=CONTENT_TYPES["pcm"],
                headers={"X-Audio-Sample-Rate": str(SAMPLE_RATE)},
            )
        return Response(
            content=_audio_bytes(audio, response_format),
            media_type=CONTENT_TYPES[response_format],
        )

    def _synthesize_full(
        self,
        pipeline: KPipeline,
        text: str,
        voice: str,
        speed: float,
        split_pattern: str,
    ) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for _, _, audio in pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern):
            chunks.append(_to_numpy(audio))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    async def _stream_pcm(
        self,
        pipeline: KPipeline,
        text: str,
        voice: str,
        speed: float,
        split_pattern: str,
    ) -> AsyncIterator[bytes]:
        """Yield PCM bytes per synthesis chunk, as soon as each chunk is ready.

        KPipeline is a blocking generator, so we advance it in a thread executor
        and hand each finished chunk to the event loop. This is sentence-level
        streaming — the first bytes arrive as soon as the first sentence is
        synthesized (typically <100ms on L4/A10G).
        """
        loop = asyncio.get_running_loop()
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
        sentinel = object()

        def _next():
            try:
                return next(generator)
            except StopIteration:
                return sentinel

        while True:
            result = await loop.run_in_executor(None, _next)
            if result is sentinel:
                break
            _, _, audio = result
            yield _pcm_bytes(_to_numpy(audio))


def _to_numpy(audio) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32)


def _pcm_bytes(waveform: np.ndarray) -> bytes:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def _audio_bytes(waveform: np.ndarray, response_format: str) -> bytes:
    buf = io.BytesIO()
    format_map = {
        "wav": ("WAV", "PCM_16"),
        "flac": ("FLAC", "PCM_16"),
        "ogg": ("OGG", "VORBIS"),
    }
    file_format, subtype = format_map[response_format]
    sf.write(buf, waveform, SAMPLE_RATE, format=file_format, subtype=subtype)
    return buf.getvalue()


def _infer_lang(voice: str) -> str:
    """Infer lang_code from voice prefix (e.g. 'af_heart' → 'a', 'bm_george' → 'b')."""
    if not voice or len(voice) < 2 or voice[1] not in ("f", "m"):
        return DEFAULT_LANG
    prefix = voice[0]
    return prefix if prefix in SUPPORTED_LANGS else DEFAULT_LANG


_VOICES = [
    {"id": v, "lang_code": v[0], "gender": "female" if v[1] == "f" else "male"}
    for v in [
        # American English
        "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
        "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
        "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
        "am_onyx", "am_puck", "am_santa",
        # British English
        "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
        "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
        # Japanese
        "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
        # Mandarin Chinese
        "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
        "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
        # Spanish
        "ef_dora", "em_alex", "em_santa",
        # French
        "ff_siwis",
        # Hindi
        "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
        # Italian
        "if_sara", "im_nicola",
        # Brazilian Portuguese
        "pf_dora", "pm_alex", "pm_santa",
    ]
]


app = KokoroTTS.bind()
