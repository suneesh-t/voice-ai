"""Pipecat TTS service for self-hosted Qwen3-TTS (qwen3-tts)."""

from dataclasses import dataclass
from typing import AsyncGenerator

import httpx
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

from loguru import logger

QWEN3_SAMPLE_RATE = 24000


@dataclass
class Qwen3TTSSettings:
    voice: str = "vivian"
    language: str = "Auto"
    instructions: str = ""


class Qwen3TTSService(TTSService):
    """HTTP-based TTS using qwen3-tts's streaming PCM endpoint.

    Sends text to /v1/audio/speech with stream=true, response_format=pcm
    and yields raw PCM chunks as TTSAudioRawFrame (16-bit signed, 24kHz mono).
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8000",
        voice: str = "vivian",
        language: str = "Auto",
        instructions: str = "",
        **kwargs,
    ):
        super().__init__(sample_rate=QWEN3_SAMPLE_RATE, **kwargs)
        self._api_url = api_url.rstrip("/")
        self._voice = voice
        self._language = language
        self._instructions = instructions
        self._client: httpx.AsyncClient | None = None

    async def start(self, frame):
        await super().start(frame)
        self._client = httpx.AsyncClient(timeout=300.0)

    async def cleanup(self):
        await super().cleanup()
        if self._client:
            await self._client.aclose()
            self._client = None

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Stream PCM audio from qwen3-tts."""
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            payload = {
                "input": text,
                "voice": self._voice,
                "language": self._language,
                "stream": True,
                "response_format": "pcm",
            }
            if self._instructions:
                payload["instructions"] = self._instructions

            await self.start_tts_usage_metrics(text)

            leftover = b""
            async with self._client.stream(
                "POST",
                f"{self._api_url}/v1/audio/speech",
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield ErrorFrame(
                        error=f"Qwen3 TTS error ({resp.status_code}): {resp.text}"
                    )
                    return

                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue

                    await self.stop_ttfb_metrics()

                    raw = leftover + chunk
                    usable = len(raw) - (len(raw) % 2)
                    leftover = raw[usable:]

                    if usable > 0:
                        yield TTSAudioRawFrame(
                            audio=raw[:usable],
                            sample_rate=QWEN3_SAMPLE_RATE,
                            num_channels=1,
                            context_id=context_id,
                        )

            if len(leftover) >= 2:
                yield TTSAudioRawFrame(
                    audio=leftover,
                    sample_rate=QWEN3_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )

        except Exception as e:
            yield ErrorFrame(error=f"Qwen3 TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
