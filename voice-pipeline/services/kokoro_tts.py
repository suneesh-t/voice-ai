"""Pipecat TTS service for self-hosted Kokoro-82M (kokoro-tts)."""

from typing import AsyncGenerator

import httpx
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

from loguru import logger

KOKORO_SAMPLE_RATE = 24000


class KokoroTTSService(TTSService):
    """HTTP-based TTS using kokoro-tts's streaming PCM endpoint.

    Sends text to /v1/audio/speech with stream=true, response_format=pcm
    and yields raw PCM chunks as TTSAudioRawFrame (16-bit signed, 24kHz mono).

    Kokoro streams at sentence boundaries (split_pattern), so TTFB is
    first-sentence synthesis time (~30-100ms on L4/A10G).
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8004",
        voice: str = "af_heart",
        lang_code: str | None = None,
        speed: float = 1.0,
        **kwargs,
    ):
        super().__init__(sample_rate=KOKORO_SAMPLE_RATE, **kwargs)
        self._api_url = api_url.rstrip("/")
        self._voice = voice
        self._lang_code = lang_code
        self._speed = speed
        self._client: httpx.AsyncClient | None = None

    async def start(self, frame):
        await super().start(frame)
        self._client = httpx.AsyncClient(timeout=120.0)

    async def cleanup(self):
        await super().cleanup()
        if self._client:
            await self._client.aclose()
            self._client = None

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            payload: dict = {
                "input": text,
                "voice": self._voice,
                "speed": self._speed,
                "stream": True,
                "response_format": "pcm",
            }
            if self._lang_code:
                payload["lang_code"] = self._lang_code

            await self.start_tts_usage_metrics(text)

            leftover = b""
            async with self._client.stream(
                "POST",
                f"{self._api_url}/v1/audio/speech",
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    logger.error(
                        f"Kokoro TTS upstream error {resp.status_code}: {resp.text[:500]}"
                    )
                    yield ErrorFrame(
                        error=f"Kokoro TTS error ({resp.status_code})"
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
                            sample_rate=KOKORO_SAMPLE_RATE,
                            num_channels=1,
                            context_id=context_id,
                        )

            if len(leftover) >= 2:
                yield TTSAudioRawFrame(
                    audio=leftover,
                    sample_rate=KOKORO_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )

        except Exception as e:
            yield ErrorFrame(error=f"Kokoro TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
