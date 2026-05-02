"""Pipecat TTS service for self-hosted OmniVoice (omni-voice)."""

from typing import AsyncGenerator

import httpx
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.tts_service import TTSService

from loguru import logger

OMNIVOICE_SAMPLE_RATE = 24000


class OmniVoiceTTSService(TTSService):
    """HTTP-based TTS using omni-voice's PCM streaming endpoint.

    Sends text to /v1/audio/speech with response_format=pcm and stream=true,
    yields raw PCM chunks as TTSAudioRawFrame (16-bit signed, 24kHz mono).
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8002",
        instruct: str = "",
        speed: float = 1.0,
        **kwargs,
    ):
        super().__init__(sample_rate=OMNIVOICE_SAMPLE_RATE, **kwargs)
        self._api_url = api_url.rstrip("/")
        self._instruct = instruct
        self._speed = speed
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
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            payload = {
                "input": text,
                "response_format": "pcm",
                "stream": True,
                "speed": self._speed,
            }
            if self._instruct:
                payload["instruct"] = self._instruct

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
                        error=f"OmniVoice TTS error ({resp.status_code}): {resp.text}"
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
                            sample_rate=OMNIVOICE_SAMPLE_RATE,
                            num_channels=1,
                            context_id=context_id,
                        )

            if len(leftover) >= 2:
                yield TTSAudioRawFrame(
                    audio=leftover,
                    sample_rate=OMNIVOICE_SAMPLE_RATE,
                    num_channels=1,
                    context_id=context_id,
                )

        except Exception as e:
            yield ErrorFrame(error=f"OmniVoice TTS error: {e}")
        finally:
            await self.stop_ttfb_metrics()
