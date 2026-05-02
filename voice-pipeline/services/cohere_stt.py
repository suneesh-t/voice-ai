"""Pipecat STT service for self-hosted Cohere Transcribe (cohere-transcribe)."""

import io
import json
import wave
from typing import AsyncGenerator

import httpx
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from loguru import logger


class CohereTranscribeSTTService(SegmentedSTTService):
    """HTTP-based STT using cohere-transcribe's streaming SSE endpoint.

    Uses SegmentedSTTService which relies on VAD to detect speech segments,
    packages them as WAV, and sends to run_stt(). We POST the WAV to our
    cohere-transcribe service with ?stream=true and assemble the streamed
    text chunks into a final transcription.
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8000",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_url = api_url.rstrip("/")
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

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Send VAD-segmented audio to cohere-transcribe and stream text back."""
        try:
            await self.start_processing_metrics()

            full_text = ""
            async with self._client.stream(
                "POST",
                f"{self._api_url}/?stream=true",
                files={"file": ("segment.wav", audio, "audio/wav")},
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield ErrorFrame(
                        error=f"Cohere transcribe error ({resp.status_code}): {resp.text}"
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        full_text += delta
                    except json.JSONDecodeError:
                        continue

            await self.stop_processing_metrics()

            text = full_text.strip()
            if text:
                logger.debug(f"Transcription: [{text}]")
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601())

        except Exception as e:
            yield ErrorFrame(error=f"Cohere transcribe STT error: {e}")
