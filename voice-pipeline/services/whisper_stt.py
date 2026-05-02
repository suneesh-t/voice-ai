"""Pipecat STT service for self-hosted Whisper Large-v3-Turbo (whisper-large-v3-turbo)."""

import json
from typing import AsyncGenerator

import httpx
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from loguru import logger


class WhisperSTTService(SegmentedSTTService):
    """HTTP-based STT using whisper-large-v3-turbo's streaming SSE endpoint.

    Same SSE format as cohere-transcribe — POST audio with ?stream=true,
    parse SSE chunks with choices[0].delta.content.
    """

    def __init__(
        self,
        *,
        api_url: str = "http://localhost:8003",
        language: str = "en",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._api_url = api_url.rstrip("/")
        self._language = language
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
        try:
            await self.start_processing_metrics()

            full_text = ""
            async with self._client.stream(
                "POST",
                f"{self._api_url}/?stream=true",
                files={"file": ("segment.wav", audio, "audio/wav")},
                data={"language": self._language},
            ) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    yield ErrorFrame(
                        error=f"Whisper STT error ({resp.status_code}): {resp.text}"
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
            yield ErrorFrame(error=f"Whisper STT error: {e}")
