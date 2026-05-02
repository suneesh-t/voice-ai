"""Voice AI pipeline: STT → LLM → TTS via Pipecat.

STT provider: cohere (default) or whisper — set STT_PROVIDER env var.
TTS provider: qwen3 (default), omnivoice, or kokoro — set TTS_PROVIDER env var.
"""

import asyncio
import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.aws.llm import AWSBedrockLLMService

load_dotenv()

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_PROMPT = """You are a helpful voice AI assistant. Keep your responses concise \
and conversational — typically 1-3 sentences. You are having a real-time voice \
conversation, so be natural, friendly, and to the point."""


def create_stt():
    provider = os.getenv("STT_PROVIDER", "cohere").lower()

    if provider == "whisper":
        from services.whisper_stt import WhisperSTTService

        logger.info("Using Whisper Large-v3-Turbo STT")
        return WhisperSTTService(
            api_url=os.getenv("WHISPER_STT_URL", "http://localhost:8003"),
            language=os.getenv("WHISPER_LANGUAGE", "en"),
        )
    else:
        from services.cohere_stt import CohereTranscribeSTTService

        logger.info("Using Cohere Transcribe STT")
        return CohereTranscribeSTTService(
            api_url=os.getenv("COHERE_STT_URL", "http://localhost:8000"),
        )


def create_tts():
    provider = os.getenv("TTS_PROVIDER", "qwen3").lower()

    if provider == "omnivoice":
        from services.omnivoice_tts import OmniVoiceTTSService

        logger.info("Using OmniVoice TTS")
        return OmniVoiceTTSService(
            api_url=os.getenv("OMNIVOICE_TTS_URL", "http://localhost:8002"),
            instruct=os.getenv("OMNIVOICE_INSTRUCT", ""),
            speed=float(os.getenv("OMNIVOICE_SPEED", "1.0")),
        )
    elif provider == "kokoro":
        from services.kokoro_tts import KokoroTTSService

        logger.info("Using Kokoro-82M TTS")
        return KokoroTTSService(
            api_url=os.getenv("KOKORO_TTS_URL", "http://localhost:8004"),
            voice=os.getenv("KOKORO_VOICE", "af_heart"),
            lang_code=os.getenv("KOKORO_LANG_CODE") or None,
            speed=float(os.getenv("KOKORO_SPEED", "1.0")),
        )
    else:
        from services.qwen3_tts import Qwen3TTSService

        logger.info("Using Qwen3 TTS")
        return Qwen3TTSService(
            api_url=os.getenv("QWEN3_TTS_URL", "http://localhost:8001"),
            voice=os.getenv("TTS_VOICE", "vivian"),
            language=os.getenv("TTS_LANGUAGE", "Auto"),
        )


async def run_bot(transport, runner_args=None):
    stt = create_stt()
    tts = create_tts()

    llm = AWSBedrockLLMService(
        settings=AWSBedrockLLMService.Settings(
            model=os.getenv("BEDROCK_MODEL_ID"),
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            system_instruction=SYSTEM_PROMPT,
        ),
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    context = LLMContext(messages=messages)
    context_pair = LLMContextAggregatorPair(context=context)

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(sample_rate=16000))

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            context_pair.user(),
            llm,
            tts,
            transport.output(),
            context_pair.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.info("Client ready — sending greeting")
        await task.queue_frames(
            [
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": "Say hello and briefly introduce yourself."}],
                    run_llm=True,
                )
            ]
        )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=True)
    await runner.run(task)


from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams

transport_params = {
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main as pipecat_main

    pipecat_main()
