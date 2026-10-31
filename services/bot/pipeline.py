"""Обработка аудио: распознавание речи и классификация.

Модели синхронные и тяжёлые, поэтому выполняются в отдельном потоке, а число
одновременных задач ограничено семафором — бот остаётся отзывчивым.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from newsbot.classifier import TopicClassifier, build_classifier
from newsbot.schemas import TopicPrediction, Transcription

if TYPE_CHECKING:
    from services.bot.config import BotSettings

log = logging.getLogger(__name__)


class SpeechRecognizer(Protocol):
    def transcribe(self, audio_path: str) -> Transcription: ...


@dataclass(frozen=True)
class PipelineResult:
    transcription: Transcription
    prediction: TopicPrediction | None  # None, если речь не распознана
    asr_sec: float
    classify_sec: float


class VoicePipeline:
    def __init__(self, asr: SpeechRecognizer, classifier: TopicClassifier, max_concurrent: int = 1) -> None:
        self._asr = asr
        self._classifier = classifier
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def run(self, audio_path: Path) -> PipelineResult:
        async with self._semaphore:
            started = time.perf_counter()
            transcription = await asyncio.to_thread(self._asr.transcribe, str(audio_path))
            asr_sec = time.perf_counter() - started

            if not transcription.text.strip():
                return PipelineResult(transcription, None, asr_sec, 0.0)

            started = time.perf_counter()
            prediction = await asyncio.to_thread(self._classifier.predict, transcription.text)
            classify_sec = time.perf_counter() - started

        return PipelineResult(transcription, prediction, asr_sec, classify_sec)


def build_pipeline(settings: "BotSettings") -> VoicePipeline:
    """Загружает модели. Вызывается один раз при старте: ошибка конфигурации видна сразу."""
    from newsbot.asr import WhisperASR

    model_path = settings.classifier_model_path if settings.classifier_backend != "stub" else None
    classifier = build_classifier(
        settings.classifier_backend, model_path=model_path, min_confidence=settings.min_confidence
    )
    log.info("classifier ready backend=%s path=%s", settings.classifier_backend, model_path)

    asr = WhisperASR(
        model_size=settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
        language=settings.whisper_language,
        beam_size=settings.whisper_beam_size,
    )
    return VoicePipeline(asr, classifier, max_concurrent=settings.max_concurrent_jobs)
