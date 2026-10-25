"""Контракт между частями системы."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Transcription:
    text: str
    audio_duration_sec: float


@dataclass(frozen=True)
class TopicPrediction:
    topic: str
    confidence: float  # 0..1
