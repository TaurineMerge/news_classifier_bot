"""Настройки бота. Читаются из переменных окружения или файла .env в корне проекта."""

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Telegram
    bot_token: SecretStr

    # ASR (faster-whisper)
    whisper_model: str = "small"  # tiny / base / small / medium / large-v3 или путь к весам
    whisper_device: str = "cpu"  # cpu / cuda
    whisper_compute_type: str = "int8"  # int8 для CPU, float16 для GPU
    whisper_language: str = "ru"
    whisper_beam_size: int = Field(5, ge=1)

    # Классификатор
    classifier_backend: str = "tfidf"  # stub / tfidf
    classifier_model_path: Path = Path("ml/models/text_baseline.joblib")
    min_confidence: float = Field(0.0, ge=0, le=1)

    # Ограничения на входное аудио
    max_duration_sec: int = Field(300, gt=0)
    max_file_size_mb: int = Field(20, gt=0)  # Bot API скачивает файлы только до 20 МБ

    # Сколько аудио обрабатывать одновременно. Whisper на CPU тяжёлый: 1 — безопасно.
    max_concurrent_jobs: int = Field(1, ge=1)

    log_level: str = "INFO"
