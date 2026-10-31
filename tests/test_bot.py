"""Тесты бота без Telegram: проверка входа, форматирование, пайплайн и обработчик."""

import asyncio
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from newsbot.classifier import FALLBACK_TOPIC
from newsbot.schemas import TopicPrediction, Transcription
from services.bot import messages
from services.bot.config import BotSettings
from services.bot.handlers import handle_voice
from services.bot.pipeline import PipelineResult, VoicePipeline

# ---------- фейковые модели ----------


class FakeASR:
    def __init__(self, text="Смартфон получил новый процессор", delay=0.0):
        self.text, self.delay = text, delay
        self.active = self.max_active = 0
        self._lock = threading.Lock()

    def transcribe(self, audio_path: str) -> Transcription:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return Transcription(text=self.text, audio_duration_sec=3.0)


class FakeClassifier:
    def __init__(self, topic="технологии", confidence=0.9):
        self.calls = 0
        self.prediction = TopicPrediction(topic, confidence)

    def predict(self, text: str) -> TopicPrediction:
        self.calls += 1
        return self.prediction


def result(text="текст", topic="технологии", confidence=0.87):
    return PipelineResult(
        Transcription(text, 3.0),
        TopicPrediction(topic, confidence) if text else None,
        asr_sec=1.0,
        classify_sec=0.01,
    )


@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "123:test")
    return BotSettings(_env_file=None, max_duration_sec=60, max_file_size_mb=1)


# ---------- проверка входа ----------


@pytest.mark.parametrize(
    ("duration", "size", "ok"),
    [
        (10, 1000, True),
        (60, 1024 * 1024, True),  # ровно на границе — можно
        (61, 1000, False),
        (10, 1024 * 1024 + 1, False),
        (None, None, True),  # Telegram не всегда присылает размер
    ],
)
def test_validate_media(duration, size, ok):
    error = messages.validate_media(duration, size, max_duration_sec=60, max_file_size_mb=1)
    assert (error is None) == ok


# ---------- форматирование ответа ----------


def test_format_reply_contains_text_and_topic():
    reply = messages.format_reply(result("Новый смартфон", "технологии", 0.87))
    assert "Новый смартфон" in reply
    assert "технологии" in reply
    assert "87%" in reply


def test_format_reply_truncates_long_text():
    reply = messages.format_reply(result("слово " * 2000))
    assert len(reply) < 4096  # лимит сообщения Telegram
    assert "…" in reply


def test_format_reply_fallback_topic():
    reply = messages.format_reply(result("что-то", FALLBACK_TOPIC, 0.0))
    assert "не удалось" in reply.lower()
    assert "0%" not in reply


def test_format_reply_empty_speech():
    assert messages.format_reply(result("")) == messages.EMPTY_SPEECH


# ---------- пайплайн ----------


def test_pipeline_runs_asr_and_classifier():
    clf = FakeClassifier()
    res = asyncio.run(VoicePipeline(FakeASR(), clf).run(Path("x.ogg")))
    assert res.transcription.text
    assert res.prediction == clf.prediction
    assert clf.calls == 1


def test_pipeline_skips_classifier_on_empty_speech():
    clf = FakeClassifier()
    res = asyncio.run(VoicePipeline(FakeASR(text="   "), clf).run(Path("x.ogg")))
    assert res.prediction is None
    assert clf.calls == 0


def test_pipeline_limits_concurrency():
    asr = FakeASR(delay=0.05)
    pipeline = VoicePipeline(asr, FakeClassifier(), max_concurrent=1)

    async def many():
        await asyncio.gather(*(pipeline.run(Path("x.ogg")) for _ in range(4)))

    asyncio.run(many())
    assert asr.max_active == 1


# ---------- обработчик голосового ----------


def make_message(duration=5, file_size=1000):
    status = SimpleNamespace(edit_text=AsyncMock())
    return SimpleNamespace(
        voice=SimpleNamespace(duration=duration, file_size=file_size, file_id="file-1"),
        audio=None,
        chat=SimpleNamespace(id=42),
        reply=AsyncMock(return_value=status),
    ), status


def make_bot():
    def download(file_id, destination):
        Path(destination).write_bytes(b"fake audio")

    return SimpleNamespace(download=AsyncMock(side_effect=download))


def test_handle_voice_success(settings):
    message, status = make_message()
    pipeline = VoicePipeline(FakeASR(), FakeClassifier())

    asyncio.run(handle_voice(message, make_bot(), pipeline, settings))

    message.reply.assert_awaited_once_with(messages.PROCESSING)
    reply = status.edit_text.await_args.args[0]
    assert "технологии" in reply


def test_handle_voice_rejects_long_audio_without_download(settings):
    message, status = make_message(duration=600)
    bot = make_bot()

    asyncio.run(handle_voice(message, bot, VoicePipeline(FakeASR(), FakeClassifier()), settings))

    bot.download.assert_not_awaited()
    assert "длинное" in message.reply.await_args.args[0]
    status.edit_text.assert_not_awaited()


def test_handle_voice_model_error_is_reported_not_raised(settings):
    message, status = make_message()
    broken = SimpleNamespace(run=AsyncMock(side_effect=RuntimeError("boom")))

    asyncio.run(handle_voice(message, make_bot(), broken, settings))

    status.edit_text.assert_awaited_once_with(messages.INTERNAL_ERROR)


# ---------- настройки ----------


def test_settings_require_token(monkeypatch):
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    with pytest.raises(ValueError):
        BotSettings(_env_file=None)


def test_settings_defaults(settings):
    assert settings.classifier_backend == "tfidf"
    assert settings.classifier_model_path == Path("ml/models/text_baseline.joblib")
    assert settings.bot_token.get_secret_value() == "123:test"
    assert "123:test" not in repr(settings)  # токен не утекает в логи
