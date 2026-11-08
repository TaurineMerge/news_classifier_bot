"""Тексты бота, проверка входа и форматирование ответа. Без зависимости от aiogram."""

from newsbot.classifier import FALLBACK_TOPIC
from services.bot.pipeline import PipelineResult

# Лимит сообщения Telegram — 4096 символов; оставляем запас под тему
MAX_TRANSCRIPT_CHARS = 3500

START = (
    "Привет! Пришлите голосовое сообщение с новостью — я расшифрую его и определю тематику.\n\n"
    "Можно и аудиофайл. Команда /help — подробнее."
)
HELP = (
    "Как пользоваться:\n"
    "1. Запишите голосовое с новостью или перешлите аудио.\n"
    "2. Подождите несколько секунд — пришлю расшифровку и тему.\n\n"
    "Ограничения: до {max_minutes} мин и до {max_mb} МБ."
)
NOT_AUDIO = "Я понимаю только голосовые и аудиофайлы 🎤"
PROCESSING = "⏳ Распознаю…"
EMPTY_SPEECH = "Не удалось разобрать речь. Попробуйте записать ещё раз, ближе к микрофону."
INTERNAL_ERROR = "Что-то пошло не так при обработке. Попробуйте ещё раз чуть позже."


def help_text(max_duration_sec: int, max_file_size_mb: int) -> str:
    return HELP.format(max_minutes=max(1, max_duration_sec // 60), max_mb=max_file_size_mb)


def validate_media(
    duration: int | None, file_size: int | None, *, max_duration_sec: int, max_file_size_mb: int
) -> str | None:
    """Возвращает текст ошибки для пользователя или None, если аудио подходит."""
    if duration is not None and duration > max_duration_sec:
        return f"Слишком длинное аудио. Максимум — {max(1, max_duration_sec // 60)} мин."
    if file_size is not None and file_size > max_file_size_mb * 1024 * 1024:
        return f"Файл больше {max_file_size_mb} МБ — Telegram не даёт ботам скачивать такие файлы."
    return None


def format_reply(result: PipelineResult) -> str:
    if result.prediction is None:
        return EMPTY_SPEECH

    text = result.transcription.text.strip()
    if len(text) > MAX_TRANSCRIPT_CHARS:
        text = text[:MAX_TRANSCRIPT_CHARS].rstrip() + "…"

    prediction = result.prediction
    if prediction.topic == FALLBACK_TOPIC:
        topic_line = "🏷 Тему определить не удалось"
    else:
        topic_line = f"🏷 Тема: {prediction.topic} (уверенность {prediction.confidence:.0%})"

    return f"📝 Расшифровка:\n{text}\n\n{topic_line}"
