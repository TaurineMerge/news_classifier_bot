"""Обработчики сообщений Telegram."""

import logging
import tempfile
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from services.bot import messages
from services.bot.config import BotSettings
from services.bot.pipeline import VoicePipeline

log = logging.getLogger(__name__)
router = Router(name="voice")


@router.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(messages.START)


@router.message(Command("help"))
async def on_help(message: Message, settings: BotSettings) -> None:
    await message.answer(messages.help_text(settings.max_duration_sec, settings.max_file_size_mb))


@router.message(F.voice | F.audio)
async def on_voice(message: Message, bot: Bot, pipeline: VoicePipeline, settings: BotSettings) -> None:
    await handle_voice(message, bot, pipeline, settings)


@router.message()
async def on_other(message: Message) -> None:
    await message.reply(messages.NOT_AUDIO)


async def handle_voice(message: Message, bot: Bot, pipeline: VoicePipeline, settings: BotSettings) -> None:
    """Логика обработки голосового. Вынесена из хендлера, чтобы её можно было тестировать."""
    media = message.voice or message.audio
    error = messages.validate_media(
        media.duration,
        media.file_size,
        max_duration_sec=settings.max_duration_sec,
        max_file_size_mb=settings.max_file_size_mb,
    )
    if error:
        await message.reply(error)
        return

    status = await message.reply(messages.PROCESSING)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio"
            await bot.download(media.file_id, destination=path)
            result = await pipeline.run(path)
    except Exception:
        log.exception("failed chat_id=%s file_id=%s", message.chat.id, media.file_id)
        await status.edit_text(messages.INTERNAL_ERROR)
        return

    await status.edit_text(messages.format_reply(result))

    # Структурированный лог: отсюда потом берутся метрики (скорость, распределение тем)
    prediction = result.prediction
    log.info(
        "processed chat_id=%s audio_sec=%.1f asr_sec=%.2f classify_sec=%.3f topic=%s confidence=%.3f",
        message.chat.id,
        result.transcription.audio_duration_sec,
        result.asr_sec,
        result.classify_sec,
        prediction.topic if prediction else "-",
        prediction.confidence if prediction else 0.0,
    )
