"""Точка входа: uv run --group bot python -m services.bot"""

import asyncio
import logging

from aiogram import Bot, Dispatcher

from services.bot.config import BotSettings
from services.bot.handlers import router
from services.bot.pipeline import build_pipeline

log = logging.getLogger("services.bot")


async def main() -> None:
    settings = BotSettings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Модели грузим до подключения к Telegram: если нет весов, бот упадёт сразу и понятно
    pipeline = build_pipeline(settings)

    bot = Bot(settings.bot_token.get_secret_value())
    dp = Dispatcher(pipeline=pipeline, settings=settings)
    dp.include_router(router)

    log.info("bot started")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
