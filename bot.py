import asyncio
import os

import httpx
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
SERVER_URL = "http://127.0.0.1:8000/predict-audio"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message):
    await message.answer("Привет! Отправь мне голосовое сообщение, и я определю его тему.")


@dp.message(lambda message: message.voice is not None)
async def handle_voice(message: Message):
    await message.answer("Слушаю и анализирую...")

    file = await bot.get_file(message.voice.file_id)
    file_path = file.file_path
    file_on_disk = f"voice_{message.from_user.id}.ogg"
    await bot.download_file(file_path, destination=file_on_disk)

    try:
        async with httpx.AsyncClient() as client:
            with open(file_on_disk, "rb") as f:
                files = {"file": (file_on_disk, f, "audio/ogg")}
                response = await client.post(SERVER_URL, files=files, timeout=60.0)

        if response.status_code == 200:
            data = response.json()
            label_int = data.get("label")
            text = data.get("text", "")
            topic = data.get("topic", "Неизвестно")

            await message.answer(f"📝 Текст: {text}\n\n🎯 Тема: {topic} (код: {label_int})")
        else:
            await message.answer("Ошибка на сервере. Проверь логи.")
    except Exception as e:
        await message.answer(f"Ошибка при отправке: {e}")
    finally:
        if os.path.exists(file_on_disk):
            os.remove(file_on_disk)


async def main():
    print("Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
