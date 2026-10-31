import importlib.util

# Тесты бота требуют группу bot (aiogram). У ML-команды её может не быть — тогда пропускаем.
collect_ignore = [] if importlib.util.find_spec("aiogram") else ["test_bot.py"]
