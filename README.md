# newsbot

Транскрибация голосовых сообщений с новостями (faster-whisper) и определение их тематики.

## Быстрый старт

Нужен [uv](https://docs.astral.sh/uv/) — менеджер зависимостей и Python:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Дальше:

```bash
git clone <url> && cd newsbot
uv sync                       # окружение + зависимости (для ML: uv sync --group ml)
uv run pre-commit install     # проверки перед каждым коммитом
uv run pytest                 # убедиться, что всё работает
```

Python ставить отдельно не нужно — uv скачает подходящую версию сам.

## Структура

```
newsbot/
├─ src/newsbot/       общий пакет: контракт и модели (рабочий код)
│  ├─ schemas.py      контракт: Transcription, TopicPrediction
│  ├─ asr.py          распознавание речи (faster-whisper)
│  └─ classifier.py   заглушка и опциональный TF-IDF backend
├─ ml/                эксперименты ML-команды → см. ml/README.md
├─ services/          запускаемые сервисы (бот появится здесь)
└─ tests/             тесты
```

Эксперименты находятся в `ml/notebooks/`, рабочий код — в `src/newsbot/`.
Всё, что использует бот, должно лежать в `src/newsbot/` и соответствовать контракту из `schemas.py`.

Baseline трёх тем и передача данных описаны в [ml/reports/DATA_HANDOFF.md](ml/reports/DATA_HANDOFF.md).
После `uv sync --locked --group ml` и получения файла модели:

```python
from newsbot.classifier import build_classifier

classifier = build_classifier("tfidf", model_path="ml/models/text_baseline.joblib")
prediction = classifier.predict("В университете начался прием абитуриентов")
print(prediction.topic, prediction.confidence)
```

Backend возвращает `TopicPrediction` с темами «образование», «развлечения», «технологии».
Пустой ввод и неизвестный словарь дают «другое». Оценки модели получены на предварительной
разметке; готовность к реальным голосовым сообщениям ими не подтверждена.

## Частые команды

| Что                                   | Команда                                             |
| ------------------------------------- | --------------------------------------------------- |
| Добавить библиотеку в общий пакет     | `uv add <пакет>`                                    |
| Добавить библиотеку для экспериментов | `uv add --group ml <пакет>`                         |
| Тесты                                 | `uv run pytest`                                     |
| Линтер и автоисправление              | `uv run ruff check --fix . && uv run ruff format .` |
| Jupyter                               | `uv run jupyter lab`                                |

## Как работаем

- В `main` напрямую не пушим — только через Pull Request.
- PR мержится, когда CI зелёный.
- После `uv add` коммитим и `pyproject.toml`, и `uv.lock`.
- Данные, веса и `.env` не коммитим.
