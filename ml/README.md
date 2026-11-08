# ML

Рабочее место ML-команды.

Готовый baseline Data: [отчёт и инструкции](reports/DATA_HANDOFF.md),
[ноутбук](notebooks/01_lenta_baseline.ipynb).

```bash
uv sync --locked --group ml
uv run --group ml python -m ml.scripts.download_lenta
uv run --group ml python -m ml.scripts.lenta_data
uv run --group ml python -m ml.scripts.train_text_baseline
```

В `ml/scripts/` находятся вспомогательные функции подготовки и обучения, которые вызывает
ноутбук; эксперимент и сравнение моделей доступны в ноутбуке. Runtime backend —
`src/newsbot/tfidf_classifier.py`. Исходный архив, JSONL, токены и CSV проверки лежат в
`ml/data/`, модель со словарём — в `ml/models/`: всё это исключено из Git.

```
ml/
├─ notebooks/   эксперименты, EDA, сравнение моделей
├─ data/        датасеты — НЕ коммитятся (в .gitignore)
└─ models/      веса — НЕ коммитятся (в .gitignore)
```

## Правила

- **Ноутбуки — для экспериментов, `src/newsbot/` — для рабочего кода.**
  Когда модель готова, перенесите её в `src/newsbot/classifier.py` (или новый модуль)
  как класс с методом `predict(text) -> TopicPrediction`. Бот возьмёт её оттуда.
- В ноутбуке можно импортировать общий код: `from newsbot.asr import WhisperASR`.
- Выводы ячеек очищаются автоматически при коммите (nbstripout в pre-commit).
- Данные и веса не коммитим: git плохо работает с большими файлами, а вычистить их
  из истории потом очень трудно. Пока храним на общем диске / сервере, позже подключим DVC.
- Новую библиотеку добавляем через `uv add --group ml <пакет>`, чтобы у всех были те же версии.
