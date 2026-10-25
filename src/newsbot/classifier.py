"""Классификатор тематики новостей.

Классификатор — это класс с методом predict(text) -> TopicPrediction.
Пока модели нет, работает StubKeywordClassifier (поиск по ключевым словам).

Необходимо:
  1. написать класс с методом predict(text) -> TopicPrediction;
  2. зарегистрировать его в build_classifier() под новым именем;
  3. добавить тесты в tests/.
"""

import re
from typing import ClassVar, Protocol

from newsbot.schemas import TopicPrediction

FALLBACK_TOPIC = "другое"


class TopicClassifier(Protocol):
    def predict(self, text: str) -> TopicPrediction: ...


class StubKeywordClassifier:
    """Временная заглушка, чтобы остальная система работала без модели."""

    KEYWORDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "политика": ("выбор", "президент", "правительств", "депутат", "закон", "министр", "санкц"),
        "экономика": ("рубл", "доллар", "курс", "инфляц", "банк", "бюджет", "цен", "рынок"),
        "спорт": ("матч", "футбол", "хоккей", "чемпионат", "команд", "гол", "турнир"),
        "технологии": ("искусственн", "нейросет", "смартфон", "приложени", "компьютер", "робот"),
        "происшествия": ("пожар", "авари", "дтп", "задержа", "погиб", "пострада"),
        "культура": ("фильм", "кино", "театр", "выставк", "концерт", "книг", "музе"),
    }

    def predict(self, text: str) -> TopicPrediction:
        words = re.findall(r"\w+", text.lower())
        scores = {
            topic: sum(1 for w in words for stem in stems if w.startswith(stem))
            for topic, stems in self.KEYWORDS.items()
        }
        total = sum(scores.values())
        if total == 0:
            return TopicPrediction(topic=FALLBACK_TOPIC, confidence=0.0)
        topic = max(scores, key=scores.get)
        return TopicPrediction(topic=topic, confidence=round(scores[topic] / total, 3))


def build_classifier(backend: str = "stub") -> TopicClassifier:
    if backend == "stub":
        return StubKeywordClassifier()
    raise ValueError(f"Unknown classifier backend: {backend!r}")
