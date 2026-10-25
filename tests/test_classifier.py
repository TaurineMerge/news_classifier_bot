import pytest

from newsbot.classifier import FALLBACK_TOPIC, StubKeywordClassifier, build_classifier
from newsbot.schemas import TopicPrediction


@pytest.fixture
def clf():
    return StubKeywordClassifier()

def test_returns_contract_type(clf):
    pred = clf.predict("любой текст")
    assert isinstance(pred, TopicPrediction)
    assert 0.0 <= pred.confidence <= 1.0

def test_sport(clf):
    assert clf.predict("Вчера сборная выиграла матч чемпионата по футболу").topic == "спорт"

def test_fallback(clf):
    pred = clf.predict("абвгд")
    assert pred.topic == FALLBACK_TOPIC
    assert pred.confidence == 0.0

def test_unknown_backend():
    with pytest.raises(ValueError):
        build_classifier("nope")
