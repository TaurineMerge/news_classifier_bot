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


def test_all_topics_recognized(clf):
    cases = {
        "политика": "Президент подписал закон о выборах",
        "экономика": "Курс доллара вырос, инфляция ускорилась",
        "технологии": "Нейросеть научилась рисовать роботов",
        "происшествия": "Пожар на складе, есть пострадавшие",
        "культура": "В театре прошла премьера фильма",
        "спорт": "Матч по футболу завершился победой",
    }
    for expected, text in cases.items():
        pred = clf.predict(text)
        assert pred.topic == expected, f"ожидали {expected!r} для текста {text!r}, получили {pred.topic!r}"


def test_prediction_is_deterministic(clf):
    text = "Сборная выиграла матч чемпионата"
    assert clf.predict(text) == clf.predict(text)


def test_case_insensitive(clf):
    assert clf.predict("ВЫБОРЫ ПРЕЗИДЕНТА") == clf.predict("выборы президента")


def test_empty_string_returns_fallback(clf):
    pred = clf.predict("")
    assert pred.topic == FALLBACK_TOPIC
    assert pred.confidence == 0.0


def test_topic_always_from_known_set(clf):
    known = set(StubKeywordClassifier.KEYWORDS) | {FALLBACK_TOPIC}
    for text in ["", "абвгд", "матч", "курс доллара", "нейросеть"]:
        assert clf.predict(text).topic in known


def test_confidence_in_range_for_all_topics(clf):
    texts = [
        "выборы президента",
        "курс доллара инфляция",
        "матч футбол чемпионат",
        "нейросеть смартфон",
        "пожар авария",
        "фильм театр",
        "абвгд",
        "",
    ]
    for text in texts:
        conf = clf.predict(text).confidence
        assert 0.0 <= conf <= 1.0


def test_build_classifier_default_is_stub():
    assert isinstance(build_classifier(), StubKeywordClassifier)


def test_build_classifier_explicit_stub():
    assert isinstance(build_classifier("stub"), StubKeywordClassifier)
