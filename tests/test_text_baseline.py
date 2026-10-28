import hashlib
import json
import wave

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.scripts.evaluate_tts import edit_distance, words
from ml.scripts.lenta_data import label_candidate, split_from_date
from ml.scripts.make_tts_sample import valid_audio
from ml.scripts.train_text_baseline import read_splits
from newsbot.classifier import build_classifier
from newsbot.schemas import TopicPrediction
from newsbot.tfidf_classifier import TfidfTopicClassifier


@pytest.mark.parametrize(
    ("title", "topic", "tag", "expected"),
    [
        ("На ЕГЭ изменят задания по математике", "Россия", "Общество", "education"),
        ("В вузах изменят порядок поступления абитуриентов", "Россия", "Общество", "education"),
        ("В общежитии университета застрелили студента", "Мир", "Все", None),
        ("В кортеж врезался студент Дипломатической академии", "Россия", "Все", None),
        ("Профессор обругал студенток за откровенную одежду", "Из жизни", "Все", None),
        ("Школьники открыли остров в Арктике", "Россия", "Все", None),
        ("Курсы акций выросли после открытия торгов", "Экономика", "Все", None),
        ("Стипендии паралимпийцам увеличили", "Спорт", "Все", None),
        ("Стипендии студентам вузов увеличили", "Россия", "Общество", "education"),
        ("Чиновник без высшего образования работал вице-мэром", "Россия", "Все", None),
        ("Школьница пропала из-за экзамена", "Россия", "Все", None),
        ("Стена школы обрушилась во время урока", "Россия", "Все", None),
        ("Певица рассказала о своем заболевании", "Культура", "Музыка", None),
        ("Певица выпустила новый альбом", "Культура", "Музыка", "entertainment"),
        ("В браузере исправили ошибки", "Наука и техника", "Софт", "technology"),
        ("Смартфон получил новый процессор", "", "Гаджеты", "technology"),
        ("Чашка кофе с главой Apple ушла с аукциона", "Наука и техника", "Гаджеты", None),
        ("Режиссер представил новый фильм", "Культура", "Кино", "entertainment"),
        ("В новую игру добавят мультиплеер", "Наука и техника", "Игры", "entertainment"),
    ],
)
def test_label_regressions(title, topic, tag, expected):
    assert label_candidate({"title": title, "topic": topic, "tags": tag})[0] == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2016/12/31", "train"),
        ("2017/01/01", "train"),
        ("2018-01-01", "validation"),
        ("1914/01/01", None),
        ("2018/02/30", None),
        ("", None),
    ],
)
def test_chronological_boundaries(value, expected):
    assert split_from_date(value) == expected


@pytest.fixture
def artifact_path(tmp_path):
    texts = [
        "университет абитуриенты обучение экзамен",
        "школа учебники учитель обучение",
        "школа экзамен урок",
        "фильм актер режиссер кино",
        "музыка концерт альбом певец",
        "кино фильм актер премьера",
        "смартфон процессор телефон компьютер",
        "компьютер браузер программа обновление",
        "телефон смартфон процессор программа",
    ]
    labels = ["education"] * 3 + ["entertainment"] * 3 + ["technology"] * 3
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    ).fit(texts, labels)
    path = tmp_path / "baseline.joblib"
    joblib.dump({"format_version": 1, "max_words": 140, "pipeline": pipeline}, path)
    return path


@pytest.mark.parametrize(
    ("text", "topic"),
    [
        ("университет абитуриенты экзамен", "образование"),
        ("актер фильм режиссер", "развлечения"),
        ("смартфон телефон процессор", "технологии"),
    ],
)
def test_saved_backend_matches_contract(artifact_path, text, topic):
    classifier = build_classifier("tfidf", model_path=artifact_path)
    prediction = classifier.predict(text)
    assert isinstance(prediction, TopicPrediction)
    assert prediction.topic == topic
    assert 0 <= prediction.confidence <= 1


@pytest.mark.parametrize("text", ["", "  \n", "абракадабрафывап", "🎤🎶"])
def test_empty_and_unknown_vocabulary_fallback(artifact_path, text):
    assert TfidfTopicClassifier(artifact_path).predict(text) == TopicPrediction("другое", 0.0)


def test_explicit_confidence_rejection(artifact_path):
    classifier = TfidfTopicClassifier(artifact_path, min_confidence=1)
    assert classifier.predict("школа кино смартфон").topic == "другое"


def test_missing_model_is_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="Baseline artifact not found"):
        TfidfTopicClassifier(tmp_path / "missing.joblib")
    with pytest.raises(ValueError, match="model_path is required"):
        build_classifier("tfidf")


def test_incompatible_artifact_rejected(tmp_path):
    path = tmp_path / "wrong.joblib"
    joblib.dump({"format_version": 99}, path)
    with pytest.raises(ValueError, match="Unsupported baseline artifact"):
        TfidfTopicClassifier(path)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold(artifact_path, threshold):
    with pytest.raises(ValueError, match="min_confidence"):
        TfidfTopicClassifier(artifact_path, min_confidence=threshold)


def test_training_input_overlap_is_rejected(tmp_path):
    meta = {"splits": {}}
    for name, year in [("train", 2016), ("validation", 2018), ("test", 2019)]:
        rows = [
            {
                "id": f"{name}-{label}",
                "url": f"{name}/{label}",
                "title": f"{name} {label}",
                "text": "identical shared input",
                "date": f"{year}/01/01",
                "label": label,
            }
            for label in ("education", "entertainment", "technology")
        ]
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))
        meta["splits"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Split overlap: text"):
        read_splits(tmp_path)


def test_changed_data_fingerprint_rejected(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"splits": {"train": {"sha256": "wrong"}}}))
    (tmp_path / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        read_splits(tmp_path)


def test_tts_rejects_empty_wav(tmp_path):
    path = tmp_path / "empty.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
    assert not valid_audio(path)


def test_word_error_distance_and_normalization():
    assert words("Ёлка, школа!") == ["елка", "школа"]
    assert edit_distance(["два", "слова"], ["два", "других", "слова"]) == 1
    assert edit_distance(["два", "слова"], []) == 2
    assert edit_distance([], ["слово"]) == 1
