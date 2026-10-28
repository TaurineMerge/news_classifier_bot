"""Load the local ML baseline without training or network requests."""

import math
from pathlib import Path
from typing import ClassVar

from newsbot.schemas import TopicPrediction


class TfidfTopicClassifier:
    """TF-IDF + a probabilistic sklearn estimator; requires the ml dependency group.

    Only load trusted project artifacts. The default confidence threshold does not
    detect every out-of-domain message; calibrating rejection needs labelled OOD data.
    """

    TOPICS: ClassVar[dict[str, str]] = {
        "education": "образование",
        "entertainment": "развлечения",
        "technology": "технологии",
    }

    def __init__(self, model_path: str | Path, min_confidence: float = 0.0) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in 0..1")
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Baseline artifact not found: {path}. See ml/README.md")
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("Install ML dependencies: uv sync --group ml") from exc
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
            raise ValueError("Unsupported baseline artifact format")
        pipeline = artifact.get("pipeline")
        if not hasattr(pipeline, "named_steps") or set(getattr(pipeline, "classes_", ())) != set(self.TOPICS):
            raise ValueError("Baseline must contain the three documented topic classes")
        if set(pipeline.named_steps) != {"tfidf", "classifier"}:
            raise ValueError("Unsupported baseline pipeline")
        max_words = artifact.get("max_words")
        if not isinstance(max_words, int) or isinstance(max_words, bool) or max_words <= 0:
            raise ValueError("Invalid max_words in baseline artifact")
        self._pipeline = pipeline
        self._max_words = max_words
        self._min_confidence = min_confidence

    def predict(self, text: str) -> TopicPrediction:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        clean = " ".join(text.split()[: self._max_words])
        if not clean:
            return TopicPrediction(topic="другое", confidence=0.0)
        features = self._pipeline.named_steps["tfidf"].transform([clean])
        if features.nnz == 0:
            return TopicPrediction(topic="другое", confidence=0.0)
        scores = self._pipeline.named_steps["classifier"].predict_proba(features)[0]
        if any(not math.isfinite(float(score)) or not 0 <= score <= 1 for score in scores):
            raise ValueError("Model returned invalid probability scores")
        index = int(scores.argmax())
        confidence = float(scores[index])
        if confidence < self._min_confidence:
            return TopicPrediction(topic="другое", confidence=0.0)
        label = str(self._pipeline.classes_[index])
        return TopicPrediction(topic=self.TOPICS[label], confidence=confidence)
