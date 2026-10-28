"""Train two CPU baselines; select on 2017 validation, evaluate on 2019.

uv run --group ml python -m ml.scripts.train_text_baseline
"""

import argparse
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from ml.scripts.lenta_data import LABELS, MAX_WORDS, normalize, split_from_date


def read_splits(data: Path) -> tuple[dict, dict]:
    metadata = json.loads((data / "metadata.json").read_text())
    splits = {}
    for name in ("train", "validation", "test"):
        path = data / f"{name}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata["splits"][name]["sha256"]:
            raise ValueError(f"Dataset fingerprint mismatch: {name}")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if {r["label"] for r in rows} != set(LABELS):
            raise ValueError(f"Missing classes in {name}")
        if any(split_from_date(r["date"]) != name for r in rows):
            raise ValueError(f"Date leakage in {name}")
        splits[name] = rows
    for field in ("id", "url", "title", "text"):
        sets = {name: {normalize(r[field]) for r in rows} for name, rows in splits.items()}
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if sets[left] & sets[right]:
                raise ValueError(f"Split overlap: {field}, {left}/{right}")
    return splits, metadata


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    max_features=40_000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            ("classifier", estimator),
        ]
    )


def evaluate(model, rows: list[dict]) -> dict:
    y = [r["label"] for r in rows]
    prediction = model.predict([r["text"] for r in rows])
    return {
        "rows": len(rows),
        "accuracy": accuracy_score(y, prediction),
        "macro_f1": f1_score(y, prediction, labels=LABELS, average="macro", zero_division=0),
        "report": classification_report(y, prediction, labels=LABELS, output_dict=True, zero_division=0),
        "labels": list(LABELS),
        "confusion_matrix": confusion_matrix(y, prediction, labels=LABELS).tolist(),
    }


def train(data: Path, model_dir: Path, report_dir: Path) -> dict:
    splits, metadata = read_splits(data)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "logistic_regression": make_pipeline(LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        "multinomial_nb": make_pipeline(MultinomialNB(alpha=0.5)),
    }
    train_x, train_y = [r["text"] for r in splits["train"]], [r["label"] for r in splits["train"]]
    validation = {}
    for name, model in models.items():
        model.fit(train_x, train_y)
        validation[name] = evaluate(model, splits["validation"])
    selected = max(models, key=lambda name: validation[name]["macro_f1"])
    model = models[selected]
    test = evaluate(model, splits["test"])
    dummy = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    dummy_test = evaluate(dummy, splits["test"])
    # Persist exactly the training-only pipeline that was evaluated.
    artifact = {
        "format_version": 1,
        "pipeline": model,
        "max_words": MAX_WORDS,
        "rule_version": metadata["rule_version"],
        "selected_model": selected,
        "label_status": metadata["label_status"],
        "versions": {"sklearn": sklearn.__version__},
    }
    model_path = model_dir / "text_baseline.joblib"
    joblib.dump(artifact, model_path, compress=3)
    reloaded = joblib.load(model_path)["pipeline"]
    test_x = [r["text"] for r in splits["test"]]
    if not np.array_equal(model.predict(test_x), reloaded.predict(test_x)):
        raise AssertionError("Saved-model prediction mismatch")
    probabilities = reloaded.predict_proba(test_x)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise AssertionError("Invalid probabilities")
    with model_path.open("rb") as stream:
        model_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    # Explicit tokens are provided for the Data deliverable; the fitted vectorizer
    # remains the authoritative inference tokenizer and vocabulary.
    vectorizer = model.named_steps["tfidf"]
    tokenizer, preprocessor = vectorizer.build_tokenizer(), vectorizer.build_preprocessor()
    token_counts = {}
    for split, rows in splits.items():
        lengths = []
        with (data / f"{split}_tokens.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                tokens = tokenizer(preprocessor(row["text"]))
                lengths.append(len(tokens))
                stream.write(
                    json.dumps({"id": row["id"], "label": row["label"], "tokens": tokens}, ensure_ascii=False)
                    + "\n"
                )
        token_counts[split] = {
            "documents": len(rows),
            "tokens": sum(lengths),
            "median_tokens": float(np.median(lengths)),
        }
    vocabulary = {word: int(index) for word, index in vectorizer.vocabulary_.items()}
    (model_dir / "vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False), encoding="utf-8")
    result = {
        "dataset": metadata,
        "training_class_counts": dict(Counter(train_y)),
        "selection_policy": "best macro-F1 on validation 2018; test never used to fit or select parameters",
        "selected_model": selected,
        "validation": validation,
        "test": test,
        "dummy_test": dummy_test,
        "tokenization": {
            "method": "sklearn word tokenizer, lowercase, unigrams and bigrams, no lemmatization",
            "vocabulary_features": len(vectorizer.vocabulary_),
            "splits": token_counts,
        },
        "artifact": {
            "file": "ml/models/text_baseline.joblib",
            "sha256": model_sha,
            "bytes": model_path.stat().st_size,
        },
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "weak labels, not human gold",
            "not a voice/ASR evaluation",
            "no near-duplicate event grouping",
            "confidence not calibrated on human labels",
        ],
        "checks": [
            "verified input file hashes",
            "no exact body/title/url/input split overlap",
            "verified chronological split",
            "saved-model parity on full test",
            "finite normalized probabilities",
        ],
    }
    (report_dir / "baseline_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "manifest.json").write_text(json.dumps(result["artifact"], indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ml/data/lenta_v3"))
    parser.add_argument("--models", type=Path, default=Path("ml/models"))
    parser.add_argument("--reports", type=Path, default=Path("ml/reports"))
    args = parser.parse_args()
    result = train(args.data, args.models, args.reports)
    print(
        json.dumps(
            {"selected_model": result["selected_model"], "test": result["test"]}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
