"""Compare three fixed classifiers using the original Lenta v3 split."""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from ml.scripts.train_text_baseline import evaluate, make_pipeline, read_splits


def compare(data: Path, output: Path) -> dict:
    splits, metadata = read_splits(data)
    estimators = {
        "logistic_regression": LogisticRegression(C=1, max_iter=1000, random_state=42),
        "multinomial_nb": MultinomialNB(alpha=0.5),
        "linear_svc": LinearSVC(C=1, random_state=42),
    }
    models, results = {}, {}
    for name, estimator in estimators.items():
        start = perf_counter()
        model = make_pipeline(estimator).fit(
            [r["text"] for r in splits["train"]],
            [r["label"] for r in splits["train"]],
        )
        elapsed = perf_counter() - start
        models[name] = model
        results[name] = {
            "parameters": estimator.get_params(),
            "fit_seconds": elapsed,
            "validation": evaluate(model, splits["validation"]),
        }
    selected = max(results, key=lambda name: results[name]["validation"]["macro_f1"])
    for name, model in models.items():
        results[name]["test"] = evaluate(model, splits["test"])
    report = {
        "dataset": metadata,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "selection": "maximum validation macro-F1 among these three fixed configurations",
        "selected_model": selected,
        "models": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ml/data/lenta_v3"))
    parser.add_argument("--output", type=Path, default=Path("ml/reports/integration_experiments.json"))
    args = parser.parse_args()
    result = compare(args.data, args.output)
    for name, run in result["models"].items():
        print(
            name,
            "validation F1",
            run["validation"]["macro_f1"],
            "test accuracy",
            run["test"]["accuracy"],
            "test F1",
            run["test"]["macro_f1"],
        )
    print("Selected:", result["selected_model"])
