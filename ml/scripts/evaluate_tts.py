"""Optional end-to-end diagnostic using the team's existing WhisperASR wrapper."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score

from newsbot.asr import WhisperASR
from newsbot.classifier import build_classifier
from newsbot.tfidf_classifier import TfidfTopicClassifier


def words(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower().replace("ё", "е"))


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def evaluate(audio_dir: Path, model_path: Path, report_dir: Path, whisper_size: str = "tiny") -> dict:
    entries = [json.loads(line) for line in (audio_dir / "manifest.jsonl").read_text().splitlines()]
    entries = [r for r in entries if r["split"] == "test"]
    if not entries:
        raise ValueError("No held-out test audio in manifest")
    classifier = build_classifier("tfidf", model_path=model_path)
    asr = WhisperASR(model_size=whisper_size, device="cpu", compute_type="int8")
    records = []
    edits = total_words = 0
    for entry in entries:
        path = (audio_dir / entry["audio"]).resolve()
        if not path.is_relative_to(audio_dir.resolve()):
            raise ValueError("Audio path escapes the dataset directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["audio_sha256"]:
            raise ValueError("Audio fingerprint mismatch")
        transcription = asr.transcribe(str(path))
        reference, hypothesis = words(entry["text"]), words(transcription.text)
        distance = edit_distance(reference, hypothesis)
        edits += distance
        total_words += len(reference)
        expected = TfidfTopicClassifier.TOPICS[entry["label"]]
        clean = classifier.predict(entry["text"])
        noisy = classifier.predict(transcription.text)
        records.append(
            {
                "parent_id": entry["parent_id"],
                "audio": entry["audio"],
                "expected_weak_topic": expected,
                "reference": entry["text"],
                "transcript": transcription.text,
                "clean_topic": clean.topic,
                "asr_topic": noisy.topic,
                "asr_confidence": noisy.confidence,
                "wer": distance / max(len(reference), 1),
                "audio_duration_sec": transcription.audio_duration_sec,
            }
        )
        print(
            f"{len(records)}/{len(entries)}: expected={expected}, text={clean.topic}, ASR={noisy.topic}",
            flush=True,
        )
    labels = list(TfidfTopicClassifier.TOPICS.values())
    y = [r["expected_weak_topic"] for r in records]
    clean_y, asr_y = [r["clean_topic"] for r in records], [r["asr_topic"] for r in records]
    result = {
        "rows": len(records),
        "split": "test",
        "whisper_model": whisper_size,
        "device": "cpu/int8",
        "clean_accuracy": accuracy_score(y, clean_y),
        "asr_accuracy": accuracy_score(y, asr_y),
        "clean_macro_f1": f1_score(y, clean_y, labels=labels, average="macro", zero_division=0),
        "asr_macro_f1": f1_score(y, asr_y, labels=labels, average="macro", zero_division=0),
        "micro_wer": edits / max(total_words, 1),
        "reference_words": total_words,
        "normalization": "lowercase, ё->е, word tokens; numbers and abbreviations not expanded",
        "limitations": [
            "small synthetic single-voice diagnostic",
            "weak topic labels",
            "not real Telegram speech",
        ],
        "source_manifest_sha256": hashlib.sha256((audio_dir / "manifest.jsonl").read_bytes()).hexdigest(),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "tts_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (audio_dir / "asr_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=Path("ml/data/tts_v3"))
    parser.add_argument("--model", type=Path, default=Path("ml/models/text_baseline.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("ml/reports"))
    parser.add_argument("--whisper", default="tiny")
    args = parser.parse_args()
    print(
        json.dumps(evaluate(args.audio, args.model, args.reports, args.whisper), ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
