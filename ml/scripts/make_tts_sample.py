"""Create a local macOS TTS pilot; audio inherits the parent document split.

uv run --group ml python -m ml.scripts.make_tts_sample --per-class 3
Use --per-class 0 to synthesize all prepared documents (much larger).
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from ml.scripts.lenta_data import LABELS
from ml.scripts.train_text_baseline import read_splits


def valid_audio(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with wave.open(str(path)) as wav:
            return wav.getnframes() > 0 and any(wav.readframes(wav.getnframes()))
    except (wave.Error, EOFError):
        return False


def synthesize(
    data: Path, out: Path, per_class: int = 3, max_words: int = 45, voice: str = "Milena", rate: int = 170
) -> dict:
    if per_class < 0 or max_words < 1 or rate < 1:
        raise ValueError("Invalid sample size, word limit or speech rate")
    if platform.system() != "Darwin" or shutil.which("say") is None:
        raise RuntimeError("This optional TTS exporter requires macOS say with an installed Russian voice")
    splits, metadata = read_splits(data)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, rows in splits.items():
        selected = []
        for label in LABELS:
            candidates = [r for r in rows if r["label"] == label]
            selected.extend(candidates[:per_class] if per_class else candidates)
        (out / split).mkdir(exist_ok=True)
        for row in selected:
            text = " ".join(row["text"].split()[:max_words])
            key = hashlib.sha256(f"{voice}|{rate}|{text}".encode()).hexdigest()[:16]
            relative = Path(split) / f"{row['id'][:16]}_{key}.wav"
            path = out / relative
            if not valid_audio(path):
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt") as source:
                    source.write(text)
                    source.flush()
                    subprocess.run(
                        [
                            "say",
                            "-v",
                            voice,
                            "-r",
                            str(rate),
                            "-f",
                            source.name,
                            "-o",
                            str(path),
                            "--data-format=LEI16@16000",
                        ],
                        check=True,
                        timeout=60,
                        capture_output=True,
                    )
                if not valid_audio(path):
                    raise ValueError(f"Speech engine produced empty or silent audio: {relative}")
            with wave.open(str(path)) as wav:
                if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                    raise ValueError(f"Unexpected WAV format: {relative}")
                duration = wav.getnframes() / wav.getframerate()
            if duration <= 0:
                raise ValueError(f"Empty generated audio: {relative}")
            manifest.append(
                {
                    "parent_id": row["id"],
                    "split": split,
                    "label": row["label"],
                    "label_status": metadata["label_status"],
                    "text": text,
                    "audio": relative.as_posix(),
                    "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "duration_sec": duration,
                    "voice": voice,
                    "words_per_minute": rate,
                    "source_url": row["url"],
                }
            )
            print(f"{split}/{row['label']}: {relative.name}", flush=True)
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8"
    )
    result = {
        "rows": len(manifest),
        "duration_sec": sum(r["duration_sec"] for r in manifest),
        "engine": "macOS say",
        "voice": voice,
        "rate": rate,
        "max_words": max_words,
        "format": "WAV mono PCM16 16000Hz",
        "parent_split_preserved": True,
        "purpose": "TTS pilot; not a natural human speech benchmark",
        "label_status": metadata["label_status"],
    }
    (out / "metadata.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("ml/data/lenta_v3"))
    parser.add_argument("--out", type=Path, default=Path("ml/data/tts_v3"))
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--max-words", type=int, default=45)
    parser.add_argument("--voice", default="Milena")
    args = parser.parse_args()
    print(
        json.dumps(
            synthesize(args.data, args.out, args.per_class, args.max_words, args.voice),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
