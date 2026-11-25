"""Пакетная транскрибация аудиофайлов из папки audios."""

import csv
import json
import logging
import time
from pathlib import Path

from newsbot.asr import WhisperASR

# Голосовые из Telegram приходят в .ogg (Opus)
AUDIO_EXTENSIONS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def transcribe_batch(
    audio_dir: Path,
    output_csv: Path,
    output_json: Path,
    model_size: str = "small",
    device: str = "cpu",
    denoise_mode: str = "auto",
    max_files: int | None = None,
) -> None:
    """
    Транскрибирует все аудиофайлы (ogg, opus, mp3, wav, m4a, flac) в audio_dir и сохраняет результаты.

    Args:
        audio_dir: папка с аудиофайлами
        output_csv: путь для CSV-результатов
        output_json: путь для JSON-результатов
        model_size: размер модели или путь к локальной модели
        device: cpu / cuda / auto
        denoise_mode: auto / on / off
        max_files: ограничить количество файлов (для тестирования)
    """
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

    # Собираем все аудиофайлы
    audio_files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)

    if max_files:
        audio_files = audio_files[:max_files]

    if not audio_files:
        logger.warning("No audio files found in %s", audio_dir)
        return

    logger.info("Found %d audio files", len(audio_files))

    # Инициализируем модель
    logger.info("Loading model: %s", model_size)
    asr = WhisperASR(
        model_size=model_size,
        device=device,
        use_ffmpeg=True,
        denoise_mode=denoise_mode,
        download_root=Path("ml/models"),
    )

    results = []
    start_time = time.perf_counter()

    for i, audio_file in enumerate(audio_files, 1):
        file_start = time.perf_counter()

        try:
            result = asr.transcribe(str(audio_file))

            elapsed = time.perf_counter() - file_start

            results.append(
                {
                    "file": audio_file.name,
                    "text": result.text,
                    "duration_sec": result.audio_duration_sec,
                    "processing_sec": round(elapsed, 3),
                }
            )

            # Логируем каждый 10-й файл или последний
            if i % 10 == 0 or i == len(audio_files):
                logger.info(
                    "[%d/%d] %s: %s (%.2fs)",
                    i,
                    len(audio_files),
                    audio_file.name,
                    result.text[:50] + "..." if len(result.text) > 50 else result.text,
                    elapsed,
                )

        except Exception as e:
            logger.error("Failed to transcribe %s: %s", audio_file.name, e)
            results.append(
                {
                    "file": audio_file.name,
                    "text": "",
                    "duration_sec": 0.0,
                    "processing_sec": 0.0,
                    "error": str(e),
                }
            )

    total_time = time.perf_counter() - start_time

    # Сохраняем в CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "text", "duration_sec", "processing_sec", "error"],
        )
        writer.writeheader()
        writer.writerows(results)

    logger.info("CSV saved to %s", output_csv)

    # Сохраняем в JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("JSON saved to %s", output_json)

    # Статистика
    successful = sum(1 for r in results if r["text"])
    failed = len(results) - successful

    logger.info("=" * 60)
    logger.info("Batch transcription complete")
    logger.info("Total files: %d", len(results))
    logger.info("Successful: %d", successful)
    logger.info("Failed: %d", failed)
    logger.info("Total time: %.2f seconds", total_time)
    logger.info("Average per file: %.2f seconds", total_time / len(results))
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch transcribe audio files")

    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("audios"),
        help="Directory with audio files (default: audios)",
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/transcriptions.csv"),
        help="Output CSV file",
    )

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/transcriptions.json"),
        help="Output JSON file",
    )

    parser.add_argument(
        "--model",
        default="small",
        help="Model size or path (default: small)",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device (default: auto)",
    )

    parser.add_argument(
        "--denoise",
        default="auto",
        choices=["auto", "on", "off"],
        help="Denoise mode (default: auto)",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    args = parser.parse_args()

    transcribe_batch(
        audio_dir=args.audio_dir,
        output_csv=args.output_csv,
        output_json=args.output_json,
        model_size=args.model,
        device=args.device,
        denoise_mode=args.denoise,
        max_files=args.max_files,
    )
