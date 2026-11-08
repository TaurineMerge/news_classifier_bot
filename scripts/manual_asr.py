from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from src.newsbot.asr import WhisperASR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ручная проверка ASR на реальном аудио"
    )

    parser.add_argument(
        "audio",
        type=Path,
        help="Путь к аудиофайлу: .ogg, .opus, .mp3, .wav, .m4a",
    )

    parser.add_argument(
        "--model",
        default="small",
        help="Размер модели или локальный путь к модели. Например: small, medium, ml/models/faster-whisper-small",
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство: auto, cpu, cuda",
    )

    parser.add_argument(
        "--denoise",
        default="auto",
        choices=["auto", "on", "off"],
        help="Режим денойзинга",
    )

    parser.add_argument(
        "--rnnoise",
        type=Path,
        default=None,
        help="Путь к весам RNNoise, например ml/models/rnnoise/bd.rnnn",
    )

    parser.add_argument(
        "--no-ffmpeg",
        action="store_true",
        help="Не использовать ffmpeg-предобработку",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.audio.exists():
        raise SystemExit(f"Audio file not found: {args.audio}")

    asr = WhisperASR(
        model_size=args.model,
        device=args.device,
        use_ffmpeg=not args.no_ffmpeg,
        denoise_mode=args.denoise,
        rnnoise_model_path=args.rnnoise,
        download_root=Path("ml/models"),
    )

    start = time.perf_counter()
    result = asr.transcribe(args.audio)
    elapsed = time.perf_counter() - start

    print("\n----- ASR RESULT -----")
    print(f"file: {args.audio}")
    print(f"device: {asr.device}")
    print(f"compute_type: {asr.compute_type}")
    print(f"audio_duration_sec: {result.audio_duration_sec:.2f}")
    print(f"elapsed_sec: {elapsed:.2f}")
    print(f"text: {result.text}")


if __name__ == "__main__":
    main()