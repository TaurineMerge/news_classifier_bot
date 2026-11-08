from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Спецсимволы ffmpeg filtergraph, которые ломают парсинг значения опции.
_FILTERGRAPH_SPECIAL = set("\\:,[]'")


class ASRError(RuntimeError):
    """Базовая ошибка ASR-пайплайна."""


def _escape_filtergraph_value(value: str) -> str:
    """
    Экранирует спецсимволы filtergraph обратным слэшем.

    Двоеточие диска ('C:') и бэкслэши Windows-пути иначе разбираются
    ffmpeg как разделители опций/экранирования и валят граф.
    Прямые слеши '/' спецсимволами не являются и не трогаются.
    """
    out = []
    for ch in value:
        if ch in _FILTERGRAPH_SPECIAL:
            out.append("\\")
        out.append(ch)
    return "".join(out)


class FFmpegAudioPreprocessor:
    """
    Предобработка аудио через ffmpeg для Whisper.
    Конвертирует в WAV 16kHz mono PCM s16le и опционально применяет денойзинг.
    """

    def __init__(
        self,
        denoise_mode: str = "auto",
        rnnoise_model_path: Optional[PathLike] = None,
        ffmpeg_bin: str = "ffmpeg",
        timeout_sec: float = 300.0,
    ) -> None:
        self._denoise_mode = str(denoise_mode).lower()
        self._rnnoise_model_path = (
            Path(rnnoise_model_path)
            if rnnoise_model_path is not None
            else Path("ml/models/rnnoise/bd.rnnn")
        )
        self._ffmpeg_bin = ffmpeg_bin
        self._timeout_sec = timeout_sec

        if self._denoise_mode not in {"auto", "on", "off"}:
            raise ASRError(f"Unsupported denoise_mode: {denoise_mode}")

    def prepare(self, input_path: PathLike) -> Path:
        input_path = Path(input_path)

        if not input_path.exists():
            raise ASRError(f"File not found: {input_path}")

        self._ensure_ffmpeg()

        denoise_filter = self._select_denoise_filter()
        output_path = self._create_temp_wav()

        cmd = self._build_command(
            input_path=input_path,
            output_path=output_path,
            denoise_filter=denoise_filter,
        )

        logger.info(
            "FFmpeg preprocessing: input=%s denoise=%s output=%s",
            input_path,
            denoise_filter or "none",
            output_path,
        )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self._timeout_sec,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            self._unlink_quiet(output_path)
            stderr = result.stderr.strip() or result.stdout.strip() or "unknown error"
            raise ASRError(f"ffmpeg failed: {stderr}")

        return output_path

    def supports_arnndn(self) -> bool:
        return self._has_filter("arnndn")

    def supports_afftdn(self) -> bool:
        return self._has_filter("afftdn")

    def _ensure_ffmpeg(self) -> None:
        if shutil.which(self._ffmpeg_bin) is None:
            raise ASRError(
                f"ffmpeg not found: {self._ffmpeg_bin}. "
                "Please install ffmpeg and add it to your system PATH."
            )

    def _has_filter(self, filter_name: str) -> bool:
        if shutil.which(self._ffmpeg_bin) is None:
            return False

        cmd = [self._ffmpeg_bin, "-hide_banner", "-filters"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10.0,
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            logger.debug("Failed to query ffmpeg filters", exc_info=True)
            return False

        return result.returncode == 0 and filter_name in result.stdout

    def _select_denoise_filter(self) -> Optional[str]:
        if self._denoise_mode == "off":
            return None

        rnnoise_available = self._rnnoise_model_path.exists()

        if rnnoise_available and self.supports_arnndn():
            return "arnndn"

        if self.supports_afftdn():
            return "afftdn"

        if self._denoise_mode == "on":
            raise ASRError("No denoiser available")

        logger.warning("No denoiser available, continuing without denoising")
        return None

    def _create_temp_wav(self) -> Path:
        fd, path = tempfile.mkstemp(prefix="newsbot_asr_", suffix=".wav")
        os.close(fd)
        return Path(path)

    def _build_command(
        self,
        input_path: Path,
        output_path: Path,
        denoise_filter: Optional[str],
    ) -> list[str]:
        if denoise_filter == "arnndn":
            # as_posix() без resolve(): относительный путь от cwd проекта
            # ('ml/models/rnnoise/bd.rnnn') не содержит ':' и '\'.
            # Экранирование на случай абсолютного пути с двоеточием диска.
            model_value = _escape_filtergraph_value(
                self._rnnoise_model_path.as_posix()
            )
            audio_filter = f"arnndn=m={model_value},aresample=16000"
        elif denoise_filter == "afftdn":
            audio_filter = "afftdn,aresample=16000"
        else:
            audio_filter = "aresample=16000"

        return [
            self._ffmpeg_bin,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(input_path),
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            audio_filter,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ]

    @staticmethod
    def _unlink_quiet(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.debug("Failed to remove temporary file %s", path, exc_info=True)