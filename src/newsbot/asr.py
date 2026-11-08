from __future__ import annotations

import logging
import os
from pathlib import Path

from faster_whisper import WhisperModel

from newsbot.audio_preprocess import ASRError, FFmpegAudioPreprocessor
from newsbot.schemas import Transcription

logger = logging.getLogger(__name__)

PathLike = str | Path

# Маркеры ошибок загрузки CUDA-рантайма (cuBLAS/cuDNN и т.п.).
# Ловим по тексту, т.к. CTranslate2 бросает их лениво, при первом forward,
# и тип исключения по версии/сборке плавает. Специфичны — ложных срабатываний
# на cpu-пути не дают (там эти библиотеки не дёргаются).
_CUDA_LIB_MARKERS = (
    "cublas",
    "cudnn",
    "cufft",
    "cusparse",
    "cusolver",
    "nvrtc",
    "curand",
    "nccl",
    "cannot be loaded",
    "cuda driver version is insufficient",
    "no cuda-capable",
    "dynamic linker",
)


def _cuda_device_count() -> int:
    """Количество CUDA-устройств по ДРАЙВЕРУ. Вынесено для мокирования в тестах.

    ВАЖНО: count>0 НЕ означает, что cublas/cudnn нужной мажорной версии
    реально загрузятся (у тебя драйвер видит GPU, а CTranslate2 4.8.5 собран
    под cuBLAS 12, при CUDA 13 dll не поднимается). Поэтому auto не доверяет
    count'у, а пробует cuda и фолбэкает по ошибке загрузки.
    """
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        logger.debug("CUDA device count unavailable", exc_info=True)
        return 0


def _is_cuda_library_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _CUDA_LIB_MARKERS)


class WhisperASR:
    """
    ASR на faster-whisper + ffmpeg-предобработка.

    Веса: по умолчанию faster-whisper скачивает модель по имени (small, medium…)
    в кэш HuggingFace (HF_HOME) при первом запуске и дальше берёт её оттуда.
    Можно передать путь к папке с весами (model_size="ml/models/whisper-small").
    Полностью офлайн: local_files_only=True или env HF_HUB_OFFLINE=1 —
    только когда веса уже лежат локально.

    Устройство: device="auto" пробует cuda и при ошибке загрузки dll
    тихо переключается на cpu; явный device="cuda" фолбэк НЕ делает
    (падает честно). NEWSBOT_FORCE_CPU=1 — аварийный выключатель для CI/сдачи.
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        compute_type: str | None = None,
        language: str = "ru",
        beam_size: int = 5,
        use_ffmpeg: bool = True,
        denoise_mode: str = "auto",
        rnnoise_model_path: PathLike | None = None,
        ffmpeg_bin: str = "ffmpeg",
        download_root: PathLike | None = None,
        ffmpeg_timeout_sec: float = 300.0,
        local_files_only: bool = False,
    ) -> None:
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.use_ffmpeg = use_ffmpeg

        self._preprocessor: FFmpegAudioPreprocessor | None = None
        if use_ffmpeg:
            self._preprocessor = FFmpegAudioPreprocessor(
                denoise_mode=denoise_mode,
                rnnoise_model_path=rnnoise_model_path,
                ffmpeg_bin=ffmpeg_bin,
                timeout_sec=ffmpeg_timeout_sec,
            )

        # None — стандартный кэш HuggingFace (HF_HOME). Веса в git не храним.
        self._download_root = Path(download_root) if download_root is not None else None
        self._local_files_only = local_files_only

        # Кандидаты (device, compute) в порядке попытки. Модель НЕ грузим
        # в __init__: cublas у CTranslate2 грузится лениво, при первом
        # forward, поэтому фолбэк должен жить вокруг инференса, а не init.
        self._candidates = self._resolve_device_compute(device, compute_type)
        self._start_idx = 0
        self._models: dict[tuple[str, str], WhisperModel] = {}

        # Стартовые атрибуты для логов/тестов до первого transcribe.
        self.device, self.compute_type = self._candidates[0]

    def load(self) -> None:
        """Загрузить модель заранее (например, при старте бота).

        Ошибка «нет весов» видна сразу, а первый пользователь не ждёт загрузку.
        Ошибки CUDA-библиотек всё равно проявятся только при первом инференсе,
        и там сработает фолбэк на CPU.
        """
        self._get_or_build(*self._candidates[self._start_idx])

    def transcribe(self, audio_path: PathLike) -> Transcription:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise ASRError(f"File not found: {audio_path}")

        prepared_path: Path | None = None
        try:
            if self.use_ffmpeg and self._preprocessor is not None:
                prepared_path = self._preprocessor.prepare(audio_path)
                inference_path = prepared_path
            else:
                inference_path = audio_path
            return self._transcribe_with_fallback(inference_path)
        finally:
            if prepared_path is not None:
                self._preprocessor._unlink_quiet(prepared_path)

    def _transcribe_with_fallback(self, inference_path: Path) -> Transcription:
        last_exc: BaseException | None = None
        # Начинаем с последнего рабочего кандидата, чтобы после падения cuda
        # на первом файле не дёргать её на каждом следующем.
        for offset, (cand_device, cand_compute) in enumerate(self._candidates[self._start_idx :]):
            abs_idx = self._start_idx + offset
            try:
                model = self._get_or_build(cand_device, cand_compute)
                segments, info = model.transcribe(
                    str(inference_path),
                    language=self.language,
                    beam_size=self.beam_size,
                    vad_filter=True,
                )
                parts: list[str] = []
                for segment in segments:  # материализация генератора: тут и летит cublas
                    text = getattr(segment, "text", None)
                    if text:
                        parts.append(str(text).strip())
                result_text = " ".join(parts).strip()
                duration = float(getattr(info, "duration", 0.0) or 0.0)

                self.device, self.compute_type = cand_device, cand_compute
                self._start_idx = abs_idx  # запомним рабочий кандидат
                logger.info(
                    "ASR inference: device=%s compute=%s audio_duration=%.2fs text_len=%d",
                    cand_device,
                    cand_compute,
                    duration,
                    len(result_text),
                )
                return Transcription(text=result_text, audio_duration_sec=duration)

            except Exception as exc:
                last_exc = exc
                if _is_cuda_library_error(exc) and abs_idx + 1 < len(self._candidates):
                    logger.warning(
                        "CUDA runtime unavailable (%s); falling back to next candidate",
                        exc,
                    )
                    continue
                raise ASRError(f"ASR failed on device={cand_device}: {exc}") from exc

        raise ASRError(f"ASR failed: {last_exc}") from last_exc

    def _get_or_build(self, device: str, compute_type: str) -> WhisperModel:
        key = (device, compute_type)
        model = self._models.get(key)
        if model is None:
            logger.info(
                "Loading Whisper model=%s device=%s compute=%s",
                self.model_size,
                device,
                compute_type,
            )
            model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=compute_type,
                download_root=str(self._download_root) if self._download_root else None,
                local_files_only=self._local_files_only,
            )
            self._models[key] = model
        return model

    def _resolve_device_compute(
        self,
        device: str,
        compute_type: str | None,
    ) -> list[tuple[str, str]]:
        force_cpu = os.getenv("NEWSBOT_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
        device = str(device).lower()

        if force_cpu or device == "cpu":
            return [("cpu", compute_type or "int8")]

        if device == "cuda":
            if _cuda_device_count() <= 0:
                raise ASRError("CUDA is not available")
            return [("cuda", compute_type or "float16")]  # без fallback: явный запрос

        if device == "auto":
            if _cuda_device_count() > 0:
                return [("cuda", compute_type or "float16"), ("cpu", "int8")]
            return [("cpu", compute_type or "int8")]

        raise ASRError(f"Unsupported device: {device}")
