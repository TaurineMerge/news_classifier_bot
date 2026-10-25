"""Распознавание речи на faster-whisper.

Пример:
    asr = WhisperASR(model_size="small")
    result = asr.transcribe("voice.ogg")
    print(result.text)

OGG/Opus из Telegram faster-whisper декодирует сам (через PyAV), отдельный ffmpeg не нужен.
"""

import logging

from newsbot.schemas import Transcription

log = logging.getLogger(__name__)


class WhisperASR:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "ru",
        beam_size: int = 5,
    ) -> None:
        """
        model_size:   tiny / base / small / medium / large-v3, либо путь к своим весам
        device:       cpu / cuda
        compute_type: int8 для CPU, float16 для GPU
        """
        # Импорт внутри, чтобы модуль можно было импортировать без загрузки модели
        from faster_whisper import WhisperModel

        log.info("Loading whisper model=%s device=%s compute=%s", model_size, device, compute_type)
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._language = language
        self._beam_size = beam_size

    def transcribe(self, audio_path: str) -> Transcription:
        segments, info = self._model.transcribe(
            audio_path,
            language=self._language,
            beam_size=self._beam_size,
            vad_filter=True,  # вырезает тишину: быстрее и меньше галлюцинаций
        )
        # segments — генератор, распознавание идёт здесь
        text = " ".join(s.text.strip() for s in segments).strip()
        return Transcription(
            text=text,
            audio_duration_sec=info.duration,
        )
