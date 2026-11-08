# tests/test_asr_pipeline.py
from __future__ import annotations

import contextlib
import dataclasses
import importlib
import inspect
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


asr_module = pytest.importorskip(
    "src.newsbot.asr",
    reason="Модуль src/newsbot/asr.py ещё не реализован",
)
preprocess_module = pytest.importorskip(
    "src.newsbot.audio_preprocess",
    reason="Модуль src/newsbot/audio_preprocess.py ещё не реализован",
)
schemas_module = pytest.importorskip(
    "src.newsbot.schemas",
    reason="Модуль src/newsbot/schemas.py ещё не реализован",
)


FFmpegAudioPreprocessor = getattr(preprocess_module, "FFmpegAudioPreprocessor", None)
if FFmpegAudioPreprocessor is None:
    pytest.fail("В src/newsbot/audio_preprocess.py должен быть класс FFmpegAudioPreprocessor")

WhisperASR = getattr(asr_module, "WhisperASR", None)
if WhisperASR is None:
    pytest.fail("В src/newsbot/asr.py должен быть класс WhisperASR")

Transcription = getattr(schemas_module, "Transcription", None)
if Transcription is None:
    pytest.fail("В src/newsbot/schemas.py должен быть класс Transcription")


ASRErrorClasses = []
for module in (asr_module, preprocess_module):
    exc = getattr(module, "ASRError", None)
    if isinstance(exc, type) and issubclass(exc, BaseException):
        ASRErrorClasses.append(exc)

ExpectedASRError = tuple(ASRErrorClasses) if ASRErrorClasses else (RuntimeError,)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_pair(cmd: list[str], flag: str, value: str) -> bool:
    """
    Проверяет, что в команде есть пара вида:
        -ar 16000
        -ac 1
    """
    for i in range(len(cmd) - 1):
        if str(cmd[i]) == flag and str(cmd[i + 1]) == str(value):
            return True
    return False


def _get_audio_filter(cmd: list[str]) -> str:
    """
    Достаёт значение аудиофильтра из команды ffmpeg.
    Поддерживает варианты:
        -af "..."
        -filter:a "..."
    """
    for flag in ("-af", "-filter:a"):
        if flag in cmd:
            idx = cmd.index(flag)
            if idx + 1 < len(cmd):
                return str(cmd[idx + 1])
    return ""


def _get_conversion_cmd(commands: list[list[str]]) -> list[str]:
    """
    Ищет реальную команду конвертации, пропуская служебные вызовы
    вида `ffmpeg -filters`.
    """
    for cmd in reversed(commands):
        if "-filters" in cmd:
            continue

        # Команда конвертации обычно содержит выходной .wav,
        # или частоту 16000, или аудиофильтр.
        if (
            any(str(x).endswith(".wav") for x in cmd)
            or _has_pair(cmd, "-ar", "16000")
            or _get_audio_filter(cmd)
        ):
            return cmd

    pytest.fail("Не найдена команда ffmpeg для конвертации аудио")


def _make_fake_run(
    commands: list[list[str]],
    *,
    arnndn_supported: bool = True,
    fail_conversion: bool = False,
):
    """
    Фейковый subprocess.run для ffmpeg.

    - Если вызывают `ffmpeg -filters`, отдаём список фильтров.
    - Если вызывают конвертацию, создаём выходной wav-файл.
    - Если fail_conversion=True, возвращаем ошибку конвертации.
    """

    def fake_run(cmd, *args, **kwargs):
        cmd_list = [str(x) for x in cmd]
        commands.append(cmd_list)

        if "-filters" in cmd_list:
            if arnndn_supported:
                stdout = (
                    "T.S arnndn           Apply noise reduction\n"
                    "T.S afftdn           Apply FFT denoising\n"
                )
            else:
                stdout = "T.S afftdn           Apply FFT denoising\n"

            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

        if fail_conversion:
            return SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="fake ffmpeg failure",
            )

        # Обычно выходной файл идёт последним аргументом.
        output_path = Path(cmd_list[-1])
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake-wav")
        except Exception:
            # Если реализация кладёт вывод не туда, тест упадёт позже
            # на проверке пути.
            pass

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return fake_run


def _patch_preprocess_io(
    monkeypatch: pytest.MonkeyPatch,
    fake_run,
    ffmpeg_path: str = "/usr/bin/ffmpeg",
) -> None:
    """
    Патчит shutil.which и subprocess.run в модуле предобработки
    и глобально, чтобы покрыть разные стили импорта.
    """

    def fake_which(name: str):
        return ffmpeg_path

    if hasattr(preprocess_module, "shutil"):
        monkeypatch.setattr(preprocess_module.shutil, "which", fake_which, raising=False)

    if hasattr(preprocess_module, "which"):
        monkeypatch.setattr(preprocess_module, "which", fake_which, raising=False)

    monkeypatch.setattr(shutil, "which", fake_which, raising=False)

    if hasattr(preprocess_module, "subprocess"):
        monkeypatch.setattr(preprocess_module.subprocess, "run", fake_run, raising=False)

    if hasattr(preprocess_module, "run"):
        monkeypatch.setattr(preprocess_module, "run", fake_run, raising=False)

    monkeypatch.setattr(subprocess, "run", fake_run, raising=False)


def _patch_whisper_model(monkeypatch: pytest.MonkeyPatch):
    """
    Подменяет WhisperModel на мок.

    Патчим и `src.newsbot.asr.WhisperModel`, и `faster_whisper.WhisperModel`,
    чтобы покрыть оба варианта импорта:
        from faster_whisper import WhisperModel
        import faster_whisper
    """
    mock_model_class = MagicMock(name="WhisperModelClass")
    mock_model_instance = MagicMock(name="WhisperModelInstance")
    mock_model_class.return_value = mock_model_instance

    patched = False

    if hasattr(asr_module, "WhisperModel"):
        monkeypatch.setattr(asr_module, "WhisperModel", mock_model_class, raising=False)
        patched = True

    with contextlib.suppress(ImportError):
        faster_whisper = importlib.import_module("faster_whisper")
        monkeypatch.setattr(faster_whisper, "WhisperModel", mock_model_class, raising=False)
        patched = True

    if not patched:
        pytest.fail("Не удалось замокать WhisperModel")

    return mock_model_class, mock_model_instance


def _set_whisper_result(
    mock_model_instance: MagicMock,
    texts: list[str],
    duration: float = 10.0,
) -> None:
    """
    Мокает ответ faster-whisper:
        segments, info = model.transcribe(...)
    """
    segments = [SimpleNamespace(text=text) for text in texts]
    info = SimpleNamespace(duration=duration)
    mock_model_instance.transcribe.return_value = (segments, info)


def _get_transcribe_path(mock_model_instance: MagicMock) -> Path:
    """
    Достаёт путь, который передали в model.transcribe(...).
    """
    call = mock_model_instance.transcribe.call_args
    if call is None:
        pytest.fail("WhisperModel.transcribe не был вызван")

    args, kwargs = call

    if args:
        return Path(args[0])

    for key in ("audio_path", "path", "file", "audio"):
        if key in kwargs:
            return Path(kwargs[key])

    pytest.fail("Не удалось определить путь, переданный в WhisperModel.transcribe")


def _patch_cuda_device_count(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    """
    Патчит проверку CUDA.

    Ожидаем один из вариантов:
        - src.newsbot.asr._cuda_device_count()
        - src.newsbot.asr.cuda_device_count()
        - src.newsbot.asr.get_cuda_device_count()
        - ctranslate2.get_cuda_device_count()
    """
    patched = False

    for attr_name in ("_cuda_device_count", "cuda_device_count", "get_cuda_device_count"):
        if hasattr(asr_module, attr_name):
            monkeypatch.setattr(asr_module, attr_name, lambda: count, raising=False)
            patched = True

    with contextlib.suppress(ImportError):
        ctranslate2 = importlib.import_module("ctranslate2")
        monkeypatch.setattr(
            ctranslate2,
            "get_cuda_device_count",
            lambda: count,
            raising=False,
        )
        patched = True

    if not patched:
        pytest.fail(
            "Реализация должна использовать ctranslate2.get_cuda_device_count "
            "или предоставить тестируемую функцию проверки CUDA, "
            "например src.newsbot.asr._cuda_device_count"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def input_audio(tmp_path: Path) -> Path:
    audio_file = tmp_path / "voice.ogg"
    audio_file.write_bytes(b"fake-audio")
    return audio_file


@pytest.fixture
def rnnoise_model(tmp_path: Path) -> Path:
    model_file = tmp_path / "bd.rnnn"
    model_file.write_bytes(b"fake-rnnoise-model")
    return model_file


# ---------------------------------------------------------------------------
# FFmpegAudioPreprocessor tests
# ---------------------------------------------------------------------------


def test_asr_error_is_defined() -> None:
    assert ASRErrorClasses, (
        "В src/newsbot/asr.py или src/newsbot/audio_preprocess.py "
        "должен быть определён класс ASRError"
    )


def test_prepare_off_no_denoise_and_converts_to_wav(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    preprocessor = FFmpegAudioPreprocessor(denoise_mode="off")
    output_path = Path(preprocessor.prepare(input_audio))

    assert output_path.suffix == ".wav"

    cmd = _get_conversion_cmd(commands)

    assert _has_pair(cmd, "-ar", "16000")
    assert _has_pair(cmd, "-ac", "1")
    assert _has_pair(cmd, "-c:a", "pcm_s16le")
    assert _has_pair(cmd, "-f", "wav")

    audio_filter = _get_audio_filter(cmd)
    assert "afftdn" not in audio_filter
    assert "arnndn" not in audio_filter


def test_prepare_auto_without_rnnoise_uses_afftdn(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands, arnndn_supported=True))

    missing_rnnoise = tmp_path / "missing.rnnn"

    preprocessor = FFmpegAudioPreprocessor(
        denoise_mode="auto",
        rnnoise_model_path=missing_rnnoise,
    )
    preprocessor.prepare(input_audio)

    cmd = _get_conversion_cmd(commands)
    audio_filter = _get_audio_filter(cmd)

    assert "afftdn" in audio_filter
    assert "arnndn" not in audio_filter


def test_prepare_auto_with_rnnoise_uses_arnndn(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
    rnnoise_model: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands, arnndn_supported=True))

    preprocessor = FFmpegAudioPreprocessor(
        denoise_mode="auto",
        rnnoise_model_path=rnnoise_model,
    )
    preprocessor.prepare(input_audio)

    cmd = _get_conversion_cmd(commands)
    audio_filter = _get_audio_filter(cmd)

    assert "arnndn" in audio_filter
    assert rnnoise_model.name in audio_filter


def test_prepare_on_without_arnndn_falls_back_to_afftdn(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands, arnndn_supported=False))

    missing_rnnoise = tmp_path / "missing.rnnn"

    preprocessor = FFmpegAudioPreprocessor(
        denoise_mode="on",
        rnnoise_model_path=missing_rnnoise,
    )
    preprocessor.prepare(input_audio)

    cmd = _get_conversion_cmd(commands)
    audio_filter = _get_audio_filter(cmd)

    assert "afftdn" in audio_filter
    assert "arnndn" not in audio_filter


def test_prepare_on_without_any_denoiser_raises(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
    tmp_path: Path,
) -> None:
    if not hasattr(FFmpegAudioPreprocessor, "supports_afftdn"):
        pytest.skip(
            "Для проверки полного отсутствия денойзера реализация должна "
            "предоставлять метод supports_afftdn()"
        )

    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    missing_rnnoise = tmp_path / "missing.rnnn"

    preprocessor = FFmpegAudioPreprocessor(
        denoise_mode="on",
        rnnoise_model_path=missing_rnnoise,
    )

    monkeypatch.setattr(preprocessor, "supports_arnndn", lambda: False, raising=False)
    monkeypatch.setattr(preprocessor, "supports_afftdn", lambda: False, raising=False)

    with pytest.raises(ExpectedASRError):
        preprocessor.prepare(input_audio)


def test_prepare_missing_input_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    preprocessor = FFmpegAudioPreprocessor()

    with pytest.raises(ExpectedASRError + (FileNotFoundError,)):
        preprocessor.prepare(tmp_path / "missing.wav")


def test_prepare_ffmpeg_not_found_raises(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    def fake_which(name: str):
        return None

    if hasattr(preprocess_module, "shutil"):
        monkeypatch.setattr(preprocess_module.shutil, "which", fake_which, raising=False)

    if hasattr(preprocess_module, "which"):
        monkeypatch.setattr(preprocess_module, "which", fake_which, raising=False)

    monkeypatch.setattr(shutil, "which", fake_which, raising=False)

    preprocessor = FFmpegAudioPreprocessor()

    with pytest.raises(ExpectedASRError + (FileNotFoundError,)):
        preprocessor.prepare(input_audio)


def test_prepare_ffmpeg_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(
        monkeypatch,
        _make_fake_run(commands, fail_conversion=True),
    )

    preprocessor = FFmpegAudioPreprocessor(denoise_mode="off")

    with pytest.raises(ExpectedASRError + (RuntimeError,)):
        preprocessor.prepare(input_audio)


# ---------------------------------------------------------------------------
# WhisperASR tests
# ---------------------------------------------------------------------------


def test_transcribe_with_ffmpeg_returns_contract_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["hello", "world"], duration=12.5)

    asr = WhisperASR(
        model_size="small",
        device="cpu",
        use_ffmpeg=True,
    )

    result = asr.transcribe(str(input_audio))

    assert isinstance(result, Transcription)
    assert result.text.strip() == "hello world"
    assert float(result.audio_duration_sec) == 12.5

    prepared_path = _get_transcribe_path(mock_model_instance)
    assert not prepared_path.exists(), "Временный WAV должен быть удалён после обработки"


def test_transcribe_without_ffmpeg_uses_original_path(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["ok"], duration=3.0)

    prepare_mock = MagicMock(return_value=input_audio)
    monkeypatch.setattr(FFmpegAudioPreprocessor, "prepare", prepare_mock, raising=False)

    asr = WhisperASR(
        model_size="small",
        device="cpu",
        use_ffmpeg=False,
    )

    result = asr.transcribe(str(input_audio))

    assert isinstance(result, Transcription)
    assert result.text == "ok"

    used_path = _get_transcribe_path(mock_model_instance)
    assert used_path == input_audio

    prepare_mock.assert_not_called()


def test_transcribe_empty_segments_returns_empty_text(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    mock_model_instance.transcribe.return_value = ([], SimpleNamespace(duration=0.0))

    asr = WhisperASR(
        model_size="small",
        device="cpu",
        use_ffmpeg=False,
    )

    result = asr.transcribe(str(input_audio))

    assert isinstance(result, Transcription)
    assert result.text == ""
    assert float(result.audio_duration_sec) == 0.0


def test_transcribe_ffmpeg_failure_raises_and_model_not_called(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(
        monkeypatch,
        _make_fake_run(commands, fail_conversion=True),
    )

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["should-not-happen"])

    asr = WhisperASR(
        model_size="small",
        device="cpu",
        use_ffmpeg=True,
    )

    with pytest.raises(ExpectedASRError + (RuntimeError,)):
        asr.transcribe(str(input_audio))

    mock_model_instance.transcribe.assert_not_called()


def test_temp_file_removed_when_model_fails(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    mock_model_instance.transcribe.side_effect = RuntimeError("model failed")

    asr = WhisperASR(
        model_size="small",
        device="cpu",
        use_ffmpeg=True,
    )

    with pytest.raises(ExpectedASRError + (RuntimeError,)):
        asr.transcribe(str(input_audio))

    if mock_model_instance.transcribe.call_args is not None:
        prepared_path = _get_transcribe_path(mock_model_instance)
        assert not prepared_path.exists(), "Временный WAV должен быть удалён даже при ошибке"


# ---------------------------------------------------------------------------
# Device selection tests
# ---------------------------------------------------------------------------


def test_device_auto_uses_cpu_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["cpu-test"])

    _patch_cuda_device_count(monkeypatch, 0)

    asr = WhisperASR(
        model_size="small",
        device="auto",
        use_ffmpeg=False,
    )

    result = asr.transcribe(str(input_audio))

    assert result.text == "cpu-test"

    init_kwargs = mock_model_class.call_args.kwargs
    assert init_kwargs.get("device") == "cpu"
    assert init_kwargs.get("compute_type") == "int8"


def test_device_auto_uses_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["cuda-test"])

    _patch_cuda_device_count(monkeypatch, 1)

    asr = WhisperASR(
        model_size="small",
        device="auto",
        use_ffmpeg=False,
    )

    result = asr.transcribe(str(input_audio))

    assert result.text == "cuda-test"

    init_kwargs = mock_model_class.call_args.kwargs
    assert init_kwargs.get("device") == "cuda"
    assert init_kwargs.get("compute_type") in {"float16", "int8_float16"}


def test_device_cpu_does_not_use_cuda(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["forced-cpu"])

    # Даже если CUDA якобы доступна, при device="cpu" должен использоваться CPU.
    _patch_cuda_device_count(monkeypatch, 1)

    asr = WhisperASR(
        model_size="small",
        device="cpu",
        use_ffmpeg=False,
    )

    result = asr.transcribe(str(input_audio))

    assert result.text == "forced-cpu"

    init_kwargs = mock_model_class.call_args.kwargs
    assert init_kwargs.get("device") == "cpu"
    assert init_kwargs.get("compute_type") == "int8"


def test_device_cuda_unavailable_raises(
    monkeypatch: pytest.MonkeyPatch,
    input_audio: Path,
) -> None:
    commands: list[list[str]] = []
    _patch_preprocess_io(monkeypatch, _make_fake_run(commands))

    mock_model_class, mock_model_instance = _patch_whisper_model(monkeypatch)
    _set_whisper_result(mock_model_instance, ["should-not-happen"])

    _patch_cuda_device_count(monkeypatch, 0)

    try:
        asr = WhisperASR(
            model_size="small",
            device="cuda",
            use_ffmpeg=False,
        )
    except ExpectedASRError + (RuntimeError, ValueError):
        # Допустимо, если реализация проверяет CUDA сразу в конструкторе.
        return

    with pytest.raises(ExpectedASRError + (RuntimeError, ValueError)):
        asr.transcribe(str(input_audio))


# ---------------------------------------------------------------------------
# Contract / schema tests
# ---------------------------------------------------------------------------


def test_transcription_contract_fields() -> None:
    field_names = {field.name for field in dataclasses.fields(Transcription)}
    assert field_names == {"text", "audio_duration_sec"}


def test_transcription_is_frozen() -> None:
    obj = Transcription(text="test", audio_duration_sec=1.0)

    frozen_error = getattr(dataclasses, "FrozenInstanceError", AttributeError)

    with pytest.raises(frozen_error):
        obj.text = "changed"


def test_no_torch_import_in_asr_modules() -> None:
    torch_import_pattern = re.compile(r"^\s*(import\s+torch|from\s+torch)\b", re.M)

    for module in (asr_module, preprocess_module):
        try:
            source = inspect.getsource(module)
        except OSError:
            pytest.skip(f"Не удалось прочитать исходник модуля {module.__name__}")

        assert not torch_import_pattern.search(source), (
            f"В модуле {module.__name__} найден импорт torch. "
            "По ТЗ torch использовать запрещено."
        )