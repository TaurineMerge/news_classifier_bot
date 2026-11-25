# ASR troubleshooting

Только ошибки, воспроизводимые на чужой машине и не выводимые из чтения кода. Формат: симптом → причина → диагностика → действие.

| Симптом | Причина | Диагностика | Действие |
|---|---|---|---|
| `ffmpeg not found` | `ffmpeg` не установлен или не в PATH | `ffmpeg -version` | Установить системно: Linux `apt install ffmpeg`; Windows `winget install Gyan.FFmpeg` + перезапуск терминала |
| `arnndn: Error initializing filters` / `Invalid data found when processing input` при `--denoise auto` | Фильтр `arnndn` есть в сборке, но не может инициализироваться (несовместимость сборки `librnnoise` и/или формата весов) | `ffmpeg -hide_banner -filters \| Select-String arnndn` | В `auto` пайплайн сам откатится на `afftdn`. Если нужен именно `arnndn` — собрать/взять `ffmpeg` с `--enable-librnnoise` и подходящими весами; иначе принять `afftdn` как штатный денойзер |
| `afftdn` отсутствует в выводе фильтров | Урезанная сборка `ffmpeg` без FFT-денойзера | `ffmpeg -hide_banner -filters \| Select-String afftdn` | В `auto` пайплайн продолжит без денойзинга (предупреждение в лог). Для обязательного шумодава (`on`) потребуется полная сборка `ffmpeg` |
| `cublas64_12.dll is not found or cannot be loaded` | Мажиор CUDA на хосте не совпадает с мажором, под который собран CTranslate2 (например CUDA 13 vs cuBLAS 12). Драйвер GPU видит, рантайм нужной версии — нет | `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"` (даст >0, но cuda нерабочая) | В `auto` пайплайн сам фолбэкнет на CPU. Для гарантированного CPU: `$env:NEWSBOT_FORCE_CPU="1"` или явный `--device cpu`. Ставить CUDA-рантайм нужного мажора — только если реально нужен GPU |
| `open() got an unexpected keyword argument 'metadata_errors'` | Несовместимость `faster-whisper` и `av` (PyAV): fw передаёт kwarg, которого нет в установленной версии `av` | `python -c "import faster_whisper, av; print(faster_whisper.__version__, av.__version__)"` | Понизить `av` до совместимой версии (`pip install "av<19"` для fw 1.2.1) либо согласовать версии через lock-файл проекта. Это ошибка окружения, а не кода пайплайна |
| Модель лезет в сеть при каждом запуске (`GET .../revision/main`) | HF-hub по умолчанию сверяет ревизию, даже когда кэш локален | модель уже скачана в кэш HF (`HF_HOME`) | Включить офлайн: `WhisperASR(..., local_files_only=True)` или env `HF_HUB_OFFLINE=1`. Первый запуск на голой машине — без офлайна, чтобы модель скачалась |
| `ModuleNotFoundError: No module named 'newsbot'` | Пакет не установлен в окружение, или (Windows) кириллица в пути к проекту ломает чтение `.pth` в Python 3.11 | `uv run python -c "import newsbot, sys; print(newsbot.__file__)"` | `uv sync`; держать проект в пути из латиницы (`C:\dev\newsbot`). Импортировать только как `newsbot.*`, не `src.newsbot.*` |
| Пустой `text` при отсутствии ошибок | Речь не распознана: тишина/шум/слишком тихая запись/нет голоса; либо VAD отрезал всё | проверить исходный файл другим проигрывателем; посмотреть лог `VAD filter removed ...` | Не баг пайплайна. Использовать валидное речевое аудио; при необходимости снизить требования к уровню сигнала на входе |

## Что НЕ является ошибкой окружения

- Кракозябры в CSV при двойном клике в Excel на русской Windows — особенность просмотра (Excel игнорирует BOM для `.csv`), данные на диске целы. Смотреть JSON или импорт через «Данные → Из текста/CSV» (см. `docs/asr_testing.md`).
- Unit-тесты проходят без `ffmpeg`/GPU/весов — внешние вызовы замоканы, это штатная изоляция.
- Откат `auto` с `arnndn` на `afftdn`, или с `cuda` на `cpu` по ошибке dll — рассчитанное поведение fallback, фиксируется в логе предупреждением, а не исключением.
