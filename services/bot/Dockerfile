# Сборка из корня репозитория:
#   docker build -f services/bot/Dockerfile -t newsbot-bot .
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/cache/huggingface

WORKDIR /app

# Сначала только зависимости — этот слой кэшируется, пока не поменялся uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --group bot --no-install-project

COPY src ./src
COPY services ./services
RUN uv sync --locked --no-dev --group bot

# Не запускаем под root. /cache — кэш весов whisper (volume), ml/models — классификатор (volume)
RUN useradd --create-home --uid 1000 bot \
    && mkdir -p /cache/huggingface /app/ml/models \
    && chown -R bot:bot /cache /app/ml
USER bot

CMD ["python", "-m", "services.bot"]
