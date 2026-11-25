#!/usr/bin/env bash
# Обновление бота на сервере до образа с указанным тегом, с автооткатом.
#
#   deploy/deploy.sh <tag>        например: deploy/deploy.sh sha-1a2b3c4
#
# Запускается из GitHub Actions по SSH, но можно и вручную на сервере.
# Если новая версия не стартовала за HEALTH_TIMEOUT секунд — возвращает предыдущую.
set -euo pipefail

TAG="${1:?usage: deploy/deploy.sh <image-tag>}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-600}"  # первый запуск качает whisper — это долго
READY_MARKER="bot started"
IMAGE="${BOT_IMAGE:-ghcr.io/taurinemerge/news_classifier_bot}"

cd "$(dirname "$0")/.."
STATE_DIR=".deploy"
mkdir -p "$STATE_DIR"
PREVIOUS_TAG="$(cat "$STATE_DIR/current_tag" 2>/dev/null || true)"

log() { echo "[deploy] $*"; }
fail() { echo "[deploy] ERROR: $*" >&2; exit 1; }

# --- проверки окружения сервера ---
command -v docker >/dev/null || fail "docker не установлен"
[[ -f .env ]] || fail ".env не найден — скопируйте .env.example в .env и заполните"
if ! grep -qE '^CLASSIFIER_BACKEND=stub' .env; then
    model_path="$(grep -E '^CLASSIFIER_MODEL_PATH=' .env | cut -d= -f2- || true)"
    model_path="${model_path:-ml/models/text_baseline.joblib}"
    [[ -f "$model_path" ]] || fail "нет модели классификатора: $model_path"
fi

container_id() { docker compose ps -q bot; }

wait_until_ready() {
    local started=$SECONDS id restarts
    while (( SECONDS - started < HEALTH_TIMEOUT )); do
        id="$(container_id)"
        if [[ -n "$id" ]]; then
            restarts="$(docker inspect -f '{{.RestartCount}}' "$id")"
            if [[ "$(docker inspect -f '{{.State.Running}}' "$id")" != "true" || "$restarts" != "0" ]]; then
                log "контейнер упал при старте"
                return 1
            fi
            if docker logs "$id" 2>&1 | grep -q "$READY_MARKER"; then
                return 0
            fi
        fi
        sleep 5
    done
    log "бот не стартовал за ${HEALTH_TIMEOUT} с"
    return 1
}

start() {
    BOT_TAG="$1" docker compose pull bot
    BOT_TAG="$1" docker compose up -d --no-build --force-recreate bot
}

# --- деплой ---
log "текущая версия: ${PREVIOUS_TAG:-нет}, новая: $TAG"
start "$TAG"

if wait_until_ready; then
    echo "$TAG" > "$STATE_DIR/current_tag"
    if [[ -n "$PREVIOUS_TAG" && "$PREVIOUS_TAG" != "$TAG" ]]; then
        echo "$PREVIOUS_TAG" > "$STATE_DIR/previous_tag"
    fi
    # Храним только текущую и предыдущую версии (для отката), остальные — удаляем
    docker images "$IMAGE" --format '{{.Tag}}' \
        | grep -vxF -e "$TAG" -e "${PREVIOUS_TAG:-$TAG}" -e latest \
        | xargs -r -I{} docker rmi "$IMAGE:{}" >/dev/null || true
    docker image prune -f >/dev/null
    log "готово: работает $TAG"
    exit 0
fi

log "последние логи новой версии:"
docker compose logs --tail 50 bot || true

if [[ -n "$PREVIOUS_TAG" && "$PREVIOUS_TAG" != "$TAG" ]]; then
    log "откатываюсь на $PREVIOUS_TAG"
    start "$PREVIOUS_TAG"
    if wait_until_ready; then
        fail "деплой $TAG не удался, работает предыдущая версия $PREVIOUS_TAG"
    fi
    fail "деплой $TAG не удался, и откат на $PREVIOUS_TAG тоже не поднялся"
fi
fail "деплой $TAG не удался, предыдущей версии для отката нет"
