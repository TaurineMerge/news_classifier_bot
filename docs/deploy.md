# Деплой бота на сервер

```
мерж PR в main
   └─► GitHub Actions (deploy.yml)
         ├─ сборка Docker-образа → ghcr.io/taurinemerge/news_classifier_bot:sha-<коммит>
         └─ SSH на сервер → deploy/deploy.sh sha-<коммит>
               ├─ docker compose pull + up
               ├─ ждёт в логах «bot started» (до 10 мин — первый раз качается whisper)
               └─ не стартовал → автоматический откат на предыдущую версию
```

На сервере живут только: клон репозитория (ради `compose.yaml` и `deploy/`), `.env`,
модель в `ml/models/` и кэш whisper в Docker volume. Код бота — внутри образа.

## Первая настройка (один раз)

Команды для Ubuntu/Debian. Выполняются на сервере.

### 1. Docker

```bash
curl -fsSL https://get.docker.com | sh
```

### 2. Пользователь для деплоя

Отдельный пользователь без sudo, но с правом запускать Docker:

```bash
sudo adduser --disabled-password --gecos "" deploy
sudo usermod -aG docker deploy
```

### 3. SSH-ключ для GitHub Actions

На **своём компьютере** создать отдельный ключ (без пароля — его будет использовать робот):

```bash
ssh-keygen -t ed25519 -C "github-deploy" -f deploy_key -N ""
```

Публичную часть положить на сервер:

```bash
sudo mkdir -p /home/deploy/.ssh
sudo tee -a /home/deploy/.ssh/authorized_keys < deploy_key.pub
sudo chown -R deploy:deploy /home/deploy/.ssh && sudo chmod 700 /home/deploy/.ssh && sudo chmod 600 /home/deploy/.ssh/authorized_keys
```

Проверить с компьютера: `ssh -i deploy_key deploy@<адрес-сервера>` — должно пустить без пароля.

### 4. Проект на сервере

```bash
sudo -iu deploy
git clone https://github.com/TaurineMerge/news_classifier_bot.git ~/newsbot
cd ~/newsbot
cp .env.example .env
nano .env          # BOT_TOKEN — боевого бота, не тестового
```

Модель классификатора (пока DVC пустой) — скопировать с компьютера:

```bash
scp ml/models/text_baseline.joblib deploy@<адрес-сервера>:~/newsbot/ml/models/
```

### 5. Секреты в GitHub

**Settings → Secrets and variables → Actions → New repository secret:**

| Секрет | Значение |
|---|---|
| `DEPLOY_HOST` | адрес сервера (IP или домен) |
| `DEPLOY_USER` | `deploy` |
| `DEPLOY_SSH_KEY` | содержимое файла `deploy_key` (приватный ключ целиком, с `-----BEGIN…` и `-----END…`) |
| `DEPLOY_PORT` | порт SSH, если не 22 (необязательно) |
| `DEPLOY_PATH` | путь к проекту, если не `~/newsbot` (необязательно) |

После этого файл `deploy_key` на компьютере можно удалить.

### 6. Первый деплой

**Actions → Deploy → Run workflow** (ветка `main`). Первый запуск дольше обычного:
сборка образа и скачивание whisper (~480 МБ для `small`).

## Обновление

Ничего делать не нужно: **мерж PR в `main` = деплой.** Статус — во вкладке Actions.
Если новая версия не стартовала, `deploy.sh` сам вернёт предыдущую, а job станет красным.

Бот на время перезапуска (несколько секунд) не отвечает; сообщения, присланные в это время,
Telegram доставит после старта.

## Откат вручную

1. Найти хеш нужного коммита в истории `main` (например, `1a2b3c4`).
2. **Actions → Deploy → Run workflow**, в поле `ref` — этот хеш.

Сборки не будет: берётся готовый образ из GHCR. Работает для любой версии, задеплоенной
после появления этого workflow.

Откат прямо на сервере (если GitHub недоступен) — на предыдущую версию:

```bash
cd ~/newsbot && deploy/deploy.sh "$(cat .deploy/previous_tag)"
```

## Обновление модели классификатора

Модель не входит в образ, поэтому её обновление — без пересборки:

```bash
scp ml/models/text_baseline.joblib deploy@<адрес-сервера>:~/newsbot/ml/models/
ssh deploy@<адрес-сервера> "cd ~/newsbot && docker compose restart bot"
```

Когда DVC remote переедет на сервер, вместо `scp` будет `dvc pull`.

## Изменение настроек

`.env` на сервере правится руками и в git не попадает:

```bash
ssh deploy@<адрес-сервера>
cd ~/newsbot && nano .env && docker compose up -d bot
```

## Полезное на сервере

| Что | Команда |
|---|---|
| Логи бота | `docker compose logs -f bot` |
| Какая версия работает | `cat .deploy/current_tag` |
| Статус контейнера | `docker compose ps` |
| Перезапуск | `docker compose restart bot` |
| Место на диске под Docker | `docker system df` |

`deploy.sh` хранит на сервере только текущий и предыдущий образы (около 1,5 ГБ каждый),
старые удаляет сам.
