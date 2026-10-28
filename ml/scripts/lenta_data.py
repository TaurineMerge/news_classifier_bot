"""Reproducible rule-based label selection for a three-topic pilot.

Run from the repository root: uv run python -m ml.scripts.lenta_data --help
"""

import argparse
import bz2
import csv
import hashlib
import json
import random
import re
from collections import Counter
from datetime import date
from pathlib import Path

SOURCE = "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2"
SOURCE_SHA256 = "c4f7f771c7c78221978943a65dfabc9cc129a72a5d477635acc16a771d1fda47"
RULE_VERSION = "lenta-three-topics-v3"
LABELS = ("education", "entertainment", "technology")
MAX_WORDS = 140


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("ё", "е").split())


def rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


EDUCATION_DIRECT = rx(
    r"\b(?:егэ|огэ|абитуриент\w*|учебник\w*|"
    r"(?:школьн\w*|учебн\w*)\s+(?:год\w*|план\w*|программ\w*|каникул\w*|заняти\w*)|"
    r"(?:высш\w*|средн\w*|дошкольн\w*|дистанционн\w*|профессиональн\w*)\s+образовани\w*|"
    r"онлайн[ -]курс\w*)\b"
)
EDUCATION_ANCHOR = rx(
    r"\b(?:школ\w*|школьни\w*|вуз\w*|университет\w*|учител\w*|преподавател\w*|студент\w*|аспирант\w*)\b"
)
EDUCATION_PROCESS = rx(
    r"\b(?:обучени\w*|учеб\w*|экзамен\w*|образовани\w*|каникул\w*|"
    r"урок\w*|абитуриент\w*|аккредитаци\w*|лицензи\w*|стипенди\w*|"
    r"поступлени\w*|отчислени\w*|общежити\w*|зач[её]т\w*|"
    r"бюджетн\w*\s+мест\w*)\b"
)
INCIDENT = rx(
    r"\b(?:уби\w*|убь\w*|застрел\w*|расстрел\w*|стрел\w*|погиб\w*|смерт\w*|умер\w*|"
    r"изби\w*|изнасил\w*|секс\w*|порно\w*|взят\w*|торговл\w*\s+детьми|"
    r"теракт\w*|террор\w*|взрыв\w*|взорв\w*|обстрел\w*|стрельб\w*|"
    r"ранен\w*|ранил\w*|арест\w*|задерж\w*|дтп|авари\w*|врезал\w*|"
    r"поконч\w*|колумбайн\w*|оскорб\w*|надруг\w*|мошеннич\w*|скандал\w*|"
    r"пропал\w*|ушл\w*\s+из\s+дома|обруш\w*|пожар\w*|нападени\w*|похит\w*)\b"
)
NON_EDUCATION = rx(
    r"\b(?:военн\w*|беспилот\w*|самолет\w*|самолёт\w*|нейросет\w*|машинн\w*|"
    r"оружи\w*|корвет\w*|арми\w*|флот\w*|президент\w*|минобороны|"
    r"злокачественн\w*|новообразовани\w*|дипломатическ\w*|спецслужб\w*|"
    r"без\s+высшего\s+образования|вице[ -]мэр\w*|паралимпи\w*|спортсмен\w*)\b"
)
ENTERTAINMENT_CONTENT = rx(
    r"\b(?:фильм\w*|кино\w*|сериал\w*|альбом\w*|концерт\w*|песн\w*|музык\w*|"
    r"сингл\w*|гастрол\w*|премьер\w*|саундтрек\w*|клип\w*|игр\w*|ремейк\w*|"
    r"геймплей\w*|мультиплеер\w*|трек\w*|мультфильм\w*|мультсериал\w*|"
    r"экранизаци\w*|евровидени\w*|график\w*|фестивал\w*)\b"
)
ENTERTAINMENT_NON_CONTENT = rx(
    r"\b(?:больниц\w*|болезн\w*|заболевани\w*|операци\w*|прощани\w*|поминк\w*|"
    r"похорон\w*|расист\w*|расизм\w*|пожен\w*|развел\w*|президент\w*|выбор\w*|"
    r"госпитализ\w*|лечени\w*|суд\w*|христианин\w*)\b"
)
TECH_OBJECT = rx(
    r"\b(?:смартфон\w*|телефон\w*|планшет\w*|процессор\w*|видеокарт\w*|ноутбук\w*|"
    r"компьютер\w*|браузер\w*|приложени\w*|сервис\w*|обновлени\w*|обновлен\w*|"
    r"операционн\w*|android\w*|ios\w*|windows\w*|iphone\w*|ipad\w*|ipod\w*|nexus\w*|"
    r"galaxy\w*|гаджет\w*|очк\w*|принтер\w*|аккумулятор\w*|заряд\w*|outlook\w*|"
    r"умн\w*\s+час\w*|программ\w*|диспле\w*|экран\w*|apple\s+watch|playstation\w*|xbox\w*)\b"
)
TECH_NON_CONTENT = rx(r"\b(?:суд\w*|иск\w*|таможн\w*|аукцион\w*|кофе|антимонопольн\w*)\b")


def label_candidate(row: dict) -> tuple[str | None, str]:
    title = normalize(row["title"])
    topic, tag = row["topic"].strip(), row["tags"].strip()
    labels = []
    if not INCIDENT.search(title):
        if (topic in {"Культура", ""} and tag in {"Кино", "Музыка"}) or (
            topic in {"Наука и техника", ""} and tag == "Игры"
        ):
            if ENTERTAINMENT_CONTENT.search(title) and not ENTERTAINMENT_NON_CONTENT.search(title):
                labels.append(("entertainment", "editorial_tag_and_content"))
        if topic in {"Наука и техника", ""} and tag in {"Гаджеты", "Софт"}:
            if TECH_OBJECT.search(title) and not TECH_NON_CONTENT.search(title):
                labels.append(("technology", "editorial_tag_and_technical_object"))
        if not NON_EDUCATION.search(title) and (
            EDUCATION_DIRECT.search(title)
            or (EDUCATION_ANCHOR.search(title) and EDUCATION_PROCESS.search(title))
        ):
            labels.append(("education", "education_title_rules"))
    if len(labels) == 1:
        return labels[0]
    return None, "conflict" if labels else "out_of_scope"


def split_from_date(value: str) -> str | None:
    match = re.match(r"(\d{4})[-/](\d{2})[-/](\d{2})", value)
    if not match:
        return None
    try:
        published = date(*map(int, match.groups()))
    except ValueError:
        return None
    if not 1999 <= published.year <= 2019:
        return None
    if published.year <= 2017:
        return "train"
    return "validation" if published.year == 2018 else "test"


def prepare(archive: Path, out: Path, seed: int = 20261028) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with archive.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != SOURCE_SHA256:
        raise ValueError("Archive checksum differs from the documented Lenta v1.1 source")
    rng = random.Random(seed)
    caps = {"train": 1500, "validation": 300, "test": 300}
    pools = {(split, label): [] for split in caps for label in LABELS}
    counts, rejected = Counter(), Counter()
    exclusion_path = Path(__file__).resolve().parents[1] / "reports" / "previous_review_ids.txt"
    reviewed_ids = set(exclusion_path.read_text().splitlines())
    seen_body, seen_title, seen_input, seen_url = set(), set(), set(), set()
    csv.field_size_limit(10_000_000)
    total = 0
    with bz2.open(archive, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            total += 1
            if total % 200_000 == 0:
                print(f"Scanned {total:,}", flush=True)
            split = split_from_date(row.get("date", ""))
            if split is None:
                rejected["invalid_date"] += 1
                continue
            label, rule = label_candidate(row)
            if label is None:
                rejected[rule] += 1
                continue
            body = row["text"].strip()
            if len(body.split()) < 30:
                rejected["short_body"] += 1
                continue
            input_text = " ".join(body.split()[:MAX_WORDS])
            body_id = hashlib.sha256(normalize(body).encode()).hexdigest()
            legacy_id = hashlib.sha256(" ".join(body.lower().split()).encode()).hexdigest()
            if split != "train" and (legacy_id in reviewed_ids or body_id in reviewed_ids):
                rejected["previously_reviewed_holdout"] += 1
                continue
            title_key, input_key, url = normalize(row["title"]), normalize(input_text), row["url"].strip()
            if not url or not title_key:
                rejected["empty_identity"] += 1
                continue
            if body_id in seen_body or title_key in seen_title or input_key in seen_input or url in seen_url:
                rejected["duplicate"] += 1
                continue
            seen_body.add(body_id)
            seen_title.add(title_key)
            seen_input.add(input_key)
            seen_url.add(url)
            item = {
                "id": body_id,
                "url": url,
                "date": row["date"],
                "title": row["title"],
                "text": input_text,
                "label": label,
                "label_source": rule,
                "topic": row["topic"],
                "tags": row["tags"],
            }
            key = (split, label)
            counts[key] += 1
            pool = pools[key]
            if len(pool) < caps[split]:
                pool.append(item)
            else:
                i = rng.randrange(counts[key])
                if i < caps[split]:
                    pool[i] = item
    metadata = {
        "rule_version": RULE_VERSION,
        "source": SOURCE,
        "source_sha256": sha,
        "source_rows": total,
        "seed": seed,
        "max_words": MAX_WORDS,
        "previous_review_exclusion_sha256": hashlib.sha256(exclusion_path.read_bytes()).hexdigest(),
        "label_status": "rule-based weak supervision; not human gold",
        "eligible": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        "rejected": dict(rejected),
        "splits": {},
        "split_policy": "train 1999-2017; validation 2018; test 2019; exact deduplication before split",
        "limitations": [
            "near-duplicate events not grouped",
            "filtered editorial domain",
            "old news",
            "no human gold labels",
        ],
    }
    for split in caps:
        per_class = min(len(pools[split, label]) for label in LABELS)
        if per_class < 30:
            raise ValueError(f"Too few examples for {split}: {per_class}")
        rows = []
        for label in LABELS:
            rng.shuffle(pools[split, label])
            rows.extend(pools[split, label][:per_class])
        rng.shuffle(rows)
        path = out / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as stream:
            for item in rows:
                stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        metadata["splits"][split] = {
            "rows": len(rows),
            "per_class": per_class,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("ml/data/lenta-ru-news.csv.bz2"))
    parser.add_argument("--out", type=Path, default=Path("ml/data/lenta_v3"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
