"""Чтение текстовых файлов без encoding ломается на русской Windows (там по умолчанию cp1251)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    p for d in ("src", "services", "ml") for p in (ROOT / d).rglob("*") if p.suffix in {".py", ".ipynb"}
]
BARE_READ_TEXT = re.compile(r"\.read_text\(\s*\)")


def test_read_text_always_has_encoding():
    offenders = [
        f"{path.relative_to(ROOT)}:{number}"
        for path in SOURCES
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if BARE_READ_TEXT.search(line)
    ]
    assert not offenders, "Укажите read_text(encoding='utf-8'): " + ", ".join(offenders)
