"""Download the pinned public archive atomically and verify its SHA-256."""

import argparse
import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

from ml.scripts.lenta_data import SOURCE, SOURCE_SHA256


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(target: Path) -> Path:
    if target.exists():
        if digest(target) != SOURCE_SHA256:
            raise ValueError("Existing archive has a different SHA-256; move it before downloading")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="lenta-", suffix=".part", dir=target.parent)
    os.close(descriptor)
    path = Path(temporary)
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "2",
                "--connect-timeout",
                "20",
                "-o",
                str(path),
                SOURCE,
            ],
            check=True,
        )
        if digest(path) != SOURCE_SHA256:
            raise ValueError("Downloaded archive checksum mismatch")
        path.replace(target)
    finally:
        path.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("ml/data/lenta-ru-news.csv.bz2"))
    args = parser.parse_args()
    print(download(args.out))


if __name__ == "__main__":
    main()
