from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MANIFEST = RESULTS / "MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def render() -> str:
    paths = sorted(
        path
        for path in RESULTS.rglob("*")
        if path.is_file() and path != MANIFEST and path.name != "README.md"
    )
    return "".join(
        f"{digest(path)}  {path.relative_to(ROOT).as_posix()}\n" for path in paths
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not MANIFEST.exists() or MANIFEST.read_text(encoding="utf-8") != expected:
            raise SystemExit("result manifest is missing or stale")
        print("result manifest is current")
        return 0
    MANIFEST.write_text(expected, encoding="utf-8")
    print(f"updated {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
