"""Build lockin-server.zip: just the code a server needs (a few hundred KB).

    python deploy/make_bundle.py

Leaves out everything private or bulky: API keys (secrets/, .env), settings,
trade history and logs (storage/), downloaded research data (data/), the
legacy bot and caches. Set the server up fresh through its own dashboard.
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "lockin-server.zip"
INCLUDE_DIRS = ["kalshi_bot", "deploy", "tests"]
INCLUDE_FILES = ["requirements.txt", "requirements-server.txt", "README.md", "RESEARCH.md", "pytest.ini",
                 ".env.example"]
SKIP_PARTS = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".key", ".pem", ".zip", ".log", ".sqlite3"}


def main() -> None:
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and not (set(p.parts) & SKIP_PARTS) and p.suffix.lower() not in SKIP_SUFFIXES:
                    arc = p.relative_to(ROOT).as_posix()
                    data = p.read_bytes()
                    if p.suffix in (".sh", ".service"):          # Linux needs Unix line endings
                        data = data.replace(b"\r\n", b"\n")
                    # real modification times: Python decides whether a cached .pyc is
                    # stale from the source's time and size, so they must change with the code
                    info = zipfile.ZipInfo(arc, date_time=time.localtime(p.stat().st_mtime)[:6])
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = (0o755 if p.suffix == ".sh" else 0o644) << 16
                    z.writestr(info, data)
                    n += 1
        for f in INCLUDE_FILES:
            if (ROOT / f).exists():
                info = zipfile.ZipInfo(f, date_time=time.localtime((ROOT / f).stat().st_mtime)[:6])
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16                 # as packed on Windows these unpack writable by anyone
                z.writestr(info, (ROOT / f).read_bytes())
                n += 1
    print(f"wrote {OUT}  ({n} files, {OUT.stat().st_size / 1024:.0f} KB)")
    print("upload it with:  scp lockin-server.zip root@YOUR_SERVER:/root/")


if __name__ == "__main__":
    main()
