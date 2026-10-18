"""Pack the benchmark code (for upload to GitHub / sending to the authors) into ../cl-benchmark_code.zip.

Excludes downloaded datasets, caches, local Kaggle build/output folders and run logs.
"""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "cl-benchmark_code.zip"
SKIP_DIRS = {"data", "__pycache__", ".git", ".venv", "build", "out", "locks", "failed", "logs", "manual"}
TOKEN = re.compile(rb"KGAT_[A-Za-z0-9]{6,}")

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(ROOT.rglob("*")):
        rel = f.relative_to(ROOT)
        if not f.is_file() or SKIP_DIRS & set(rel.parts) or f.suffix in {".pyc", ".tmp"}:
            continue
        if f.suffix in {".py", ".md", ".txt", ".json", ".csv", ".cfg", ".toml", ".yml", ".yaml"} and TOKEN.search(f.read_bytes()):
            raise SystemExit(f"API token found in {rel}")
        z.write(f, ("cl-benchmark" / rel).as_posix())
        n += 1
print("wrote", OUT, n, "files", f"{OUT.stat().st_size / 1e6:.1f} MB")
