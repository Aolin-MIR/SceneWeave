"""Portable storage helpers for the exported SceneWeave controllers."""
from pathlib import Path
import hashlib
import json
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
OUT = ROOT / "data"
RAW = ROOT / "runs/raw"
TOOLS = ROOT / "tools"

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
