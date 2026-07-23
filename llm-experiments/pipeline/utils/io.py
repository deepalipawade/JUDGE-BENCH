from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    try:
        temp.replace(path)
    except PermissionError:
        fallback = path.with_suffix(path.suffix + f".{int(time.time())}.bak")
        try:
            temp.replace(fallback)
            print(f"[WARN] PermissionError on {path}; backup saved to {fallback}")
        except Exception as exc:
            print(f"[ERROR] Failed to save: {exc}")
    except Exception as exc:
        print(f"[ERROR] Unexpected error saving JSON: {exc}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_records(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    records = data.get("records", [])
    return [r for r in records if isinstance(r, dict)]
