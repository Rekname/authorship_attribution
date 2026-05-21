from __future__ import annotations
import json
from typing import Any
from src.config import CONFIG_PATH


def save_training_config(
    *,
    classifier_metrics: dict[str, dict[str, Any]],
    n_authors: int,
) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps({
        "n_authors": n_authors,
        "classifier_metrics": classifier_metrics,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def load_training_config() -> dict[str, Any] | None:
    if not CONFIG_PATH.exists():
        return None
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
