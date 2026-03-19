"""
Model versioning – keeps the last N trained models with metadata so
we can roll back if a retrained model degrades.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from kalshi_bot.config import MODEL_DIR, ML
from kalshi_bot.ml.predictor import Predictor


class ModelStore:
    """Manages versioned snapshots of the predictor."""

    META_FILE = MODEL_DIR / "versions.json"

    def __init__(self):
        self._versions: list[dict] = self._load_meta()

    def _load_meta(self) -> list[dict]:
        if self.META_FILE.exists():
            try:
                return json.loads(self.META_FILE.read_text())
            except Exception:
                pass
        return []

    def _save_meta(self):
        self.META_FILE.write_text(json.dumps(self._versions, indent=2))

    def save_version(self, predictor: Predictor, metrics: dict) -> str:
        tag = f"v{int(time.time())}"
        predictor.save(tag)

        entry = {
            "tag": tag,
            "ts": time.time(),
            "metrics": metrics,
            "n_samples": predictor.n_train_samples,
            "val_acc": metrics.get("val_acc", 0),
        }
        self._versions.append(entry)

        # also save as 'latest'
        predictor.save("latest")

        # prune old versions
        while len(self._versions) > ML.model_version_keep:
            old = self._versions.pop(0)
            old_path = MODEL_DIR / f"predictor_{old['tag']}.pkl"
            if old_path.exists():
                old_path.unlink()

        self._save_meta()
        logger.info("Saved model version {} (val_acc={:.2%})", tag, entry["val_acc"])
        return tag

    def load_best(self, predictor: Predictor) -> bool:
        """Load the version with highest val_acc."""
        if not self._versions:
            return predictor.load("latest")
        best = max(self._versions, key=lambda v: v.get("val_acc", 0))
        return predictor.load(best["tag"])

    def load_latest(self, predictor: Predictor) -> bool:
        return predictor.load("latest")

    def rollback(self, predictor: Predictor) -> bool:
        """Load the previous version (second-to-last)."""
        if len(self._versions) < 2:
            logger.warning("No previous version to roll back to")
            return False
        prev = self._versions[-2]
        logger.info("Rolling back to {}", prev["tag"])
        return predictor.load(prev["tag"])

    @property
    def latest_accuracy(self) -> float:
        if self._versions:
            return self._versions[-1].get("val_acc", 0.0)
        return 0.0

    @property
    def version_count(self) -> int:
        return len(self._versions)
