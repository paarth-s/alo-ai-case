"""Tiny thread-safe JSON cache for LLM calls, so repeat runs (and the regression harness) cost nothing."""
import json
import threading
from pathlib import Path


class DiskCache:
    def __init__(self, path: str | Path):
        self.path, self.lock = Path(path), threading.Lock()
        try:
            self.data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            self.data = {}

    def get_or_compute(self, key: str, fn):
        with self.lock:
            if key in self.data:
                return self.data[key]
        value = fn()  # outside the lock, so parallel calls don't serialise
        with self.lock:
            self.data[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False))
        return value
