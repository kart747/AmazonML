"""Logging, timing/RAM telemetry and simple stage caching (resumability)."""
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil

from . import config

_PROC = psutil.Process(os.getpid())


def log(msg):
    rss = _PROC.memory_info().rss / 2**30
    print(f"[{time.strftime('%H:%M:%S')}] [rss {rss:5.1f}G] {msg}", flush=True)


class PeakRSS:
    """Samples this process's RSS in a background thread to record the peak."""

    def __init__(self, interval=0.5):
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(interval,), daemon=True)

    def _run(self, interval):
        while not self._stop.is_set():
            self.peak = max(self.peak, _PROC.memory_info().rss)
            time.sleep(interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, _PROC.memory_info().rss)


@contextmanager
def stage(name, report=None):
    """Time a stage, track its peak RSS and append a JSON line to work/telemetry.jsonl."""
    log(f"START {name}")
    t0 = time.time()
    with PeakRSS() as pk:
        yield
    rec = {"stage": name, "seconds": round(time.time() - t0, 1), "peak_rss_gb": round(pk.peak / 2**30, 2)}
    if report:
        rec.update(report)
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.WORK_DIR / "telemetry.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"END {name}: {rec['seconds']}s, peak RSS {rec['peak_rss_gb']} GB")


def done(path: Path) -> bool:
    """A stage output counts as complete only when its '.done' marker exists."""
    return Path(str(path) + ".done").exists()


def mark_done(path: Path):
    Path(str(path) + ".done").touch()


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
