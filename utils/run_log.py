"""Timestamped run directories and terminal capture for Gradio inference."""

from __future__ import annotations

import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

RUNS_ROOT = os.path.join(".", "results", "object_placement", "runs")
LATEST_RUN_POINTER = os.path.join(RUNS_ROOT, "latest_run.txt")


def _now() -> datetime:
    return datetime.now()


def _stamp(dt: datetime | None = None) -> str:
    dt = dt or _now()
    return dt.strftime("%Y-%m-%d_%H%M%S")


def make_timestamped_run_dir(stage: str, *, runs_root: str = RUNS_ROOT) -> str:
    """Create ``runs/<timestamp>_<stage>/`` and return its path."""
    safe_stage = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stage.strip()) or "run"
    run_id = f"{_stamp()}_{safe_stage}"
    run_dir = os.path.join(runs_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _write_latest_pointer(run_dir: str) -> None:
    os.makedirs(os.path.dirname(LATEST_RUN_POINTER), exist_ok=True)
    with open(LATEST_RUN_POINTER, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(run_dir) + "\n")


class _TimestampedTee:
    """Mirror stdout/stderr to a log file with ISO timestamps on each line."""

    def __init__(self, stream, log_file, *, prefix: str):
        self._stream = stream
        self._log_file = log_file
        self._prefix = prefix
        self._buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self._stream.write(data)
        self._stream.flush()

        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            ts = _now().isoformat(timespec="seconds")
            self._log_file.write(f"{ts} [{self._prefix}] {line}\n")
        self._log_file.flush()
        return len(data)

    def flush(self) -> None:
        self._stream.flush()
        if self._buffer:
            ts = _now().isoformat(timespec="seconds")
            self._log_file.write(f"{ts} [{self._prefix}] {self._buffer}\n")
            self._buffer = ""
        self._log_file.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")


@contextmanager
def capture_run_log(
    run_dir: str,
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
    mark_latest: bool = False,
) -> Iterator["RunLogger"]:
    """Capture stdout/stderr into ``run_dir/run.log`` for one Analyze/Generate click."""
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    manifest_path = os.path.join(run_dir, "run_manifest.json")

    started = _now()
    meta: dict[str, Any] = {
        "run_id": os.path.basename(run_dir),
        "stage": stage,
        "started_at": started.isoformat(timespec="seconds"),
        "run_dir": os.path.abspath(run_dir),
        "log_file": os.path.abspath(log_path),
    }
    if metadata:
        meta.update(metadata)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"=== run start stage={stage} run_dir={os.path.abspath(run_dir)} ===\n")
        log_file.flush()

        logger = RunLogger(run_dir=run_dir, log_file=log_file, stage=stage)
        logger.log(f"Run directory: {os.path.abspath(run_dir)}")

        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = _TimestampedTee(old_out, log_file, prefix="stdout")
        sys.stderr = _TimestampedTee(old_err, log_file, prefix="stderr")

        status = "ok"
        error_text = ""
        try:
            yield logger
        except Exception as exc:
            status = "error"
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            logger.log(f"Run failed: {exc}")
            log_file.write(error_text)
            log_file.flush()
            raise
        finally:
            sys.stdout = old_out
            sys.stderr = old_err

            finished = _now()
            meta["finished_at"] = finished.isoformat(timespec="seconds")
            meta["duration_sec"] = round((finished - started).total_seconds(), 3)
            meta["status"] = status
            if error_text:
                meta["error"] = error_text.strip()[-4000:]

            logger.merge_manifest(meta)
            write_run_manifest(manifest_path, meta)

            log_file.write(
                f"=== run end stage={stage} status={status} "
                f"duration_sec={meta['duration_sec']} ===\n"
            )
            log_file.flush()

            if mark_latest and status == "ok":
                _write_latest_pointer(run_dir)


class RunLogger:
    def __init__(self, *, run_dir: str, log_file, stage: str):
        self.run_dir = run_dir
        self._log_file = log_file
        self.stage = stage
        self._manifest_extra: dict[str, Any] = {}

    def log(self, message: str, *, also_print: bool = True) -> None:
        ts = _now().isoformat(timespec="seconds")
        line = f"[gradio-run {self.stage}] {message}"
        formatted = f"{ts} {line}"
        self._log_file.write(formatted + "\n")
        self._log_file.flush()
        if also_print:
            print(formatted, flush=True)

    def set_manifest(self, **fields: Any) -> None:
        self._manifest_extra.update(fields)

    def merge_manifest(self, manifest: dict[str, Any]) -> None:
        manifest.update(self._manifest_extra)


def write_run_manifest(path: str, manifest: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
