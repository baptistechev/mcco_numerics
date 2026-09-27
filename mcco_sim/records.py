"""Output directory: JSON-lines records, committed units (resume), params snapshot, provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import troma

from .params import Params, params_hash

# JSON-lines files are split into parts of at most this size (GitHub warns above 50 MB, refuses above 100 MB)
PART_BYTES = 50 * 1024 ** 2


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def dumps(obj) -> str:
    return json.dumps(obj, default=_json_default)


def _dict_hash(values: dict) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=_json_default).encode()).hexdigest()[:16]


def _git_info(path: Path) -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return {"path": str(path), "commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(git("status", "--porcelain")) if git("rev-parse", "HEAD") else None}


def _hardware() -> dict:
    info = {"hostname": socket.gethostname(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "processor": platform.processor()}
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        with open("/proc/meminfo") as fh:
            info["mem_total_kb"] = int(fh.readline().split()[1])
    except OSError:
        pass
    return info


def _troma_source() -> dict | None:
    """How troma was installed (PEP 610 direct_url.json: git URL + commit, or local path)."""
    try:
        text = importlib.metadata.distribution("troma").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    return json.loads(text) if text else None


def provenance(params: Params, stage: str, workers: int) -> dict:
    troma_root = Path(troma.__file__).resolve().parents[2]
    troma_git = _git_info(troma_root) if (troma_root / ".git").exists() else None  # local checkout only
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "stage": stage,
        "workers": workers,
        "blas_threads": os.environ.get("OMP_NUM_THREADS"),
        "params_hash": params_hash(params),
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                     "troma": troma.__version__, "troma_file": troma.__file__},
        "troma_source": _troma_source(),
        "git": {"troma_lib": troma_git, "mcco_paper": _git_info(Path(__file__).resolve().parents[1])},
        "hardware": _hardware(),
    }


class RecordStore:
    """Read-only view of an output directory (no params check): used for plotting old results."""

    def __init__(self, out_dir: str | Path) -> None:
        self.dir = Path(out_dir)

    def path(self, name: str) -> Path:
        return self.dir / name

    def parts(self, name: str) -> list[Path]:
        """Existing parts of a JSON-lines file, in order: ``runs.jsonl``, ``runs.001.jsonl``, ``runs.002.jsonl``, ..."""
        stem, ext = name.rsplit(".", 1)
        first = self.path(name)
        rest = sorted(self.dir.glob(f"{stem}.[0-9][0-9][0-9].{ext}"))
        return ([first] if first.exists() else []) + rest

    def lines(self, name: str):
        """Lines of a JSON-lines file across all its parts."""
        for path in self.parts(name):
            with open(path) as fh:
                yield from fh

    def done_units(self) -> set[str]:
        return {json.loads(line)["unit"] for line in self.lines("progress.jsonl") if line.strip()}

    def load(self, name: str, fields: list[str] | None = None, contains: list[str] | None = None) -> list[dict]:
        """Records of committed units only (a crash mid-write can leave uncommitted lines).
        With ``fields``, each record keeps only those keys (missing keys give None).
        With ``contains``, only lines holding every given substring are parsed (fast prefilter of
        large files; filter again on the parsed values if a substring could be ambiguous)."""
        if not self.parts(name):
            return []
        done = self.done_units()
        lines = self.lines(name)
        if contains:
            lines = (line for line in lines if all(c in line for c in contains))
        records = (rec for rec in map(json.loads, lines) if rec.get("unit") in done)
        if fields is None:
            return list(records)
        return [{k: rec.get(k) for k in fields} for rec in records]

    def read_json(self, name: str):
        with open(self.path(name)) as fh:
            return json.load(fh)


class Output(RecordStore):
    """An output directory tied to one set of params (``params.json`` + a copy of the params file)."""

    def __init__(self, out_dir: str | Path, params: Params, params_file: str | Path | None = None) -> None:
        super().__init__(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.params = params
        stored = self.dir / "params.json"
        resolved = json.loads(dumps(params.to_dict()))
        if stored.exists():
            with open(stored) as fh:
                if _dict_hash(json.load(fh)) != _dict_hash(resolved):
                    raise SystemExit(f"{stored} differs from the given params; use another --out directory.")
        else:
            with open(stored, "w") as fh:
                json.dump(resolved, fh, indent=2)
            if params_file is not None:
                shutil.copyfile(params_file, self.dir / "params.py")

    def append(self, name: str, records: list[dict]) -> None:
        data = "".join(dumps(rec) + "\n" for rec in records).encode()
        parts = self.parts(name)
        target = parts[-1] if parts else self.path(name)
        if target.exists() and target.stat().st_size > 0 and target.stat().st_size + len(data) > PART_BYTES:
            stem, ext = name.rsplit(".", 1)
            target = self.path(f"{stem}.{len(parts):03d}.{ext}")
        with open(target, "ab") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())

    def write_json(self, name: str, obj) -> None:
        with open(self.path(name), "w") as fh:
            fh.write(json.dumps(obj, indent=2, default=_json_default))
