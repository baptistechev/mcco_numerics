"""Output directory: JSON-lines records, committed units (resume), params snapshot, provenance."""

from __future__ import annotations

import hashlib
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


def provenance(params: Params, stage: str, workers: int) -> dict:
    troma_root = Path(troma.__file__).resolve().parents[2]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "stage": stage,
        "workers": workers,
        "blas_threads": os.environ.get("OMP_NUM_THREADS"),
        "params_hash": params_hash(params),
        "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                     "troma": troma.__version__, "troma_file": troma.__file__},
        "git": {"troma_lib": _git_info(troma_root), "mcco_paper": _git_info(Path(__file__).resolve().parents[1])},
        "hardware": _hardware(),
    }


class Output:
    """An output directory tied to one set of params (``params.json`` + a copy of the params file)."""

    def __init__(self, out_dir: str | Path, params: Params, params_file: str | Path | None = None) -> None:
        self.dir = Path(out_dir)
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

    def path(self, name: str) -> Path:
        return self.dir / name

    def append(self, name: str, records: list[dict]) -> None:
        with open(self.path(name), "a") as fh:
            for rec in records:
                fh.write(dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def done_units(self) -> set[str]:
        path = self.path("progress.jsonl")
        if not path.exists():
            return set()
        with open(path) as fh:
            return {json.loads(line)["unit"] for line in fh if line.strip()}

    def load(self, name: str) -> list[dict]:
        """Records of committed units only (a crash mid-write can leave uncommitted lines)."""
        path = self.path(name)
        if not path.exists():
            return []
        done = self.done_units()
        with open(path) as fh:
            return [rec for rec in map(json.loads, fh) if rec.get("unit") in done]

    def write_json(self, name: str, obj) -> None:
        with open(self.path(name), "w") as fh:
            fh.write(json.dumps(obj, indent=2, default=_json_default))

    def read_json(self, name: str):
        with open(self.path(name)) as fh:
            return json.load(fh)
