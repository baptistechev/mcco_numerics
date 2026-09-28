"""Sketches: the troma sketch maps of an instance, and the exact linear algebra of Phi used by
Problem II and the theory checks."""

from __future__ import annotations

import time

import numpy as np
from troma import ConstraintSketchMap, ExplicitSketchMap, get_optimizer

from .instances import Instance
from .params import Params
from .seeds import derive_seed


def sketch_rows(params: Params, spec: dict) -> int:
    N = params.N
    if spec["type"] == "nearest_neighbors":
        return (N - spec["k"] + 1) * 2 ** spec["k"]
    rows = spec["rows"]
    if isinstance(rows, str):
        return sketch_rows(params, params.SKETCHES[rows])
    return int(rows)


def sketch_names(params: Params, descriptor: dict, context: str) -> list[str]:
    """Sketches run on an instance: ``"e1"`` (E1 ensembles, subset by ``e1_instances``) or
    ``"single"`` (sweep, E2 and the E3 instance, sketches with ``single_instance``)."""
    if context == "e1":
        return [name for name, spec in params.SKETCHES.items()
                if spec["e1_instances"] is None or descriptor["instance_id"] < spec["e1_instances"]]
    if context == "single":
        return [name for name, spec in params.SKETCHES.items() if spec["single_instance"]]
    raise ValueError(f"Unknown context {context}")


class StructuredPhi:
    """0/1 nearest-neighbour sketch of troma (row = window i, pattern c; index i*2^k + c)."""

    kind = "structured"

    def __init__(self, N: int, k: int) -> None:
        self.N, self.k = N, k
        self.W = N - k + 1
        self.P = 2 ** k
        self.m = self.W * self.P
        self._xs = np.arange(2 ** N, dtype=np.int64)

    def codes(self, i: int, xs: np.ndarray | None = None) -> np.ndarray:
        xs = self._xs if xs is None else xs
        return (xs >> (self.N - i - self.k)) & (self.P - 1)

    def _window_view(self, v: np.ndarray, i: int) -> np.ndarray:
        """v indexed as (bits before window i, window pattern, bits after): MSB-first index order."""
        return v.reshape(2 ** i, self.P, 2 ** (self.N - i - self.k))

    def apply(self, v: np.ndarray) -> np.ndarray:
        """Phi @ v for a dense v over {0,1}^N (row (i, c) sums v over the x with pattern c in window i)."""
        v = np.asarray(v, dtype=float)
        return np.concatenate([self._window_view(v, i).sum(axis=(0, 2)) for i in range(self.W)])

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Phi^T @ y, i.e. (Phi^T y)(x) = sum over windows i of y[i, pattern of x in window i]."""
        y = np.asarray(y, dtype=float).reshape(self.W, self.P)
        out = np.zeros(2 ** self.N)
        for i in range(self.W):
            self._window_view(out, i)[...] += y[i][None, :, None]
        return out

    def column(self, x: int) -> np.ndarray:
        col = np.zeros(self.m)
        for i in range(self.W):
            col[i * self.P + int(self.codes(i, np.array([x]))[0])] = 1.0
        return col

    def weighted_gram(self, w: np.ndarray) -> np.ndarray:
        """A = Phi diag(w) Phi^T."""
        A = np.zeros((self.m, self.m))
        for i in range(self.W):
            ci = self.codes(i) * self.P
            for j in range(i, self.W):
                block = np.bincount(ci + self.codes(j), weights=w, minlength=self.P ** 2).reshape(self.P, self.P)
                A[i * self.P:(i + 1) * self.P, j * self.P:(j + 1) * self.P] = block
                A[j * self.P:(j + 1) * self.P, i * self.P:(i + 1) * self.P] = block.T
        return A

    def quad_diag(self, A: np.ndarray) -> np.ndarray:
        """phi_x^T A phi_x for every x."""
        out = np.zeros(2 ** self.N)
        codes = [self.codes(i) for i in range(self.W)]
        for i in range(self.W):
            for j in range(self.W):
                block = A[i * self.P:(i + 1) * self.P, j * self.P:(j + 1) * self.P]
                out += block[codes[i], codes[j]]
        return out

    def g_range(self) -> float:
        """max over s, x, x' of |G_{s,x} - G_{s,x'}| (G entries in [0, W])."""
        return float(self.W)


class DensePhi:
    """Dense sketch matrix (troma ExplicitSketchMap.map), processed in column chunks."""

    kind = "dense"

    def __init__(self, matrix: np.ndarray, chunk: int) -> None:
        self.M = matrix
        self.m = matrix.shape[0]
        self.chunk = chunk

    def _chunks(self):
        n = self.M.shape[1]
        for start in range(0, n, self.chunk):
            yield slice(start, min(start + self.chunk, n))

    def apply(self, v: np.ndarray) -> np.ndarray:
        return self.M @ v

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return self.M.T @ np.asarray(y, dtype=float)

    def column(self, x: int) -> np.ndarray:
        return np.array(self.M[:, x], dtype=float)

    def weighted_gram(self, w: np.ndarray) -> np.ndarray:
        A = np.zeros((self.m, self.m))
        for c in self._chunks():
            A += (self.M[:, c] * w[c]) @ self.M[:, c].T
        return A

    def quad_diag(self, A: np.ndarray) -> np.ndarray:
        out = np.empty(self.M.shape[1])
        for c in self._chunks():
            out[c] = np.einsum("ij,ij->j", self.M[:, c], A @ self.M[:, c])
        return out

    def max_col_norm_sq(self) -> float:
        return float(max(np.einsum("ij,ij->j", self.M[:, c], self.M[:, c]).max() for c in self._chunks()))

    def g_range(self) -> float:
        """Cauchy-Schwarz bound: |G_{s,x} - G_{s,x'}| <= 2 max_x ||phi_x||^2."""
        return 2.0 * self.max_col_norm_sq()


class SketchSet:
    """The troma sketch maps of one instance, their decoders, and their Phi."""

    def __init__(self, params: Params, inst: Instance, names: list[str] | None = None) -> None:
        self.params = params
        self.N = inst.N
        self.names = list(params.SKETCHES) if names is None else list(names)
        self.maps: dict = {}
        self.phis: dict = {}
        self.optimizers: dict = {}
        self.seeds: dict = {}
        self.setup_time: dict = {}             # seconds to build each sketch (compute estimate)
        for name in self.names:
            start = time.perf_counter()
            spec = params.SKETCHES[name]
            if spec["type"] == "nearest_neighbors":
                self.maps[name] = ConstraintSketchMap(
                    sketch_length=self.N, interaction_size=spec["k"], constraints="nearest_neighbors")
                self.phis[name] = StructuredPhi(self.N, spec["k"])
                self.optimizers[name] = get_optimizer("spin_chain_nn_max")
                self.seeds[name] = None
            elif spec["type"] == "gaussian":
                seed = derive_seed(params, "random_sketch", inst.d["instance_seed"])
                sketch_map = ExplicitSketchMap(sketch_length=self.N)
                sketch_map.random_sketch(sketch_rows(params, spec), random_state=seed)
                self.maps[name] = sketch_map
                self.phis[name] = DensePhi(sketch_map.map, params.THEORY_CHUNK_COLUMNS)
                self.optimizers[name] = get_optimizer("brute_force_max")
                self.seeds[name] = seed
            else:
                raise ValueError(f"Unknown sketch type {spec['type']}")
            self.setup_time[name] = time.perf_counter() - start


def problem2(phi, y: list[float] | np.ndarray, maximizers: np.ndarray) -> dict:
    """Problem II: argmax_x (Phi^T y)(x). Success = unique argmax that is a maximizer of f."""
    score = phi.adjoint(np.asarray(y, dtype=float))
    top = score.max()
    tol = 1e-12 * max(abs(top), 1e-300)
    argmax_set = np.flatnonzero(score >= top - tol)
    argmax = int(argmax_set[0])
    tie = bool(argmax_set.size > 1)
    return {"problem2_argmax": argmax, "problem2_tie": tie,
            "problem2_success": bool((not tie) and np.isin(argmax, maximizers))}
