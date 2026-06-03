#!/usr/bin/env python3
"""Iterative solver benchmark (Jacobi method)."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class SolverResult:
    n: int
    iterations: int
    time_ms: float
    residual: float
    mode: str


def jacobi_solve(n: int = 100, max_iter: int = 1000, tol: float = 1e-6) -> SolverResult:
    A = np.diag(np.ones(n, dtype=np.float32) * 4) + np.diag(np.ones(n-1, dtype=np.float32) * -1, 1) + np.diag(np.ones(n-1, dtype=np.float32) * -1, -1)
    b = np.ones(n, dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    D = np.diag(A)
    R = A - np.diag(D)

    t0 = time.perf_counter()
    for it in range(1, max_iter + 1):
        x_new = (b - R @ x) / D
        residual = float(np.linalg.norm(x_new - x))
        x = x_new
        if residual < tol:
            dt = time.perf_counter() - t0
            return SolverResult(n=n, iterations=it, time_ms=dt*1000, residual=residual, mode="cpu")
    dt = time.perf_counter() - t0
    return SolverResult(n=n, iterations=max_iter, time_ms=dt*1000, residual=residual, mode="cpu")


def run_solver_suite() -> list[SolverResult]:
    return [jacobi_solve(n=n) for n in [50, 100, 200, 500]]


def print_results(results):
    table = Table(title="Iterative Solver (Jacobi)")
    table.add_column("N", justify="right")
    table.add_column("Iterations", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Residual", justify="right")
    table.add_column("Mode")
    for r in results:
        table.add_row(str(r.n), str(r.iterations), f"{r.time_ms:.2f}", f"{r.residual:.2e}", r.mode)
    console.print(table)


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
