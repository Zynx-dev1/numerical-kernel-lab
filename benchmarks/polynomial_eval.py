#!/usr/bin/env python3
"""Polynomial evaluation benchmark (Horner scheme)."""
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
class PolyResult:
    degree: int
    n_points: int
    time_ms: float
    throughput: float
    mode: str


def horner_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Horner scheme polynomial evaluation."""
    result = np.zeros_like(x)
    for c in coeffs[::-1]:
        result = result * x + c
    return result


def run_poly_suite() -> list[PolyResult]:
    results = []
    x = np.linspace(-1, 1, 1000000, dtype=np.float32)
    for degree in [4, 8, 16, 32, 64]:
        coeffs = np.random.randn(degree + 1).astype(np.float32)
        for _ in range(3):
            horner_eval(coeffs, x)
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            horner_eval(coeffs, x)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        ops = len(x) * degree
        results.append(PolyResult(degree=degree, n_points=len(x), time_ms=avg*1000, throughput=round(ops/(avg*1e9), 2), mode="cpu"))
    return results


def print_results(results):
    table = Table(title="Polynomial Evaluation (Horner)")
    table.add_column("Degree", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("Throughput (Gops/s)", justify="right")
    table.add_column("Mode")
    for r in results:
        table.add_row(str(r.degree), str(r.n_points), f"{r.time_ms:.3f}", str(r.throughput), r.mode)
    console.print(table)


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))
