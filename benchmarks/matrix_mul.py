#!/usr/bin/env python3
"""Matrix multiplication benchmark."""
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

DEFAULT_SIZES = [256, 512, 1024, 2048]
ITERS = 5


@dataclass
class MatrixResult:
    size: int
    time_ms: float
    tflops: float
    mode: str


def run_matrix_suite(sizes=None, use_gpu=False, iters=ITERS):
    sizes = sizes or DEFAULT_SIZES
    results = []
    for s in sizes:
        a = np.random.randn(s, s).astype(np.float32)
        b = np.random.randn(s, s).astype(np.float32)
        for _ in range(2):
            np.dot(a, b)
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            np.dot(a, b)
            times.append(time.perf_counter() - t0)
        avg = np.mean(times)
        tflops = (2 * s**3) / (avg * 1e12)
        results.append(MatrixResult(size=s, time_ms=avg*1000, tflops=round(tflops, 4), mode="cpu"))
    return results


def print_results(results):
    table = Table(title="Matrix Multiply Benchmark")
    table.add_column("Size", justify="right")
    table.add_column("Time (ms)", justify="right")
    table.add_column("TFLOPS", justify="right")
    table.add_column("Mode")
    for r in results:
        table.add_row(f"{r.size}x{r.size}", f"{r.time_ms:.2f}", str(r.tflops), r.mode)
    console.print(table)


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in results], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--steps", type=int, default=ITERS)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    results = run_matrix_suite(args.sizes, use_gpu=not args.cpu, iters=args.steps)
    print_results(results)
    save_results(results, Path("results/matrix_results.json"))


if __name__ == "__main__":
    main()
