#!/usr/bin/env python3
"""Main entry point for numerical-kernel-lab."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="numerical-kernel-lab -- numerical kernel experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_all.py                     # Run all
  python run_all.py --suite dot solver  # Specific suites
  python run_all.py --cpu               # CPU mode
        """,
    )
    parser.add_argument("--suite", nargs="+",
                        choices=["dot", "matrix", "solver", "poly", "all"],
                        default=["all"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", "-o", default="results")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    suites = args.suite if "all" not in args.suite else ["dot", "matrix", "solver", "poly"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    console.print(Panel.fit(
        f"[bold white]numerical-kernel-lab[/bold white]\n"
        f"Suites: {', '.join(suites)}\n"
        f"Mode: {'CPU' if args.cpu else 'GPU (ROCm/HIP)'}",
        title="[red]AMD[/red] Numerical Kernels",
        border_style="red",
    ))

    t_start = time.time()

    if "dot" in suites:
        console.print("\n[bold]=== Dot Product ===[/bold]")
        from benchmarks.dot_product import run_dot_suite, print_results, save_results
        r = run_dot_suite(use_gpu=not args.cpu, iters=args.iters)
        print_results(r)
        save_results(r, output / "dot_results.json")

    if "matrix" in suites:
        console.print("\n[bold]=== Matrix Multiply ===[/bold]")
        from benchmarks.matrix_mul import run_matrix_suite, print_results as mp, save_results as ms
        r = run_matrix_suite(use_gpu=not args.cpu, iters=args.iters)
        mp(r)
        ms(r, output / "matrix_results.json")

    if "solver" in suites:
        console.print("\n[bold]=== Iterative Solver ===[/bold]")
        from benchmarks.iterative_solver import run_solver_suite, print_results as sp, save_results as ss
        r = run_solver_suite()
        sp(r)
        ss(r, output / "solver_results.json")

    if "poly" in suites:
        console.print("\n[bold]=== Polynomial Eval ===[/bold]")
        from benchmarks.polynomial_eval import run_poly_suite, print_results as pp, save_results as ps
        r = run_poly_suite()
        pp(r)
        ps(r, output / "poly_results.json")

    console.print(f"\n[green]Done in {time.time() - t_start:.1f}s[/green]")
    from utils.report_generator import generate_report
    generate_report(output)


if __name__ == "__main__":
    main()
