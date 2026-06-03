#!/usr/bin/env python3
"""Generate benchmark reports."""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

from rich.console import Console

console = Console()


def generate_report(results_dir: Path):
    output = results_dir / "BENCHMARK_REPORT.md"
    json_files = sorted(results_dir.glob("*_results.json"))
    if not json_files:
        console.print("[yellow]No results found[/yellow]")
        return

    lines = ["# Benchmark Report", f"\nGenerated: {datetime.now().isoformat()}\n"]
    for jf in json_files:
        data = json.loads(jf.read_text())
        name = jf.stem.replace("_results", "").replace("_", " ").title()
        lines.append(f"## {name}\n")
        if isinstance(data, list) and data:
            keys = list(data[0].keys())
            lines.append("| " + " | ".join(keys) + " |")
            lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
            for row in data:
                lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
            lines.append("")

    output.write_text("\n".join(lines))
    console.print(f"[green]Report: {output}[/green]")
