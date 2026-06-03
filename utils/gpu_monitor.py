#!/usr/bin/env python3
"""GPU monitoring via ROCm SMI."""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class GPUState:
    gpu_id: int
    name: str = "Unknown"
    temperature_c: float = 0.0
    power_w: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    utilization_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)


def query_gpus() -> list[GPUState]:
    try:
        r = subprocess.run(["rocm-smi", "--showallinfo", "--json"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            data = json.loads(r.stdout)
            gpus = []
            for i, dev in enumerate(data.get("card", [data])):
                gpus.append(GPUState(
                    gpu_id=i,
                    name=dev.get("Card series", "Unknown"),
                    temperature_c=float(dev.get("Temperature (Sensor edge) (C)", 0)),
                    power_w=float(dev.get("Average Graphics Package Power (W)", 0)),
                    memory_used_mb=float(dev.get("VRAM Total Used Memory (B)", 0)) / 1e6,
                    memory_total_mb=float(dev.get("VRAM Total Memory (B)", 0)) / 1e6,
                    utilization_pct=float(dev.get("GPU use (%)", 0)),
                ))
            return gpus
    except Exception:
        pass
    return [GPUState(gpu_id=0, name="Simulated", temperature_c=45.0, power_w=75.0,
                     memory_used_mb=2048, memory_total_mb=16384, utilization_pct=12.0)]


def print_gpu_state(gpus: list[GPUState]):
    table = Table(title="GPU Status")
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Temp (C)", justify="right")
    table.add_column("Power (W)", justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("Util %", justify="right")
    for g in gpus:
        vram = f"{g.memory_used_mb:.0f}/{g.memory_total_mb:.0f} MB"
        table.add_row(str(g.gpu_id), g.name, f"{g.temperature_c:.0f}", f"{g.power_w:.0f}", vram, f"{g.utilization_pct:.0f}")
    console.print(table)
