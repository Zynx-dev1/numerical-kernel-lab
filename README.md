# numerical-kernel-lab

A small lab for numerical kernel experiments and benchmarking.

## Overview

This repository contains lightweight experiments around numerical computation.

It currently includes:
- a Python baseline benchmark
- a HIP kernel example
- structured benchmark notes
- documentation for experiment direction

## Goals

- build a simple numerical experimentation workflow
- document kernel behavior clearly
- compare CPU baselines with GPU-oriented examples
- create a foundation for more advanced kernel work

## Repository layout

```
numerical-kernel-lab/
├── README.md
├── LICENSE
├── .gitignore
├── benchmarks/
│   └── results.md
├── docs/
│   ├── design.md
│   └── notes.md
├── examples/
│   └── dot_product_hip.cpp
└── scripts/
    ├── run_all.sh
    └── bench.py
```

## Quick start

### Python baseline benchmark

```bash
python scripts/bench.py --mode dot --n 2000000 --steps 3
```

### HIP example

```bash
hipcc examples/dot_product_hip.cpp -o dot_product_hip
./dot_product_hip
```

## Notes

This repository favors simple, readable numerical experiments.
