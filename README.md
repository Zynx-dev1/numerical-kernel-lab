# numerical-kernel-lab

A small lab for numerical kernel experiments and benchmarking.

## Overview

This repository focuses on lightweight numerical computation experiments.

It includes:
- a Python baseline benchmark
- a HIP kernel example
- structured benchmark notes
- documentation for experiment direction

## Key features

- reproducible benchmark workflow
- simple numerical examples
- CPU baseline and HIP kernel comparison
- clean technical structure

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

### Run baseline benchmark

```bash
python scripts/bench.py --mode dot --n 2000000 --steps 3
```

### Build and run HIP example

```bash
hipcc examples/dot_product_hip.cpp -o dot_product_hip
./dot_product_hip
```

## Why this project exists

This repository is intended as a simple numerical benchmarking lab for future kernel experimentation.

## License

MIT
