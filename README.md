# numerical-kernel-lab

[![HIP](https://img.shields.io/badge/HIP-kernel%20lab-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Laboratory for numerical kernel experiments — dot products, reductions, matrix operations, and iterative solvers on AMD GPUs.**

A structured environment for developing and benchmarking numerical computation kernels.

---

## Architecture

```
+--------------------------------------------------------------+
|                        run_all.py                             |
|                   (orchestrator + CLI)                         |
+----------+----------+---------------+-------------------------+
|  Dot     | Matrix   | Iterative     | Polynomial              |
|  Product | Multiply | Solvers       | Evaluation              |
|  FP32    | FP16/32  | Jacobi/Gauss  | Horner/Bernstein        |
+----------+----------+---------------+-------------------------+
|                    utils/                                      |
|  gpu_monitor.py    |  report_generator.py                     |
|  ROCm SMI wrapper  |  markdown + charts                       |
+--------------------------------------------------------------+
```

## Features

| Kernel | What it measures | Key metrics |
|:---|:---|:---|
| **Dot Product** | Reduction performance | GFLOPS, latency |
| **Matrix Multiply** | GEMM performance | TFLOPS, GFLOPS |
| **Iterative Solver** | Convergence + throughput | Iterations, time |
| **Polynomial Eval** | Horner scheme throughput | Ops/sec |
| **GPU Monitor** | Hardware state | Temp, power, VRAM |

---

## Quick Start

```bash
git clone https://github.com/Zynx-dev1/numerical-kernel-lab.git
cd numerical-kernel-lab
pip install -r requirements.txt

python run_all.py
python run_all.py --suite dot matrix
python run_all.py --cpu
```

### HIP Examples

```bash
hipcc examples/dot_product_hip.cpp -o dot_product_hip && ./dot_product_hip
hipcc examples/matrix_mul_hip.cpp -o matrix_mul_hip && ./matrix_mul_hip
hipcc examples/reduction_hip.cpp -o reduction_hip && ./reduction_hip
```

---

## Project Structure

```
numerical-kernel-lab/
+-- benchmarks/
|   +-- __init__.py
|   +-- dot_product.py          # Dot product benchmark
|   +-- matrix_mul.py           # Matrix multiply benchmark
|   +-- iterative_solver.py     # Jacobi/Gauss-Seidel
|   +-- polynomial_eval.py      # Horner/Bernstein evaluation
+-- examples/
|   +-- dot_product_hip.cpp     # HIP dot product kernel
|   +-- matrix_mul_hip.cpp      # HIP matrix multiply kernel
|   +-- reduction_hip.cpp       # HIP reduction kernel
+-- utils/
|   +-- __init__.py
|   +-- gpu_monitor.py          # ROCm SMI wrapper
|   +-- report_generator.py     # Report generation
+-- results/                    # Benchmark output
+-- run_all.py                  # Main entry point
+-- Dockerfile
+-- requirements.txt
+-- pyproject.toml
+-- LICENSE
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check .
python run_all.py --cpu
```

---

## Roadmap

- [ ] Conjugate gradient solver
- [ ] FFT kernel benchmarks
- [ ] Sparse matrix operations
- [ ] Multi-GPU reduction
- [ ] Mixed-precision experiments

---

## License

MIT -- see [LICENSE](LICENSE).
