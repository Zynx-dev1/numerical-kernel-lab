#!/usr/bin/env python3
import argparse
import time


def dot_bench(n: int, steps: int):
    a = [1.0] * n
    b = [2.0] * n

    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        s = 0.0
        for i in range(n):
            s += a[i] * b[i]
        dt = time.perf_counter() - t0
        times.append(dt)

    avg = sum(times) / len(times)
    return {
        "mode": "dot",
        "n": n,
        "steps": steps,
        "avg_sec": round(avg, 6),
        "times": [round(t, 6) for t in times],
    }


def main():
    parser = argparse.ArgumentParser(description="numerical-kernel-lab bench")
    parser.add_argument("--mode", default="dot", choices=["dot"])
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "dot":
        result = dot_bench(args.n, args.steps)

    print("[bench] result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
