"""Run benchmark from CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvbench.benchmark import run_benchmark

if __name__ == "__main__":
    result = run_benchmark("results/benchmark_results.json")
    print(f"Results saved to results/benchmark_results.json")
