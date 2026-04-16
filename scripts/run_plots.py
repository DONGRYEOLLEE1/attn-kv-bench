"""Generate plots from benchmark results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kvbench.plots import render_plots

if __name__ == "__main__":
    paths = render_plots("results/benchmark_results.json", "results/plots")
    for name, path in paths.items():
        print(f"  {name}: {path}")
