from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokendock.plots import render_plots


def main():
    result = render_plots(
        results_path=str(ROOT / "results" / "benchmark_results.json"),
        output_dir=str(ROOT / "results" / "plots"),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
