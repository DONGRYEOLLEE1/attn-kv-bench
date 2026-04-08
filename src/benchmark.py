from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokendock.benchmark import run_benchmark


def main():
    output = ROOT / "results" / "benchmark_results.json"
    result = run_benchmark(str(output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()

