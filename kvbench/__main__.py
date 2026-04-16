"""CLI entry point."""

from __future__ import annotations

import argparse
import json

from .benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="attn-kv-bench: Attention & KV Cache Serving Benchmark")
    sub = parser.add_subparsers(dest="command")
    bench = sub.add_parser("benchmark", help="Run the serving benchmark")
    bench.add_argument("--output", default="results/benchmark_results.json")
    args = parser.parse_args()

    if args.command == "benchmark":
        result = run_benchmark(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
