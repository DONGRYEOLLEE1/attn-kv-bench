"""Plot benchmark results into PNG files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_plots(results_path: str = "results/benchmark_results.json", output_dir: str = "results/plots"):
    data = load_results(results_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    engines = data["engines"]
    names = ["baseline_mha_no_cache", "gqa_contiguous_cache", "gqa_paged_cache"]
    labels = ["Baseline\nMHA/no-cache", "Optimized\nGQA+KV", "Optimized\nGQA+PagedKV"]

    ttft = [engines[name]["mean_ttft_ms"] for name in names]
    tps = [engines[name]["mean_tokens_per_s"] for name in names]
    total = [engines[name]["mean_total_time_ms"] for name in names]
    kv_bytes = [engines[name]["peak_kv_bytes"] for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(labels, ttft, color=["#8d99ae", "#2a9d8f", "#457b9d"])
    axes[0].set_title("Mean TTFT (ms)")
    axes[0].set_ylabel("ms")

    axes[1].bar(labels, tps, color=["#8d99ae", "#2a9d8f", "#457b9d"])
    axes[1].set_title("Mean Decode Throughput")
    axes[1].set_ylabel("tokens/s")

    axes[2].bar(labels, total, color=["#8d99ae", "#2a9d8f", "#457b9d"])
    axes[2].set_title("Mean End-to-End Latency")
    axes[2].set_ylabel("ms")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=10)

    fig.tight_layout()
    fig.savefig(out / "latency_throughput.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, kv_bytes, color=["#8d99ae", "#2a9d8f", "#457b9d"])
    ax.set_title("Peak KV Cache Bytes")
    ax.set_ylabel("bytes")
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(out / "kv_memory.png", dpi=180)
    plt.close(fig)

    return {
        "latency_throughput_png": str(out / "latency_throughput.png"),
        "kv_memory_png": str(out / "kv_memory.png"),
    }

