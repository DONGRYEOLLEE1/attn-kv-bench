"""Plot benchmark results into PNG files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


ENGINES = ["baseline_mha_no_cache", "gqa_contiguous_cache", "gqa_paged_cache", "mla_latent_cache"]
LABELS = ["Baseline\nMHA", "GQA\nContiguous", "GQA\nPaged", "MLA\nLatent"]
COLORS = ["#8d99ae", "#2a9d8f", "#457b9d", "#e76f51"]


def render_plots(results_path: str = "results/benchmark_results.json", output_dir: str = "results/plots"):
    data = load_results(results_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    engines = data["engines"]
    ttft = [engines[n]["mean_ttft_ms"] for n in ENGINES]
    tps = [engines[n]["mean_tokens_per_s"] for n in ENGINES]
    total = [engines[n]["mean_total_time_ms"] for n in ENGINES]
    kv_bytes = [engines[n]["peak_kv_bytes"] for n in ENGINES]

    # Latency & Throughput
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(LABELS, ttft, color=COLORS)
    axes[0].set_title("Mean TTFT (ms)")
    axes[0].set_ylabel("ms")

    axes[1].bar(LABELS, tps, color=COLORS)
    axes[1].set_title("Mean Decode Throughput")
    axes[1].set_ylabel("tokens/s")

    axes[2].bar(LABELS, total, color=COLORS)
    axes[2].set_title("Mean End-to-End Latency")
    axes[2].set_ylabel("ms")

    for ax in axes:
        ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(out / "latency_throughput.png", dpi=180)
    plt.close(fig)

    # KV Memory
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(LABELS, kv_bytes, color=COLORS)
    ax.set_title("Peak KV Cache Bytes")
    ax.set_ylabel("bytes")
    for bar, val in zip(bars, kv_bytes):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", labelrotation=10)
    fig.tight_layout()
    fig.savefig(out / "kv_memory.png", dpi=180)
    plt.close(fig)

    # KV elements per token comparison
    if "kv_cache_elements_per_token" in data:
        elems = data["kv_cache_elements_per_token"]
        names = list(elems.keys())
        values = list(elems.values())
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(names, values, color=["#8d99ae", "#2a9d8f", "#e76f51"])
        ax.set_title("KV Cache Elements per Token")
        ax.set_ylabel("elements")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / "kv_elements.png", dpi=180)
        plt.close(fig)

    return {
        "latency_throughput_png": str(out / "latency_throughput.png"),
        "kv_memory_png": str(out / "kv_memory.png"),
        "kv_elements_png": str(out / "kv_elements.png"),
    }
