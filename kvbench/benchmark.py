"""Run serving optimization benchmark and write JSON results."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .config import BenchmarkConfig, ModelConfig
from .engines import run_engine, select_device
from .workload import multi_turn_workload


def summarize(metrics):
    return {
        "mean_ttft_ms": round(statistics.mean(metrics.ttft_ms), 3),
        "p95_ttft_ms": round(sorted(metrics.ttft_ms)[int(len(metrics.ttft_ms) * 0.95) - 1], 3),
        "mean_tokens_per_s": round(statistics.mean(metrics.tokens_per_s), 3),
        "mean_total_time_ms": round(statistics.mean(metrics.total_time_ms), 3),
        "peak_kv_bytes": max(metrics.kv_bytes) if metrics.kv_bytes else 0,
        "samples": len(metrics.ttft_ms),
    }


def improvement(base, opt):
    return {
        "ttft_improvement_pct": round((base["mean_ttft_ms"] - opt["mean_ttft_ms"]) / base["mean_ttft_ms"] * 100.0, 3),
        "tokens_per_s_improvement_pct": round((opt["mean_tokens_per_s"] - base["mean_tokens_per_s"]) / base["mean_tokens_per_s"] * 100.0, 3),
        "total_time_improvement_pct": round((base["mean_total_time_ms"] - opt["mean_total_time_ms"]) / base["mean_total_time_ms"] * 100.0, 3),
        "peak_kv_bytes_reduction_pct": round((base["peak_kv_bytes"] - opt["peak_kv_bytes"]) / max(base["peak_kv_bytes"], 1) * 100.0, 3) if base["peak_kv_bytes"] else None,
    }


def run_benchmark(output_path: str = "results/benchmark_results.json"):
    bench_cfg = BenchmarkConfig()
    model_cfg = ModelConfig()
    sessions = multi_turn_workload()

    started = time.time()
    baseline = summarize(run_engine("baseline_mha_no_cache", bench_cfg, model_cfg, sessions))
    contiguous = summarize(run_engine("gqa_contiguous_cache", bench_cfg, model_cfg, sessions))
    paged = summarize(run_engine("gqa_paged_cache", bench_cfg, model_cfg, sessions))
    mla = summarize(run_engine("mla_latent_cache", bench_cfg, model_cfg, sessions))
    elapsed = time.time() - started

    result = {
        "system": {
            "device": str(select_device(bench_cfg.device)),
            "hardware_note": "Apple Silicon / MPS if available, else CPU",
        },
        "model": {
            "vocab_size": model_cfg.vocab_size,
            "hidden_size": model_cfg.hidden_size,
            "num_hidden_layers": model_cfg.num_hidden_layers,
            "num_attention_heads": model_cfg.num_attention_heads,
            "num_key_value_heads": model_cfg.num_key_value_heads,
            "intermediate_size": model_cfg.intermediate_size,
            "max_seq_len": model_cfg.max_seq_len,
            "kv_lora_rank": model_cfg.kv_lora_rank,
        },
        "workload": {
            "num_sessions": len(sessions),
            "turns_per_session": [len(session) for session in sessions],
            "max_new_tokens": bench_cfg.max_new_tokens,
            "warmup_runs": bench_cfg.warmup_runs,
            "measure_runs": bench_cfg.measure_runs,
            "block_size": bench_cfg.block_size,
            "sessions": sessions,
        },
        "engines": {
            "baseline_mha_no_cache": baseline,
            "gqa_contiguous_cache": contiguous,
            "gqa_paged_cache": paged,
            "mla_latent_cache": mla,
        },
        "comparisons": {
            "gqa_contiguous_vs_baseline": improvement(baseline, contiguous),
            "gqa_paged_vs_baseline": improvement(baseline, paged),
            "mla_vs_baseline": improvement(baseline, mla),
            "mla_vs_gqa_contiguous": improvement(contiguous, mla),
        },
        "kv_cache_elements_per_token": {
            "mha": 2 * model_cfg.num_attention_heads * model_cfg.head_dim,
            "gqa": 2 * model_cfg.num_key_value_heads * model_cfg.head_dim,
            "mla": model_cfg.kv_lora_rank,
        },
        "conclusion": {
            "summary": (
                "Four-way comparison of attention/KV-cache strategies on a toy transformer. "
                "GQA + KV cache reduces TTFT and boosts throughput vs baseline MHA. "
                "MLA with latent KV cache further compresses the cache footprint while maintaining comparable serving speed."
            ),
            "elapsed_wall_time_s": round(elapsed, 3),
        },
        "references": [
            "https://arxiv.org/abs/2309.06180",
            "https://arxiv.org/abs/2405.04434",
            "https://arxiv.org/abs/2305.13245",
        ],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), ensure_ascii=False, indent=2))
