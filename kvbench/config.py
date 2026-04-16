"""Benchmark configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 260
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    intermediate_size: int = 768
    max_seq_len: int = 1024
    dropout: float = 0.0

    # MLA parameters
    kv_lora_rank: int = 48
    v_head_dim: int = 32

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


@dataclass
class BenchmarkConfig:
    seed: int = 42
    device: str = "auto"
    max_new_tokens: int = 16
    block_size: int = 16
    warmup_runs: int = 1
    measure_runs: int = 3

    def to_dict(self):
        return asdict(self)
