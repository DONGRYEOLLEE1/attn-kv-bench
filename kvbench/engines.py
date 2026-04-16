"""Baseline, GQA, and MLA serving engines with KV cache variants."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BenchmarkConfig, ModelConfig
from .tokenizer import ByteTokenizer


# ─── Utilities ───────────────────────────────────────────────

def sync_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


# ─── Common Layers ───────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x.to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ─── KV Caches ───────────────────────────────────────────────

class ContiguousKVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        self.keys = [None] * num_layers
        self.values = [None] * num_layers
        self.lengths = [0] * num_layers
        self.capacities = [0] * num_layers
        self.device = device
        self.dtype = dtype
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        batch, kv_heads, seq_len, head_dim = k_new.shape
        needed = self.lengths[layer_idx] + seq_len
        if self.capacities[layer_idx] < needed:
            new_capacity = max(needed, max(1, self.capacities[layer_idx] * 2))
            new_k = torch.zeros((batch, kv_heads, new_capacity, head_dim), device=self.device, dtype=self.dtype)
            new_v = torch.zeros((batch, kv_heads, new_capacity, head_dim), device=self.device, dtype=self.dtype)
            if self.keys[layer_idx] is not None:
                old_len = self.lengths[layer_idx]
                new_k[:, :, :old_len] = self.keys[layer_idx][:, :, :old_len]
                new_v[:, :, :old_len] = self.values[layer_idx][:, :, :old_len]
            self.keys[layer_idx] = new_k
            self.values[layer_idx] = new_v
            self.capacities[layer_idx] = new_capacity
        start = self.lengths[layer_idx]
        end = start + seq_len
        self.keys[layer_idx][:, :, start:end] = k_new
        self.values[layer_idx][:, :, start:end] = v_new
        self.lengths[layer_idx] = end

    def get(self, layer_idx: int):
        if self.keys[layer_idx] is None:
            return None, None
        used = self.lengths[layer_idx]
        return self.keys[layer_idx][:, :, :used], self.values[layer_idx][:, :, :used]

    def kv_bytes(self) -> int:
        total = 0
        for tensor in self.keys + self.values:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


class PagedKVCache:
    def __init__(self, num_layers: int, num_kv_heads: int, head_dim: int, block_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks = math.ceil(max_seq_len / block_size)
        self.device = device
        self.dtype = dtype
        self.k_pools = [torch.zeros((self.max_blocks, num_kv_heads, block_size, head_dim), device=device, dtype=dtype) for _ in range(num_layers)]
        self.v_pools = [torch.zeros((self.max_blocks, num_kv_heads, block_size, head_dim), device=device, dtype=dtype) for _ in range(num_layers)]
        self.lengths = [0] * num_layers

    def _grow(self):
        new_blocks = self.max_blocks * 2
        for layer_idx in range(self.num_layers):
            new_k = torch.zeros((new_blocks, self.num_kv_heads, self.block_size, self.head_dim), device=self.device, dtype=self.dtype)
            new_v = torch.zeros((new_blocks, self.num_kv_heads, self.block_size, self.head_dim), device=self.device, dtype=self.dtype)
            new_k[: self.max_blocks] = self.k_pools[layer_idx]
            new_v[: self.max_blocks] = self.v_pools[layer_idx]
            self.k_pools[layer_idx] = new_k
            self.v_pools[layer_idx] = new_v
        self.max_blocks = new_blocks

    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        seq_len = k_new.shape[2]
        offset = 0
        while offset < seq_len:
            block_index = self.lengths[layer_idx] // self.block_size
            if block_index >= self.max_blocks:
                self._grow()
            inner_offset = self.lengths[layer_idx] % self.block_size
            space = self.block_size - inner_offset
            take = min(space, seq_len - offset)
            self.k_pools[layer_idx][block_index, :, inner_offset : inner_offset + take] = k_new[0, :, offset : offset + take]
            self.v_pools[layer_idx][block_index, :, inner_offset : inner_offset + take] = v_new[0, :, offset : offset + take]
            offset += take
            self.lengths[layer_idx] += take

    def get(self, layer_idx: int):
        length = self.lengths[layer_idx]
        if length == 0:
            return None, None
        used_blocks = math.ceil(length / self.block_size)
        full_k = self.k_pools[layer_idx][:used_blocks].transpose(0, 1).reshape(self.num_kv_heads, used_blocks * self.block_size, self.head_dim)
        full_v = self.v_pools[layer_idx][:used_blocks].transpose(0, 1).reshape(self.num_kv_heads, used_blocks * self.block_size, self.head_dim)
        return full_k[:, :length].unsqueeze(0), full_v[:, :length].unsqueeze(0)

    def kv_bytes(self) -> int:
        total = 0
        for layer_idx in range(self.num_layers):
            used_blocks = math.ceil(self.lengths[layer_idx] / self.block_size)
            if used_blocks == 0:
                continue
            k = self.k_pools[layer_idx][:used_blocks]
            v = self.v_pools[layer_idx][:used_blocks]
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
        return total


class LatentKVCache:
    """MLA-specific cache: stores compressed latent instead of full K, V."""

    def __init__(self, num_layers: int, latent_dim: int, device: torch.device, dtype: torch.dtype):
        self.latents = [None] * num_layers
        self.lengths = [0] * num_layers
        self.capacities = [0] * num_layers
        self.device = device
        self.dtype = dtype
        self.latent_dim = latent_dim

    def append(self, layer_idx: int, compressed: torch.Tensor):
        """compressed: [batch, seq_len, latent_dim]"""
        batch, seq_len, dim = compressed.shape
        needed = self.lengths[layer_idx] + seq_len
        if self.capacities[layer_idx] < needed:
            new_capacity = max(needed, max(1, self.capacities[layer_idx] * 2))
            new_buf = torch.zeros((batch, new_capacity, dim), device=self.device, dtype=self.dtype)
            if self.latents[layer_idx] is not None:
                old_len = self.lengths[layer_idx]
                new_buf[:, :old_len] = self.latents[layer_idx][:, :old_len]
            self.latents[layer_idx] = new_buf
            self.capacities[layer_idx] = new_capacity
        start = self.lengths[layer_idx]
        end = start + seq_len
        self.latents[layer_idx][:, start:end] = compressed
        self.lengths[layer_idx] = end

    def get(self, layer_idx: int) -> torch.Tensor | None:
        if self.latents[layer_idx] is None:
            return None
        used = self.lengths[layer_idx]
        return self.latents[layer_idx][:, :used]

    def kv_bytes(self) -> int:
        total = 0
        for tensor in self.latents:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total


# ─── Attention Modules ───────────────────────────────────────

class StandardAttention(nn.Module):
    """Standard multi-head or grouped-query attention."""

    def __init__(self, config: ModelConfig, num_key_value_heads: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        kv_hidden = num_key_value_heads * self.head_dim
        self.k_proj = nn.Linear(config.hidden_size, kv_hidden, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_hidden, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int, cache=None):
        batch, seq_len, hidden = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_new = self.k_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_new = self.v_proj(hidden_states).view(batch, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if cache is None:
            full_k, full_v = k_new, v_new
        else:
            cache.append(layer_idx, k_new, v_new)
            full_k, full_v = cache.get(layer_idx)

        if self.num_key_value_heads != self.num_heads:
            full_k = repeat_kv(full_k, self.num_key_value_groups)
            full_v = repeat_kv(full_v, self.num_key_value_groups)

        scores = (q @ full_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.tril(torch.ones((seq_len, full_k.shape[2]), device=hidden_states.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        out = attn @ full_v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.o_proj(out)


class MLAAttention(nn.Module):
    """Multi-head Latent Attention: low-rank KV compression.

    Caches compressed latent (kv_lora_rank) instead of full K, V.
    At attention time, decompresses latent → per-head K and V.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.softmax_scale = self.head_dim ** -0.5

        # Q: standard projection
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)

        # KV down-projection: hidden → compressed latent
        self.kv_down_proj = nn.Linear(config.hidden_size, config.kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank)

        # KV up-projection: latent → per-head K and V
        self.kv_up_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int, cache=None):
        batch, seq_len, _ = hidden_states.shape

        # Q
        q = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # KV compression
        compressed = self.kv_down_proj(hidden_states)  # [batch, seq, kv_lora_rank]

        if cache is None:
            all_compressed = compressed
        else:
            cache.append(layer_idx, compressed)
            all_compressed = cache.get(layer_idx)

        # KV decompression → per-head K, V
        kv = self.kv_up_proj(self.kv_norm(all_compressed))
        kv = kv.view(batch, -1, self.num_heads, self.head_dim + self.v_head_dim).transpose(1, 2)
        k, v = kv.split([self.head_dim, self.v_head_dim], dim=-1)

        # Standard attention
        scores = (q @ k.transpose(-2, -1)) * self.softmax_scale
        kv_len = k.shape[2]
        causal = torch.tril(torch.ones((seq_len, kv_len), device=hidden_states.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.v_head_dim)
        return self.o_proj(out)


# ─── Transformer Blocks & Model ──────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, attn_module: nn.Module):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = attn_module
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, layer_idx: int, cache=None):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), layer_idx, cache)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class ToyLM(nn.Module):
    def __init__(self, config: ModelConfig, attn_type: str = "mha"):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        layers = []
        for _ in range(config.num_hidden_layers):
            if attn_type == "mha":
                attn = StandardAttention(config, num_key_value_heads=config.num_attention_heads)
            elif attn_type == "gqa":
                attn = StandardAttention(config, num_key_value_heads=config.num_key_value_heads)
            elif attn_type == "mla":
                attn = MLAAttention(config)
            else:
                raise ValueError(f"Unknown attn_type: {attn_type}")
            layers.append(TransformerBlock(config, attn))
        self.layers = nn.ModuleList(layers)

        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.vocab_size, config.hidden_size, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, cache=None):
        hidden = self.embed_tokens(input_ids)
        for idx, layer in enumerate(self.layers):
            hidden = layer(hidden, idx, cache)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)


# ─── Engine Metrics ──────────────────────────────────────────

@dataclass
class EngineMetrics:
    ttft_ms: list
    tokens_per_s: list
    total_time_ms: list
    kv_bytes: list


# ─── Engines ─────────────────────────────────────────────────

class BaselineEngine:
    def __init__(self, config: ModelConfig, device: torch.device):
        self.device = device
        self.model = ToyLM(config, attn_type="mha").to(device).eval()

    @torch.no_grad()
    def generate_turn(self, input_ids: list[int], max_new_tokens: int):
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        sync_device(self.device)
        t0 = time.perf_counter()
        logits = self.model(ids)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        sync_device(self.device)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        generated = [next_id]
        decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            ids = torch.cat([ids, torch.tensor([[generated[-1]]], device=self.device)], dim=1)
            logits = self.model(ids)
            generated.append(int(torch.argmax(logits[:, -1, :], dim=-1).item()))
        sync_device(self.device)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        tps = (max_new_tokens - 1) / max(decode_ms / 1000.0, 1e-9)
        return generated, ttft_ms, decode_ms + ttft_ms, tps, 0


class CachedGQAEngine:
    def __init__(self, config: ModelConfig, device: torch.device, paged: bool, block_size: int):
        self.device = device
        self.model = ToyLM(config, attn_type="gqa").to(device).eval()
        self.block_size = block_size
        self.paged = paged
        self.config = config
        self.reset_session()

    def reset_session(self):
        if self.paged:
            self.cache = PagedKVCache(self.config.num_hidden_layers, self.config.num_key_value_heads,
                                      self.config.head_dim, self.block_size, self.config.max_seq_len,
                                      self.device, torch.float32)
        else:
            self.cache = ContiguousKVCache(self.config.num_hidden_layers, self.config.num_key_value_heads,
                                           self.config.head_dim, self.device, torch.float32)
        self.history_tokens = []

    @torch.no_grad()
    def prefill(self, new_ids: list[int]):
        if not new_ids:
            return None
        ids = torch.tensor([new_ids], dtype=torch.long, device=self.device)
        logits = self.model(ids, cache=self.cache)
        self.history_tokens.extend(new_ids)
        return logits

    @torch.no_grad()
    def decode_step(self, last_token_id: int):
        ids = torch.tensor([[last_token_id]], dtype=torch.long, device=self.device)
        logits = self.model(ids, cache=self.cache)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        self.history_tokens.append(last_token_id)
        return next_id

    @torch.no_grad()
    def generate_turn(self, delta_ids: list[int], max_new_tokens: int):
        sync_device(self.device)
        t0 = time.perf_counter()
        logits = self.prefill(delta_ids)
        if logits is None:
            raise ValueError("delta_ids must not be empty")
        first_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        sync_device(self.device)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        generated = [first_id]
        decode_start = time.perf_counter()
        current = first_id
        for _ in range(max_new_tokens - 1):
            current = self.decode_step(current)
            generated.append(current)
        sync_device(self.device)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        tps = (max_new_tokens - 1) / max(decode_ms / 1000.0, 1e-9)
        kv_bytes = self.cache.kv_bytes()
        return generated, ttft_ms, decode_ms + ttft_ms, tps, kv_bytes


class CachedMLAEngine:
    """MLA engine: compressed latent KV cache."""

    def __init__(self, config: ModelConfig, device: torch.device):
        self.device = device
        self.model = ToyLM(config, attn_type="mla").to(device).eval()
        self.config = config
        self.reset_session()

    def reset_session(self):
        self.cache = LatentKVCache(
            self.config.num_hidden_layers,
            self.config.kv_lora_rank,
            self.device,
            torch.float32,
        )
        self.history_tokens = []

    @torch.no_grad()
    def prefill(self, new_ids: list[int]):
        if not new_ids:
            return None
        ids = torch.tensor([new_ids], dtype=torch.long, device=self.device)
        logits = self.model(ids, cache=self.cache)
        self.history_tokens.extend(new_ids)
        return logits

    @torch.no_grad()
    def decode_step(self, last_token_id: int):
        ids = torch.tensor([[last_token_id]], dtype=torch.long, device=self.device)
        logits = self.model(ids, cache=self.cache)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        self.history_tokens.append(last_token_id)
        return next_id

    @torch.no_grad()
    def generate_turn(self, delta_ids: list[int], max_new_tokens: int):
        sync_device(self.device)
        t0 = time.perf_counter()
        logits = self.prefill(delta_ids)
        if logits is None:
            raise ValueError("delta_ids must not be empty")
        first_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        sync_device(self.device)
        ttft_ms = (time.perf_counter() - t0) * 1000.0

        generated = [first_id]
        decode_start = time.perf_counter()
        current = first_id
        for _ in range(max_new_tokens - 1):
            current = self.decode_step(current)
            generated.append(current)
        sync_device(self.device)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        tps = (max_new_tokens - 1) / max(decode_ms / 1000.0, 1e-9)
        kv_bytes = self.cache.kv_bytes()
        return generated, ttft_ms, decode_ms + ttft_ms, tps, kv_bytes


# ─── Engine Runner ───────────────────────────────────────────

def build_prompt_history(history_turns: list[str], assistant_prefix: str = "Assistant:"):
    return "\n".join(history_turns + [assistant_prefix])


def run_engine(engine_name: str, benchmark_config: BenchmarkConfig, model_config: ModelConfig, sessions: list[list[str]]):
    tokenizer = ByteTokenizer()
    device = select_device(benchmark_config.device)

    if engine_name == "baseline_mha_no_cache":
        engine = BaselineEngine(model_config, device)
    elif engine_name == "gqa_contiguous_cache":
        engine = CachedGQAEngine(model_config, device, paged=False, block_size=benchmark_config.block_size)
    elif engine_name == "gqa_paged_cache":
        engine = CachedGQAEngine(model_config, device, paged=True, block_size=benchmark_config.block_size)
    elif engine_name == "mla_latent_cache":
        engine = CachedMLAEngine(model_config, device)
    else:
        raise ValueError(engine_name)

    is_baseline = engine_name == "baseline_mha_no_cache"
    metrics = EngineMetrics(ttft_ms=[], tokens_per_s=[], total_time_ms=[], kv_bytes=[])

    for run_type in ["warmup", "measure"]:
        runs = benchmark_config.warmup_runs if run_type == "warmup" else benchmark_config.measure_runs
        for _ in range(runs):
            for session in sessions:
                if hasattr(engine, "reset_session"):
                    engine.reset_session()
                history = []
                for turn in session:
                    if is_baseline:
                        history.append(turn)
                        prompt = build_prompt_history(history)
                        input_ids = tokenizer.encode(prompt)
                        generated, ttft_ms, total_ms, tps, kv_bytes = engine.generate_turn(input_ids, benchmark_config.max_new_tokens)
                        history.append("Assistant: " + tokenizer.decode(generated))
                    else:
                        delta = turn + "\nAssistant:"
                        delta_ids = tokenizer.encode(delta)
                        generated, ttft_ms, total_ms, tps, kv_bytes = engine.generate_turn(delta_ids, benchmark_config.max_new_tokens)
                        history.extend([turn, "Assistant: " + tokenizer.decode(generated)])

                    if run_type == "measure":
                        metrics.ttft_ms.append(ttft_ms)
                        metrics.tokens_per_s.append(tps)
                        metrics.total_time_ms.append(total_ms)
                        metrics.kv_bytes.append(kv_bytes)

    return metrics
