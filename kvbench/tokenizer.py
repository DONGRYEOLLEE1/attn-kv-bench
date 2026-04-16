"""Simple byte tokenizer for serving benchmarks."""

from __future__ import annotations


class ByteTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    offset = 4
    vocab_size = 260

    def encode(self, text: str):
        return [self.bos_id] + [byte + self.offset for byte in text.encode("utf-8")] + [self.eos_id]

    def decode(self, ids):
        payload = []
        for token_id in ids:
            if token_id < self.offset:
                continue
            payload.append(token_id - self.offset)
        return bytes(payload).decode("utf-8", errors="ignore")
