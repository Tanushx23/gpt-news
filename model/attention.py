import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Single head of self-attention.
    Each token looks at all other tokens and decides
    how much to attend to each one.
    """
    def __init__(self, d_model, head_size, context_len, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.key   = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_len, context_len))
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        if kv_cache is not None:
            if "k" in kv_cache:
                k = torch.cat([kv_cache["k"], k], dim=1)
                v = torch.cat([kv_cache["v"], v], dim=1)
            kv_cache["k"] = k
            kv_cache["v"] = v

        scale = k.shape[-1] ** -0.5
        scores = q @ k.transpose(-2, -1) * scale

        T_q, T_k = scores.shape[-2], scores.shape[-1]
        scores = scores.masked_fill(
            self.mask[:T_q, :T_k] == 0, float("-inf")
        )

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = weights @ v
        return out, kv_cache


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.
    Each head can focus on different aspects of the input.
    """
    def __init__(self, d_model, num_heads, context_len, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.head_size = d_model // num_heads
        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            SelfAttention(d_model, self.head_size, context_len, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, kv_caches=None):
        head_outputs = []
        new_caches = []

        for i, head in enumerate(self.heads):
            cache = kv_caches[i] if kv_caches is not None else None
            out, cache = head(x, cache)
            head_outputs.append(out)
            new_caches.append(cache)

        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out, new_caches
