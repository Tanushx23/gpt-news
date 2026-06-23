import torch
import torch.nn as nn
from model.attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Simple feedforward network applied after attention.
    Gives each token a chance to "think" independently
    after gathering information from other tokens via attention.
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 4x expansion is standard in transformer literature
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),               # smoother than ReLU, used in GPT-2
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One full transformer block:
    LayerNorm → MultiHeadAttention → residual
    LayerNorm → FeedForward → residual

    Pre-norm architecture (norm before attention) —
    more stable training than original post-norm paper.
    """
    def __init__(self, d_model, num_heads, context_len, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads, context_len, dropout)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x, kv_caches=None):
        # Attention with residual connection
        attn_out, kv_caches = self.attention(self.norm1(x), kv_caches)
        x = x + attn_out

        # FeedForward with residual connection
        x = x + self.ff(self.norm2(x))

        return x, kv_caches
