import torch
import torch.nn as nn
import torch.nn.functional as F
from model.transformer import TransformerBlock

class GPT(nn.Module):
    """
    Full GPT model:
    Token Embedding + Positional Embedding
    → N Transformer Blocks
    → LayerNorm
    → Linear output head (vocab size)
    """
    def __init__(self, vocab_size, d_model, num_heads, num_layers, 
                 context_len, dropout=0.1):
        super().__init__()
        self.context_len = context_len

        # Token embedding — maps each token ID to a d_model vector
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embedding — learned position encodings
        self.pos_embedding = nn.Embedding(context_len, d_model)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_len, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm before output
        self.norm = nn.LayerNorm(d_model)

        # Output head — projects d_model back to vocab_size
        # (gives probability score for each token in vocabulary)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying — share weights between token embedding and output head
        # This is a standard trick that reduces parameters and improves performance
        self.token_embedding.weight = self.head.weight

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        device = idx.device

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)           # (B, T, d_model)
        pos = torch.arange(T, device=device)
        pos_emb = self.pos_embedding(pos)             # (T, d_model)
        x = tok_emb + pos_emb                         # (B, T, d_model)

        # Pass through all transformer blocks
        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, cache = block(x, cache)
            new_kv_caches.append(cache)

        # Final norm + output projection
        x = self.norm(x)
        logits = self.head(x)                         # (B, T, vocab_size)

        # If targets provided, compute loss (training mode)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, V),
                targets.view(B * T)
            )

        return logits, loss, new_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, 
                 top_k=None, top_p=None):
        """
        Generate new tokens autoregressively.
        Implements temperature, top-k, and top-p (nucleus) sampling.
        """
        for _ in range(max_new_tokens):
            # Crop context to context_len if needed
            idx_cond = idx[:, -self.context_len:]

            # Forward pass
            logits, _, _ = self(idx_cond)

            # Get logits for last token only
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply temperature — controls randomness
            logits = logits / temperature

            # Top-k sampling — keep only top k tokens
            if top_k is not None:
                top_k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, top_k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            # Top-p (nucleus) sampling — keep tokens covering top p probability mass
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                # Remove tokens beyond the top_p threshold
                sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx
