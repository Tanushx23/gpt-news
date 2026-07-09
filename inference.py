import torch
import torch.nn as nn
from model.gpt import GPT
from tokenizers import Tokenizer
import os

def load_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = GPT(
        vocab_size  = config["vocab_size"],
        d_model     = config["d_model"],
        num_heads   = config["num_heads"],
        num_layers  = config["num_layers"],
        context_len = config["context_len"],
        dropout     = 0.0
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from step {checkpoint.get('step', '?')}")
    print(f"Val loss at checkpoint: {checkpoint.get('val_loss', '?'):.4f}")
    return model, config


def generate_headline(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 60,
    temperature    = 0.8,
    top_k          = 50,
    top_p          = 0.9,
    device         = "cuda"
):
    model.eval()

    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long).to(device)

    # Find newline token id
    newline_id = tokenizer.encode("\n").ids
    newline_id = newline_id[0] if newline_id else None

    prompt_len = len(encoded.ids)
    generated_ids = idx[0].tolist()
    tokens_generated = 0

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -model.context_len:]
            logits, _, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits = logits.masked_fill(
                    logits < values[:, -1].unsqueeze(-1), float("-inf")
                )

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            # Only stop at newline if we have generated enough tokens
            # beyond the prompt — prevents stopping immediately
            if newline_id is not None and token_id == newline_id:
                if tokens_generated >= 5:
                    break
                else:
                    # Skip this newline, continue generating
                    idx = torch.cat([idx, next_token], dim=1)
                    generated_ids.append(token_id)
                    tokens_generated += 1
                    continue

            idx = torch.cat([idx, next_token], dim=1)
            generated_ids.append(token_id)
            tokens_generated += 1

    result = tokenizer.decode(generated_ids).strip()

    # Clean up any embedded newlines
    result = result.replace("\n", " ").strip()

    return result
