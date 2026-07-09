import torch
import torch.nn.functional as F

from model.gpt import GPT


def load_model(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    model = GPT(
        vocab_size  = config["vocab_size"],
        d_model     = config["d_model"],
        num_heads   = config["num_heads"],
        num_layers  = config["num_layers"],
        context_len = config["context_len"],
        dropout     = 0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded from step {checkpoint.get('step', '?')}")
    print(f"Val loss at checkpoint: {checkpoint.get('val_loss', '?'):.4f}")
    return model, config


def _is_low_quality(text, min_words=3):
    """
    Cheap, fast heuristics to catch degenerate generations worth
    retrying rather than showing to a user:
      - too short to be a real headline (model stopped almost immediately)
      - the same word repeated back-to-back (a common failure mode of
        small language models under sampling)
      - a single word/token repeated for most of the output
    This is intentionally conservative — it only catches clear failures,
    not just "boring" output, since some retry budget is limited.
    """
    if not text:
        return True

    words = text.split()
    if len(words) < min_words:
        return True

    # Immediate word-level repetition: "the the the..." or "says says..."
    for i in range(len(words) - 1):
        if words[i].lower() == words[i + 1].lower():
            return True

    # One word dominating the whole output (e.g. mostly "the the the X")
    if len(words) >= 5:
        most_common_count = max(words.count(w) for w in set(words))
        if most_common_count / len(words) > 0.5:
            return True

    return False


def generate_headline(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 60,
    temperature    = 0.8,
    top_k          = 50,
    top_p          = 0.9,
    device         = "cuda",
    max_retries    = 3,
):
    """
    Generates a headline, retrying (with slightly higher temperature
    each time) if the output looks degenerate — too short or stuck
    repeating a word. Returns the best attempt if all retries still
    look poor, rather than silently returning nothing.
    """
    best_result = ""
    for attempt in range(max_retries):
        # nudge temperature up slightly on retries to break out of
        # whatever narrow distribution produced the bad output
        attempt_temp = temperature + (0.1 * attempt)
        result = _generate_headline_once(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=attempt_temp,
            top_k=top_k, top_p=top_p, device=device,
        )
        if not _is_low_quality(result):
            return result
        if len(result) > len(best_result):
            best_result = result

    return best_result


def _generate_headline_once(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 60,
    temperature    = 0.8,
    top_k          = 50,
    top_p          = 0.9,
    device         = "cuda",
):
    """
    Generates a headline starting from `prompt`, stopping cleanly at the
    model's learned [EOS] token.

    Note: the tokenizer normalizes casing internally (see
    data/train_tokenizer.py), so the prompt's casing no longer affects
    tokenization — "supreme court" and "Supreme Court" now tokenize
    identically. This replaces the old behavior where lowercase prompts
    were shredded into out-of-distribution character fragments.
    """
    model.eval()

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")

    prompt_ids = tokenizer.encode(prompt.strip()).ids
    idx = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long).to(device)

    generated_ids = []

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
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.scatter(logits, 1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = next_token.item()

            if token_id == eos_id:
                break

            idx = torch.cat([idx, next_token], dim=1)
            generated_ids.append(token_id)

    result = tokenizer.decode(generated_ids).strip()

    # Tokenizer output is lowercase (normalized) — capitalize for display.
    if result:
        result = result[0].upper() + result[1:]

    return result
