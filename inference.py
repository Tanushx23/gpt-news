import torch
from model.gpt import GPT
from tokenizers import Tokenizer
import re
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


def clean_text(text):
    """Fix BPE decoding artifacts — remove spaces before punctuation."""
    text = re.sub(r" ([,;:'\.\!\?])", r"", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_headline(
    model,
    tokenizer,
    prompt,
    max_new_tokens = 50,
    temperature    = 0.8,
    top_k          = 50,
    top_p          = 0.9,
    device         = "cuda"
):
    model.eval()

    encoded = tokenizer.encode(prompt)
    idx = torch.tensor([encoded.ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model.generate(
            idx,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_k          = top_k,
            top_p          = top_p
        )

    generated_ids = output[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    generated_text = clean_text(generated_text)

    return generated_text
