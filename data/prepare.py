import random
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class HeadlineDataset(Dataset):
    """
    Sliding-window next-token-prediction dataset over a pre-built
    token stream. The stream is built elsewhere (see build_token_streams)
    so that train/val separation happens BEFORE windowing.
    """
    def __init__(self, token_ids, context_len):
        self.context_len = context_len
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.context_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return x, y


def build_token_streams(text_file, tokenizer_file, val_split=0.1, seed=42):
    """
    Splits at the HEADLINE level, not the token/window level, before any
    sliding windows are built.

    Why this matters: if you tokenize the whole corpus into one long
    stream and THEN build overlapping windows (stride 1) and THEN
    random_split those windows, adjacent windows differ by a single
    token and can land on opposite sides of the split — i.e. val
    "unseen" examples are near-duplicates of train examples. Splitting
    headlines first means train and val never share a headline, so the
    windows built from each are genuinely disjoint.

    Each headline is wrapped in [BOS] ... [EOS] so the model gets an
    explicit signal for where a headline starts and ends, instead of
    inferring it from a bare newline character.
    """
    tok = Tokenizer.from_file(tokenizer_file)

    bos_id = tok.token_to_id("[BOS]")
    eos_id = tok.token_to_id("[EOS]")
    if bos_id is None or eos_id is None:
        raise ValueError(
            "Tokenizer is missing [BOS]/[EOS] special tokens. "
            "Retrain it with data/train_tokenizer.py first."
        )

    with open(text_file, "r", encoding="utf-8") as f:
        headlines = [line.strip() for line in f if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(headlines)

    val_size = int(len(headlines) * val_split)
    val_headlines = headlines[:val_size]
    train_headlines = headlines[val_size:]

    def encode_stream(headline_list):
        ids = []
        # batch encoding is much faster than calling .encode() per line
        for enc in tok.encode_batch(headline_list):
            ids.append(bos_id)
            ids.extend(enc.ids)
            ids.append(eos_id)
        return ids

    train_ids = encode_stream(train_headlines)
    val_ids = encode_stream(val_headlines)

    print(f"Train headlines: {len(train_headlines):,} | tokens: {len(train_ids):,}")
    print(f"Val headlines:   {len(val_headlines):,} | tokens: {len(val_ids):,}")

    return train_ids, val_ids


def get_dataloader(text_file, tokenizer_file, context_len, batch_size,
                    val_split=0.1, seed=42, num_workers=2):
    train_ids, val_ids = build_token_streams(text_file, tokenizer_file, val_split, seed)

    train_dataset = HeadlineDataset(train_ids, context_len)
    val_dataset = HeadlineDataset(val_ids, context_len)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train windows: {len(train_dataset):,} | Val windows: {len(val_dataset):,}")

    return train_loader, val_loader
