"""
Trains a fresh ByteLevel BPE tokenizer with lowercase normalization.

Why this exists:
The original tokenizer was case-sensitive, so "Supreme Court" (as it
appears in real headlines) and "supreme court" (as a user might type it
into a prompt) tokenized completely differently — the lowercase version
shredded into near-random subword fragments the model had never seen at
the start of a sequence. Lowercasing everything before BPE removes this
fragility entirely: casing at inference time no longer matters, because
it never mattered during training either.

Run this ONCE, before train.py, on the same headlines.txt used for
training. It overwrites bpe_tokenizer.json.
"""
from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders, normalizers
from tokenizers.models import BPE

VOCAB_SIZE = 8000
TEXT_FILE = "headlines.txt"           # one headline per line
OUTPUT_FILE = "bpe_tokenizer.json"

def train_tokenizer(text_file=TEXT_FILE, output_file=OUTPUT_FILE, vocab_size=VOCAB_SIZE):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Lowercase everything before BPE — removes casing as a source of
    # out-of-distribution tokenization at inference time.
    tokenizer.normalizer = normalizers.Lowercase()

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )

    tokenizer.train(files=[text_file], trainer=trainer)
    tokenizer.save(output_file)

    print(f"Tokenizer trained. Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Saved to {output_file}")

    # Sanity check — casing should no longer matter
    for probe in ["supreme court", "Supreme Court", "SUPREME COURT"]:
        enc = tokenizer.encode(probe)
        print(f"  {probe!r} -> {enc.tokens}")

if __name__ == "__main__":
    train_tokenizer()
