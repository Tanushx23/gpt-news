# GPT News Headline Generator

A GPT-2 style transformer language model built **from scratch** in PyTorch, trained on 3.6M Indian news headlines from the Times of India dataset (2001-2022).

## Features

- **Transformer architecture from scratch** — self-attention, multi-head attention, residual connections, layer normalization, all implemented manually in PyTorch (no `nn.Transformer`)
- **Custom BPE tokenizer** — ByteLevel BPE tokenizer trained on the corpus (vocab size 8,000) with **lowercase normalization**, so prompt casing never affects generation quality
- **KV-caching** — key-value cache implemented in the attention module for efficient autoregressive inference
- **Custom sampling strategies** — temperature, top-k, and top-p (nucleus) sampling implemented from scratch
- **Cosine LR schedule with warmup** — linear warmup for the first 300 steps, then cosine decay to 0
- **Selective weight decay** — LayerNorm and bias parameters excluded from weight decay, standard GPT-2 practice
- **Gradient clipping** — prevents exploding gradients during training
- **Streamlit UI** — interactive demo, headline generation paired with a Groq LLaMA-3.3-70B call for the descriptive sub-text

## Model Architecture

| Parameter | Value |
|---|---|
| Parameters | 13.76M |
| Layers | 6 |
| Attention heads | 6 |
| Embedding dim | 384 |
| Context length | 128 tokens |
| Vocabulary size | 8,000 |
| Dropout | 0.2 |

## Training

Trained on Kaggle (T4 GPU) for 16,000 steps (~142 minutes), batch size 128, ~4.5 epochs over the cleaned corpus.

**Final val loss: 3.7793** (perplexity ≈ 44)

## Dataset

[India News Headlines Dataset](https://www.kaggle.com/datasets/therohk/india-headlines-news-dataset) — Times of India headlines, 2001-2022 (CC0 license).

- Raw: 3,876,557 headlines
- After deduplication + length filtering: **3,604,051 headlines**
- Split 90/10 at the **headline level** before building training windows (see "What was fixed" below)

## What was fixed

An earlier version of this project trained on the full dataset but produced clipped, sometimes garbage output. Debugging traced it to a few specific issues, not the model architecture or dataset size:

**1. Tokenizer was case-sensitive.** Real headlines are sentence-cased ("Supreme Court asks Centre..."), so the BPE merges learned clean whole-word tokens for that casing. A lowercase prompt like `"supreme court"` tokenized into unrelated character fragments (`s`, `up`, `reme`, `court`) that the model had essentially never seen at the start of a sequence — producing near-random output. **Fix:** retrained the tokenizer with a lowercase normalizer (`data/train_tokenizer.py`), so casing no longer affects tokenization at all.

**2. Train/val split leaked.** The original pipeline tokenized the whole corpus into one long stream, built overlapping sliding windows (stride 1), and only *then* did a random train/val split — meaning adjacent windows differing by a single token could land on opposite sides of the split. Val loss looked better than it should have. **Fix:** split headlines into train/val *before* windowing (`data/prepare.py`), so no headline (or its windows) appears in both sets.

**3. No explicit sequence boundaries.** Headlines were joined with a bare `\n`, so the model had to infer "headline starts/ends here" from a plain whitespace character. Generation used a fragile heuristic ("skip newline unless 5+ tokens generated") to avoid stopping too early, which sometimes clipped headlines mid-thought anyway. **Fix:** each headline is wrapped in explicit `[BOS]`/`[EOS]` tokens, and generation now stops cleanly on a learned `[EOS]` (`inference.py`).

**4. Optimizer/schedule gaps.** No LR warmup, and weight decay applied uniformly (including to LayerNorm/bias params). **Fix:** added 300-step linear warmup before cosine decay, and excluded 1-D parameters from weight decay.

**5. KV-cache masking bug.** The causal mask slicing was incorrect for cached single-token decoding steps (dead code at the time, since `generate()` didn't use the cache, but fixed for correctness in `model/attention.py`).

## Generated Examples

*(post-fix, run through `generate_headline()`)*

| Prompt | Generated |
|---|---|
| supreme court | Refuses to quash fir against kandhamal mla |
| Supreme Court | Says no to quashing of pil in a week |
| RBI cuts | Rate of time to recover rs 186 crore in march |
| india vs australia | : aussie triumphs |
| delhi government | To seek government nod for rt-pcr tests |

Note the model's job here is only the headline text — the descriptive paragraph shown in the UI is generated separately by Groq's Llama-3.3-70B, not by this model.

## Project Structure

```
gpt-news/
├── model/
│   ├── attention.py      # SelfAttention + MultiHeadAttention with KV-cache
│   ├── transformer.py    # TransformerBlock (pre-norm architecture)
│   └── gpt.py             # Full GPT model + generate() with sampling
├── data/
│   ├── prepare.py         # HeadlineDataset + headline-level train/val split
│   └── train_tokenizer.py # Trains lowercase-normalized BPE tokenizer
├── train.py                # Training loop, warmup + cosine LR, selective weight decay
├── inference.py            # Model loading + EOS-based text generation
├── app.py                  # Streamlit UI
└── bpe_tokenizer.json      # Trained BPE tokenizer (lowercase-normalized)
```

## Setup

```bash
pip install torch tokenizers streamlit huggingface_hub groq
```

## Training from scratch

```bash
python data/train_tokenizer.py   # builds bpe_tokenizer.json from headlines.txt
python train.py                  # trains the model, ~2-2.5 hrs on a T4
```

## Run Streamlit App

```bash
streamlit run app.py
```

## Key Design Decisions

**Why BPE over character-level tokenization?**
BPE compresses sequences 6x compared to character-level, allowing the model to attend over much more context within the same window size.

**Why lowercase-normalize the tokenizer?**
Removes prompt casing as a source of out-of-distribution input — see "What was fixed" above.

**Why split train/val at the headline level, not the window level?**
Overlapping sliding windows built from the same underlying text are near-duplicates of each other. Splitting after windowing leaks near-identical examples across the split; splitting before windowing (at the headline level) guarantees train and val never share source text.

**Why KV-caching?**
During autoregressive generation, without caching, keys and values for all previous tokens are recomputed at every step — O(n²) cost. KV-cache stores them and only computes for the new token — O(n) cost. This is how production LLM serving works.

**Why top-p over top-k?**
Top-p is more adaptive — when the model is confident it naturally narrows options; when uncertain it keeps more options open. Top-k uses a fixed cutoff regardless of the probability distribution shape.

**Why pre-norm architecture?**
Applying LayerNorm before attention/feedforward (rather than after, as in the original 2017 paper) leads to more stable training gradients, especially in deeper networks.
