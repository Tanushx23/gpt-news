import streamlit as st
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from model.gpt import GPT
from inference import load_model, generate_headline

st.set_page_config(
    page_title="Indian News Headline Generator",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Indian News Headline Generator")
st.caption("GPT trained from scratch on 150,000 Indian news headlines")

# ── Load model (cached) ──────────────────────────────────────
@st.cache_resource
def load_assets():
    device = "cpu"  # Streamlit Cloud has no GPU

    # Download from HuggingFace
    model_path = hf_hub_download(
        repo_id="tanush23x/gpt-news-headlines",
        filename="best_model.pt"
    )
    tokenizer_path = hf_hub_download(
        repo_id="tanush23x/gpt-news-headlines",
        filename="bpe_tokenizer.json"
    )

    model, config = load_model(model_path, device)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device

with st.spinner("Loading model from HuggingFace..."):
    model, tokenizer, device = load_assets()

st.success("Model loaded!")

# ── Sidebar controls ─────────────────────────────────────────
st.sidebar.header("Generation Settings")

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1, max_value=2.0, value=0.8, step=0.1,
    help="Higher = more creative. Lower = more focused."
)
top_k = st.sidebar.slider(
    "Top-K",
    min_value=1, max_value=100, value=50, step=1,
    help="Keep only top K most likely tokens."
)
top_p = st.sidebar.slider(
    "Top-P (Nucleus)",
    min_value=0.1, max_value=1.0, value=0.9, step=0.05,
    help="Keep tokens covering top P probability mass."
)
max_tokens = st.sidebar.slider(
    "Max new tokens",
    min_value=10, max_value=100, value=40, step=5
)
num_variants = st.sidebar.slider(
    "Number of variants",
    min_value=1, max_value=5, value=3
)

# ── Main input ───────────────────────────────────────────────
st.subheader("Enter a prompt")
prompt = st.text_input(
    "Start of headline",
    placeholder="e.g. Modi, RBI cuts, Delhi, Supreme Court...",
    value="Modi"
)

# ── Quick prompts ────────────────────────────────────────────
st.write("**Quick prompts:**")
cols = st.columns(5)
suggestions = ["Modi", "RBI", "Delhi", "India vs", "Supreme Court"]
for i, suggestion in enumerate(suggestions):
    if cols[i].button(suggestion):
        prompt = suggestion

# ── Generate ─────────────────────────────────────────────────
if st.button("Generate Headlines", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt first!")
    else:
        with st.spinner("Generating..."):
            st.subheader("Generated Headlines")
            for i in range(num_variants):
                result = generate_headline(
                    model, tokenizer, prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    device=device
                )
                st.markdown(f"**Variant {i+1}:**")
                st.info(result)

# ── About ────────────────────────────────────────────────────
with st.expander("About this model"):
    st.markdown("""
    **Architecture:** GPT-2 style transformer built from scratch in PyTorch

    **Training data:** 150,000 Indian news headlines (Times of India, 2001-2022)

    **Tokenizer:** Custom ByteLevel BPE tokenizer (vocab size 8,000)

    **Model size:** 13.76M parameters

    **Key features:**
    - Transformer architecture implemented from scratch (no nn.Transformer)
    - Custom BPE tokenization — same approach as GPT-2
    - KV-caching for efficient inference
    - Top-k and Top-p nucleus sampling from scratch
    - Cosine LR scheduling + gradient clipping
    - Two training runs with hyperparameter ablation
    """)
