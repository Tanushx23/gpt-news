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
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-banner h1 { color: #e94560; font-size: 2.5rem; font-weight: 800; margin: 0; }
    .header-banner p { color: #a8b2d8; font-size: 1rem; margin-top: 0.5rem; }
    .stat-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #e94560;
    }
    .stat-box h3 { color: #e94560; font-size: 1.5rem; font-weight: 700; margin: 0; }
    .stat-box p { color: #666; font-size: 0.8rem; margin: 0.2rem 0 0 0; }
    .headline-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #e94560;
    }
    .headline-card .variant-label {
        color: #e94560;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
    }
    .headline-card .headline-text {
        color: #1a1a2e;
        font-size: 1.1rem;
        font-weight: 600;
        line-height: 1.4;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(max_entries=1)
def load_assets():
    device = "cpu"
    model_path = hf_hub_download(
        repo_id="tanush23x/gpt-news-headlines",
        filename="best_model.pt",
        force_download=True
    )
    tokenizer_path = hf_hub_download(
        repo_id="tanush23x/gpt-news-headlines",
        filename="bpe_tokenizer.json",
        force_download=True
    )
    model, config = load_model(model_path, device)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device

with st.spinner("Loading model..."):
    model, tokenizer, device = load_assets()

st.markdown("""
<div class="header-banner">
    <h1>📰 Indian News Headline Generator</h1>
    <p>GPT-2 style transformer built from scratch · 300K Indian news headlines · 13.76M parameters</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='stat-box'><h3>13.76M</h3><p>Parameters</p></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='stat-box'><h3>300K</h3><p>Training Headlines</p></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='stat-box'><h3>~77</h3><p>Perplexity</p></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='stat-box'><h3>8K</h3><p>Vocabulary Size</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1.5])

with left:
    st.markdown("### ✍️ Enter a Prompt")
    prompt = st.text_input(
        "",
        placeholder="e.g. Modi, RBI cuts, Delhi, Supreme Court...",
        value="Modi",
        label_visibility="collapsed"
    )

    st.markdown("**Quick prompts:**")
    cols = st.columns(5)
    suggestions = ["Modi", "RBI", "Delhi", "India vs", "SC"]
    for i, s in enumerate(suggestions):
        if cols[i].button(s, key=f"btn_{s}"):
            prompt = s

    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("🚀 Generate Headlines", type="primary")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1)
    top_k = st.slider("Top-K", 1, 100, 50)
    top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05)
    max_tokens = st.slider("Max tokens", 10, 100, 50, 5)
    num_variants = st.slider("Variants", 1, 5, 3)

with right:
    st.markdown("### 📋 Generated Headlines")

    if generate:
        if not prompt.strip():
            st.warning("Please enter a prompt first!")
        else:
            with st.spinner("Generating..."):
                for i in range(num_variants):
                    result = generate_headline(
                        model, tokenizer, prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        device=device
                    )
                    st.markdown(f"""
                    <div class="headline-card">
                        <div class="variant-label">Variant {i+1}</div>
                        <div class="headline-text">{result}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding: 3rem; color: #aaa;">
            <div style="font-size: 3rem;">📰</div>
            <p>Enter a prompt and click Generate to see headlines</p>
        </div>
        """, unsafe_allow_html=True)

with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Architecture:** GPT-2 style transformer built from scratch in PyTorch

    **Key features:**
    - Self-attention, multi-head attention, residual connections, layer norm implemented manually
    - Custom ByteLevel BPE tokenizer (vocab 8K)
    - KV-caching for efficient inference
    - Top-k and Top-p nucleus sampling from scratch
    - Newline-token stopping for clean headline boundaries

    **Training:** T4 GPU · 300K headlines · 6000 steps · ~28 mins · Val loss: 4.34 · Perplexity: ~77
    """)
