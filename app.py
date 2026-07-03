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

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; }
    
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-banner h1 {
        color: #e94560;
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0;
    }
    .header-banner p {
        color: #a8b2d8;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Stats row */
    .stat-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #e94560;
    }
    .stat-box h3 {
        color: #e94560;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-box p {
        color: #666;
        font-size: 0.8rem;
        margin: 0.2rem 0 0 0;
    }
    
    /* Headline cards */
    .headline-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #e94560;
        transition: transform 0.2s;
    }
    .headline-card:hover {
        transform: translateX(4px);
        box-shadow: 0 5px 20px rgba(233,69,96,0.15);
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
    
    /* Prompt input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
        padding: 0.6rem 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e94560;
    }
    
    /* Generate button */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }

    /* Quick prompt buttons */
    .stButton > button[kind="secondary"] {
        background: #f0f2f6;
        color: #1a1a2e;
        font-size: 0.85rem;
        padding: 0.4rem 0.8rem;
    }

    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource(max_entries=1)
def load_assets():
    device = "cpu"
    model_path = hf_hub_download(
    repo_id="tanush23x/gpt-news-headlines",
    filename="best_model.pt",
        force_download=True,
    force_download=True  # add this
)
tokenizer_path = hf_hub_download(
    repo_id="tanush23x/gpt-news-headlines",
    filename="bpe_tokenizer.json",
        force_download=True,
    force_download=True  # add this
)
    model, config = load_model(model_path, device)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device

with st.spinner("Loading model..."):
    model, tokenizer, device = load_assets()

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>📰 Indian News Headline Generator</h1>
    <p>A GPT-2 style transformer built from scratch · Trained on 300K Indian news headlines · 13.76M parameters</p>
</div>
""", unsafe_allow_html=True)

# ── Stats row ────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><h3>13.76M</h3><p>Parameters</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><h3>300K</h3><p>Training Headlines</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><h3>~77</h3><p>Perplexity</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><h3>8K</h3><p>Vocabulary Size</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main layout ──────────────────────────────────────────────
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
    temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1,
        help="Higher = more creative. Lower = more focused.")
    top_k = st.slider("Top-K", 1, 100, 50,
        help="Keep only top K most likely tokens.")
    top_p = st.slider("Top-P (Nucleus)", 0.1, 1.0, 0.9, 0.05,
        help="Keep tokens covering top P probability mass.")
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

# ── About ────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Architecture:** GPT-2 style transformer built **from scratch** in PyTorch

    **Key features:**
    - Self-attention, multi-head attention, residual connections, layer norm — all implemented manually
    - Custom ByteLevel BPE tokenizer (vocab 8K) — same approach as GPT-2
    - KV-caching for efficient autoregressive inference
    - Top-k and Top-p nucleus sampling implemented from scratch
    - Trained on 300K Indian news headlines (Times of India, 2001–2022)
    - 3 training runs with hyperparameter ablation documented
    - Newline-token stopping for clean headline boundaries

    **Training:** Google Colab T4 GPU · 6000 steps · ~28 minutes · Val loss: 4.34 · Perplexity: ~77
    """)
