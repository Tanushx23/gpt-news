import streamlit as st
import torch
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
from groq import Groq
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

st.markdown("""
<style>
    /* Force dark background everywhere */
    .stApp {
        background-color: #0d1117 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
    }
    .main .block-container {
        background-color: #0d1117 !important;
        padding: 2rem 3rem;
        max-width: 800px;
    }

    /* All text white */
    body, p, span, label, div {
        color: #e6edf3 !important;
    }

    /* Header */
    .top-header {
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 2rem;
    }
    .top-header .badge {
        background: #e94560;
        color: white !important;
        font-size: 0.7rem;
        font-weight: 700;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        display: inline-block;
        margin-bottom: 0.8rem;
    }
    .top-header h1 {
        color: #e6edf3 !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .top-header p {
        color: #8b949e !important;
        font-size: 0.9rem;
        margin: 0.5rem 0 0 0;
    }

    /* Stats */
    .stats-row {
        display: flex;
        gap: 1px;
        background: #21262d;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 2rem;
    }
    .stat-item {
        flex: 1;
        background: #161b22;
        padding: 1rem;
        text-align: center;
    }
    .stat-item h3 {
        color: #e94560 !important;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 0;
    }
    .stat-item p {
        color: #8b949e !important;
        font-size: 0.72rem;
        margin: 0.2rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Input area */
    .input-section {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .input-section h3 {
        color: #e6edf3 !important;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 0.8rem 0;
    }

    /* Text input */
    .stTextInput > div > div > input {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 0 3px rgba(233,69,96,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #484f58 !important;
    }

    /* Quick prompt buttons */
    div[data-testid="column"] .stButton > button {
        background: #21262d !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.5rem !important;
        width: 100% !important;
        font-weight: 500 !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #30363d !important;
        border-color: #e94560 !important;
    }

    /* Generate button */
    div[data-testid="stVerticalBlock"] > div:last-child .stButton > button,
    .generate-btn .stButton > button {
        background: linear-gradient(135deg, #e94560, #c0392b) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
    }

    /* Headline cards */
    .news-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .news-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        background: linear-gradient(180deg, #e94560, #c0392b);
        border-radius: 4px 0 0 4px;
    }
    .news-card .card-num {
        color: #e94560 !important;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }
    .news-card .card-headline {
        color: #e6edf3 !important;
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.45;
        margin-bottom: 0.7rem;
    }
    .news-card .card-sub {
        color: #8b949e !important;
        font-size: 0.85rem;
        line-height: 1.55;
        padding-top: 0.7rem;
        border-top: 1px solid #21262d;
    }

    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem 0;
        color: #484f58 !important;
    }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .empty-state p { color: #484f58 !important; font-size: 0.9rem; }

    /* Expander */
    .streamlit-expanderHeader {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 10px !important;
        color: #8b949e !important;
    }
    .streamlit-expanderContent {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-top: none !important;
    }

    /* Select slider */
    .stSlider > div { color: #8b949e !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource(max_entries=1)
def load_assets_v4():
    device = "cpu"
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

with st.spinner("Loading model..."):
    model, tokenizer, device = load_assets_v4()

# ── Groq sub-description ─────────────────────────────────────
def get_sub_description(headline):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Write a 1-2 sentence news brief for this Indian news headline. Be concise and journalistic. Do not start with the headline itself: {headline}"
            }],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div class="badge">AI · Built from Scratch</div>
    <h1>📰 News Headline Generator</h1>
    <p>GPT-2 transformer trained on 300K Times of India headlines · 13.76M parameters</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-item"><h3>13.76M</h3><p>Parameters</p></div>
    <div class="stat-item"><h3>300K</h3><p>Headlines</p></div>
    <div class="stat-item"><h3>~77</h3><p>Perplexity</p></div>
    <div class="stat-item"><h3>8K</h3><p>Vocabulary</p></div>
    <div class="stat-item"><h3>4.34</h3><p>Val Loss</p></div>
</div>
""", unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────
st.markdown('<div class="input-section"><h3>Enter a prompt</h3>', unsafe_allow_html=True)

prompt = st.text_input(
    "",
    placeholder="e.g. Modi, RBI cuts, Delhi, Supreme Court...",
    value="Modi",
    label_visibility="collapsed"
)

st.markdown("**Quick prompts:**")
cols = st.columns(6)
suggestions = ["Modi", "RBI", "Delhi", "India vs", "Supreme Court", "Railways"]
for i, s in enumerate(suggestions):
    if cols[i].button(s, key=f"q_{s}"):
        prompt = s

st.markdown('</div>', unsafe_allow_html=True)

num_variants = st.select_slider(
    "Number of headlines",
    options=[1, 2, 3],
    value=3
)

generate = st.button("🚀 Generate Headlines")

# ── Output ───────────────────────────────────────────────────
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt!")
    else:
        for i in range(num_variants):
            with st.spinner(f"Generating headline {i+1}..."):
                headline = generate_headline(
                    model, tokenizer, prompt,
                    max_new_tokens=60,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    device=device
                )
                sub = get_sub_description(headline)

            st.markdown(f"""
            <div class="news-card">
                <div class="card-num">Headline {i+1}</div>
                <div class="card-headline">{headline}</div>
                {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">📰</div>
        <p>Enter a prompt above and click Generate</p>
    </div>
    """, unsafe_allow_html=True)

# ── About ────────────────────────────────────────────────────
with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Architecture:** GPT-2 style transformer built from scratch in PyTorch

    - Self-attention, multi-head attention, residual connections, layer norm — all manual
    - Custom ByteLevel BPE tokenizer (8K vocab) trained on corpus
    - KV-caching for efficient autoregressive inference
    - Top-k and Top-p nucleus sampling from scratch
    - Newline-token stopping for clean headline boundaries
    - Sub-descriptions generated via Groq LLaMA API

    **Training:** T4 GPU · 300K headlines · 6000 steps · 28 mins · Val loss 4.34 · Perplexity ~77
    """)
