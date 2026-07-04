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
    .stApp { background-color: #0d1117 !important; }
    .main .block-container {
        background-color: #0d1117 !important;
        padding: 2rem 2.5rem;
        max-width: 820px;
    }
    body, p, span, label, div, h1, h2, h3 { color: #e6edf3 !important; }

    .top-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 1.8rem;
    }
    .top-header .badges {
        display: flex;
        gap: 0.5rem;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .badge {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .badge-red { background: #e94560; color: white !important; }
    .badge-blue { background: #1f6feb; color: white !important; }
    .badge-green { background: #238636; color: white !important; }
    .top-header h1 {
        color: #e6edf3 !important;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .top-header p { color: #8b949e !important; font-size: 0.88rem; margin: 0; }

    .stats-row {
        display: flex;
        gap: 1px;
        background: #21262d;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1.8rem;
    }
    .stat-item {
        flex: 1;
        padding: 0.9rem 0.5rem;
        text-align: center;
    }
    .stat-item.s1 { background: #161b22; border-bottom: 3px solid #e94560; }
    .stat-item.s2 { background: #161b22; border-bottom: 3px solid #1f6feb; }
    .stat-item.s3 { background: #161b22; border-bottom: 3px solid #f78166; }
    .stat-item.s4 { background: #161b22; border-bottom: 3px solid #238636; }
    .stat-item.s5 { background: #161b22; border-bottom: 3px solid #a371f7; }
    .stat-item h3 { font-size: 1.3rem; font-weight: 700; margin: 0; }
    .stat-item.s1 h3 { color: #e94560 !important; }
    .stat-item.s2 h3 { color: #1f6feb !important; }
    .stat-item.s3 h3 { color: #f78166 !important; }
    .stat-item.s4 h3 { color: #238636 !important; }
    .stat-item.s5 h3 { color: #a371f7 !important; }
    .stat-item p { color: #8b949e !important; font-size: 0.68rem; margin: 0.2rem 0 0 0; text-transform: uppercase; letter-spacing: 0.08em; }

    .input-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
    }
    .input-card .section-label {
        color: #8b949e !important;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.6rem;
    }

    .stTextInput > div > div > input {
        background-color: #0d1117 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1rem !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1f6feb !important;
        box-shadow: 0 0 0 3px rgba(31,111,235,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #484f58 !important; }

    /* Quick prompt buttons — distinct style */
    .quick-btn .stButton > button {
        background: #21262d !important;
        color: #58a6ff !important;
        border: 1px solid #30363d !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        padding: 0.3rem 0.6rem !important;
        font-weight: 500 !important;
        width: 100% !important;
    }
    .quick-btn .stButton > button:hover {
        background: #1f6feb !important;
        color: white !important;
        border-color: #1f6feb !important;
    }

    /* Generate button */
    .gen-btn .stButton > button {
        background: linear-gradient(135deg, #238636, #2ea043) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.02em !important;
    }
    .gen-btn .stButton > button:hover {
        background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    }

    .news-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
    }
    .news-card.c1::before { content: ""; position: absolute; top:0; left:0; width:4px; height:100%; background: #e94560; }
    .news-card.c2::before { content: ""; position: absolute; top:0; left:0; width:4px; height:100%; background: #1f6feb; }
    .news-card.c3::before { content: ""; position: absolute; top:0; left:0; width:4px; height:100%; background: #238636; }
    .news-card.c4::before { content: ""; position: absolute; top:0; left:0; width:4px; height:100%; background: #a371f7; }
    .news-card.c5::before { content: ""; position: absolute; top:0; left:0; width:4px; height:100%; background: #f78166; }

    .card-num { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.5rem; }
    .c1 .card-num { color: #e94560 !important; }
    .c2 .card-num { color: #58a6ff !important; }
    .c3 .card-num { color: #3fb950 !important; }
    .c4 .card-num { color: #d2a8ff !important; }
    .c5 .card-num { color: #f78166 !important; }

    .card-headline { color: #e6edf3 !important; font-size: 1.12rem; font-weight: 700; line-height: 1.45; margin-bottom: 0.7rem; }
    .card-sub { color: #8b949e !important; font-size: 0.85rem; line-height: 1.55; padding-top: 0.7rem; border-top: 1px solid #21262d; }

    .empty-state { text-align: center; padding: 3rem 0; }
    .empty-state .icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
    .empty-state p { color: #484f58 !important; font-size: 0.9rem; }

    div[data-testid="stExpander"] { background: #161b22 !important; border: 1px solid #21262d !important; border-radius: 10px !important; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Session state for prompt ─────────────────────────────────
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

def set_prompt(p):
    st.session_state.prompt = p

# ── Load model ───────────────────────────────────────────────
@st.cache_resource(max_entries=1)
def load_assets_v5():
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
    model, tokenizer, device = load_assets_v5()

# ── Groq ─────────────────────────────────────────────────────
def get_sub_description(headline):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Write a 1-2 sentence news brief for this Indian news headline. Be concise and journalistic. Do not repeat the headline: {headline}"
            }],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <div class="badges">
        <span class="badge badge-red">AI</span>
        <span class="badge badge-blue">PyTorch</span>
        <span class="badge badge-green">Built from Scratch</span>
    </div>
    <h1>📰 News Headline Generator</h1>
    <p>GPT-2 transformer trained on 300K Times of India headlines · 13.76M parameters</p>
</div>
""", unsafe_allow_html=True)

# ── Stats ────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-item s1"><h3>13.76M</h3><p>Parameters</p></div>
    <div class="stat-item s2"><h3>300K</h3><p>Headlines</p></div>
    <div class="stat-item s3"><h3>~77</h3><p>Perplexity</p></div>
    <div class="stat-item s4"><h3>8K</h3><p>Vocabulary</p></div>
    <div class="stat-item s5"><h3>4.34</h3><p>Val Loss</p></div>
</div>
""", unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────
st.markdown('<div class="input-card"><div class="section-label">Enter a prompt</div>', unsafe_allow_html=True)

prompt = st.text_input(
    "",
    placeholder="e.g. Modi, RBI cuts, Delhi, Supreme Court, Indian Railways...",
    key="prompt",
    label_visibility="collapsed"
)

st.markdown('<div class="section-label" style="margin-top:1rem">Quick prompts</div>', unsafe_allow_html=True)

suggestions = ["Modi", "RBI", "Delhi", "India vs", "Supreme Court", "Railways"]
cols = st.columns(6)
for i, s in enumerate(suggestions):
    with cols[i]:
        st.markdown('<div class="quick-btn">', unsafe_allow_html=True)
        if st.button(s, key=f"q_{s}"):
            set_prompt(s)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

num_variants = st.select_slider(
    "Number of headlines to generate",
    options=[1, 2, 3],
    value=3
)

st.markdown('<div class="gen-btn">', unsafe_allow_html=True)
generate = st.button("⚡ Generate Headlines")
st.markdown('</div>', unsafe_allow_html=True)

# ── Output ───────────────────────────────────────────────────
colors = ["c1", "c2", "c3", "c4", "c5"]

if generate:
    current_prompt = st.session_state.prompt
    if not current_prompt.strip():
        st.warning("Please enter a prompt or click a quick prompt!")
    else:
        st.markdown("---")
        for i in range(num_variants):
            with st.spinner(f"Generating headline {i+1}..."):
                headline = generate_headline(
                    model, tokenizer, current_prompt,
                    max_new_tokens=60,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    device=device
                )
                sub = get_sub_description(headline)

            c = colors[i % len(colors)]
            st.markdown(f"""
            <div class="news-card {c}">
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

with st.expander("ℹ️ About this model"):
    st.markdown("""
    **Architecture:** GPT-2 style transformer built from scratch in PyTorch

    - Self-attention, multi-head attention, residual connections, layer norm — all manual
    - Custom ByteLevel BPE tokenizer (8K vocab) trained on corpus
    - KV-caching for efficient autoregressive inference
    - Top-k and Top-p nucleus sampling from scratch
    - Newline-token stopping for clean headline boundaries
    - Sub-descriptions via Groq LLaMA API

    **Training:** T4 GPU · 300K headlines · 6000 steps · 28 mins · Val loss 4.34 · Perplexity ~77
    """)
