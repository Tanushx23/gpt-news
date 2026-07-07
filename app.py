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
    layout="wide"
)

st.markdown("""
<style>
    /* Base */
    .stApp { background-color: #080b10 !important; }
    section[data-testid="stSidebar"] { display: none; }
    .main .block-container {
        background-color: #080b10 !important;
        padding: 0 !important;
        max-width: 100% !important;
    }
    * { box-sizing: border-box; }
    body, p, span, label, div, h1, h2, h3, li, strong, em {
        color: #e2e8f0 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

    /* Topbar */
    .topbar {
        background: rgba(8, 11, 16, 0.95);
        border-bottom: 1px solid #1e293b;
        padding: 1rem 3rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        z-index: 100;
        backdrop-filter: blur(12px);
    }
    .topbar-logo {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0 !important;
        letter-spacing: -0.3px;
    }
    .topbar-logo span { color: #6366f1 !important; }
    .topbar-meta {
        font-size: 0.72rem;
        color: #475569 !important;
        letter-spacing: 0.05em;
    }
    .topbar-badge {
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #818cf8 !important;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        letter-spacing: 0.05em;
    }

    /* Two column wrapper */
    .page-grid {
        display: grid;
        grid-template-columns: 420px 1fr;
        min-height: calc(100vh - 57px);
    }

    /* Left panel */
    .left-panel {
        background: #0d1117;
        border-right: 1px solid #1e293b;
        padding: 2.5rem 2rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    .panel-label {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #475569 !important;
        margin-bottom: 0.5rem;
    }
    .hero-text {
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.2;
        color: #f1f5f9 !important;
        letter-spacing: -0.8px;
    }
    .hero-text .accent { color: #6366f1 !important; }
    .hero-sub {
        font-size: 0.85rem;
        color: #64748b !important;
        line-height: 1.6;
        margin-top: -0.5rem;
    }

    /* Stats */
    .mini-stats {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
    }
    .mini-stat {
        background: #161b22;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 0.75rem 1rem;
    }
    .mini-stat .val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0 !important;
    }
    .mini-stat .lbl {
        font-size: 0.65rem;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.1rem;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #1e293b, transparent);
        margin: 0;
    }

    /* Input */
    .stTextInput > div > div > input {
        background-color: #161b22 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        outline: none !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #334155 !important;
    }

    /* Quick chips */
    .chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.3rem;
    }
    .chip {
        background: #161b22;
        border: 1px solid #1e293b;
        border-radius: 20px;
        padding: 0.3rem 0.85rem;
        font-size: 0.78rem;
        color: #94a3b8 !important;
        cursor: pointer;
        transition: all 0.15s;
        display: inline-block;
    }
    .chip:hover {
        border-color: #6366f1;
        color: #818cf8 !important;
        background: rgba(99,102,241,0.08);
    }

    /* Generate button */
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        letter-spacing: 0.01em !important;
        transition: opacity 0.2s !important;
        box-shadow: 0 4px 24px rgba(99,102,241,0.25) !important;
    }
    .stButton > button:hover {
        opacity: 0.9 !important;
        box-shadow: 0 6px 32px rgba(99,102,241,0.4) !important;
    }

    /* Right panel */
    .right-panel {
        background: #080b10;
        padding: 2.5rem 2.5rem;
    }
    .output-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1e293b;
    }
    .output-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94a3b8 !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .output-dot {
        width: 6px; height: 6px;
        background: #6366f1;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        box-shadow: 0 0 8px #6366f1;
    }

    /* News cards */
    .news-card {
        background: #0d1117;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .news-card:hover {
        border-color: #334155;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    }
    .news-card .glow {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 2px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, transparent);
    }
    .card-num {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: #6366f1 !important;
        margin-bottom: 0.5rem;
        margin-top: 0.3rem;
    }
    .card-headline {
        font-size: 1.15rem;
        font-weight: 700;
        line-height: 1.45;
        color: #f1f5f9 !important;
        margin-bottom: 0.7rem;
        letter-spacing: -0.2px;
    }
    .card-sub {
        font-size: 0.83rem;
        line-height: 1.6;
        color: #64748b !important;
        padding-top: 0.7rem;
        border-top: 1px solid #1e293b;
    }

    /* Empty state */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        text-align: center;
        gap: 1rem;
    }
    .empty-icon {
        width: 60px; height: 60px;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.6rem;
        margin: 0 auto;
    }
    .empty-title {
        font-size: 1rem;
        font-weight: 600;
        color: #334155 !important;
    }
    .empty-sub {
        font-size: 0.8rem;
        color: #1e293b !important;
        max-width: 280px;
    }

    /* Expander */
    div[data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        margin-top: 1.5rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

if "prompt_val" not in st.session_state:
    st.session_state.prompt_val = ""
if "results" not in st.session_state:
    st.session_state.results = []

@st.cache_resource(max_entries=1)
def load_assets_v7():
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
    model, tokenizer, device = load_assets_v7()

def get_sub(headline):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": f"Write a 1-2 sentence Indian news brief for this headline. Be concise and journalistic. Do not repeat the headline: {headline}"}],
            max_tokens=80
        )
        return r.choices[0].message.content.strip()
    except:
        return ""

# Topbar
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">📰 News<span>GPT</span></div>
    <div class="topbar-meta">GPT-2 · Built from Scratch · PyTorch</div>
    <div class="topbar-badge">13.76M Parameters</div>
</div>
""", unsafe_allow_html=True)

# Two column layout
left, right = st.columns([1.1, 1.8])

with left:
    st.markdown("""
    <div style="padding: 2rem 1rem 0 1rem;">
        <div class="panel-label">Indian News Headline Generator</div>
        <div class="hero-text">Generate news<br>headlines with <span class="accent">AI</span></div>
        <div class="hero-sub" style="margin-top:0.8rem">
            A GPT-2 transformer trained from scratch on 300,000
            Times of India headlines — generating realistic Indian
            news in seconds.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 1rem 1rem 0 1rem;">
    <div class="mini-stats">
        <div class="mini-stat"><div class="val">300K</div><div class="lbl">Headlines trained</div></div>
        <div class="mini-stat"><div class="val">~77</div><div class="lbl">Perplexity</div></div>
        <div class="mini-stat"><div class="val">8K</div><div class="lbl">Vocabulary size</div></div>
        <div class="mini-stat"><div class="val">4.34</div><div class="lbl">Val loss</div></div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 0 1rem;'>", unsafe_allow_html=True)
    st.markdown('<div class="panel-label" style="margin-bottom:0.3rem">Enter a topic or keyword</div>', unsafe_allow_html=True)

    prompt = st.text_input(
        "",
        value=st.session_state.prompt_val,
        placeholder="Search the news...",
        label_visibility="collapsed"
    )

    st.markdown('<div class="panel-label" style="margin-bottom:0.3rem; margin-top:0.5rem">Quick topics</div>', unsafe_allow_html=True)

    suggestions = ["Modi", "RBI", "Delhi", "India vs", "Supreme Court", "Railways", "Budget", "Election"]
    cols = st.columns(4)
    for i, s in enumerate(suggestions):
        with cols[i % 4]:
            if st.button(s, key=f"chip_{s}", use_container_width=True):
                st.session_state.prompt_val = s
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    num_variants = st.radio(
        "Headlines to generate",
        options=[1, 2, 3],
        index=2,
        horizontal=True,
        label_visibility="visible"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    generate = st.button("⚡ Generate Headlines", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("About this model"):
        st.markdown("""
        **Architecture:** GPT-2 style transformer built from scratch — self-attention, multi-head attention, residual connections, layer norm.

        **Features:** Custom ByteLevel BPE tokenizer · KV-caching · Top-k/Top-p sampling · Newline stopping · Groq LLaMA sub-descriptions

        **Training:** T4 GPU · 300K headlines · 6000 steps · Val loss 4.34
        """)

with right:
    st.markdown("""
    <div style="padding: 2rem 1.5rem 0 1.5rem;">
    <div class="output-header">
        <div class="output-title"><span class="output-dot"></span>Generated Output</div>
    </div>
    """, unsafe_allow_html=True)

    if generate:
        current_prompt = st.session_state.prompt_val or prompt
        if not current_prompt.strip():
            st.warning("Enter a topic or click a quick topic to get started.")
        else:
            st.session_state.results = []
            for i in range(num_variants):
                with st.spinner(f"Writing headline {i+1}..."):
                    headline = generate_headline(
                        model, tokenizer, current_prompt,
                        max_new_tokens=60,
                        temperature=0.8,
                        top_k=50,
                        top_p=0.9,
                        device=device
                    )
                    sub = get_sub(headline)
                    st.session_state.results.append((headline, sub))

    if st.session_state.results:
        for i, (headline, sub) in enumerate(st.session_state.results):
            st.markdown(f"""
            <div class="news-card">
                <div class="glow"></div>
                <div class="card-num">Headline {i+1}</div>
                <div class="card-headline">{headline}</div>
                {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">📰</div>
            <div class="empty-title">No headlines yet</div>
            <div class="empty-sub">Enter a topic on the left and click Generate to see AI-written headlines appear here.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
