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
    page_title="NewsGPT",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ---- Token system ----
       bg-base:      #101216  (layered dark grey, not pure black)
       bg-surface:   #191c22  (cards)
       bg-hover:     #1f232b  (card hover)
       border:       rgba(255,255,255,0.09)
       border-hover: rgba(129,140,248,0.35)
       text-primary: #f1f5f9
       text-secondary: #a3adc2  (readable grey, not near-invisible)
       text-tertiary:  #6b7686
       accent-indigo: #818cf8 / #6366f1
       accent-cyan:   #22d3ee  (contrast highlight for stats)
    */

    .stApp { background-color: #101216 !important; }
    .main .block-container {
        background-color: #101216 !important;
        padding: 76px 0 0 0 !important;
        max-width: 100% !important;
    }
    * { font-family: 'Inter', -apple-system, sans-serif !important; }
    body, p, span, label, div, h1, h2, h3, li, strong {
        color: #f1f5f9 !important;
    }

    /* Topbar */
    .topbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 999;
        background: rgba(16, 18, 22, 0.88);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(255,255,255,0.08);
        padding: 0 3rem;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .topbar-logo {
        font-size: 1rem;
        font-weight: 700;
        color: #f1f5f9 !important;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .logo-dot {
        width: 8px; height: 8px;
        background: #6366f1;
        border-radius: 50%;
        box-shadow: 0 0 10px #6366f1;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .topbar-link {
        font-size: 0.82rem;
        color: #8b94a7 !important;
        font-weight: 500;
    }
    .topbar-cta {
        background: #f1f5f9;
        color: #101216 !important;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.4rem 1.1rem;
        border-radius: 6px;
    }

    /* Left */
    .left-col {
        padding: 4rem 4rem 3rem 4rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        border-right: 1px solid rgba(255,255,255,0.07);
        position: relative;
        overflow: hidden;
    }
    .left-glow {
        position: absolute;
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(99,102,241,0.14) 0%, transparent 70%);
        top: 15%; left: -120px;
        pointer-events: none;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(129,140,248,0.3);
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #a5b4fc !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.8rem;
        width: fit-content;
    }
    .eyebrow-dot {
        width: 5px; height: 5px;
        background: #6366f1;
        border-radius: 50%;
        box-shadow: 0 0 6px #6366f1;
    }
    .hero-h1 {
        font-size: 3.6rem;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -2px;
        color: #f1f5f9 !important;
        margin-bottom: 1.2rem;
    }
    .hero-h1 .purple { color: #a5b4fc !important; }
    .hero-desc {
        font-size: 1rem;
        color: #a3adc2 !important;
        line-height: 1.7;
        max-width: 420px;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    /* Input */
    .input-wrap {
        position: relative;
        margin-bottom: 1rem;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.045) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 12px !important;
        color: #f1f5f9 !important;
        font-size: 1rem !important;
        padding: 1rem 1.2rem !important;
        transition: all 0.2s !important;
        height: 54px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #818cf8 !important;
        background: rgba(99,102,241,0.08) !important;
        box-shadow: 0 0 0 4px rgba(99,102,241,0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #5b6577 !important;
        font-weight: 400;
    }

    /* Generate button */
    .stButton > button {
        background: #f1f5f9 !important;
        color: #101216 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0 1.5rem !important;
        font-size: 0.9rem !important;
        font-weight: 700 !important;
        height: 54px !important;
        width: 100% !important;
        letter-spacing: -0.2px !important;
        transition: all 0.15s !important;
    }
    .stButton > button:hover {
        background: #c7d2fe !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.25) !important;
    }

    /* Stats row */
    .stats-strip {
        display: flex;
        gap: 2rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-top: 1.5rem;
    }
    .stat-s {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }
    .stat-s .v {
        font-size: 1.2rem;
        font-weight: 800;
        color: #22d3ee !important;
        letter-spacing: -0.5px;
    }
    .stat-s .l {
        font-size: 0.7rem;
        color: #6b7686 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    /* Right column */
    .right-col {
        background: #101216;
        padding: 4rem 4rem 3rem 4rem;
        display: flex;
        flex-direction: column;
    }
    .right-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6b7686 !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .right-label-dot {
        width: 5px; height: 5px;
        background: #6366f1;
        border-radius: 50%;
        box-shadow: 0 0 6px #6366f1;
    }

    /* Cards */
    .news-card {
        background: #191c22;
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, background 0.2s;
    }
    .news-card:hover {
        border-color: rgba(129,140,248,0.35);
        background: #1f232b;
    }
    .card-glow {
        position: absolute;
        top: 0; left: 0;
        right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, #6366f1, transparent);
        opacity: 0.7;
    }
    .card-n {
        font-size: 0.65rem;
        font-weight: 700;
        color: #22d3ee !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
    }
    .card-h {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.45;
        color: #f1f5f9 !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.75rem;
    }
    .card-d {
        font-size: 0.82rem;
        line-height: 1.6;
        color: #8b94a7 !important;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.07);
        font-weight: 400;
    }

    /* Empty */
    .empty-wrap {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 4rem 0;
    }
    .empty-icon-box {
        width: 64px; height: 64px;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(129,140,248,0.2);
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
    }
    .empty-t {
        font-size: 0.95rem;
        font-weight: 600;
        color: #a3adc2 !important;
    }
    .empty-s {
        font-size: 0.8rem;
        color: #6b7686 !important;
        text-align: center;
        max-width: 260px;
        line-height: 1.6;
    }

    div[data-testid="stExpander"] {
        background: #191c22 !important;
        border: 1px solid rgba(255,255,255,0.09) !important;
        border-radius: 10px !important;
        margin-top: 2rem;
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
def load_assets_v8():
    device = "cpu"
    model_path = hf_hub_download(repo_id="tanush23x/gpt-news-headlines", filename="best_model.pt")
    tokenizer_path = hf_hub_download(repo_id="tanush23x/gpt-news-headlines", filename="bpe_tokenizer.json")
    model, config = load_model(model_path, device)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device

with st.spinner(""):
    model, tokenizer, device = load_assets_v8()

def get_sub(headline):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role":"user","content":f"Write a 1-2 sentence Indian news brief for this headline. Be concise and journalistic. Do not repeat the headline: {headline}"}],
            max_tokens=80
        )
        return r.choices[0].message.content.strip()
    except:
        return ""

# Topbar
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-dot"></div>
        NewsGPT
    </div>
    <div class="topbar-right">
        <span class="topbar-link">13.76M Parameters</span>
        <span class="topbar-link">300K Headlines</span>
        <span class="topbar-cta">PyTorch · Built from Scratch</span>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown("""
    <div class="left-col">
        <div class="left-glow"></div>
        <div class="eyebrow"><div class="eyebrow-dot"></div>AI · Indian News · GPT-2</div>
        <div class="hero-h1">Generate<br>Indian News<br><span class="purple">Headlines.</span></div>
        <div class="hero-desc">
            A GPT-2 transformer built from scratch and trained on
            300,000 Times of India headlines. Type any topic and
            watch AI write realistic Indian news.
        </div>
    </div>
    """, unsafe_allow_html=True)

    prompt = st.text_input(
        "",
        value=st.session_state.prompt_val,
        placeholder="Search the news...",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([3,1])
    with col1:
        num_variants = st.radio("", [1,2,3], index=2, horizontal=True, label_visibility="collapsed")
    with col2:
        generate = st.button("Generate →", use_container_width=True)

    st.markdown("""
    <div class="stats-strip">
        <div class="stat-s"><div class="v">300K</div><div class="l">Headlines</div></div>
        <div class="stat-s"><div class="v">13.76M</div><div class="l">Parameters</div></div>
        <div class="stat-s"><div class="v">~77</div><div class="l">Perplexity</div></div>
        <div class="stat-s"><div class="v">4.34</div><div class="l">Val Loss</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About this model"):
        st.markdown("""
        GPT-2 transformer built from scratch in PyTorch — self-attention, multi-head attention, residual connections, layer norm, KV-caching, top-k/top-p sampling, ByteLevel BPE tokenizer. Sub-descriptions via Groq LLaMA.

        **Training:** T4 GPU · 300K headlines · 6000 steps · ~28 mins
        """)

with right:
    st.markdown("""
    <div class="right-col">
    <div class="right-label"><div class="right-label-dot"></div>Generated Headlines</div>
    """, unsafe_allow_html=True)

    if generate:
        current_prompt = prompt or st.session_state.prompt_val
        if not current_prompt.strip():
            st.warning("Enter a topic to generate headlines.")
        else:
            st.session_state.results = []
            for i in range(num_variants):
                with st.spinner(f"Writing headline {i+1}..."):
                    h = generate_headline(
                        model, tokenizer, current_prompt,
                        max_new_tokens=60, temperature=0.8,
                        top_k=50, top_p=0.9, device=device
                    )
                    d = get_sub(h)
                    st.session_state.results.append((h, d))

    if st.session_state.results:
        for i, (h, d) in enumerate(st.session_state.results):
            st.markdown(f"""
            <div class="news-card">
                <div class="card-glow"></div>
                <div class="card-n">Headline {i+1}</div>
                <div class="card-h">{h}</div>
                {"<div class='card-d'>" + d + "</div>" if d else ""}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-wrap">
            <div class="empty-icon-box">📰</div>
            <div class="empty-t">No headlines yet</div>
            <div class="empty-s">Type a topic on the left and click Generate to see AI-written headlines appear here.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    