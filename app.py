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

    .stApp { background-color: #fafafa !important; }
    .main .block-container {
        background-color: #fafafa !important;
        padding: 76px 0 140px 0 !important;
        max-width: 100% !important;
    }
    *:not([data-testid="stIconMaterial"]) { font-family: 'Inter', -apple-system, sans-serif !important; }
    [data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

    .left-col, .right-col, .stMarkdown, div[data-testid="stExpander"] {
        color: #111318;
    }

    .topbar {
        position: fixed;
        top: 0; left: 0; right: 0;
        z-index: 999;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid #e5e7eb;
        padding: 0 3rem;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .topbar-logo {
        font-size: 1rem;
        font-weight: 700;
        color: #111318 !important;
        letter-spacing: -0.3px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .logo-dot {
        width: 8px; height: 8px;
        background: #4f46e5;
        border-radius: 50%;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }
    .topbar-link {
        font-size: 0.82rem;
        color: #6b7280 !important;
        font-weight: 500;
    }
    .topbar-cta {
        background: #111318;
        color: #ffffff !important;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.4rem 1.1rem;
        border-radius: 6px;
    }

    .left-col {
        background: #ffffff;
        padding: 4rem 4rem 3rem 4rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        position: relative;
        overflow: hidden;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.72rem;
        font-weight: 600;
        color: #4338ca !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 1.8rem;
        width: fit-content;
    }
    .eyebrow-dot {
        width: 5px; height: 5px;
        background: #4f46e5;
        border-radius: 50%;
    }
    .hero-h1 {
        font-size: 3.6rem;
        font-weight: 900;
        line-height: 1.05;
        letter-spacing: -2px;
        color: #111318 !important;
        margin-bottom: 1.2rem;
    }
    .hero-h1 .purple { color: #4f46e5 !important; }
    .hero-desc {
        font-size: 1rem;
        color: #6b7280 !important;
        line-height: 1.7;
        max-width: 420px;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }

    .stTextInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
        color: #111318 !important;
        font-size: 1rem !important;
        padding: 1rem 1.2rem !important;
        transition: all 0.2s !important;
        height: 54px !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 4px rgba(79,70,229,0.12) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
        font-weight: 400;
    }

    .stRadio [role="radiogroup"] label div:first-child {
        border-color: #4f46e5 !important;
    }
    .stRadio [data-baseweb="radio"] div[aria-checked="true"] {
        background-color: #4f46e5 !important;
        border-color: #4f46e5 !important;
    }
    .stRadio label span { color: #111318 !important; }

    .stButton > button {
        background: #4f46e5 !important;
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
    .stButton > button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .stButton > button:hover {
        background: #4338ca !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 20px rgba(79,70,229,0.3) !important;
    }

    .stats-strip {
        display: flex;
        gap: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e5e7eb;
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
        color: #0d9488 !important;
        letter-spacing: -0.5px;
    }
    .stat-s .l {
        font-size: 0.7rem;
        color: #9ca3af !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
    }

    .right-col {
        background: #f1f2f7;
        padding: 4rem 4rem 3rem 4rem;
        display: flex;
        flex-direction: column;
        min-height: calc(100vh - 76px - 140px);
        border-left: 1px solid #e5e7eb;
    }
    .right-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6b7280 !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .right-label-dot {
        width: 5px; height: 5px;
        background: #4f46e5;
        border-radius: 50%;
    }

    .news-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .news-card:hover {
        border-color: #c7d2fe;
        box-shadow: 0 4px 14px rgba(79,70,229,0.08);
    }
    .card-n {
        font-size: 0.65rem;
        font-weight: 700;
        color: #0d9488 !important;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.6rem;
        margin-top: 0.2rem;
    }
    .card-h {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.45;
        color: #111318 !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.75rem;
    }
    .card-d {
        font-size: 0.82rem;
        line-height: 1.6;
        color: #6b7280 !important;
        padding-top: 0.75rem;
        border-top: 1px solid #e5e7eb;
        font-weight: 400;
    }

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
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
    }
    .empty-t {
        font-size: 0.95rem;
        font-weight: 600;
        color: #374151 !important;
    }
    .empty-s {
        font-size: 0.8rem;
        color: #9ca3af !important;
        text-align: center;
        max-width: 260px;
        line-height: 1.6;
    }

    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 10px !important;
        margin-top: 2rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    a[href*="streamlit.io"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

if "prompt_val" not in st.session_state:
    st.session_state.prompt_val = ""
if "results" not in st.session_state:
    st.session_state.results = []

@st.cache_resource(max_entries=1)
def load_assets_v11():
    device = "cpu"
    model_path = hf_hub_download(repo_id="tanush23x/gpt-news-headlines", filename="best_model.pt")
    tokenizer_path = hf_hub_download(repo_id="tanush23x/gpt-news-headlines", filename="bpe_tokenizer.json")
    model, config = load_model(model_path, device)
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return model, tokenizer, device

with st.spinner(""):
    model, tokenizer, device = load_assets_v11()

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

st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="logo-dot"></div>
        NewsGPT
    </div>
    <div class="topbar-right">
        <span class="topbar-link">13.76M Parameters</span>
        <span class="topbar-link">3.8M Headlines</span>
        <span class="topbar-cta">PyTorch · Built from Scratch</span>
    </div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1])

with left:
    st.markdown("""
    <div class="left-col">
        <div class="eyebrow"><div class="eyebrow-dot"></div>AI · Indian News · GPT-2</div>
        <div class="hero-h1">Generate<br>Indian News<br><span class="purple">Headlines.</span></div>
        <div class="hero-desc">
            A GPT-2 transformer built from scratch and trained on
            3.8M Times of India headlines. Type any topic and
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
        <div class="stat-s"><div class="v">3.8M</div><div class="l">Headlines</div></div>
        <div class="stat-s"><div class="v">13.76M</div><div class="l">Parameters</div></div>
        <div class="stat-s"><div class="v">~49</div><div class="l">Perplexity</div></div>
        <div class="stat-s"><div class="v">3.91</div><div class="l">Val Loss</div></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("About this model"):
        st.markdown("""
        GPT-2 transformer built from scratch in PyTorch — self-attention, multi-head attention, residual connections, layer norm, KV-caching, top-k/top-p sampling, ByteLevel BPE tokenizer. Sub-descriptions via Groq LLaMA.

        **Training:** T4 GPU · 300K headlines · 6000 steps · ~28 mins
        """)

with right:
    gen_warning = False
    if generate:
        current_prompt = prompt or st.session_state.prompt_val
        if not current_prompt.strip():
            gen_warning = True
        else:
            st.session_state.results = []
            for i in range(num_variants):
                with st.spinner(f"Writing headline {i+1}..."):
                    h = generate_headline(
                        model, tokenizer, current_prompt,
                        max_new_tokens=60, temperature=0.75,
                        top_k=50, top_p=0.9, device=device
                    )
                    d = get_sub(h)
                    st.session_state.results.append((h, d))

    if gen_warning:
        st.warning("Enter a topic to generate headlines.")

    if st.session_state.results:
        cards_html = ""
        for i, (h, d) in enumerate(st.session_state.results):
            sub_html = f"<div class='card-d'>{d}</div>" if d else ""
            cards_html += f"""<div class="news-card">
            <div class="card-n">Headline {i+1}</div>
            <div class="card-h">{h}</div>
            {sub_html}
            </div>
            """
        body_html = cards_html
    else:
        body_html = """<div class="empty-wrap">
        <div class="empty-icon-box">📰</div>
        <div class="empty-t">No headlines yet</div>
        <div class="empty-s">Type a topic on the left and click Generate to see AI-written headlines appear here.</div>
        </div>
        """

    st.markdown(f"""
    <div class="right-col">
        <div class="right-label"><div class="right-label-dot"></div>Generated Headlines</div>
        {body_html}
    </div>
    """, unsafe_allow_html=True)
