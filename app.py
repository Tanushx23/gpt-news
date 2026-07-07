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
        max-width: 800px;
    }
    body, p, span, label, div, h1, h2, h3, li, strong {
        color: #e6edf3 !important;
    }
    .top-header {
        border-bottom: 2px solid #e94560;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .paper-date {
        font-family: Georgia, serif;
        font-size: 0.72rem;
        color: #8b949e !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 0.4rem;
    }
    .paper-title {
        font-family: Georgia, serif;
        font-size: 2rem;
        font-weight: 700;
        color: #e6edf3 !important;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .paper-sub {
        font-family: Georgia, serif;
        font-size: 0.82rem;
        color: #8b949e !important;
        margin-top: 0.3rem;
        font-style: italic;
    }
    .stats-row {
        display: flex;
        gap: 1px;
        background: #21262d;
        border-radius: 8px;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .stat-item {
        flex: 1;
        padding: 0.75rem 0.5rem;
        text-align: center;
        background: #161b22;
    }
    .stat-item h3 {
        font-size: 1.2rem !important;
        font-weight: 700;
        margin: 0;
    }
    .stat-item p {
        font-size: 0.65rem !important;
        margin: 0.15rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e !important;
    }
    .s1 h3 { color: #e94560 !important; border-bottom: 2px solid #e94560; padding-bottom: 4px; }
    .s2 h3 { color: #1f6feb !important; border-bottom: 2px solid #1f6feb; padding-bottom: 4px; }
    .s3 h3 { color: #f78166 !important; border-bottom: 2px solid #f78166; padding-bottom: 4px; }
    .s4 h3 { color: #238636 !important; border-bottom: 2px solid #238636; padding-bottom: 4px; }
    .s5 h3 { color: #a371f7 !important; border-bottom: 2px solid #a371f7; padding-bottom: 4px; }
    .stTextInput > div > div > input {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #e6edf3 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1rem !important;
        font-family: Georgia, serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 0 2px rgba(233,69,96,0.2) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #484f58 !important;
        font-style: italic;
    }
    .news-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin: 0.8rem 0;
        position: relative;
        overflow: hidden;
    }
    .news-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
    }
    .nc1::before { background: #e94560; }
    .nc2::before { background: #1f6feb; }
    .nc3::before { background: #238636; }
    .card-section {
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
        font-family: Georgia, serif;
    }
    .nc1 .card-section { color: #e94560 !important; }
    .nc2 .card-section { color: #58a6ff !important; }
    .nc3 .card-section { color: #3fb950 !important; }
    .card-headline {
        font-family: Georgia, serif;
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.4;
        color: #e6edf3 !important;
        margin-bottom: 0.6rem;
    }
    .card-sub {
        font-family: Georgia, serif;
        font-size: 0.82rem;
        line-height: 1.5;
        color: #8b949e !important;
        padding-top: 0.6rem;
        border-top: 1px solid #21262d;
        font-style: italic;
    }
    .empty-state {
        text-align: center;
        padding: 2.5rem 0;
        border: 1px dashed #21262d;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .empty-state p { color: #484f58 !important; font-style: italic; font-family: Georgia, serif; }
    div[data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #21262d !important;
        border-radius: 8px !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "prompt_val" not in st.session_state:
    st.session_state.prompt_val = ""

@st.cache_resource(max_entries=1)
def load_assets_v6():
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
    model, tokenizer, device = load_assets_v6()

def get_sub_description(headline):
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Write a 1-2 sentence Indian news brief for this headline. Be concise and journalistic. Do not repeat the headline: {headline}"
            }],
            max_tokens=80
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

st.markdown("""
<div class="top-header">
    <div class="paper-date">AI · Built from Scratch · PyTorch</div>
    <div class="paper-title">📰 News Headline Generator</div>
    <div class="paper-sub">GPT-2 transformer trained on 300K Times of India headlines</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-item s1"><h3>13.76M</h3><p>Parameters</p></div>
    <div class="stat-item s2"><h3>300K</h3><p>Headlines</p></div>
    <div class="stat-item s3"><h3>~77</h3><p>Perplexity</p></div>
    <div class="stat-item s4"><h3>8K</h3><p>Vocabulary</p></div>
    <div class="stat-item s5"><h3>4.34</h3><p>Val Loss</p></div>
</div>
""", unsafe_allow_html=True)

prompt = st.text_input(
    "",
    value=st.session_state.prompt_val,
    placeholder="Start a headline... e.g. Modi, RBI cuts, Delhi, Supreme Court",
    label_visibility="collapsed"
)

cols = st.columns(6)
suggestions = ["Modi", "RBI", "Delhi", "India vs", "SC", "Railways"]
labels =      ["Modi", "RBI", "Delhi", "India vs", "Supreme Court", "Railways"]
for i, (s, label) in enumerate(zip(suggestions, labels)):
    with cols[i]:
        if st.button(label, key=f"qp_{s}", use_container_width=True):
            st.session_state.prompt_val = s
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    num_variants = st.radio(
        "Headlines to generate",
        options=[1, 2, 3],
        index=2,
        horizontal=True
    )
with col2:
    generate = st.button("Generate ⚡", type="primary", use_container_width=True)

colors = ["nc1", "nc2", "nc3"]
sections = ["Politics", "Economy", "India"]

if generate:
    current_prompt = st.session_state.prompt_val or prompt
    if not current_prompt.strip():
        st.warning("Enter a prompt or click a quick prompt above.")
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
            sec = sections[i % len(sections)]
            st.markdown(f"""
            <div class="news-card {c}">
                <div class="card-section">{sec}</div>
                <div class="card-headline">{headline}</div>
                {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <p>Enter a prompt above and click Generate</p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("About this model"):
    st.markdown("""
    **Architecture:** GPT-2 style transformer built from scratch in PyTorch — self-attention, multi-head attention, residual connections, and layer normalization implemented manually without `nn.Transformer`.

    **Key features:** Custom ByteLevel BPE tokenizer · KV-caching · Top-k/Top-p nucleus sampling · Newline-token stopping · Groq LLaMA sub-descriptions

    **Training:** T4 GPU · 300K headlines · 6000 steps · 28 mins · Val loss 4.34 · Perplexity ~77
    """)
