"""
app.py
=======
Streamlit UI for the Codeforces RAG Chatbot.
"""

import os
import sys
import json
import time

import streamlit as st
from dotenv import load_dotenv

# ── pysqlite3 patch for Streamlit Cloud ─────────────────────────────────────
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

load_dotenv()

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CF Assistant — Codeforces RAG Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

  /* ── Global reset ── */
  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #e6edf3;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

  /* ── Header gradient ── */
  .cf-header {
    background: linear-gradient(135deg, #1f6feb 0%, #388bfd 40%, #58a6ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2rem;
    line-height: 1.2;
    margin-bottom: 0.25rem;
  }
  .cf-subtitle {
    color: #8b949e;
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border: 1px solid #21262d;
    background: #161b22;
  }
  [data-testid="stChatMessage"][data-testid*="user"] {
    background: #1f2a3a;
    border-color: #1f6feb44;
  }

  /* ── Code blocks ── */
  code, pre {
    font-family: 'Fira Code', monospace !important;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
    font-size: 0.85rem;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
  }
  .metric-card:hover { border-color: #1f6feb; }
  .metric-title { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #58a6ff; }
  .metric-sub { font-size: 0.75rem; color: #6e7681; margin-top: 0.1rem; }

  /* ── Tag pills ── */
  .tag-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    background: #1f2a3a;
    border: 1px solid #1f6feb55;
    color: #58a6ff;
    font-size: 0.72rem;
    margin: 2px;
    font-weight: 500;
  }

  /* ── Rating badge ── */
  .rating-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.8rem;
    font-family: 'Fira Code', monospace;
  }
  .rating-easy { background: #1b3a2d; color: #3fb950; border: 1px solid #3fb95055; }
  .rating-med  { background: #3a2e1b; color: #d29922; border: 1px solid #d2992255; }
  .rating-hard { background: #3a1b1b; color: #f85149; border: 1px solid #f8514955; }

  /* ── Divider ── */
  .section-divider {
    height: 1px;
    background: linear-gradient(90deg, #1f6feb33, #58a6ff55, #1f6feb33);
    margin: 1rem 0;
    border: none;
  }

  /* ── Input area ── */
  .stChatInput textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
  }
  .stChatInput textarea:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 3px #1f6feb22 !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: opacity 0.2s ease !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* ── Status banner ── */
  .status-ok {
    background: #1b3a2d;
    border: 1px solid #3fb95055;
    border-radius: 8px;
    padding: 8px 14px;
    color: #3fb950;
    font-size: 0.82rem;
    margin-bottom: 0.75rem;
  }
  .status-err {
    background: #3a1b1b;
    border: 1px solid #f8514955;
    border-radius: 8px;
    padding: 8px 14px;
    color: #f85149;
    font-size: 0.82rem;
    margin-bottom: 0.75rem;
  }
  
  /* ── Hint level selector ── */
  .stSelectbox label, .stSlider label { color: #8b949e !important; font-size: 0.82rem !important; }
  
  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #1f6feb; }
</style>
""", unsafe_allow_html=True)


# ── Lazy-load agent loop ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_agent(provider: str, api_key: str):
    from agent.loop import AgentLoop
    return AgentLoop(provider=provider, api_key=api_key)


# ── Helper: rating → CSS class ───────────────────────────────────────────────
def rating_class(rating):
    if rating is None or rating < 0:
        return "rating-med", "N/A"
    if rating <= 1400:
        return "rating-easy", str(rating)
    if rating <= 1900:
        return "rating-med", str(rating)
    return "rating-hard", str(rating)


# ── Helper: render a problem card ─────────────────────────────────────────────
def render_problem_card(pid: str, name: str, tags: list, rating):
    cls, rat_str = rating_class(rating)
    tags_html = "".join(f'<span class="tag-pill">{t}</span>' for t in tags[:5])
    return f"""
    <div class="metric-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <div style="font-weight:600; color:#e6edf3;">{pid}: {name}</div>
        <span class="rating-badge {cls}">{rat_str}</span>
      </div>
      <div style="margin-top:6px;">{tags_html}</div>
    </div>
    """


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="cf-header">⚡ CF Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="cf-subtitle">Codeforces RAG Chatbot · Powered by Gemini</div>', unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Provider status & Key Input
    provider = os.getenv("LLM_PROVIDER", "gemini")
    env_key = os.getenv("GEMINI_API_KEY") if provider == "gemini" else os.getenv("GROQ_API_KEY")
    
    user_api_key = st.text_input(
        f"{provider.title()} API Key", 
        value=env_key or "", 
        type="password", 
        placeholder=f"Enter your {provider.title()} API key",
        help="Your key is not stored and is only used for this session."
    )
    
    active_ok = bool(user_api_key.strip())
    if active_ok:
        st.markdown(f'<div class="status-ok">✓ {provider.title()} connected</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="status-err">✗ {provider.title()} key missing</div>', unsafe_allow_html=True)

    st.markdown("**⚙ Settings**")
    hint_level = st.select_slider(
        "Default hint level",
        options=["Nudge", "Key Insight", "Pseudocode", "Full Solution"],
        value="Nudge",
        help="Controls how much detail the assistant gives by default"
    )
    show_tool_calls = st.checkbox("Show tool call trace", value=False)
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Quick search
    st.markdown("**🔍 Quick Problem Search**")
    search_query = st.text_input("Query", placeholder="e.g. graph BFS 1400-1600", label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        tag_filter = st.text_input("Tag", placeholder="graphs", label_visibility="collapsed")
    with col2:
        rating_max = st.number_input("Max rating", value=2000, step=100, label_visibility="collapsed")
    
    if st.button("Search", use_container_width=True):
        if search_query:
            with st.spinner("Searching..."):
                from retrieval.retriever import HybridRetriever
                retriever = HybridRetriever()
                results = retriever.hybrid_search(
                    query=search_query,
                    k=6,
                    tag_filter=tag_filter or None,
                    rating_max=int(rating_max)
                )
                st.session_state["search_results"] = results
        else:
            st.warning("Enter a search query first")

    # Show search results
    if "search_results" in st.session_state and st.session_state["search_results"]:
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown(f"**{len(st.session_state['search_results'])} results**")
        for r in st.session_state["search_results"]:
            meta = r.metadata
            pid = r.problem_id
            tags = json.loads(meta.get("tags", "[]")) if isinstance(meta.get("tags"), str) else []
            st.markdown(render_problem_card(
                pid, pid, tags, meta.get("rating")
            ), unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    if st.button("🗑 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown('<h1 class="cf-header" style="font-size:1.7rem;">⚡ Codeforces RAG Chatbot</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="cf-subtitle">Ask me about any Codeforces problem · I\'ll search, retrieve, and guide you with progressive hints</p>',
    unsafe_allow_html=True
)

# Initialise session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if active_ok and "agent" not in st.session_state:
    with st.spinner("Warming up the assistant…"):
        st.session_state["agent"] = get_agent(provider, user_api_key)

# Eval metrics banner
eval_path = "eval/results"
if os.path.exists(eval_path):
    files = sorted([f for f in os.listdir(eval_path) if "retrieval" in f], reverse=True)
    if files:
        with open(os.path.join(eval_path, files[0])) as f:
            eval_data = json.load(f)
        summ = eval_data.get("summary", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-title">MRR</div>
              <div class="metric-value">{summ.get('mean_mrr', 0):.3f}</div>
              <div class="metric-sub">Mean Reciprocal Rank</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-title">Recall@5</div>
              <div class="metric-value">{summ.get('mean_recall@5', 0):.1%}</div>
              <div class="metric-sub">Top 5 retrieval accuracy</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-title">Recall@10</div>
              <div class="metric-value">{summ.get('mean_recall@10', 0):.1%}</div>
              <div class="metric-sub">Top 10 retrieval accuracy</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# Render chat history
for msg in st.session_state["messages"]:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "⚡"):
        st.markdown(content)
        if show_tool_calls and "tool_calls" in msg:
            with st.expander("🔧 Tool calls"):
                st.json(msg["tool_calls"])

# Suggested prompts (only shown when chat is empty)
if not st.session_state["messages"]:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    suggestions = [
        ("🔍 Find problems", "Find me DP problems rated 1400-1600"),
        ("📖 Explain", "What is problem 2041C about?"),
        ("💡 Hint", "I'm stuck on 2041C. Give me a nudge."),
        ("🖥 Run code", "Check this Python solution for 2041C:\n```python\nprint('hello')\n```"),
    ]
    for i, (label, prompt) in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(label, key=f"suggest_{i}", use_container_width=True):
                st.session_state["pending_prompt"] = prompt
                st.rerun()

# Handle pending prompt from buttons
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Inject hint level into prompt
    hint_map = {"Nudge": 1, "Key Insight": 2, "Pseudocode": 3, "Full Solution": 4}
    if hint_map[hint_level] > 1 and "nudge" not in prompt.lower() and "hint" not in prompt.lower():
        augmented = prompt + f"\n\n[Hint level preference: {hint_level}]"
    else:
        augmented = prompt

    with st.chat_message("assistant", avatar="⚡"):
        if not active_ok:
            st.error("Please provide an API key in the sidebar.")
            st.stop()
        with st.spinner("Thinking…"):
            try:
                agent = get_agent(provider, user_api_key)
                history = [m for m in st.session_state["messages"][:-1] if m["role"] in ("user", "assistant")]
                response = agent.chat(augmented, history=history)
            except Exception as e:
                response = f"⚠️ Error: {e}"
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
    st.rerun()

# Chat input
chat_placeholder = "Ask about any Codeforces problem…" if active_ok else "Please enter your API key in the sidebar to chat."
if user_input := st.chat_input(chat_placeholder, key="chat_input", disabled=not active_ok):
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    hint_map = {"Nudge": 1, "Key Insight": 2, "Pseudocode": 3, "Full Solution": 4}
    if hint_map[hint_level] > 1:
        augmented = user_input + f"\n\n[Hint level preference: {hint_level}]"
    else:
        augmented = user_input

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Thinking…"):
            try:
                agent = get_agent(provider, user_api_key) # guarantee fresh instance based on key
                history = [m for m in st.session_state["messages"][:-1] if m["role"] in ("user", "assistant")]
                response = agent.chat(augmented, history=history)
            except Exception as e:
                response = f"⚠️ Error: {e}"
        st.markdown(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
