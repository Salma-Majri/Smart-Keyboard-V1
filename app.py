"""Lab 02 - Smart Keyboard V1 (Streamlit UI Modernized).
Run: streamlit run app.py
"""
import streamlit as st
from src.pipeline import NGramModel, EOS, load_documents, backoff_top_k

# ─────────────────────────────────────────────────────────────────── Page Config
st.set_page_config(
    page_title="Smart Keyboard V1",
    page_icon="⌨️",
    layout="centered"
)

# ─────────────────────────────────────────────────────────────────── CSS Modern Styling
st.markdown("""
<style>
/* Background principal Dark Futuristic */
html, body, [data-testid="stApp"] {
    background: radial-gradient(circle at 50% 10%, #1e1b4b 0%, #0f172a 60%, #020617 100%);
    color: #f8fafc;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Glassmorphism Container */
.glass-container {
    background: rgba(30, 41, 59, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(16px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
}

/* Hero Title */
.hero-title {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -1.5px;
    margin-bottom: 0rem;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.2rem;
    margin-bottom: 1.5rem;
}

/* Badge for Active Model Order */
.model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #a855f7);
    color: #ffffff;
    border-radius: 999px;
    padding: 4px 16px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 2px 10px rgba(168, 85, 247, 0.4);
    margin-left: 8px;
}

/* Text Input Styling */
[data-testid="stTextInput"] input {
    background: rgba(15, 23, 42, 0.9) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    color: #f8fafc !important;
    font-size: 1.2rem !important;
    padding: 0.9rem 1.2rem !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
}
[data-testid="stTextInput"] input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 20px rgba(129, 140, 248, 0.35), inset 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* Interactive Suggestion Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.8rem 1rem !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 25px rgba(124, 58, 237, 0.55) !important;
    background: linear-gradient(135deg, #6366f1 0%, #9333ea 100%) !important;
}

/* Card Metrics Styling */
[data-testid="stMetric"] {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    text-align: center !important;
}
[data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
}
[data-testid="stMetricLabel"] {
    color: #f1f5f9 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────── Model Loader
@st.cache_resource
def get_models():
    docs = load_documents()
    sentences = [s for d in docs.values() for s in d]
    return [NGramModel(sentences, n, 0.1) for n in (3, 2, 1)], sentences

(tri, bi, uni), sentences = get_models()

# ─────────────────────────────────────────────────────────────────── Header
st.markdown('<p class="hero-title">⌨️ Smart Keyboard V1</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Next-word suggestions from a trigram model with add-k smoothing and back-off.</p>', 
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────── Main
st.markdown('<div class="glass-container">', unsafe_allow_html=True)

if "typed_text" not in st.session_state:
    st.session_state.typed_text = "i want to"

# Unique single text input
text = st.text_input(
    "Type the beginning of a sentence:",
    value=st.session_state.typed_text,
    key="unique_sentence_input"
)
st.session_state.typed_text = text

words = text.lower().replace(".", " . ").split()
unknown = [w for w in words if w not in uni.vocab]

if unknown:
    st.warning(f"⚠️ Not in the training vocabulary (treated as `<unk>`): {', '.join(unknown)}")

if not words:
    st.info("ℹ️ Type at least one word – showing how sentences usually start.")

suggestions, order = backoff_top_k(tri, bi, uni, words, 3)

st.markdown(
    f'<div style="margin-top: 1.5rem; margin-bottom: 1rem; font-size: 1.1rem; color: #cbd5e1;">'
    f'<b>Top-3 suggestions</b> · used model: <span class="model-badge">{order}</span>'
    f'</div>', 
    unsafe_allow_html=True
)

cols = st.columns(3)
for col, (w, p) in zip(cols, suggestions):
    label = "(end of sentence)" if w == EOS else w
    with col:
        st.metric(label, f"{p:.1%}")
        btn_label = "⏎ End" if w == EOS else f"+ {w}"
        if st.button(btn_label, key=f"btn_{w}", use_container_width=True):
            if w != EOS:
                st.session_state.typed_text = (text.rstrip() + " " + w)
            else:
                st.session_state.typed_text = text.rstrip() + " ."
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

st.caption("🔒 Probabilities are model estimates on a synthetic corpus — not real-world typing statistics.")
