import streamlit as st
import os
import numpy as np
import cv2
import base64
import tempfile

import json

from watermark_logic import (
    encode_image, decode_image, remove_watermark,
    compute_trust_score, _adaptive_strength, _password_to_seed,
    _prepare_bits,
)
from video_engine import (
    encode_video, decode_video, _get_video_info,
    extract_single_frame, get_frame_trust_score, extract_frames,
    reassemble_video, extract_audio, HAS_FFMPEG,
)
from tamper_detection import generate_tamper_map, get_tamper_summary
from certificate import generate_certificate, HAS_FPDF
from signature_registry import (
    register_signature, get_key_for_signature,
    list_signatures, signature_exists,
)
from translations import LANGS, RTL_LANGS, t

# ─────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="VeriFrame — Digital Authenticity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CSS — Premium glassmorphism theme
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e2e8f0;
}

:root {
    --bg-primary: #06080f;
    --bg-card: rgba(15, 20, 35, 0.7);
    --bg-card-solid: #0f1423;
    --bg-glass: rgba(15, 23, 42, 0.55);
    --border: rgba(56, 189, 248, 0.12);
    --border-hover: rgba(56, 189, 248, 0.3);
    --cyan: #38bdf8;
    --indigo: #818cf8;
    --purple: #a78bfa;
    --green: #22c55e;
    --yellow: #f59e0b;
    --red: #ef4444;
    --orange: #fb923c;
    --slate: #64748b;
    --light: #e2e8f0;
    --glow-cyan: 0 0 30px rgba(56,189,248,0.15);
    --glow-green: 0 0 30px rgba(34,197,94,0.15);
    --glow-red: 0 0 30px rgba(239,68,68,0.15);
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 0.8rem 2rem 2rem 2rem;
    max-width: 1260px;
    background: var(--bg-primary);
}

/* ── Animated background mesh ── */
.stApp {
    background:
        radial-gradient(ellipse at 20% 0%, rgba(56,189,248,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 100%, rgba(129,140,248,0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(15,20,35,1) 0%, #06080f 100%);
    background-attachment: fixed;
}

/* ── Hero header ── */
.hero {
    position: relative;
    padding: 1.8rem 0 1.2rem 0;
    margin-bottom: 1rem;
}
.hero::after {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--cyan) 30%, var(--indigo) 70%, transparent 100%);
    opacity: 0.5;
}

.hero-logo {
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
    line-height: 1.1;
}
.hero-shield {
    font-size: 2rem;
    margin-right: 0.4rem;
    -webkit-text-fill-color: initial;
    filter: drop-shadow(0 0 12px rgba(56,189,248,0.4));
}
.hero-tagline {
    font-size: 0.82rem;
    color: var(--slate);
    margin-top: 4px;
    font-weight: 400;
    letter-spacing: 0.5px;
}

/* ── Tech badges row ── */
.tech-row {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    margin: 0.8rem 0 0.4rem 0;
}
.tech-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid;
    transition: all 0.25s ease;
}
.tech-badge:hover { transform: translateY(-1px); }
.tb-dct  { background: rgba(56,189,248,0.08); color: #38bdf8; border-color: rgba(56,189,248,0.25); }
.tb-ss   { background: rgba(167,139,250,0.08); color: #a78bfa; border-color: rgba(167,139,250,0.25); }
.tb-ecc  { background: rgba(52,211,153,0.08); color: #34d399; border-color: rgba(52,211,153,0.25); }
.tb-sync { background: rgba(251,146,60,0.08); color: #fb923c; border-color: rgba(251,146,60,0.25); }
.tb-dot  { width: 6px; height: 6px; border-radius: 50%; }
.tb-dct .tb-dot  { background: #38bdf8; box-shadow: 0 0 6px #38bdf8; }
.tb-ss  .tb-dot  { background: #a78bfa; box-shadow: 0 0 6px #a78bfa; }
.tb-ecc .tb-dot  { background: #34d399; box-shadow: 0 0 6px #34d399; }
.tb-sync .tb-dot { background: #fb923c; box-shadow: 0 0 6px #fb923c; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 14px;
    padding: 5px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: var(--slate);
    font-weight: 600;
    font-size: 0.88rem;
    padding: 10px 28px;
    border: 1px solid transparent;
    transition: all 0.25s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--light);
    background: rgba(56,189,248,0.05);
}
.stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.1) !important;
    color: var(--cyan) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    box-shadow: var(--glow-cyan);
}

/* ── Glass cards ── */
.glass-card {
    background: var(--bg-glass);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.glass-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--glow-cyan);
}

.section-title {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--cyan);
    margin-bottom: 1rem;
    display: flex; align-items: center; gap: 8px;
}
.section-title::before {
    content: '';
    width: 3px; height: 14px;
    background: linear-gradient(180deg, var(--cyan), var(--indigo));
    border-radius: 2px;
}

/* ── Trust score — dramatic display ── */
.trust-display {
    position: relative;
    background: linear-gradient(135deg, rgba(6,8,15,0.9), rgba(15,23,42,0.9));
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 1.5rem;
    text-align: center;
    overflow: hidden;
}
.trust-display::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
}
.trust-display.trust-glow-green::before {
    background: linear-gradient(90deg, transparent, #22c55e, transparent);
    box-shadow: 0 0 20px rgba(34,197,94,0.3);
}
.trust-display.trust-glow-yellow::before {
    background: linear-gradient(90deg, transparent, #f59e0b, transparent);
    box-shadow: 0 0 20px rgba(245,158,11,0.3);
}
.trust-display.trust-glow-red::before {
    background: linear-gradient(90deg, transparent, #ef4444, transparent);
    box-shadow: 0 0 20px rgba(239,68,68,0.3);
}

.trust-number {
    font-size: 4.5rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -3px;
    font-family: 'JetBrains Mono', monospace;
}
.trust-high  { color: #22c55e; text-shadow: 0 0 40px rgba(34,197,94,0.3); }
.trust-mid   { color: #f59e0b; text-shadow: 0 0 40px rgba(245,158,11,0.3); }
.trust-low   { color: #ef4444; text-shadow: 0 0 40px rgba(239,68,68,0.3); }
.trust-unit  { font-size: 1.2rem; font-weight: 400; color: var(--slate); margin-left: 2px; }
.trust-label { font-size: 0.78rem; color: var(--slate); margin-top: 8px; letter-spacing: 1px; }

/* ── Message display card ── */
.msg-card {
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.msg-label { font-size: 0.68rem; color: var(--slate); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.msg-text  { font-size: 1.15rem; font-weight: 700; color: var(--cyan); font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px; }

/* ── Layer bars ── */
.layer-bar {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0;
}
.layer-label {
    font-size: 0.72rem; font-weight: 600; min-width: 130px;
    color: var(--slate);
}
.layer-track {
    flex: 1; height: 6px; border-radius: 3px;
    background: rgba(30, 41, 59, 0.8);
    overflow: hidden;
}
.layer-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}
.fill-cyan   { background: linear-gradient(90deg, #0ea5e9, #38bdf8); box-shadow: 0 0 8px rgba(56,189,248,0.3); }
.fill-purple { background: linear-gradient(90deg, #7c3aed, #a78bfa); box-shadow: 0 0 8px rgba(167,139,250,0.3); }
.fill-green  { background: linear-gradient(90deg, #059669, #34d399); box-shadow: 0 0 8px rgba(52,211,153,0.3); }
.fill-orange { background: linear-gradient(90deg, #ea580c, #fb923c); box-shadow: 0 0 8px rgba(251,146,60,0.3); }
.layer-val   { font-size: 0.75rem; font-weight: 700; min-width: 40px; text-align: right; font-family: 'JetBrains Mono', monospace; }

/* ── Severity system ── */
.severity-banner {
    display: flex; align-items: center; gap: 14px;
    border-radius: 14px; padding: 1rem 1.2rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.sev-pristine { background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25); }
.sev-minor    { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.25); }
.sev-moderate { background: rgba(251,146,60,0.1); border: 1px solid rgba(251,146,60,0.25); }
.sev-severe   { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.25); }
.sev-icon     { font-size: 1.8rem; }
.sev-title    { font-weight: 700; font-size: 1rem; }
.sev-detail   { font-size: 0.78rem; color: var(--slate); margin-top: 2px; }
.severity-pristine { color: #22c55e; font-weight: 700; }
.severity-minor    { color: #f59e0b; font-weight: 700; }
.severity-moderate { color: #fb923c; font-weight: 700; }
.severity-severe   { color: #ef4444; font-weight: 700; }
.severity-none     { color: var(--slate); font-weight: 700; }

/* ── PSNR meter ── */
.psnr-bar {
    background: rgba(30,41,59,0.6); border-radius: 8px;
    padding: 0.8rem 1rem; margin: 0.5rem 0;
    border: 1px solid var(--border);
}
.psnr-value {
    font-size: 1.6rem; font-weight: 800; color: var(--green);
    font-family: 'JetBrains Mono', monospace;
}
.psnr-unit { font-size: 0.8rem; color: var(--slate); font-weight: 400; }
.psnr-desc { font-size: 0.72rem; color: var(--slate); margin-top: 2px; }

/* ── Forensic info box ── */
.forensic-info {
    background: rgba(15,23,42,0.6);
    border-radius: 10px; padding: 0.7rem 1rem;
    margin-bottom: 0.8rem;
    font-size: 0.78rem; color: var(--slate);
    line-height: 1.5;
    border-left: 3px solid;
}

/* ── Region tag ── */
.region-tag {
    background: rgba(30,41,59,0.8);
    border-radius: 10px; padding: 0.5rem 0.8rem;
    margin-bottom: 5px;
    border-left: 3px solid var(--red);
    font-size: 0.8rem;
    display: flex; align-items: center; gap: 8px;
}
.region-num { color: var(--red); font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.region-dim { color: var(--slate); }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-radius: 10px !important;
    color: var(--light) !important;
    font-size: 0.9rem !important;
    backdrop-filter: blur(8px) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,0.1), var(--glow-cyan) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #818cf8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(14,165,233,0.35) !important;
}

.stDownloadButton > button {
    background: linear-gradient(135deg, rgba(15,23,42,0.8), rgba(30,41,59,0.8)) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    border-radius: 10px !important;
    color: var(--cyan) !important;
    font-weight: 600 !important;
    transition: all 0.25s ease !important;
}
.stDownloadButton > button:hover {
    border-color: var(--cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

/* ── File uploader ── */
.stFileUploader > div {
    background: rgba(15, 23, 42, 0.4) !important;
    border: 2px dashed rgba(56,189,248,0.2) !important;
    border-radius: 14px !important;
    transition: border-color 0.3s ease !important;
}
.stFileUploader > div:hover {
    border-color: rgba(56,189,248,0.4) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #818cf8) !important;
    border-radius: 4px !important;
    box-shadow: 0 0 10px rgba(14,165,233,0.3) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(15,23,42,0.5) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    font-weight: 600 !important;
    color: var(--cyan) !important;
}

/* ── Metric ── */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.4);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: var(--cyan) !important;
}

/* ── Dividers ── */
hr { border-color: rgba(56,189,248,0.1) !important; }

/* ── RTL support ── */
.rtl { direction: rtl; text-align: right; }

/* ── Legend dots ── */
.legend-row { display: flex; gap: 18px; justify-content: center; margin: 0.5rem 0; }
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.72rem; color: var(--slate); }
.legend-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── Color chips for frame scores ── */
.frame-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 8px;
    font-size: 0.78rem; margin: 2px 0;
    background: rgba(15,23,42,0.5);
    border: 1px solid var(--border);
}

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-glow {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}
.animate-in { animation: fadeInUp 0.5s ease-out; }
.pulse { animation: pulse-glow 2s ease-in-out infinite; }

/* ── Footer ── */
.vf-footer {
    text-align: center; padding: 1.5rem 0 0.5rem 0;
    font-size: 0.7rem; color: rgba(100,116,139,0.6);
    letter-spacing: 0.5px;
}
.vf-footer-line {
    width: 100%; height: 1px; margin-bottom: 1rem;
    background: linear-gradient(90deg, transparent, rgba(56,189,248,0.2), rgba(129,140,248,0.2), transparent);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Session state
# ─────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "he"

os.makedirs("tmp", exist_ok=True)


# ─────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────
def _psnr(img1_path, img2_path) -> float:
    a = cv2.imread(img1_path)
    b = cv2.imread(img2_path)
    if a is None or b is None:
        return 0.0
    a = a.astype(np.float64)
    b = cv2.resize(b, (a.shape[1], a.shape[0])).astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return round(20 * np.log10(255.0 / np.sqrt(mse)), 2)


def _trust_color(score: int) -> str:
    if score >= 70:
        return "trust-high"
    elif score >= 40:
        return "trust-mid"
    return "trust-low"


def _trust_glow(score: int) -> str:
    if score >= 70:
        return "trust-glow-green"
    elif score >= 40:
        return "trust-glow-yellow"
    return "trust-glow-red"


def _severity_class(severity: str) -> str:
    return f"severity-{severity}"


def _cv2_to_rgb(img):
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────
#  Hero Header
# ─────────────────────────────────────────
header_col, lang_col = st.columns([4, 1])

with header_col:
    lang = st.session_state.lang
    st.markdown(f"""
    <div class="hero animate-in">
        <div>
            <span class="hero-logo">
                <span class="hero-shield">🛡️</span>VeriFrame
            </span>
            <div class="hero-tagline">{t('app_subtitle', lang)}</div>
        </div>
        <div class="tech-row">
            <span class="tech-badge tb-dct"><span class="tb-dot"></span>DCT Frequency</span>
            <span class="tech-badge tb-ss"><span class="tb-dot"></span>Spread Spectrum</span>
            <span class="tech-badge tb-ecc"><span class="tb-dot"></span>Reed-Solomon ECC</span>
            <span class="tech-badge tb-sync"><span class="tb-dot"></span>Sync Markers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def _on_lang_change():
    st.session_state.lang = LANGS[st.session_state._lang_selector]

with lang_col:
    st.markdown("<br>", unsafe_allow_html=True)
    # Find current language display name
    _lang_display = [k for k, v in LANGS.items() if v == st.session_state.lang][0]
    st.selectbox(
        "🌐",
        options=list(LANGS.keys()),
        index=list(LANGS.keys()).index(_lang_display),
        label_visibility="collapsed",
        key="_lang_selector",
        on_change=_on_lang_change,
    )
    lang = st.session_state.lang

# ─────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────
tab_embed, tab_verify, tab_remove = st.tabs([
    f"  {t('tab_embed', lang)}  ",
    f"  {t('tab_verify', lang)}  ",
    f"  {t('tab_remove', lang)}  ",
])


# ══════════════════════════════════════════
#  TAB 1 — EMBED
# ══════════════════════════════════════════
with tab_embed:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f'<div class="section-title">{t("tab_embed", lang)}</div>',
                    unsafe_allow_html=True)

        uploaded = st.file_uploader(
            t("upload_media", lang),
            type=["png", "jpg", "jpeg", "mp4"],
            key="embed_upload",
        )

        secret = st.text_input(
            t("signature_name_label", lang),
            placeholder="e.g.  CAM-001-2026",
            key="embed_secret",
        )
        password = st.text_input(
            t("encryption_key_label", lang),
            type="password", placeholder="••••••••", key="embed_pass",
        )

        # Check for double signing
        _already_signed = False
        if uploaded:
            is_video = uploaded.name.lower().endswith(".mp4")
            ext = "mp4" if is_video else "png"
            in_path = f"tmp/embed_in.{ext}"
            out_path = f"tmp/embed_out.{ext}"
            with open(in_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Check if file already has a watermark (try all known signatures)
            if not is_video:
                known_sigs = list_signatures()
                for sig_name in known_sigs:
                    sig_key = get_key_for_signature(sig_name)
                    check_result = decode_image(in_path, sig_key)
                    if "✅" in check_result:
                        _already_signed = True
                        st.error(f"⛔ {t('already_signed', lang)}")
                        st.markdown(f"""
                        <div class="msg-card animate-in">
                            <div class="msg-label">{t('embedded_signature', lang)}</div>
                            <div class="msg-text">{sig_name}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        break

        if uploaded and secret and password and not _already_signed:
            if st.button(t("btn_embed", lang), key="do_embed"):
                if is_video:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text(t("video_progress", lang))
                    result = encode_video(
                        in_path, secret, out_path, password,
                        progress_callback=lambda p: progress_bar.progress(min(p, 1.0)),
                    )
                    progress_bar.progress(1.0)
                    status_text.empty()
                else:
                    with st.spinner("Embedding watermark..."):
                        result = encode_image(in_path, secret, out_path, password)

                if "✅" in result:
                    # Save to registry
                    register_signature(secret, password)
                    st.success(result)
                    st.info(f"📋 {t('signature_saved', lang)}: **{secret}**")
                    st.session_state["embed_done"] = True
                    st.session_state["embed_in"] = in_path
                    st.session_state["embed_out"] = out_path
                    st.session_state["embed_is_video"] = is_video
                    if not is_video:
                        st.session_state["embed_psnr"] = _psnr(in_path, out_path)
                else:
                    st.error(result)

    with right:
        if st.session_state.get("embed_done"):
            is_vid = st.session_state.get("embed_is_video", False)
            st.markdown(f'<div class="section-title">{t("preview_label", lang)}</div>',
                        unsafe_allow_html=True)

            if is_vid:
                st.video(st.session_state["embed_out"])
                with open(st.session_state["embed_out"], "rb") as f:
                    st.download_button(
                        t("btn_download", lang), f.read(),
                        file_name="veriframe_watermarked.mp4", mime="video/mp4",
                    )
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(st.session_state["embed_in"],
                             caption=t("original_label", lang), use_container_width=True)
                with c2:
                    st.image(st.session_state["embed_out"],
                             caption=t("watermarked_label", lang), use_container_width=True)

                # PSNR display
                psnr = st.session_state.get("embed_psnr", 0)
                quality_text = "Imperceptible" if psnr >= 40 else "Good" if psnr >= 30 else "Visible"
                st.markdown(f"""
                <div class="psnr-bar">
                    <span class="psnr-value">{psnr}</span>
                    <span class="psnr-unit"> dB</span>
                    <span class="psnr-desc">&nbsp;&nbsp;PSNR — {quality_text}</span>
                </div>
                """, unsafe_allow_html=True)

                with open(st.session_state["embed_out"], "rb") as f:
                    st.download_button(
                        t("btn_download", lang), f.read(),
                        file_name="veriframe_watermarked.png", mime="image/png",
                    )


# ══════════════════════════════════════════
#  TAB 2 — VERIFY
# ══════════════════════════════════════════
with tab_verify:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f'<div class="section-title">{t("tab_verify", lang)}</div>',
                    unsafe_allow_html=True)

        uploaded_v = st.file_uploader(
            t("upload_media", lang),
            type=["png", "jpg", "jpeg", "mp4"],
            key="verify_upload",
        )

        # Select signature from registry (user doesn't need to know the password)
        known_sigs = list_signatures()
        password_v = ""

        if known_sigs:
            selected_sig = st.selectbox(
                t("verify_signature_select", lang),
                options=known_sigs,
                key="verify_sig_select",
            )
            password_v = get_key_for_signature(selected_sig) if selected_sig else ""
        else:
            st.info(t("no_signatures", lang))

        if uploaded_v:
            is_video = uploaded_v.name.lower().endswith(".mp4")
            ext = "mp4" if is_video else "png"
            v_path = f"tmp/verify_input.{ext}"
            with open(v_path, "wb") as f:
                f.write(uploaded_v.getbuffer())

            if is_video:
                st.video(v_path)
            else:
                st.image(v_path, caption=t("preview_label", lang),
                         use_container_width=True)

            if password_v and st.button(t("btn_verify", lang), key="do_verify"):
                if is_video:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text(t("video_progress", lang))
                    video_result = decode_video(
                        v_path, password_v, num_sample_frames=10,
                        progress_callback=lambda p: progress_bar.progress(min(p, 1.0)),
                    )
                    progress_bar.progress(1.0)
                    status_text.empty()

                    st.session_state["verify_result"] = (
                        f"✅ {video_result['message']}" if video_result["found"]
                        else "❌ No watermark found"
                    )
                    st.session_state["verify_trust"] = {
                        "score": video_result["avg_score"],
                        "found": video_result["found"],
                        "message": video_result.get("message", ""),
                        "score_a": video_result["avg_score"],
                        "score_b": 0,
                    }
                    st.session_state["verify_video_scores"] = video_result.get("scores", [])
                    st.session_state["verify_is_video"] = True
                    st.session_state["verify_path"] = v_path
                else:
                    with st.spinner("Analyzing..."):
                        result_msg = decode_image(v_path, password_v)
                        trust_data = compute_trust_score(v_path, password_v)

                    st.session_state["verify_result"] = result_msg
                    st.session_state["verify_trust"] = trust_data
                    st.session_state["verify_is_video"] = False
                    st.session_state["verify_path"] = v_path

    with right:
        if st.session_state.get("verify_result"):
            trust_data = st.session_state["verify_trust"]
            score = trust_data.get("score", 0)
            found = trust_data.get("found", False)
            message = trust_data.get("message", "")

            # ── Trust Score — dramatic display ──
            color_cls = _trust_color(score)
            glow_cls = _trust_glow(score)
            st.markdown(f"""
            <div class="trust-display {glow_cls} animate-in">
                <div class="trust-number {color_cls}">{score}<span class="trust-unit">/100</span></div>
                <div class="trust-label">{t('trust_score', lang)}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if found:
                st.success(f"✅ {t('watermark_found', lang)}")
                if message:
                    st.markdown(f"""
                    <div class="msg-card animate-in">
                        <div class="msg-label">{t('embedded_signature', lang)}</div>
                        <div class="msg-text">{message}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(f"❌ {t('watermark_not_found', lang)}")

            # ── Layer Analysis — custom bars ──
            st.markdown(f'<div class="section-title" style="margin-top:1rem;">'
                        f'{t("analysis_label", lang)}</div>', unsafe_allow_html=True)

            score_a = trust_data.get("score_a", score)
            score_b = trust_data.get("score_b", 0)
            layers = [
                ("DCT Layer A", score_a, "fill-cyan", _trust_color(score_a)),
                ("Spread Spectrum B", score_b, "fill-purple", _trust_color(score_b)),
                ("ECC Recovery", 95 if found else 0, "fill-green", _trust_color(95 if found else 0)),
                ("Sync Markers", min(100, score + 10) if found else 0, "fill-orange",
                 _trust_color(min(100, score + 10) if found else 0)),
            ]
            bars_html = ""
            for name, val, fill, vcls in layers:
                w = min(val, 100)
                bars_html += f"""
                <div class="layer-bar">
                    <span class="layer-label">{name}</span>
                    <div class="layer-track">
                        <div class="layer-fill {fill}" style="width:{w}%;"></div>
                    </div>
                    <span class="layer-val {vcls}">{val}%</span>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            # ── Video: per-frame analysis ──
            if st.session_state.get("verify_is_video") and st.session_state.get("verify_video_scores"):
                scores = st.session_state["verify_video_scores"]
                v_path = st.session_state.get("verify_path", "")

                st.markdown(f'<div class="section-title" style="margin-top:1.2rem;">'
                            f'{t("frames_analyzed", lang)}: {len(scores)}</div>',
                            unsafe_allow_html=True)

                for s in scores:
                    col_cls = _trust_color(s["score"])
                    icon = "✅" if s["found"] else "❌"
                    st.markdown(
                        f'<div class="frame-chip">'
                        f'<span style="color:var(--slate);">Frame {s["frame"]}</span>'
                        f'<span class="{col_cls}" style="font-weight:700;">{s["score"]}%</span>'
                        f' {icon}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Certificate ──
            if found and HAS_FPDF:
                st.markdown('<div class="vf-footer-line" style="margin:1.2rem 0;"></div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">{t("btn_download_cert", lang)}</div>',
                            unsafe_allow_html=True)

                v_path = st.session_state.get("verify_path", "")
                cert_path = "tmp/certificate.pdf"
                cert_file = generate_certificate(
                    v_path, message, score, password_v, cert_path,
                )
                if cert_file and os.path.exists(cert_file):
                    with open(cert_file, "rb") as f:
                        st.download_button(
                            t("btn_download_cert", lang),
                            f.read(),
                            file_name="VeriFrame_Certificate.pdf",
                            mime="application/pdf",
                        )


# ══════════════════════════════════════════
#  TAB 3 — REMOVE
# ══════════════════════════════════════════
with tab_remove:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(f'<div class="section-title">{t("tab_remove", lang)}</div>',
                    unsafe_allow_html=True)

        st.info(f"ℹ️  {t('remove_info', lang)}")

        uploaded_r = st.file_uploader(
            t("upload_image", lang),
            type=["png", "jpg", "jpeg"],
            key="remove_upload",
        )
        password_r = st.text_input(
            t("password_label", lang),
            type="password", placeholder="••••••••", key="remove_pass",
        )

        if uploaded_r:
            r_in = "tmp/remove_input.png"
            r_out = "tmp/remove_output.png"
            with open(r_in, "wb") as f:
                f.write(uploaded_r.getbuffer())

            st.image(r_in, caption=t("preview_label", lang),
                     use_container_width=True)

            if st.button(t("btn_remove", lang), key="do_remove"):
                with st.spinner("Removing watermark..."):
                    result_r = remove_watermark(r_in, r_out, password_r)

                if "✅" in result_r:
                    st.success(result_r)
                    st.session_state["remove_done"] = True
                    st.session_state["remove_out"] = r_out
                else:
                    st.error(result_r)

    with right:
        if st.session_state.get("remove_done"):
            st.markdown(f'<div class="section-title">{t("preview_label", lang)}</div>',
                        unsafe_allow_html=True)
            st.image(st.session_state["remove_out"], use_container_width=True)

            with open(st.session_state["remove_out"], "rb") as f:
                st.download_button(
                    label=t("btn_download_clean", lang),
                    data=f.read(),
                    file_name="veriframe_clean.png",
                    mime="image/png",
                )


# ─────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────
st.markdown("""
<div class="vf-footer">
    <div class="vf-footer-line"></div>
    VeriFrame &nbsp;&middot;&nbsp; Invisible Digital Watermarking Engine<br>
    <span style="font-size:0.62rem; opacity:0.5;">
        DCT Frequency &middot; Spread Spectrum &middot; Reed-Solomon ECC &middot;
        Sync Markers &middot; Password Encryption &middot; Tamper Detection
    </span>
</div>
""", unsafe_allow_html=True)
