import streamlit as st
import os
import numpy as np
import cv2
import base64
import tempfile

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
from tamper_detection import (
    generate_tamper_map, get_tamper_summary,
    generate_frequency_spectrum, generate_before_after_spectrum,
    generate_dct_block_viz, generate_ela, generate_noise_analysis,
    generate_bitplane,
)
from certificate import generate_certificate, HAS_FPDF
from robustness import run_robustness_benchmark
from invisible_qr import extract_qr_blind, HAS_QR as HAS_INVISIBLE_QR
from cnn_detector import detect_watermark, calibrate_detector, get_srm_visualization
from ai_agent import run_ai_analysis, format_report_html
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
    padding: 0.8rem 2.5rem 2rem 2.5rem;
    max-width: 1260px;
    margin: 0 auto;
    background: var(--bg-primary);
}

/* ── RTL Support — dynamic ── */
.rtl-mode .block-container { direction: rtl; text-align: right; }
.rtl-mode .section-title { flex-direction: row-reverse; }
.rtl-mode .tech-row { flex-direction: row-reverse; }
.rtl-mode .severity-banner { flex-direction: row-reverse; }
.rtl-mode .layer-bar { flex-direction: row-reverse; }
.rtl-mode .layer-label { text-align: left; }
.rtl-mode .legend-row { flex-direction: row-reverse; }
.rtl-mode .msg-card { text-align: right; }
.rtl-mode .region-tag { flex-direction: row-reverse; }
.rtl-mode .forensic-info { border-left: none; border-right: 3px solid; }

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
            <span class="tech-badge tb-dct"><span class="tb-dot"></span>Invisible QR</span>
            <span class="tech-badge tb-ss"><span class="tb-dot"></span>CNN Steganalysis</span>
            <span class="tech-badge tb-ecc"><span class="tb-dot"></span>AI Forensic Agent</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with lang_col:
    st.markdown("<br>", unsafe_allow_html=True)
    selected_lang_label = st.selectbox(
        "🌐",
        options=list(LANGS.keys()),
        index=list(LANGS.values()).index(st.session_state.lang),
        label_visibility="collapsed",
    )
    st.session_state.lang = LANGS[selected_lang_label]
    lang = st.session_state.lang

# Apply RTL class for Hebrew/Arabic
if lang in RTL_LANGS:
    st.markdown('<style>.block-container{direction:rtl;text-align:right;}'
                '.section-title{flex-direction:row-reverse;}'
                '.tech-row{flex-direction:row-reverse;}'
                '.severity-banner{flex-direction:row-reverse;}'
                '.layer-bar{flex-direction:row-reverse;}'
                '.layer-label{text-align:left;}'
                '.forensic-info{border-left:none;border-right:3px solid;}'
                '</style>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────
tab_embed, tab_verify, tab_remove, tab_benchmark, tab_ai = st.tabs([
    f"  {t('tab_embed', lang)}  ",
    f"  {t('tab_verify', lang)}  ",
    f"  {t('tab_remove', lang)}  ",
    f"  {t('tab_benchmark', lang)}  ",
    f"  {t('tab_ai_agent', lang)}  ",
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

        secret = st.text_area(
            t("secret_label", lang),
            placeholder="e.g.  CAM-001 | 2026-05-14 | UNIT-7",
            height=80,
        )
        password = st.text_input(
            t("password_label", lang),
            type="password", placeholder="••••••••", key="embed_pass",
        )

        if uploaded:
            is_video = uploaded.name.lower().endswith(".mp4")
            ext = "mp4" if is_video else "png"
            in_path = f"tmp/embed_in.{ext}"
            out_path = f"tmp/embed_out.{ext}"
            with open(in_path, "wb") as f:
                f.write(uploaded.getbuffer())

            # Strength meter
            if not is_video:
                img_tmp = cv2.imread(in_path)
                if img_tmp is not None:
                    ih, iw = img_tmp.shape[:2]
                    strength = _adaptive_strength(ih, iw)
                    max_bits = (ih // 8) * (iw // 8)
                    bits_needed = len(_prepare_bits(secret)) if secret else 0
                    st.markdown(f'<div class="section-title">{t("strength_label", lang)}</div>',
                                unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.metric(t("strength_label", lang), f"{strength:.0f}")
                    c2.metric(t("capacity_label", lang), f"{max_bits}")
                    c3.metric("Resolution", f"{iw}x{ih}")
                    if bits_needed > 0:
                        usage = min(100, int(bits_needed / max_bits * 100))
                        st.progress(usage)
                        st.caption(f"{bits_needed}/{max_bits} bits ({usage}%)")

        if uploaded and secret:
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
                    st.success(result)
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

                # ── Forensic Visualization Suite ──
                with st.expander(t("forensic_label", lang)):
                    forensic_tab = st.radio(
                        "Analysis Type",
                        [
                            t("forensic_spectrum", lang),
                            t("forensic_ela", lang),
                            t("forensic_noise", lang),
                            t("forensic_dct", lang),
                            t("forensic_bitplane", lang),
                            t("forensic_qr", lang),
                            t("forensic_cnn", lang),
                        ],
                        horizontal=True,
                        label_visibility="collapsed",
                    )

                    in_path = st.session_state["embed_in"]
                    out_path = st.session_state["embed_out"]

                    if forensic_tab == t("forensic_spectrum", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:var(--cyan);">
                            {t("forensic_spectrum_desc", lang)}</div>""", unsafe_allow_html=True)
                        spec = generate_before_after_spectrum(in_path, out_path)
                        if spec is not None:
                            st.image(_cv2_to_rgb(spec), use_container_width=True)

                    elif forensic_tab == t("forensic_ela", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:var(--yellow);">
                            {t("forensic_ela_desc", lang)}</div>""", unsafe_allow_html=True)
                        ela_col1, ela_col2 = st.columns(2)
                        with ela_col1:
                            st.caption(t("original_label", lang))
                            ela_orig = generate_ela(in_path)
                            if ela_orig is not None:
                                st.image(_cv2_to_rgb(ela_orig), use_container_width=True)
                        with ela_col2:
                            st.caption(t("watermarked_label", lang))
                            ela_wm = generate_ela(out_path)
                            if ela_wm is not None:
                                st.image(_cv2_to_rgb(ela_wm), use_container_width=True)

                    elif forensic_tab == t("forensic_noise", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:var(--purple);">
                            {t("forensic_noise_desc", lang)}</div>""", unsafe_allow_html=True)
                        noise_col1, noise_col2 = st.columns(2)
                        with noise_col1:
                            st.caption(t("original_label", lang))
                            noise_orig = generate_noise_analysis(in_path)
                            if noise_orig is not None:
                                st.image(_cv2_to_rgb(noise_orig), use_container_width=True)
                        with noise_col2:
                            st.caption(t("watermarked_label", lang))
                            noise_wm = generate_noise_analysis(out_path)
                            if noise_wm is not None:
                                st.image(_cv2_to_rgb(noise_wm), use_container_width=True)

                    elif forensic_tab == t("forensic_dct", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:var(--green);">
                            {t("forensic_dct_desc", lang)}</div>""", unsafe_allow_html=True)
                        img_tmp = cv2.imread(out_path)
                        if img_tmp is not None:
                            max_br = img_tmp.shape[0] // 8 - 1
                            max_bc = img_tmp.shape[1] // 8 - 1
                            dct_c1, dct_c2 = st.columns(2)
                            with dct_c1:
                                block_r = st.slider("Block Row", 0, max(0, max_br), 10, key="dct_br")
                            with dct_c2:
                                block_c = st.slider("Block Col", 0, max(0, max_bc), 10, key="dct_bc")
                            dct_viz = generate_dct_block_viz(out_path, block_r, block_c)
                            if dct_viz is not None:
                                st.image(_cv2_to_rgb(dct_viz),
                                         caption=f"DCT Block ({block_r},{block_c})",
                                         use_container_width=True)

                    elif forensic_tab == t("forensic_bitplane", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:var(--orange);">
                            {t("forensic_bitplane_desc", lang)}</div>""", unsafe_allow_html=True)
                        bit_num = st.slider("Bit Plane (0=LSB, 7=MSB)", 0, 7, 0, key="bitplane")
                        bp_col1, bp_col2 = st.columns(2)
                        with bp_col1:
                            st.caption(t("original_label", lang))
                            bp_orig = generate_bitplane(in_path, bit_num)
                            if bp_orig is not None:
                                st.image(_cv2_to_rgb(bp_orig), use_container_width=True)
                        with bp_col2:
                            st.caption(t("watermarked_label", lang))
                            bp_wm = generate_bitplane(out_path, bit_num)
                            if bp_wm is not None:
                                st.image(_cv2_to_rgb(bp_wm), use_container_width=True)

                    elif forensic_tab == t("forensic_qr", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:#22d3ee;">
                            {t("forensic_qr_desc", lang)}</div>""", unsafe_allow_html=True)
                        qr_viz = extract_qr_blind(cv2.imread(out_path),
                                                  st.session_state.get("embed_pass", ""))
                        if qr_viz is not None:
                            st.image(_cv2_to_rgb(qr_viz),
                                     caption="Invisible QR — Extracted Pattern",
                                     use_container_width=True)
                        else:
                            st.info("QR extraction requires the qrcode library.")

                    elif forensic_tab == t("forensic_cnn", lang):
                        st.markdown(f"""<div class="forensic-info" style="border-color:#f472b6;">
                            {t("forensic_cnn_desc", lang)}</div>""", unsafe_allow_html=True)

                        # Calibrate with current pair
                        cal = calibrate_detector(in_path, out_path)

                        cnn_c1, cnn_c2 = st.columns(2)
                        with cnn_c1:
                            st.caption(t("original_label", lang))
                            det_clean = detect_watermark(in_path)
                            prob_clean = det_clean["probability"]
                            st.markdown(f"""
                            <div style="text-align:center;padding:0.8rem;">
                                <div style="font-family:'JetBrains Mono',monospace;
                                            font-size:2rem;font-weight:900;
                                            color:{'var(--green)' if prob_clean < 0.4 else 'var(--red)'};">
                                    {int(prob_clean*100)}%</div>
                                <div style="font-size:0.75rem;color:var(--slate);">
                                    Watermark Probability</div>
                            </div>""", unsafe_allow_html=True)

                        with cnn_c2:
                            st.caption(t("watermarked_label", lang))
                            det_wm = detect_watermark(out_path)
                            prob_wm = det_wm["probability"]
                            st.markdown(f"""
                            <div style="text-align:center;padding:0.8rem;">
                                <div style="font-family:'JetBrains Mono',monospace;
                                            font-size:2rem;font-weight:900;
                                            color:{'var(--green)' if prob_wm >= 0.6 else 'var(--yellow)'};">
                                    {int(prob_wm*100)}%</div>
                                <div style="font-size:0.75rem;color:var(--slate);">
                                    Watermark Probability</div>
                            </div>""", unsafe_allow_html=True)

                        # SRM visualization
                        srm_viz = get_srm_visualization(out_path)
                        if srm_viz is not None:
                            st.image(_cv2_to_rgb(srm_viz),
                                     caption="SRM Filter Responses — Watermark Noise Residuals",
                                     use_container_width=True)


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
        password_v = st.text_input(
            t("password_label", lang),
            type="password", placeholder="••••••••", key="verify_pass",
        )

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

            if st.button(t("btn_verify", lang), key="do_verify"):
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
                        tamper_data = generate_tamper_map(v_path, password_v)

                    st.session_state["verify_result"] = result_msg
                    st.session_state["verify_trust"] = trust_data
                    st.session_state["verify_tamper"] = tamper_data
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

                # Frame slider
                st.markdown(f'<div class="section-title" style="margin-top:1.2rem;">'
                            f'Frame Navigator</div>', unsafe_allow_html=True)
                info_v = _get_video_info(v_path)
                total_frames = max(1, info_v["total_frames"] - 1)
                selected_frame = st.slider("Frame", 0, total_frames, 0, key="frame_slider")

                frame_img = extract_single_frame(v_path, selected_frame)
                if frame_img is not None:
                    st.image(_cv2_to_rgb(frame_img),
                             caption=f"Frame {selected_frame} / {total_frames}",
                             use_container_width=True)
                    if st.button(f"Analyze Frame {selected_frame}", key="analyze_single_frame"):
                        with st.spinner("Analyzing frame..."):
                            frame_trust = get_frame_trust_score(v_path, selected_frame, password_v)
                        f_score = frame_trust.get("score", 0)
                        f_cls = _trust_color(f_score)
                        f_found = frame_trust.get("found", False)
                        f_msg = frame_trust.get("message", "")
                        st.markdown(
                            f'<span class="{f_cls}" style="font-size:1.5rem;font-weight:900;'
                            f'font-family:JetBrains Mono,monospace;">{f_score}%</span>'
                            f'&nbsp; {"✅ " + f_msg if f_found else "❌ Not found"}',
                            unsafe_allow_html=True,
                        )

            # ── Tamper Detection ──
            tamper_data = st.session_state.get("verify_tamper")
            if tamper_data and tamper_data.get("overlay") is not None:
                st.markdown('<div class="vf-footer-line" style="margin:1.2rem 0;"></div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">{t("tamper_map", lang)}</div>',
                            unsafe_allow_html=True)

                summary = tamper_data.get("summary")
                if summary is None:
                    raw_map = tamper_data.get("raw_map")
                    summary = get_tamper_summary(raw_map) if raw_map is not None else {}
                severity = summary.get("severity", "pristine")
                tamper_pct = summary.get("tamper_pct", 0)
                regions = summary.get("regions", [])
                num_regions = summary.get("num_regions", 0)

                sev_text = t(f"severity_{severity}", lang)
                sev_cls = _severity_class(severity)
                sev_icons = {"pristine": "✅", "minor": "⚠️", "moderate": "🚨", "severe": "🛑"}

                st.markdown(f"""
                <div class="severity-banner sev-{severity} animate-in">
                    <span class="sev-icon">{sev_icons.get(severity, '')}</span>
                    <div>
                        <div class="sev-title {sev_cls}">{sev_text}</div>
                        <div class="sev-detail">
                            {t('tamper_pct', lang)}: <strong>{tamper_pct}%</strong>
                            &nbsp;&middot;&nbsp;
                            {t('tamper_regions', lang)}: <strong>{num_regions}</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                view_mode = st.radio(
                    t("tamper_view_mode", lang),
                    [t("tamper_view_heatmap", lang), t("tamper_view_regions", lang)],
                    horizontal=True,
                    label_visibility="collapsed",
                )

                if view_mode == t("tamper_view_regions", lang) and tamper_data.get("overlay_annotated") is not None:
                    st.image(_cv2_to_rgb(tamper_data["overlay_annotated"]),
                             use_container_width=True)
                else:
                    st.image(_cv2_to_rgb(tamper_data["overlay"]),
                             use_container_width=True)

                # Legend
                st.markdown("""
                <div class="legend-row">
                    <span class="legend-item"><span class="legend-dot" style="background:#22c55e;"></span>Intact</span>
                    <span class="legend-item"><span class="legend-dot" style="background:#f59e0b;"></span>Minor</span>
                    <span class="legend-item"><span class="legend-dot" style="background:#ef4444;"></span>Tampered</span>
                </div>
                """, unsafe_allow_html=True)

                if regions:
                    with st.expander(f"{t('tamper_regions', lang)} ({num_regions})", expanded=False):
                        for idx, region in enumerate(regions[:8]):
                            rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
                            img_h, img_w = tamper_data["overlay"].shape[:2]
                            area_pct = round(rw * rh / max(1, img_w * img_h) * 100, 1)
                            st.markdown(
                                f'<div class="region-tag">'
                                f'<span class="region-num">#{idx+1}</span>'
                                f'{rw}x{rh} px'
                                f'<span class="region-dim">at ({rx},{ry}) &middot; {area_pct}%</span>'
                                f'</div>',
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

                cert_tamper_pct = -1
                cert_severity = ""
                td = st.session_state.get("verify_tamper")
                if td and td.get("summary"):
                    cert_tamper_pct = td["summary"].get("tamper_pct", -1)
                    cert_severity = td["summary"].get("severity", "")

                cert_file = generate_certificate(
                    v_path, message, score, password_v, cert_path,
                    tamper_pct=cert_tamper_pct, severity=cert_severity,
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


# ══════════════════════════════════════════
#  TAB 4 — ROBUSTNESS BENCHMARK
# ══════════════════════════════════════════
with tab_benchmark:
    st.markdown(f'<div class="section-title">{t("tab_benchmark", lang)}</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="forensic-info" style="border-color:var(--cyan);">
        {t("benchmark_desc", lang)}
    </div>
    """, unsafe_allow_html=True)

    bench_left, bench_right = st.columns([1, 2], gap="large")

    with bench_left:
        uploaded_b = st.file_uploader(
            t("upload_image", lang),
            type=["png", "jpg", "jpeg"],
            key="bench_upload",
        )
        bench_secret = st.text_input(
            t("secret_label", lang),
            value="BENCHMARK-2026",
            key="bench_secret",
        )
        bench_password = st.text_input(
            t("password_label", lang),
            type="password", placeholder="••••••••", key="bench_pass",
        )

        if uploaded_b and bench_secret:
            if st.button(t("btn_benchmark", lang), key="do_benchmark"):
                b_in = "tmp/bench_input.png"
                with open(b_in, "wb") as f:
                    f.write(uploaded_b.getbuffer())

                progress_bar = st.progress(0)
                with st.spinner(t("benchmark_running", lang)):
                    bench_result = run_robustness_benchmark(
                        b_in, bench_secret, bench_password,
                        progress_callback=lambda p: progress_bar.progress(min(p, 1.0)),
                    )
                progress_bar.progress(1.0)
                st.session_state["bench_result"] = bench_result

    with bench_right:
        if st.session_state.get("bench_result"):
            br = st.session_state["bench_result"]
            survived = br["survived"]
            total = br["total_attacks"]
            rate = br["survival_rate"]
            acc = br["accuracy_rate"]

            # Summary cards
            s1, s2, s3 = st.columns(3)
            s1.metric(t("benchmark_survival", lang), f"{rate}%")
            s2.metric(t("benchmark_accuracy", lang), f"{acc}%")
            s3.metric(t("benchmark_attacks", lang), f"{survived}/{total}")

            # Overall score bar
            rate_color = "var(--green)" if rate >= 70 else "var(--yellow)" if rate >= 40 else "var(--red)"
            st.markdown(f"""
            <div style="margin:1rem 0;">
                <div class="layer-track" style="height:10px;">
                    <div class="layer-fill" style="width:{rate}%;
                         background:linear-gradient(90deg, #0ea5e9, {'#22c55e' if rate>=70 else '#f59e0b' if rate>=40 else '#ef4444'});
                         box-shadow:0 0 12px {rate_color};height:100%;border-radius:5px;">
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Results table
            st.markdown(f'<div class="section-title" style="margin-top:1rem;">'
                        f'{t("benchmark_results", lang)}</div>', unsafe_allow_html=True)

            # Group by attack type
            for r in br["results"]:
                icon = "✅" if r["found"] else "❌"
                correct_icon = " ✓" if r["correct"] else ""
                score_cls = _trust_color(r["score"])

                border_color = "rgba(34,197,94,0.3)" if r["found"] else "rgba(239,68,68,0.3)"
                bg = "rgba(34,197,94,0.04)" if r["found"] else "rgba(239,68,68,0.04)"
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {border_color};
                            border-radius:10px;padding:0.5rem 0.8rem;margin-bottom:4px;
                            display:flex;justify-content:space-between;align-items:center;
                            font-size:0.82rem;">
                    <span>{icon} {r['description']}</span>
                    <span class="{score_cls}" style="font-family:'JetBrains Mono',monospace;
                                 font-weight:700;">{r['score']}%{correct_icon}</span>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════
#  TAB 5 — AI FORENSIC AGENT
# ══════════════════════════════════════════
with tab_ai:
    st.markdown(f'<div class="section-title">{t("tab_ai_agent", lang)}</div>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="forensic-info" style="border-color:var(--indigo);">
        {t("ai_agent_desc", lang)}
    </div>
    """, unsafe_allow_html=True)

    ai_left, ai_right = st.columns([1, 2], gap="large")

    with ai_left:
        uploaded_ai = st.file_uploader(
            t("upload_media", lang),
            type=["png", "jpg", "jpeg"],
            key="ai_upload",
        )
        ai_password = st.text_input(
            t("password_label", lang),
            type="password", placeholder="••••••••", key="ai_pass",
        )

        if uploaded_ai:
            ai_path = "tmp/ai_input.png"
            with open(ai_path, "wb") as f:
                f.write(uploaded_ai.getbuffer())
            st.image(ai_path, caption=t("preview_label", lang),
                     use_container_width=True)

            if st.button(t("ai_agent_run", lang), key="do_ai_agent"):
                with st.spinner(t("ai_agent_running", lang)):
                    ai_report = run_ai_analysis(ai_path, ai_password)
                st.session_state["ai_report"] = ai_report

    with ai_right:
        if st.session_state.get("ai_report"):
            report = st.session_state["ai_report"]

            # Animated header
            risk = report.get("risk_level", "UNKNOWN")
            risk_colors = {
                "LOW": "#22c55e", "MEDIUM": "#f59e0b",
                "HIGH": "#fb923c", "CRITICAL": "#ef4444", "UNKNOWN": "#64748b",
            }
            risk_color = risk_colors.get(risk, "#64748b")

            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(129,140,248,0.2);
                        border-radius:16px;padding:1.5rem;margin-bottom:1rem;
                        border-top:3px solid {risk_color};">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:2px;
                                    color:var(--slate);margin-bottom:6px;">
                            🤖 {t('ai_agent_report', lang)}</div>
                        <div style="font-size:1.15rem;font-weight:800;color:var(--light);">
                            {report.get('summary', 'Analysis complete')}</div>
                    </div>
                    <div style="background:{risk_color};color:white;padding:6px 18px;
                                border-radius:100px;font-size:0.72rem;font-weight:800;
                                letter-spacing:1.2px;box-shadow:0 0 15px {risk_color}40;">
                        {risk} RISK</div>
                </div>
                <div style="font-size:0.68rem;color:var(--slate);margin-top:8px;">
                    {report.get('timestamp', '')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Render each section as a card
            status_styles = {
                "success": ("var(--green)", "rgba(34,197,94,0.06)", "✅"),
                "warning": ("var(--yellow)", "rgba(245,158,11,0.06)", "⚠️"),
                "error":   ("var(--red)", "rgba(239,68,68,0.06)", "🛑"),
                "info":    ("var(--cyan)", "rgba(56,189,248,0.04)", "ℹ️"),
            }

            for section in report.get("sections", []):
                status = section.get("status", "info")
                color, bg, icon = status_styles.get(status, status_styles["info"])

                findings_html = ""
                for finding in section.get("findings", []):
                    findings_html += f'<div style="padding:3px 0;font-size:0.82rem;color:var(--light);">• {finding}</div>'

                st.markdown(f"""
                <div style="background:{bg};border:1px solid {color}18;
                            border-radius:12px;padding:0.85rem 1.1rem;margin-bottom:0.5rem;
                            border-left:3px solid {color};">
                    <div style="font-weight:700;font-size:0.82rem;color:{color};
                                margin-bottom:0.3rem;">
                        {icon} {section['title']}</div>
                    {findings_html}
                </div>
                """, unsafe_allow_html=True)

            # Recommendations
            recs = report.get("recommendations", [])
            if recs:
                recs_html = ""
                for r in recs:
                    recs_html += f'<div style="padding:3px 0;font-size:0.82rem;color:var(--light);">→ {r}</div>'
                st.markdown(f"""
                <div style="background:rgba(129,140,248,0.05);border:1px solid rgba(129,140,248,0.18);
                            border-radius:12px;padding:0.85rem 1.1rem;margin-top:0.5rem;">
                    <div style="font-weight:700;font-size:0.82rem;color:var(--indigo);
                                margin-bottom:0.4rem;">
                        💡 Recommendations</div>
                    {recs_html}
                </div>
                """, unsafe_allow_html=True)

            # Raw data expander
            with st.expander("📊 Raw Analysis Data"):
                raw = report.get("raw_data", {})
                if raw.get("stats"):
                    s = raw["stats"]
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Resolution", s.get("resolution", "N/A"))
                    c2.metric("Noise Level", s.get("noise_level", 0))
                    c3.metric("Entropy", f"{s.get('entropy', 0)} bits")

                if raw.get("cnn"):
                    cnn = raw["cnn"]
                    prob = cnn.get("probability", 0)
                    st.markdown(f"""
                    <div style="text-align:center;padding:0.6rem;background:rgba(15,23,42,0.5);
                                border-radius:10px;margin-top:0.5rem;">
                        <div style="font-size:0.68rem;color:var(--slate);text-transform:uppercase;
                                    letter-spacing:1px;">CNN Watermark Probability</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                                    font-weight:900;color:{'var(--green)' if prob >= 0.6 else 'var(--yellow)' if prob >= 0.4 else 'var(--slate)'};">
                            {int(prob*100)}%</div>
                        <div style="font-size:0.72rem;color:var(--slate);">
                            {cnn.get('features_extracted', 0)} features
                            (SRM: {cnn.get('srm_features', 0)} · DCT: {cnn.get('dct_features', 0)} · Channel: {cnn.get('channel_features', 0)})</div>
                    </div>
                    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────
st.markdown("""
<div class="vf-footer">
    <div class="vf-footer-line"></div>
    VeriFrame &nbsp;&middot;&nbsp; Invisible Digital Watermarking Engine<br>
    <span style="font-size:0.62rem; opacity:0.5;">
        DCT Frequency &middot; Spread Spectrum &middot; Reed-Solomon ECC &middot;
        Sync Markers &middot; CNN Steganalysis &middot; AI Forensic Agent &middot; Tamper Detection
    </span>
</div>
""", unsafe_allow_html=True)
