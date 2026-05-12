"""
app/ui.py — reusable UI helpers and CSS loader.
"""
import streamlit as st
from i18n import T

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]        { font-family: 'Inter', sans-serif !important; }
.stApp                            { background-color: #f0f4f8; }
.block-container                  { padding-top: 1.4rem !important;
                                    padding-bottom: 2.5rem !important;
                                    max-width: 1240px; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"]         { background: linear-gradient(170deg, #0f2541 0%, #1e3a5f 60%, #155e75 100%) !important; }
[data-testid="stSidebar"] *       { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label           { font-weight: 600; font-size: 0.78rem;
                                    text-transform: uppercase; letter-spacing: .07em; color: #94a3b8 !important; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label  { font-size: 0.95rem !important;
                                    text-transform: none !important; letter-spacing: normal !important;
                                    font-weight: 500 !important; color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr      { border-color: rgba(255,255,255,.12) !important; }
[data-testid="stSidebar"] .stFileUploader label      { color: #94a3b8 !important;
                                    font-size: 0.78rem !important; text-transform: uppercase;
                                    letter-spacing: .07em; font-weight: 600 !important; }

/* ── KPI card ─────────────────────────────────────────────────────────────── */
.kpi-card   { background: #fff; border-radius: 14px; padding: 20px 22px;
              box-shadow: 0 2px 14px rgba(0,0,0,.07); border-top: 3px solid #14b8a6;
              transition: transform .18s ease, box-shadow .18s ease; }
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,.11); }
.kpi-label  { font-size: .72rem; font-weight: 700; color: #64748b;
              text-transform: uppercase; letter-spacing: .07em; margin-bottom: 5px; }
.kpi-value  { font-size: 1.85rem; font-weight: 800; color: #1e3a5f; line-height: 1.1; }
.kpi-sub    { font-size: .77rem; color: #94a3b8; margin-top: 5px; }
.kpi-icon   { font-size: 1.5rem; float: right; opacity: .25; margin-top: -32px; }

/* ── Section header ───────────────────────────────────────────────────────── */
.sec-head   { font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
              margin: 1.6rem 0 .55rem; padding-bottom: 7px;
              border-bottom: 2px solid #e2e8f0; }

/* ── Health badges ────────────────────────────────────────────────────────── */
.badge      { display:inline-block; padding: 2px 11px; border-radius: 20px;
              font-size: .72rem; font-weight: 700; }
.badge-ok   { background:#dcfce7; color:#166534; }
.badge-warn { background:#fef9c3; color:#854d0e; }
.badge-err  { background:#fee2e2; color:#991b1b; }
.health-row { display:flex; align-items:center; gap:10px; padding: 7px 0;
              border-bottom: 1px solid #f1f5f9; font-size:.85rem; color:#475569; }
.health-row:last-child { border-bottom: none; }
.health-label { min-width: 175px; font-weight: 500; }

/* ── Accuracy badge ───────────────────────────────────────────────────────── */
.acc-box    { background:#fff; border-radius:12px; padding:14px 20px;
              box-shadow:0 2px 12px rgba(0,0,0,.07); text-align:center;
              border-top: 3px solid #14b8a6; }
.acc-val    { font-size:1.55rem; font-weight:800; color:#14b8a6; display:block; }
.acc-lbl    { font-size:.7rem; font-weight:700; color:#94a3b8;
              text-transform:uppercase; letter-spacing:.07em; }

/* ── Summary box ──────────────────────────────────────────────────────────── */
.sum-box    { background: linear-gradient(135deg, #1e3a5f 0%, #0e7490 100%);
              border-radius:16px; padding:26px 32px; color:#fff; text-align:center; margin:1rem 0; }
.sum-big    { font-size:3rem; font-weight:800; display:block; line-height:1.1; }
.sum-sub    { font-size:.92rem; opacity:.82; margin-top:4px; }

/* ── Step cards (About) ───────────────────────────────────────────────────── */
.step-card  { background:#fff; border-radius:14px; padding:26px 22px;
              box-shadow:0 2px 14px rgba(0,0,0,.07); text-align:center; height:100%; }
.step-num   { width:46px; height:46px; border-radius:50%; color:#fff; font-weight:800;
              font-size:1.1rem; display:inline-flex; align-items:center; justify-content:center;
              margin-bottom:14px;
              background: linear-gradient(135deg, #1e3a5f, #14b8a6); }
.step-title { font-size:1.0rem; font-weight:700; color:#1e3a5f; margin-bottom:7px; }
.step-desc  { font-size:.84rem; color:#64748b; line-height:1.55; }

/* ── Hero banner ──────────────────────────────────────────────────────────── */
.hero       { background: linear-gradient(135deg, #0f2541 0%, #14b8a6 100%);
              border-radius:18px; padding:48px 40px; color:#fff; margin-bottom:1.8rem; }
.hero h1    { font-size:2.4rem; font-weight:800; margin:0 0 8px; }
.hero .tag  { font-size:1.05rem; opacity:.85; letter-spacing:.12em; text-transform:uppercase; }
.hero .desc { font-size:1.0rem; opacity:.9; margin-top:16px; max-width:600px; line-height:1.6; }

/* ── Finance progress bar ─────────────────────────────────────────────────── */
.prog-wrap  { background:#e2e8f0; border-radius:20px; height:22px; overflow:hidden;
              margin:6px 0 2px; }
.prog-bar   { height:100%; border-radius:20px; transition:width .4s ease;
              background: linear-gradient(90deg,#14b8a6,#0e7490); }
.prog-bar-warn { background: linear-gradient(90deg,#f59e0b,#d97706); }
.prog-bar-ok   { background: linear-gradient(90deg,#22c55e,#16a34a); }
.prog-label { font-size:.72rem; color:#64748b; font-weight:600; }

/* ── Finance card (green accent) ─────────────────────────────────────────── */
.fin-card   { background:#fff; border-radius:14px; padding:18px 20px;
              box-shadow:0 2px 14px rgba(0,0,0,.07); border-top:3px solid #22c55e; }
.fin-label  { font-size:.72rem; font-weight:700; color:#64748b;
              text-transform:uppercase; letter-spacing:.07em; margin-bottom:4px; }
.fin-value  { font-size:1.7rem; font-weight:800; color:#166534; line-height:1.1; }
.fin-sub    { font-size:.75rem; color:#94a3b8; margin-top:4px; }

/* ── Info pill ────────────────────────────────────────────────────────────── */
.info-pill  { display:inline-block; background:#e0f2fe; color:#0369a1;
              border-radius:20px; padding:3px 13px; font-size:.78rem; font-weight:600;
              margin:2px; }

/* ── RTL / Arabic support ─────────────────────────────────────────────────── */
.rtl        { direction:rtl; text-align:right; }
.ar-font    { font-family:'Segoe UI','Arial','Tahoma',sans-serif !important; }
.rtl .kpi-label, .rtl .kpi-value, .rtl .kpi-sub,
.rtl .fin-label, .rtl .fin-value, .rtl .fin-sub { text-align:right; }

/* ── Performance banner ──────────────────────────────────────────────────── */
.perf-banner{ border-radius:18px; padding:28px 36px; margin:0.6rem 0 1.2rem;
              display:flex; align-items:center; gap:24px; }
.perf-icon  { font-size:3.2rem; line-height:1; flex-shrink:0; }
.perf-label { font-size:2rem; font-weight:800; line-height:1.1; }
.perf-sub   { font-size:.92rem; opacity:.82; margin-top:5px; }
.perf-pct   { font-size:1.1rem; font-weight:700; margin-top:8px; }

/* ── Product ranking ─────────────────────────────────────────────────────── */
.rank-row   { display:flex; align-items:center; gap:14px; padding:10px 16px;
              background:#fff; border-radius:12px; margin-bottom:8px;
              box-shadow:0 1px 6px rgba(0,0,0,.06); }
.rank-medal { font-size:1.5rem; width:32px; text-align:center; flex-shrink:0; }
.rank-name  { flex:1; font-weight:600; color:#1e3a5f; font-size:.95rem; }
.rank-qty   { font-weight:800; color:#14b8a6; font-size:1.05rem; min-width:60px; text-align:right; }
.rank-share { font-size:.78rem; color:#94a3b8; min-width:40px; text-align:right; }
.rank-trend { font-size:1.1rem; min-width:28px; text-align:center; }

/* ── Day badge ────────────────────────────────────────────────────────────── */
.day-good   { color:#166534; font-weight:700; }
.day-avg    { color:#854d0e; font-weight:700; }
.day-bad    { color:#991b1b; font-weight:700; }

/* ── Mobile responsive (≤ 768 px) ────────────────────────────────────────── */
@media (max-width: 768px) {
    .block-container    { padding: 0.4rem 0.7rem !important; }
    .kpi-card           { padding: 12px 12px; }
    .kpi-value          { font-size: 1.3rem; }
    .kpi-icon           { display: none; }
    .fin-card           { padding: 12px 12px; }
    .fin-value          { font-size: 1.3rem; }
    .perf-banner        { flex-direction: column; text-align: center; padding: 16px 14px; gap: 10px; }
    .perf-label         { font-size: 1.4rem; }
    .perf-pct           { font-size: .95rem; }
    .sum-box            { padding: 18px 16px; }
    .sum-big            { font-size: 2rem; }
    .sum-sub            { font-size: .78rem; }
    .sec-head           { font-size: .92rem; }
    .rank-row           { padding: 8px 10px; gap: 8px; }
    .rank-qty           { font-size: .9rem; }
    h1                  { font-size: 1.5rem !important; }
    input[type="number"]{ font-size: 1.1rem !important; min-height: 42px; }
}

/* ── Connection status badge ─────────────────────────────────────────────── */
#fc-conn-badge {
    position: fixed; bottom: 14px; right: 14px; z-index: 9999;
    padding: 5px 14px; border-radius: 20px; font-size: 12px;
    font-weight: 700; font-family: sans-serif;
    box-shadow: 0 2px 10px rgba(0,0,0,.18);
    transition: background .4s, color .4s;
    cursor: default; user-select: none;
}

/* ── Saisie rapide ────────────────────────────────────────────────────────── */
.entry-card { background:#fff; border-radius:14px; padding:18px 20px;
              box-shadow:0 2px 12px rgba(0,0,0,.07);
              border-left:4px solid #14b8a6; margin-bottom:10px; }
.entry-prod { font-weight:700; color:#1e3a5f; font-size:.97rem; }
.entry-qty  { color:#14b8a6; font-weight:800; font-size:1.1rem; }
.entry-date { color:#94a3b8; font-size:.78rem; }

/* ── Lite mode banner ────────────────────────────────────────────────────── */
.lite-banner{ background:#fef3c7; border:1px solid #fcd34d; border-radius:10px;
              padding:10px 16px; color:#92400e; font-size:.85rem; font-weight:600;
              margin-bottom:.8rem; }

/* ── Conseils cards ──────────────────────────────────────────────────────── */
.conseil-hero { background:linear-gradient(135deg,#1e3a5f 0%,#0e7490 100%);
               border-radius:18px; padding:28px 32px; color:#fff;
               margin-bottom:1.2rem; }
.conseil-hero-title { font-size:1.8rem; font-weight:800; margin-bottom:4px; }
.conseil-hero-sub   { font-size:.9rem; opacity:.85; }
.conseil-section { background:#fff; border-radius:16px; padding:24px 26px;
                   box-shadow:0 2px 14px rgba(0,0,0,.07);
                   border-left:4px solid #14b8a6; margin-bottom:1.2rem; }
.conseil-section-title { font-size:1.05rem; font-weight:700; color:#1e3a5f;
                          margin-bottom:12px; }
.produit-du-jour { text-align:center; padding:24px 16px; }
.pdj-name   { font-size:2rem; font-weight:800; color:#1e3a5f; margin:10px 0 4px; }
.pdj-score  { font-size:.82rem; color:#94a3b8; font-weight:600;
              text-transform:uppercase; letter-spacing:.05em; }
.pdj-badge  { display:inline-block; background:#dcfce7; color:#166534;
              border-radius:20px; padding:4px 16px; font-size:.85rem;
              font-weight:700; margin-top:8px; }
.alert-row  { display:flex; align-items:center; gap:12px; padding:10px 14px;
              background:#fff7ed; border:1px solid #fed7aa; border-radius:10px;
              margin-bottom:8px; }
.alert-icon { font-size:1.4rem; flex-shrink:0; }
.alert-text { flex:1; }
.alert-prod { font-weight:700; color:#9a3412; font-size:.95rem; }
.alert-detail { font-size:.8rem; color:#c2410c; }
.promo-row  { display:flex; align-items:center; gap:14px; padding:10px 14px;
              background:#fefce8; border:1px solid #fde68a; border-radius:10px;
              margin-bottom:8px; }
.promo-prod { font-weight:700; color:#854d0e; flex:1; font-size:.95rem; }
.promo-price { font-size:1.1rem; font-weight:800; color:#d97706; min-width:90px; text-align:right; }
.promo-pct  { font-size:.78rem; color:#92400e; min-width:55px; text-align:right; }
.prep-row   { display:flex; align-items:center; gap:12px; padding:10px 14px;
              background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px;
              margin-bottom:8px; }
.prep-prod  { font-weight:700; color:#166634; flex:1; font-size:.95rem; }
.prep-qty   { font-size:1.1rem; font-weight:800; color:#16a34a; min-width:70px; text-align:right; }
.prep-ratio { font-size:.8rem; color:#15803d; min-width:65px; text-align:right; }
.season-card { background:#fff; border:1px solid #e2e8f0; border-radius:12px;
               padding:16px 18px; }
.season-card-title { font-weight:700; color:#1e3a5f; margin-bottom:6px; font-size:.95rem; }
.season-idea { display:flex; align-items:baseline; gap:8px; padding:4px 0; }
.season-idea-name { font-weight:600; color:#0e7490; font-size:.9rem; }

/* ── Onboarding ─────────────────────────────────────────────────────────── */
.onboard-box { background:#fff; border-radius:20px; padding:56px 40px;
               box-shadow:0 4px 24px rgba(0,0,0,.08); text-align:center;
               max-width:520px; margin:60px auto; }
.onboard-emoji { font-size:4rem; margin-bottom:16px; display:block; }
.onboard-title { font-size:1.8rem; font-weight:800; color:#1e3a5f; margin-bottom:8px; }
.onboard-sub   { font-size:.95rem; color:#64748b; margin-bottom:28px; line-height:1.55; }

/* ── Hide Streamlit chrome ────────────────────────────────────────────────── */
#MainMenu   { visibility:hidden; }
footer      { visibility:hidden; }
</style>
"""


def load_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _kpi(label: str, value: str, sub: str = "", icon: str = "") -> str:
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    sub_html  = f'<div class="kpi-sub">{sub}</div>'  if sub  else ""
    return (
        f'<div class="kpi-card">{icon_html}'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>{sub_html}</div>'
    )


def _sec(title: str) -> None:
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


def _fin_kpi(label: str, value: str, sub: str = "", icon: str = "") -> str:
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    sub_html  = f'<div class="fin-sub">{sub}</div>'  if sub  else ""
    return (
        f'<div class="fin-card">{icon_html}'
        f'<div class="fin-label">{label}</div>'
        f'<div class="fin-value">{value}</div>{sub_html}</div>'
    )


def _progress_bar(pct: float, label: str) -> str:
    clamped = min(pct, 100)
    cls = "prog-bar-ok" if pct >= 100 else ("prog-bar-warn" if pct >= 70 else "prog-bar")
    return (
        f'<div class="prog-label">{label}</div>'
        f'<div class="prog-wrap"><div class="{cls}" style="width:{clamped:.1f}%"></div></div>'
    )


def _mad(v: float) -> str:
    return f"{v:,.0f} MAD"


def rtl_wrap(html: str, lang: str) -> str:
    if lang == "ar":
        return f'<div class="rtl ar-font">{html}</div>'
    return html
