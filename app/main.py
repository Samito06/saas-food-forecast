"""
app/main.py — FoodCast | Production-ready SaaS dashboard
Predict · Decide · Grow
Run: streamlit run app/main.py
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pandas.util import hash_pandas_object
from prophet import Prophet

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).parent.parent
DEFAULT_CSV  = ROOT / "data" / "ventes.csv"
REQUIRED_COLS = {"date", "produit", "quantite_vendue"}
WEEKEND_DAYS  = {"Friday", "Saturday"}
BUFFER_RATE   = 1.15
PEAK_THRESH   = 1.30

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="FoodCast — Sales Intelligence",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Injection ─────────────────────────────────────────────────────────────
st.markdown("""
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

/* ── Step cards (About page) ──────────────────────────────────────────────── */
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

/* ── Info pill ────────────────────────────────────────────────────────────── */
.info-pill  { display:inline-block; background:#e0f2fe; color:#0369a1;
              border-radius:20px; padding:3px 13px; font-size:.78rem; font-weight:600;
              margin:2px; }

/* ── Hide Streamlit chrome ────────────────────────────────────────────────── */
#MainMenu   { visibility:hidden; }
footer      { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═════════════════════════════════════════════════════════════════════════════

def _kpi(label: str, value: str, sub: str = "", icon: str = "") -> str:
    icon_html = f'<div class="kpi-icon">{icon}</div>' if icon else ""
    sub_html  = f'<div class="kpi-sub">{sub}</div>'  if sub  else ""
    return (f'<div class="kpi-card">{icon_html}'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>{sub_html}</div>')


def _sec(title: str) -> None:
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_validate(file_bytes: bytes | None) -> tuple[pd.DataFrame | None, list[str]]:
    """Load CSV, run validation checks. Returns (df_or_None, error_list)."""
    errors: list[str] = []
    try:
        src = io.BytesIO(file_bytes) if file_bytes is not None else DEFAULT_CSV
        df  = pd.read_csv(src, parse_dates=["date"])

        missing_cols = REQUIRED_COLS - set(df.columns)
        if missing_cols:
            errors.append(f"Missing columns: **{', '.join(sorted(missing_cols))}**. "
                          f"Expected: `date`, `produit`, `quantite_vendue`.")
            return None, errors

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            errors.append("Column `date` could not be parsed. Expected format: `YYYY-MM-DD`.")
            return None, errors

        if not pd.api.types.is_numeric_dtype(df["quantite_vendue"]):
            errors.append("Column `quantite_vendue` must be numeric.")
            return None, errors

        n_days = df["date"].nunique()
        if n_days < 30:
            errors.append(f"Dataset too small: **{n_days} unique dates**. "
                          "At least 30 days of data are required.")
            return None, errors

        return df, errors

    except Exception as exc:
        errors.append(f"Could not read file: `{exc}`")
        return None, errors


@st.cache_data(show_spinner=False)
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("date")["quantite_vendue"]
        .sum().reset_index()
        .rename(columns={"date": "ds", "quantite_vendue": "y"})
        .sort_values("ds").reset_index(drop=True)
    )


@st.cache_resource(show_spinner=False)
def get_model(data_hash: str, _daily: pd.DataFrame) -> Prophet:
    """Train Prophet on the full daily series. Cached per dataset."""
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
    )
    m.fit(_daily)
    return m


@st.cache_data(show_spinner=False)
def run_forecast(data_hash: str, horizon: int, _model: Prophet,
                 _daily: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions from history start through history end + horizon days."""
    last_date = _daily["ds"].max()
    future = pd.DataFrame({
        "ds": pd.date_range(_daily["ds"].min(),
                            last_date + pd.Timedelta(days=horizon), freq="D")
    })
    pred = _model.predict(future)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(show_spinner=False)
def compute_mae(data_hash: str, _daily: pd.DataFrame) -> float | None:
    """Holdout MAE: train on [all - last 30], evaluate on last 30 days."""
    if len(_daily) < 61:
        return None
    cutoff  = len(_daily) - 30
    train   = _daily.iloc[:cutoff].copy()
    test    = _daily.iloc[cutoff:].copy()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                daily_seasonality=False, seasonality_mode="multiplicative")
    m.fit(train)
    pred = m.predict(test[["ds"]])
    return round(float(np.mean(np.abs(pred["yhat"].values - test["y"].values))), 1)


def check_data_health(df: pd.DataFrame) -> list[dict]:
    checks = []

    nulls = int(df.isnull().sum().sum())
    checks.append({"label": "Missing values",
                   "status": "ok" if nulls == 0 else "warn",
                   "detail": "None detected" if nulls == 0 else f"{nulls} null values"})

    neg = int((df["quantite_vendue"] < 0).sum())
    checks.append({"label": "Negative quantities",
                   "status": "ok" if neg == 0 else "warn",
                   "detail": "None" if neg == 0 else f"{neg} rows with qty < 0"})

    expected_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    gaps = len(expected_days) - df["date"].dt.date.nunique()
    checks.append({"label": "Date continuity",
                   "status": "ok" if gaps == 0 else "warn",
                   "detail": "No gaps" if gaps == 0 else f"{gaps} missing dates"})

    n_products = df["produit"].nunique()
    checks.append({"label": "Products detected",
                   "status": "ok",
                   "detail": f"{n_products} unique product(s): " +
                              ", ".join(sorted(df["produit"].unique()))})

    zero_days = int((df.groupby("date")["quantite_vendue"].sum() == 0).sum())
    checks.append({"label": "Zero-sales days (closed)",
                   "status": "ok",
                   "detail": f"{zero_days} day(s) with total sales = 0"})

    return checks


def make_sample_csv() -> bytes:
    rows = []
    products = ["Pizza Margherita", "Chicken Sandwich", "Burger", "Caesar Salad"]
    qtys     = [28, 22, 17, 13]
    for i in range(5):
        date_str = f"2024-01-0{i+1}"
        for p, q in zip(products, qtys):
            rows.append({"date": date_str, "produit": p, "quantite_vendue": q + i})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _risk_level(row: pd.Series, avg: float) -> str:
    """
    Risk is primarily demand-driven (ratio to average) so it stays
    meaningful regardless of CI width, which varies with forecast horizon.
    """
    ratio = row["yhat"] / max(avg, 1)
    if ratio > PEAK_THRESH:           # >130 % of average
        return "High"
    if ratio > 1.10 or row["is_weekend"]:   # >110 % or weekend
        return "Medium"
    return "Low"


# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 4px">
      <div style="font-size:1.6rem;font-weight:800;color:#fff;letter-spacing:-.5px;">
        🍕 FoodCast
      </div>
      <div style="font-size:.72rem;color:#14b8a6;font-weight:700;letter-spacing:.18em;
                  text-transform:uppercase;margin-top:2px;">
        Predict &middot; Decide &middot; Grow
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "Forecast", "Recommendations", "About"],
    )
    st.divider()

    uploaded = st.file_uploader(
        "DATA SOURCE",
        type="csv",
        help="Required columns: `date`, `produit`, `quantite_vendue`",
    )
    if uploaded is None:
        st.markdown(
            '<p style="font-size:.78rem;color:#64748b;margin-top:6px;">'
            'Using built-in demo dataset.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<p style="font-size:.78rem;color:#34d399;margin-top:6px;">'
            f'Loaded: <b>{uploaded.name}</b></p>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        '<p style="font-size:.72rem;color:#64748b;">v1.0 &nbsp;|&nbsp; '
        'Built with Streamlit + Prophet</p>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Data loading & validation gate
# ═════════════════════════════════════════════════════════════════════════════
file_bytes = uploaded.read() if uploaded is not None else None
df, load_errors = load_and_validate(file_bytes)

if df is None:
    st.error("### Could not load data")
    for e in load_errors:
        st.markdown(f"- {e}")
    st.info(
        "**Expected CSV format:**\n\n"
        "| date | produit | quantite_vendue |\n"
        "|------|---------|----------------|\n"
        "| 2024-01-01 | Pizza Margherita | 28 |\n\n"
        "Download a sample file from the **About** page."
    )
    st.stop()

# Shared derived data used by all pages
daily     = aggregate_daily(df)
data_hash = str(hash_pandas_object(daily).sum())
avg_open  = float(daily[daily["y"] > 0]["y"].mean())
last_date = daily["ds"].max()
fc_start  = last_date + pd.Timedelta(days=1)


# ═════════════════════════════════════════════════════════════════════════════
# Page 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">Sales Overview</h1>',
        unsafe_allow_html=True,
    )
    src_label = f"<b>{uploaded.name}</b>" if uploaded else "demo dataset"
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">'
        f'Data source: {src_label} &nbsp;|&nbsp; '
        f'{len(df):,} rows &nbsp;|&nbsp; '
        f'{daily["ds"].min().date()} &rarr; {daily["ds"].max().date()}'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── 4 KPI cards ──────────────────────────────────────────────────────────
    total_sales  = int(df["quantite_vendue"].sum())
    best_product = df.groupby("produit")["quantite_vendue"].sum().idxmax()
    avg_daily    = round(float(daily[daily["y"] > 0]["y"].mean()), 1)

    df_open = df[df["quantite_vendue"] > 0].copy()
    df_open["day_name"] = df_open["date"].dt.day_name()
    busiest_day = df_open.groupby("day_name")["quantite_vendue"].mean().idxmax()

    c1, c2, c3, c4 = st.columns(4)
    for col, html in zip(
        [c1, c2, c3, c4],
        [
            _kpi("Total Units Sold",    f"{total_sales:,}",  f"{daily['ds'].min().year}–{daily['ds'].max().year}", "📦"),
            _kpi("Best-Selling Product", best_product,        "by total volume", "🏆"),
            _kpi("Avg Daily Sales",      f"{avg_daily}",      "units/day (open days)", "📈"),
            _kpi("Busiest Day",          busiest_day,         "highest avg sales", "📅"),
        ],
    ):
        col.markdown(html, unsafe_allow_html=True)

    # ── Sales trend with date range selector ─────────────────────────────────
    _sec("Sales Trend")
    min_d, max_d = daily["ds"].min().date(), daily["ds"].max().date()
    date_range = st.date_input(
        "Filter date range",
        value=(min_d, max_d),
        min_value=min_d,
        max_value=max_d,
        label_visibility="collapsed",
    )
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_d, end_d = date_range
        mask = (daily["ds"].dt.date >= start_d) & (daily["ds"].dt.date <= end_d)
        filtered = daily[mask]
    else:
        filtered = daily

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=filtered["ds"], y=filtered["y"],
        mode="lines", fill="tozeroy",
        fillcolor="rgba(20,184,166,.10)",
        line=dict(color="#14b8a6", width=1.8),
        name="Daily sales",
    ))
    fig_trend.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis_title="", yaxis_title="Units Sold",
        margin=dict(t=20, b=30),
        height=300,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Product breakdown: pie + bar ─────────────────────────────────────────
    _sec("Product Breakdown")
    by_product = (
        df.groupby("produit")["quantite_vendue"].sum()
        .reset_index().sort_values("quantite_vendue", ascending=False)
        .rename(columns={"produit": "Product", "quantite_vendue": "Total"})
    )
    PALETTE = ["#14b8a6", "#1e3a5f", "#0ea5e9", "#6366f1"]

    col_pie, col_bar = st.columns(2)
    with col_pie:
        fig_pie = px.pie(
            by_product, names="Product", values="Total",
            color_discrete_sequence=PALETTE,
            hole=0.42,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            by_product, x="Product", y="Total",
            color="Product", color_discrete_sequence=PALETTE,
            text_auto=True,
        )
        fig_bar.update_layout(
            showlegend=False, template="plotly_white",
            margin=dict(t=20, b=20), height=320,
            xaxis_title="", yaxis_title="Total Units",
        )
        fig_bar.update_traces(textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Data health ──────────────────────────────────────────────────────────
    _sec("Data Health")
    checks = check_data_health(df)
    badge_map = {"ok": "badge-ok", "warn": "badge-warn", "err": "badge-err"}
    label_map = {"ok": "OK", "warn": "Warning", "err": "Error"}
    rows_html = ""
    for c in checks:
        cls = badge_map[c["status"]]
        lbl = label_map[c["status"]]
        rows_html += (
            f'<div class="health-row">'
            f'<span class="health-label">{c["label"]}</span>'
            f'<span class="badge {cls}">{lbl}</span>'
            f'<span style="color:#94a3b8">{c["detail"]}</span>'
            f'</div>'
        )
    st.markdown(
        f'<div style="background:#fff;border-radius:12px;padding:14px 20px;'
        f'box-shadow:0 2px 12px rgba(0,0,0,.06);">{rows_html}</div>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 2 — Forecast
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Forecast":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">Sales Forecast</h1>',
        unsafe_allow_html=True,
    )

    # ── Controls row ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns([2, 2, 4])
    with ctrl1:
        horizon = st.selectbox(
            "Forecast horizon",
            options=[7, 14, 30],
            index=2,
            format_func=lambda x: f"{x} days",
        )
    with ctrl2:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        with st.spinner("Computing accuracy..."):
            mae = compute_mae(data_hash, daily)
        mae_str = f"{mae}" if mae is not None else "N/A"
        st.markdown(
            f'<div class="acc-box"><span class="acc-val">{mae_str}</span>'
            f'<span class="acc-lbl">MAE (units) &mdash; 30-day holdout</span></div>',
            unsafe_allow_html=True,
        )

    # ── Train & forecast ──────────────────────────────────────────────────────
    with st.spinner("Training Prophet model..."):
        model = get_model(data_hash, daily)
        pred  = run_forecast(data_hash, horizon, model, daily)

    pred_future = pred[pred["ds"] >= fc_start].copy()

    # ── Chart ─────────────────────────────────────────────────────────────────
    _sec(f"Historical Sales + {horizon}-Day Forecast")
    fig = go.Figure()

    # CI ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([pred["ds"], pred["ds"][::-1]]),
        y=pd.concat([pred["yhat_upper"], pred["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(20,184,166,.13)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip", name="95% CI",
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=pred["ds"], y=pred["yhat"],
        mode="lines", line=dict(color="#14b8a6", width=2, dash="dot"),
        name="Prophet forecast",
    ))
    # Actuals
    fig.add_trace(go.Scatter(
        x=daily["ds"], y=daily["y"],
        mode="lines", line=dict(color="#1e3a5f", width=1.2),
        name="Actual sales",
    ))
    fig.add_vline(
        x=fc_start.timestamp() * 1000,
        line=dict(color="crimson", dash="dash", width=1.5),
        annotation_text=f"Forecast start ({fc_start.date()})",
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis_title="", yaxis_title="Total Units Sold",
        height=420, margin=dict(t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Weekly summary table ──────────────────────────────────────────────────
    _sec("Weekly Forecast Summary")
    wk = pred_future.copy()
    wk["week_start"] = wk["ds"].dt.to_period("W").apply(lambda p: p.start_time)
    wk["week_end"]   = wk["ds"].dt.to_period("W").apply(lambda p: p.end_time)
    weekly = (
        wk.groupby(["week_start", "week_end"])
        .agg(
            Days       = ("ds",   "count"),
            Total      = ("yhat", lambda x: int(x.clip(lower=0).sum().round())),
            Avg_Daily  = ("yhat", lambda x: round(float(x.clip(lower=0).mean()), 1)),
            CI_Lower   = ("yhat_lower", lambda x: int(x.clip(lower=0).sum().round())),
            CI_Upper   = ("yhat_upper", lambda x: int(x.clip(lower=0).sum().round())),
        )
        .reset_index()
    )
    weekly["Week"] = weekly.apply(
        lambda r: f"{r['week_start'].strftime('%b %d')} – {r['week_end'].strftime('%b %d')}", axis=1
    )
    weekly = weekly[["Week", "Days", "Total", "Avg_Daily", "CI_Lower", "CI_Upper"]]
    weekly.columns = ["Week", "Days", "Total Forecast", "Avg Daily", "CI Lower", "CI Upper"]
    st.dataframe(weekly.reset_index(drop=True), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# Page 3 — Recommendations
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Recommendations":

    st.markdown(
        '<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">Stock Recommendations</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b;font-size:.88rem;margin-bottom:1rem;">'
        'Next 7 days &nbsp;|&nbsp; +15% safety buffer &nbsp;|&nbsp; '
        '95% confidence intervals from Prophet</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("Computing recommendations..."):
        model = get_model(data_hash, daily)
        pred  = run_forecast(data_hash, 30, model, daily)

    next7 = pred[pred["ds"] >= fc_start].head(7).copy()
    next7["day_name"]      = next7["ds"].dt.day_name()
    next7["predicted_qty"] = next7["yhat"].clip(lower=0).round(0).astype(int)
    next7["suggested"]     = (next7["predicted_qty"] * BUFFER_RATE).round(0).astype(int)
    next7["is_weekend"]    = next7["day_name"].isin(WEEKEND_DAYS)
    next7["exceeds_avg"]   = next7["yhat"] > avg_open * PEAK_THRESH
    next7["risk"]          = next7.apply(lambda r: _risk_level(r, avg_open), axis=1)

    # ── Summary box ───────────────────────────────────────────────────────────
    total_stock = int(next7["suggested"].sum())
    total_pred  = int(next7["predicted_qty"].sum())
    st.markdown(
        f'<div class="sum-box">'
        f'<span class="sum-big">{total_stock:,} units</span>'
        f'<span class="sum-sub">Total stock to prepare this week '
        f'(forecast: {total_pred:,} + 15% buffer)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Warnings ──────────────────────────────────────────────────────────────
    high_days = next7[next7["exceeds_avg"]]
    if not high_days.empty:
        for _, row in high_days.iterrows():
            pct = (row["yhat"] / avg_open - 1) * 100
            st.warning(
                f"**{row['ds'].strftime('%A %d %b')}** — forecast of "
                f"**{row['predicted_qty']} units** is **{pct:.0f}%** above "
                "the daily average. Plan extra staffing."
            )
    else:
        st.success("No extraordinary demand expected. Normal operations should suffice.")

    # ── Color-coded table ─────────────────────────────────────────────────────
    _sec("Daily Stock Plan")

    display = next7[[
        "ds", "day_name", "predicted_qty", "suggested",
        "is_weekend", "risk",
        "yhat_lower", "yhat_upper",
    ]].copy()
    display["ds"]        = display["ds"].dt.strftime("%Y-%m-%d")
    display["yhat_lower"] = display["yhat_lower"].clip(lower=0).round(0).astype(int)
    display["yhat_upper"] = display["yhat_upper"].round(0).astype(int)
    display["is_weekend"] = display["is_weekend"].map({True: "Weekend", False: ""})
    display.columns = [
        "Date", "Day", "Forecast (qty)", "Stock to Prepare",
        "Peak Day", "Risk Level", "Lower CI", "Upper CI",
    ]

    RISK_COLOR  = {"High": "#fee2e2", "Medium": "#fef9c3", "Low": "#dcfce7"}
    PEAK_COLOR  = "#fee2e2"
    NORM_COLOR  = "#dcfce7"
    WKND_COLOR  = "#fef9c3"

    def _style_rows(row):
        if row["Risk Level"] == "High":
            bg = PEAK_COLOR
        elif row["Peak Day"] == "Weekend":
            bg = WKND_COLOR
        else:
            bg = NORM_COLOR
        return [f"background-color:{bg}"] * len(row)

    def _style_risk(val):
        colors = {"High": "color:#991b1b;font-weight:700",
                  "Medium": "color:#854d0e;font-weight:700",
                  "Low": "color:#166534;font-weight:700"}
        return colors.get(val, "")

    styled = (
        display.reset_index(drop=True)
        .style
        .apply(_style_rows, axis=1)
        .map(_style_risk, subset=["Risk Level"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown(
        f'<p style="font-size:.78rem;color:#94a3b8;">'
        f'Historical avg (open days): <b>{avg_open:.0f} units/day</b> &nbsp;|&nbsp; '
        f'Buffer: <b>+15%</b> &nbsp;|&nbsp; '
        f'Peak threshold: <b>&gt;{PEAK_THRESH*100:.0f}% of avg</b> &nbsp;|&nbsp; '
        f'Risk = demand level + forecast uncertainty</p>',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Page 4 — About
# ═════════════════════════════════════════════════════════════════════════════
elif page == "About":

    # Hero
    st.markdown("""
    <div class="hero">
      <h1>FoodCast</h1>
      <div class="tag">Predict &nbsp;&middot;&nbsp; Decide &nbsp;&middot;&nbsp; Grow</div>
      <div class="desc">
        AI-powered sales forecasting designed for small food businesses.
        Upload your sales history, get accurate demand forecasts, and make
        smarter stocking decisions &mdash; in minutes, not spreadsheets.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # What & Who
    col_what, col_who = st.columns(2)
    with col_what:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 2px 12px rgba(0,0,0,.06);height:100%;">
          <div style="font-size:1.05rem;font-weight:700;color:#1e3a5f;margin-bottom:10px;">
            What is FoodCast?
          </div>
          <p style="color:#475569;font-size:.88rem;line-height:1.65;">
            FoodCast is a data-driven dashboard that turns raw sales CSV files
            into actionable intelligence. It combines time-series forecasting
            (Facebook Prophet) with an intuitive interface so any restaurant or
            food business owner can forecast demand without any data science
            background.
          </p>
        </div>
        """, unsafe_allow_html=True)

    with col_who:
        st.markdown("""
        <div style="background:#fff;border-radius:14px;padding:24px;
                    box-shadow:0 2px 12px rgba(0,0,0,.06);height:100%;">
          <div style="font-size:1.05rem;font-weight:700;color:#1e3a5f;margin-bottom:10px;">
            Who is it for?
          </div>
          <p style="color:#475569;font-size:.88rem;line-height:1.65;">
            Designed for <b>small restaurant owners</b>, <b>food truck operators</b>,
            <b>catering managers</b>, and any food business that tracks daily product
            sales and wants to reduce waste, avoid stockouts, and plan staffing more
            efficiently.
          </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # How it works
    st.markdown(
        '<div class="sec-head">How it works</div>',
        unsafe_allow_html=True,
    )
    s1, s2, s3 = st.columns(3)
    for col, num, title, desc in [
        (s1, "1", "Upload your data",
         "Drag and drop a CSV file with your daily sales history. "
         "The only columns needed are <code>date</code>, <code>produit</code>, "
         "and <code>quantite_vendue</code>. No account or login required."),
        (s2, "2", "Analyze patterns",
         "FoodCast automatically detects weekly cycles, seasonal peaks, "
         "and product rankings. Explore KPIs, trend charts, and a full "
         "data-quality health check on the Overview page."),
        (s3, "3", "Act on forecasts",
         "The Recommendations page translates AI forecasts into a concrete "
         "daily stock plan with a +15% safety buffer, risk levels, and "
         "alerts for high-demand days &mdash; so you always prepare the right amount."),
    ]:
        col.markdown(
            f'<div class="step-card">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-title">{title}</div>'
            f'<div class="step-desc">{desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # CSV format & download
    st.markdown('<div class="sec-head">Expected CSV Format</div>', unsafe_allow_html=True)
    fmt_col, dl_col = st.columns([3, 1])
    with fmt_col:
        st.markdown("""
        | Column | Type | Example |
        |---|---|---|
        | `date` | Date (YYYY-MM-DD) | `2024-01-15` |
        | `produit` | Text | `Pizza Margherita` |
        | `quantite_vendue` | Integer | `28` |

        One row per product per day. Zero-sales days (closed) are handled automatically.
        """)
        st.markdown(
            '<span class="info-pill">Minimum 30 days</span>'
            '<span class="info-pill">Any number of products</span>'
            '<span class="info-pill">UTF-8 encoding</span>',
            unsafe_allow_html=True,
        )
    with dl_col:
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.download_button(
            label="Download sample CSV",
            data=make_sample_csv(),
            file_name="sample_ventes.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Tech stack
    st.markdown('<div class="sec-head">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        '<span class="info-pill">Python 3.12</span>'
        '<span class="info-pill">Streamlit</span>'
        '<span class="info-pill">Facebook Prophet</span>'
        '<span class="info-pill">Plotly</span>'
        '<span class="info-pill">Pandas</span>'
        '<span class="info-pill">NumPy</span>',
        unsafe_allow_html=True,
    )
