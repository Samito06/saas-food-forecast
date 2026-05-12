"""
pages/forecast.py — Prévisions page: Forecast global · Stock à préparer · Par produit · Liste de courses.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.util import hash_pandas_object

from i18n import T, Tlist
from ui import _kpi, _sec, _mad
from constants import BUFFER_RATE, PEAK_THRESH, WEEKEND_DAYS, UNIT_OPTIONS, RECIPES
from logic.forecasting import (
    get_model, run_forecast, get_product_model, run_product_forecast,
    compute_mae, get_upcoming_peaks, is_ramadan_period, _get_holiday_name,
)
from logic.data_loading import aggregate_by_product
from logic.insights import _risk_level
from logic.inventory import compute_stockout_date


def render(
    df: pd.DataFrame,
    daily: pd.DataFrame,
    data_hash: str,
    avg_open: float,
    last_date: pd.Timestamp,
    fc_start: pd.Timestamp,
    catalog: dict,
    active_prods: list[str],
    active_csv: list[str],
    ramadan_mode: bool,
    lite_mode: bool,
    lang: str,
) -> None:

    tab_global, tab_reco, tab_prod, tab_shop = st.tabs([
        T("tab_forecast_global"),
        T("tab_reco"),
        T("tab_per_product"),
        T("tab_shopping"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Tab A — Prévision globale
    # ══════════════════════════════════════════════════════════════════════════
    with tab_global:
        st.markdown(
            f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1rem;">{T("forecast_sub")}</p>',
            unsafe_allow_html=True,
        )

        if lite_mode:
            st.markdown(f'<div class="lite-banner">{T("forecast_lite_banner")}</div>', unsafe_allow_html=True)
            _sec(T("forecast_lite_section"))
            lite_df = daily.tail(30).copy()
            fig_lite = go.Figure()
            fig_lite.add_trace(go.Scatter(x=lite_df["ds"], y=lite_df["y"], mode="lines+markers",
                                           fill="tozeroy", fillcolor="rgba(20,184,166,.10)",
                                           line=dict(color="#14b8a6", width=1.8)))
            fig_lite.add_hline(y=avg_open, line=dict(color="#94a3b8", dash="dash"))
            fig_lite.update_layout(template="plotly_white", height=280, margin=dict(t=10, b=10),
                                    xaxis_title="", yaxis_title=T("units"))
            st.plotly_chart(fig_lite, use_container_width=True)
            return

        ctrl1, ctrl2, _ = st.columns([2, 2, 4])
        with ctrl1:
            horizon = st.selectbox(
                T("forecast_horizon"), options=[7, 14, 30], index=2,
                format_func=lambda x: T("forecast_horizon_fmt").format(n=x),
            )
        with ctrl2:
            with st.spinner("..."):
                mae = compute_mae(data_hash, daily)
            if mae is None:
                acc_val, acc_sub = T("forecast_acc_na"), T("forecast_acc_nodata")
            elif mae < 15:
                acc_val, acc_sub = T("forecast_acc_good"), T("forecast_acc_sub").format(mae=mae)
            elif mae < 25:
                acc_val, acc_sub = T("forecast_acc_ok"),   T("forecast_acc_sub").format(mae=mae)
            else:
                acc_val, acc_sub = T("forecast_acc_fair"),  T("forecast_acc_sub").format(mae=mae)
            st.markdown(
                f'<div class="acc-box"><span class="acc-val">{acc_val}</span>'
                f'<span class="acc-lbl">{T("forecast_acc_label")} &mdash; {acc_sub}</span></div>',
                unsafe_allow_html=True,
            )

        with st.spinner(T("forecast_spinner")):
            model = get_model(data_hash, daily)
            pred  = run_forecast(data_hash, horizon, model, daily)

        pred_future = pred[pred["ds"] >= fc_start].copy()

        _sec(T("forecast_chart_sec").format(n=horizon))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([pred["ds"], pred["ds"][::-1]]),
            y=pd.concat([pred["yhat_upper"], pred["yhat_lower"][::-1]]),
            fill="toself", fillcolor="rgba(20,184,166,.13)",
            line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name=T("forecast_range"),
        ))
        fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat"], mode="lines",
                                  line=dict(color="#14b8a6", width=2, dash="dot"), name=T("forecast_pred")))
        fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode="lines",
                                  line=dict(color="#1e3a5f", width=1.2), name=T("forecast_history")))
        fig.add_vline(x=fc_start.timestamp() * 1000, line=dict(color="crimson", dash="dash", width=1.5),
                      annotation_text=T("forecast_fc_start").format(d=fc_start.date()),
                      annotation_position="top right")
        fig.update_layout(template="plotly_white", hovermode="x unified",
                           xaxis_title="", yaxis_title=T("units"), height=420, margin=dict(t=30, b=30),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        peaks = get_upcoming_peaks(pred_future, avg_open, n_days=horizon)
        if peaks:
            _sec(T("forecast_peaks"))
            for p in peaks:
                label       = p["date"].strftime("%A %d %b")
                reasons_str = " · ".join(p["reasons"])
                st.warning(f"**{label}** — {p['yhat']} {T('units')} · {reasons_str}")

        if ramadan_mode:
            ram_days = [r for _, r in pred_future.iterrows() if is_ramadan_period(r["ds"])]
            if ram_days:
                _sec(T("forecast_ram_sec"))
                st.info(T("forecast_ram_msg").format(n=len(ram_days)))

        _sec(T("forecast_weekly"))
        wk = pred_future.copy()
        wk["week_start"] = wk["ds"].dt.to_period("W").apply(lambda p: p.start_time)
        wk["week_end"]   = wk["ds"].dt.to_period("W").apply(lambda p: p.end_time)
        weekly = (
            wk.groupby(["week_start", "week_end"])
            .agg(Days=("ds","count"),
                 Total=("yhat", lambda x: int(x.clip(lower=0).sum().round())),
                 Avg_Daily=("yhat", lambda x: round(float(x.clip(lower=0).mean()), 1)),
                 CI_Lower=("yhat_lower", lambda x: int(x.clip(lower=0).sum().round())),
                 CI_Upper=("yhat_upper", lambda x: int(x.clip(lower=0).sum().round())))
            .reset_index()
        )
        weekly["Week"] = weekly.apply(
            lambda r: f"{r['week_start'].strftime('%b %d')} – {r['week_end'].strftime('%b %d')}", axis=1)
        weekly = weekly[["Week","Days","Total","Avg_Daily","CI_Lower","CI_Upper"]]
        weekly.columns = Tlist("forecast_weekly_cols") or weekly.columns.tolist()
        st.dataframe(weekly.reset_index(drop=True), use_container_width=True, hide_index=True)

        _sec(T("forecast_monthly"))
        mo = pred_future.copy()
        mo["month"] = mo["ds"].dt.to_period("M")
        monthly = (
            mo.groupby("month")
            .agg(Total=("yhat", lambda x: int(x.clip(lower=0).sum().round())),
                 Avg_Daily=("yhat", lambda x: round(float(x.clip(lower=0).mean()), 1)),
                 CI_Lower=("yhat_lower", lambda x: int(x.clip(lower=0).sum().round())),
                 CI_Upper=("yhat_upper", lambda x: int(x.clip(lower=0).sum().round())))
            .reset_index()
        )
        monthly["Mois"] = monthly["month"].dt.strftime("%B %Y")
        monthly = monthly[["Mois","Total","Avg_Daily","CI_Lower","CI_Upper"]]
        monthly.columns = Tlist("forecast_monthly_cols") or monthly.columns.tolist()
        st.dataframe(monthly, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab B — Stock à préparer (Recommendations)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_reco:
        st.markdown(
            f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1rem;">{T("reco_sub")}</p>',
            unsafe_allow_html=True,
        )
        with st.spinner(T("reco_spinner")):
            model_r = get_model(data_hash, daily)
            pred_r  = run_forecast(data_hash, 30, model_r, daily)

        next7 = pred_r[pred_r["ds"] >= fc_start].head(7).copy()
        next7["day_name"]      = next7["ds"].dt.day_name()
        next7["predicted_qty"] = next7["yhat"].clip(lower=0).round(0).astype(int)
        next7["suggested"]     = (next7["predicted_qty"] * BUFFER_RATE).round(0).astype(int)
        next7["is_weekend"]    = next7["day_name"].isin(WEEKEND_DAYS)
        next7["exceeds_avg"]   = next7["yhat"] > avg_open * PEAK_THRESH
        next7["risk"]          = next7.apply(lambda r: _risk_level(r, avg_open), axis=1)

        total_stock = int(next7["suggested"].sum())
        total_pred  = int(next7["predicted_qty"].sum())
        st.markdown(
            f'<div class="sum-box">'
            f'<span class="sum-big">{total_stock:,} {T("units")}</span>'
            f'<span class="sum-sub">{T("reco_summary_sub").format(pred=total_pred)}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        alerted = False
        for _, row in next7[next7["exceeds_avg"]].iterrows():
            pct    = (row["yhat"] / avg_open - 1) * 100
            extras = []
            h = _get_holiday_name(row["ds"])
            if h:              extras.append(f"public holiday: {h}")
            if is_ramadan_period(row["ds"]): extras.append("Ramadan")
            extra_str = f" · {', '.join(extras)}" if extras else ""
            st.warning(f"**{row['ds'].strftime('%A %d %b')}** — {row['predicted_qty']} {T('units')} forecast (**+{pct:.0f}%** vs avg){extra_str}.")
            alerted = True

        for _, row in next7[~next7["exceeds_avg"]].iterrows():
            h   = _get_holiday_name(row["ds"])
            ram = is_ramadan_period(row["ds"])
            if h or ram:
                tags = ([f"Public holiday: **{h}**"] if h else []) + (["**Ramadan**"] if ram else [])
                st.info(f"**{row['ds'].strftime('%A %d %b')}** — {' · '.join(tags)}")
                alerted = True

        if not alerted:
            st.success(T("reco_no_peak"))

        _sec(T("reco_plan"))
        display = next7[["ds","day_name","predicted_qty","suggested","is_weekend","risk","yhat_lower","yhat_upper"]].copy()
        display["ds"]         = display["ds"].dt.strftime("%Y-%m-%d")
        display["yhat_lower"] = display["yhat_lower"].clip(lower=0).round(0).astype(int)
        display["yhat_upper"] = display["yhat_upper"].round(0).astype(int)
        display["is_weekend"] = display["is_weekend"].map({True: "Weekend", False: ""})
        display.columns = [
            T("reco_col_date"), T("reco_col_day"), T("reco_col_expected"), T("reco_col_stock"),
            T("reco_col_peak"), T("reco_col_risk"), "Min", "Max",
        ]
        risk_col = T("reco_col_risk")
        RISK_BG  = {T("reco_risk_high"): "#fee2e2", T("reco_risk_med"): "#fef9c3", T("reco_risk_low"): "#dcfce7"}
        RISK_FG  = {T("reco_risk_high"): "color:#991b1b;font-weight:700",
                    T("reco_risk_med"):  "color:#854d0e;font-weight:700",
                    T("reco_risk_low"):  "color:#166534;font-weight:700"}

        def _style_rows(row):
            return [f"background-color:{RISK_BG.get(row[risk_col], '#dcfce7')}"] * len(row)

        st.dataframe(
            display.reset_index(drop=True).style
            .apply(_style_rows, axis=1)
            .map(lambda v: RISK_FG.get(v, ""), subset=[risk_col]),
            use_container_width=True, hide_index=True,
        )
        st.caption(T("reco_footer").format(avg=f"{avg_open:.0f}"))

    # ══════════════════════════════════════════════════════════════════════════
    # Tab C — Prévision par produit
    # ══════════════════════════════════════════════════════════════════════════
    with tab_prod:
        if lite_mode:
            st.markdown(f'<div class="lite-banner">{T("analyse_prod_lite")}</div>', unsafe_allow_html=True)
        else:
            prod_options = sorted(df["produit"].unique().tolist())
            if not prod_options:
                st.warning(T("analyse_prod_empty"))
            else:
                sel_prod     = st.selectbox(T("analyse_prod_pick"), prod_options, label_visibility="collapsed")
                prod_horizon = st.radio(T("analyse_horizon"), [7, 14, 30], horizontal=True,
                                        format_func=lambda x: T("analyse_horizon_fmt").format(n=x))
                prod_daily = aggregate_by_product(df, sel_prod)
                if len(prod_daily) < 30:
                    st.warning(T("analyse_prod_nodata").format(p=sel_prod, n=len(prod_daily)))
                else:
                    with st.spinner(T("analyse_prod_spinner").format(p=sel_prod)):
                        prod_model = get_product_model(data_hash, sel_prod, prod_daily)
                        prod_pred  = run_product_forecast(data_hash, sel_prod, prod_horizon, prod_model, prod_daily)

                    prod_future = prod_pred[prod_pred["ds"] >= fc_start].copy()
                    avg_prod    = float(prod_daily[prod_daily["y"] > 0]["y"].mean())
                    next7_total = int(prod_future.head(7)["yhat"].clip(lower=0).sum().round())

                    cp1, cp2 = st.columns(2)
                    cp1.markdown(_kpi(T("analyse_prod_hist"),  f"{avg_prod:.1f}", T("analyse_prod_units"), "📈"), unsafe_allow_html=True)
                    cp2.markdown(_kpi(T("analyse_prod_next7"), f"{next7_total:,}", T("analyse_prod_total"), "🔮"), unsafe_allow_html=True)

                    fig_p = go.Figure()
                    fig_p.add_trace(go.Scatter(
                        x=pd.concat([prod_pred["ds"], prod_pred["ds"][::-1]]),
                        y=pd.concat([prod_pred["yhat_upper"], prod_pred["yhat_lower"][::-1]]),
                        fill="toself", fillcolor="rgba(20,184,166,.13)",
                        line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name=T("analyse_interval"),
                    ))
                    fig_p.add_trace(go.Scatter(x=prod_pred["ds"], y=prod_pred["yhat"], mode="lines",
                                               line=dict(color="#14b8a6", width=2, dash="dot"), name=T("analyse_pred_label")))
                    fig_p.add_trace(go.Scatter(x=prod_daily["ds"], y=prod_daily["y"], mode="lines",
                                               line=dict(color="#1e3a5f", width=1.2), name=T("analyse_hist_label")))
                    fig_p.add_vline(x=fc_start.timestamp() * 1000,
                                    line=dict(color="crimson", dash="dash", width=1.5),
                                    annotation_text=T("analyse_fc_start"), annotation_position="top right")
                    fig_p.update_layout(template="plotly_white", hovermode="x unified",
                                         xaxis_title="", yaxis_title=T("units"), height=380, margin=dict(t=30, b=30),
                                         legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_p, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab D — Liste de courses (Shopping List)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_shop:
        _render_shopping(df, daily, data_hash, avg_open, last_date, fc_start,
                         catalog, active_prods, active_csv, lang)


def _render_shopping(
    df, daily, data_hash, avg_open, last_date, fc_start,
    catalog, active_prods, active_csv, lang,
) -> None:
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">{T("shop_sub")}</p>',
        unsafe_allow_html=True,
    )

    with st.spinner(T("shop_spinner")):
        model_s = get_model(data_hash, daily)
        pred_s  = run_forecast(data_hash, 7, model_s, daily)

    next7       = pred_s[pred_s["ds"] >= fc_start].head(7).copy()
    next7_dates = next7["ds"].tolist()

    product_shares = df.groupby("produit")["quantite_vendue"].sum().div(df["quantite_vendue"].sum())
    products_list  = active_prods if active_prods else sorted(df["produit"].unique())

    # ── Ingrédients par plat ──────────────────────────────────────────────────
    _sec(T("shop_ingredients_title"))
    st.caption(T("shop_ingredients_caption"))

    recipe_tabs   = st.tabs(products_list)
    edited_recipes: dict[str, pd.DataFrame] = {}

    for tab, product in zip(recipe_tabs, products_list):
        rkey_p = f"recipe_{product}_{data_hash}"
        if rkey_p not in st.session_state:
            if product in RECIPES:
                rows = [{"Ingredient": ing, "Amount per serving": float(qty), "Unit": unit}
                        for ing, (qty, unit) in RECIPES[product].items()]
            else:
                rows = [{"Ingredient": "", "Amount per serving": 0.0, "Unit": "g"}]
            st.session_state[rkey_p] = pd.DataFrame(rows)

        with tab:
            edited = st.data_editor(
                st.session_state[rkey_p],
                column_config={
                    "Ingredient":         st.column_config.TextColumn(T("shop_col_ing")),
                    "Amount per serving": st.column_config.NumberColumn("Amount", min_value=0, format="%.1f"),
                    "Unit":               st.column_config.SelectboxColumn(T("shop_col_unit"), options=UNIT_OPTIONS),
                },
                num_rows="dynamic", hide_index=True, use_container_width=True,
                key=f"recipe_editor_{product}_{data_hash}",
            )
            st.session_state[rkey_p] = edited
            edited_recipes[product]  = edited

    ingredient_unit: dict[str, str] = {}
    for product, rdf in edited_recipes.items():
        valid = rdf.dropna(subset=["Ingredient"])
        valid = valid[valid["Ingredient"].astype(str).str.strip().ne("")]
        for _, row in valid.iterrows():
            ingredient_unit.setdefault(str(row["Ingredient"]), str(row["Unit"]))
    all_ingredients = list(ingredient_unit.keys())

    if not all_ingredients:
        st.warning(T("shop_no_ing"))
        return

    # ── Stock actuel ──────────────────────────────────────────────────────────
    _sec(T("shop_stock_title"))
    st.caption(T("shop_stock_caption"))
    skey = f"stock_{data_hash}"
    if skey not in st.session_state:
        st.session_state[skey] = {}
    for ing in all_ingredients:
        st.session_state[skey].setdefault(ing, 0)

    stock_df = pd.DataFrame([
        {T("shop_col_ing"): ing, T("shop_col_unit"): ingredient_unit[ing],
         T("shop_kpi_ok"):  int(st.session_state[skey].get(ing, 0))}
        for ing in all_ingredients
    ])
    edited_stock = st.data_editor(
        stock_df,
        column_config={
            T("shop_col_ing"):  st.column_config.TextColumn(T("shop_col_ing"), disabled=True),
            T("shop_col_unit"): st.column_config.TextColumn(T("shop_col_unit"), disabled=True),
            T("shop_kpi_ok"):   st.column_config.NumberColumn(T("shop_kpi_ok"), min_value=0, step=1),
        },
        hide_index=True, use_container_width=True, key=f"stock_editor_{data_hash}",
    )
    for _, row in edited_stock.iterrows():
        st.session_state[skey][row[T("shop_col_ing")]] = int(row[T("shop_kpi_ok")])
    current_stocks = {ing: int(st.session_state[skey].get(ing, 0)) for ing in all_ingredients}

    # ── Calcul des besoins ─────────────────────────────────────────────────────
    ingredient_daily: dict[str, list[float]] = {ing: [] for ing in all_ingredients}
    for d_idx in range(len(next7)):
        yhat_total = float(next7.iloc[d_idx]["yhat"])
        day: dict[str, float] = {ing: 0.0 for ing in all_ingredients}
        for product, rdf in edited_recipes.items():
            share    = float(product_shares.get(product, 0))
            prod_qty = yhat_total * share
            valid    = rdf.dropna(subset=["Ingredient"])
            valid    = valid[valid["Ingredient"].astype(str).str.strip().ne("")]
            for _, row in valid.iterrows():
                ing = str(row["Ingredient"])
                if ing in day:
                    day[ing] += prod_qty * float(row["Amount per serving"])
        for ing in all_ingredients:
            ingredient_daily[ing].append(day[ing])

    ingredient_total  = {ing: sum(ingredient_daily[ing]) for ing in all_ingredients}
    ingredient_needed = {ing: round(ingredient_total[ing] * BUFFER_RATE) for ing in all_ingredients}
    to_order_map      = {ing: max(0, ingredient_needed[ing] - current_stocks[ing]) for ing in all_ingredients}

    n_to_buy = sum(1 for v in to_order_map.values() if v > 0)
    n_ok     = len(all_ingredients) - n_to_buy
    soonest_dt = None
    for ing in all_ingredients:
        s = compute_stockout_date(current_stocks[ing], ingredient_daily[ing], next7_dates)
        if not s.startswith("OK"):
            d = pd.Timestamp(s)
            if soonest_dt is None or d < soonest_dt:
                soonest_dt = d
    soonest_str = soonest_dt.strftime("%b %d") if soonest_dt else "—"

    kk1, kk2, kk3 = st.columns(3)
    for col, html in zip([kk1, kk2, kk3], [
        _kpi(T("shop_kpi_tobuy"), str(n_to_buy), f"/ {len(all_ingredients)}", "🛒"),
        _kpi(T("shop_kpi_ok"),    str(n_ok),     "", "✅"),
        _kpi(T("shop_kpi_first"), soonest_str,   "", "⚠️"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    # Daily needs table
    _sec(T("shop_needs_title"))
    day_labels = [pd.Timestamp(d).strftime("%a %m/%d") for d in next7_dates]
    needs_rows = []
    for ing in all_ingredients:
        row_d = {T("shop_col_ing"): ing, T("shop_col_unit"): ingredient_unit[ing]}
        for label, qty in zip(day_labels, ingredient_daily[ing]):
            row_d[label] = round(qty)
        row_d["7j Total"]      = round(ingredient_total[ing])
        row_d["+15% buffer"]   = ingredient_needed[ing]
        needs_rows.append(row_d)
    needs_df = pd.DataFrame(needs_rows)
    st.dataframe(
        needs_df.style.map(lambda _: "font-weight:700", subset=["7j Total", "+15% buffer"]),
        use_container_width=True, hide_index=True,
    )

    # Shopping order
    _sec(T("shop_order_title"))
    S_OK, S_ORD, S_CRI = T("shop_status_ok"), T("shop_status_order"), T("shop_status_critical")
    S_BG = {S_OK: "#dcfce7", S_ORD: "#fef9c3", S_CRI: "#fee2e2"}
    S_FG = {S_OK: "color:#166534;font-weight:700", S_ORD: "color:#854d0e;font-weight:700",
            S_CRI: "color:#991b1b;font-weight:700"}
    order_rows = []
    for ing in all_ingredients:
        need   = ingredient_needed[ing]
        stock  = current_stocks[ing]
        order  = to_order_map[ing]
        status = S_OK if order == 0 else (S_CRI if order > need * 0.5 else S_ORD)
        order_rows.append({T("shop_col_ing"): ing, T("shop_col_unit"): ingredient_unit[ing],
                            T("shop_col_need"): need, T("shop_col_have"): stock,
                            T("shop_col_buy"): order, T("shop_col_status"): status})
    order_df = pd.DataFrame(order_rows)
    status_col = T("shop_col_status")
    st.dataframe(
        order_df.style
        .apply(lambda r: [f"background-color:{S_BG.get(r[status_col], '')}"] * len(r), axis=1)
        .map(lambda v: S_FG.get(v, ""), subset=[status_col]),
        use_container_width=True, hide_index=True,
    )

    # Stockout dates
    _sec(T("shop_stockout_title"))
    URG_BG = {T("shop_urg_ok"): "#dcfce7", T("shop_urg_soon"): "#fee2e2",
              T("shop_urg_week"): "#fef9c3", T("shop_urg_later"): "#fff7ed"}
    URG_FG = {T("shop_urg_ok"):   "color:#166534;font-weight:700",
              T("shop_urg_soon"): "color:#991b1b;font-weight:700",
              T("shop_urg_week"): "color:#854d0e;font-weight:700",
              T("shop_urg_later"):"color:#9a3412;font-weight:700"}

    stockout_rows = []
    for ing in all_ingredients:
        s     = compute_stockout_date(current_stocks[ing], ingredient_daily[ing], next7_dates)
        avg_d = round(ingredient_total[ing] / 7, 1)
        if s.startswith("OK"):
            date_str, delay, urg = "—", "—", T("shop_urg_ok")
        else:
            delta    = (pd.Timestamp(s) - fc_start).days + 1
            date_str = s
            delay    = f"D+{delta}"
            urg      = T("shop_urg_soon") if delta <= 2 else (T("shop_urg_week") if delta <= 4 else T("shop_urg_later"))
        stockout_rows.append({
            T("shop_col_ing"):     ing,
            T("shop_col_unit"):    ingredient_unit[ing],
            T("shop_col_instock"): current_stocks[ing],
            T("shop_col_perday"):  avg_d,
            T("shop_col_runout"):  date_str,
            "In":                  delay,
            T("shop_col_urgency"): urg,
            "_bg": URG_BG.get(urg, "#fff"),
        })

    stockout_df = pd.DataFrame(stockout_rows)
    urg_col     = T("shop_col_urgency")
    disp_cols   = [T("shop_col_ing"), T("shop_col_unit"), T("shop_col_instock"),
                   T("shop_col_perday"), T("shop_col_runout"), "In", urg_col]
    st.dataframe(
        stockout_df[disp_cols]
        .style.apply(
            lambda r: [f"background-color:{stockout_df.at[r.name, '_bg']}"] * len(r), axis=1)
        .map(lambda v: URG_FG.get(v, ""), subset=[urg_col]),
        use_container_width=True, hide_index=True,
    )
    st.divider()
    st.caption(T("shop_caption"))
