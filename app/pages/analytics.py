"""
pages/analytics.py — Analyses page: Overview · Tendances · Catalogue.
"""
import datetime as _dt

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from i18n import T, Tlist, day_label
from ui import _kpi, _sec
from logic.forecasting import (
    MAROC_FIXED_HOLIDAYS, ISLAMIC_HOLIDAYS, RAMADAN_DATES,
    get_moroccan_holidays, is_ramadan_period,
    aggregate_by_product, get_product_model, run_product_forecast,
)
from logic.data_loading import check_data_health, aggregate_by_product


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
    is_ar   = lang == "ar"
    wrap_cl = "rtl ar-font" if is_ar else ""

    st.markdown(
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.4rem;">'
        f'{T("nav_analytics")}</h1>',
        unsafe_allow_html=True,
    )

    tab_ov, tab_tr, tab_cat = st.tabs([
        T("tab_overview"), T("tab_trends"), T("tab_catalogue"),
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Tab A — Vue d'ensemble (Overview)
    # ══════════════════════════════════════════════════════════════════════════
    with tab_ov:
        src_label = f"<b>{st.session_state.get('_uploaded_name','')}</b>" if st.session_state.get('_uploaded_name') else "démo"
        st.markdown(
            f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">'
            f'{T("overview_sub").format(n=f"{len(df):,}", start=daily["ds"].min().date(), end=daily["ds"].max().date())}'
            f'</p>',
            unsafe_allow_html=True,
        )

        total_sales  = int(df["quantite_vendue"].sum())
        best_product = df.groupby("produit")["quantite_vendue"].sum().idxmax()
        avg_daily    = round(float(daily[daily["y"] > 0]["y"].mean()), 1)
        df_open      = df[df["quantite_vendue"] > 0].copy()
        df_open["day_name"] = df_open["date"].dt.day_name()
        busiest_day  = df_open.groupby("day_name")["quantite_vendue"].mean().idxmax()

        c1, c2, c3, c4 = st.columns(4)
        for col, html in zip([c1, c2, c3, c4], [
            _kpi(T("overview_kpi_total"),   f"{total_sales:,}", f"{daily['ds'].min().year}–{daily['ds'].max().year}", "📦"),
            _kpi(T("overview_kpi_best"),    best_product,       "", "🏆"),
            _kpi(T("overview_kpi_avg"),     f"{avg_daily}",     T("units") + "/day", "📈"),
            _kpi(T("overview_kpi_busiest"), busiest_day,        "", "📅"),
        ]):
            col.markdown(html, unsafe_allow_html=True)

        _sec(T("overview_trend"))
        min_d, max_d = daily["ds"].min().date(), daily["ds"].max().date()
        date_range = st.date_input(
            T("overview_filter"), value=(min_d, max_d),
            min_value=min_d, max_value=max_d, label_visibility="collapsed",
        )
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            s_d, e_d = date_range
            filtered = daily[(daily["ds"].dt.date >= s_d) & (daily["ds"].dt.date <= e_d)]
        else:
            filtered = daily

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=filtered["ds"], y=filtered["y"], mode="lines", fill="tozeroy",
            fillcolor="rgba(20,184,166,.10)", line=dict(color="#14b8a6", width=1.8),
            name=T("kpi_sold"),
        ))
        fig_trend.update_layout(
            template="plotly_white", hovermode="x unified",
            xaxis_title="", yaxis_title=T("units"),
            margin=dict(t=20, b=30), height=300,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        _sec(T("overview_breakdown"))
        by_product = (
            df[df["produit"].isin(active_csv)]
            .groupby("produit")["quantite_vendue"].sum()
            .reset_index().sort_values("quantite_vendue", ascending=False)
            .rename(columns={"produit": "Product", "quantite_vendue": "Total"})
        )
        PALETTE = ["#14b8a6", "#1e3a5f", "#0ea5e9", "#6366f1"]
        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(by_product, names="Product", values="Total",
                             color_discrete_sequence=PALETTE, hole=0.42)
            fig_pie.update_traces(textposition="outside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20), height=320)
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_bar:
            fig_bar = px.bar(by_product, x="Product", y="Total",
                             color="Product", color_discrete_sequence=PALETTE, text_auto=True)
            fig_bar.update_layout(showlegend=False, template="plotly_white",
                                   margin=dict(t=20, b=20), height=320,
                                   xaxis_title="", yaxis_title=T("overview_kpi_total"))
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

        _sec(T("overview_health"))
        checks   = check_data_health(df)
        badge_map = {"ok": "badge-ok", "warn": "badge-warn", "err": "badge-err"}
        label_map = {
            "ok":   T("overview_health_ok"),
            "warn": T("overview_health_warn"),
            "err":  T("overview_health_err"),
        }
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

    # ══════════════════════════════════════════════════════════════════════════
    # Tab B — Tendances
    # ══════════════════════════════════════════════════════════════════════════
    with tab_tr:
        # Semaine en cours vs semaine dernière
        _sec(T("analyse_week"))
        this_wk_start = last_date - pd.Timedelta(days=6)
        last_wk_start = last_date - pd.Timedelta(days=13)
        last_wk_end   = last_date - pd.Timedelta(days=7)
        this_wk = daily[daily["ds"] >= this_wk_start].copy()
        last_wk = daily[(daily["ds"] >= last_wk_start) & (daily["ds"] <= last_wk_end)].copy()
        this_wk["day_label"] = this_wk["ds"].dt.day_name().str[:3]
        last_wk["day_label"] = last_wk["ds"].dt.day_name().str[:3]

        wk_cur  = int(this_wk["y"].sum())
        wk_prev = int(last_wk["y"].sum())
        wk_delta = wk_cur - wk_prev
        wk_pct   = (wk_delta / max(wk_prev, 1)) * 100
        arrow = "▲" if wk_delta >= 0 else "▼"

        col_kw1, col_kw2, col_kw3 = st.columns(3)
        col_kw1.markdown(_kpi(T("analyse_cur_week"),  f"{wk_cur:,}",  T("analyse_cur_week_sub"),  "📅"), unsafe_allow_html=True)
        col_kw2.markdown(_kpi(T("analyse_prev_week"), f"{wk_prev:,}", T("analyse_prev_week_sub"), "📆"), unsafe_allow_html=True)
        col_kw3.markdown(_kpi(T("analyse_delta"),     f"{arrow} {abs(wk_pct):.1f}%",
                               f"{arrow} {abs(wk_delta):,} {T('units')}", "📊"), unsafe_allow_html=True)

        fig_ww = go.Figure()
        fig_ww.add_trace(go.Bar(x=last_wk["day_label"], y=last_wk["y"],
                                 name=T("analyse_prev_week"), marker_color="#94a3b8"))
        fig_ww.add_trace(go.Bar(x=this_wk["day_label"], y=this_wk["y"],
                                 name=T("analyse_cur_week"), marker_color="#14b8a6"))
        fig_ww.update_layout(barmode="group", template="plotly_white",
                              height=280, margin=dict(t=20, b=20),
                              xaxis_title="", yaxis_title=T("analyse_units_sold"),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_ww, use_container_width=True)

        # Heure de pointe
        _sec(T("analyse_hour"))
        if "heure" in df.columns:
            df_h = df.copy()
            df_h["heure_int"] = pd.to_numeric(df_h["heure"], errors="coerce").dropna().astype(int)
            hourly = (
                df_h.groupby("heure_int")["quantite_vendue"].mean().reset_index()
                .rename(columns={"heure_int": "Heure", "quantite_vendue": "Moy. unités"})
            )
            best_hour = int(hourly.loc[hourly["Moy. unités"].idxmax(), "Heure"])
            st.markdown(
                _kpi(T("analyse_best_hour"), f"{best_hour:02d}h00",
                     T("analyse_peak_avg").format(qty=hourly["Moy. unités"].max()), "🕐"),
                unsafe_allow_html=True,
            )
            fig_hour = px.bar(hourly, x="Heure", y="Moy. unités",
                              color_discrete_sequence=["#14b8a6"], text_auto=".0f")
            fig_hour.update_layout(template="plotly_white", height=260, margin=dict(t=20, b=20),
                                    xaxis=dict(tickmode="linear", dtick=1))
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.info(T("analyse_no_hour"))

        # Calendrier des fêtes
        _sec(T("analyse_calendar"))
        current_year = _dt.date.today().year
        cal_years    = [current_year, current_year + 1]
        holiday_rows = []
        for yr in cal_years:
            for month, day, name in MAROC_FIXED_HOLIDAYS:
                holiday_rows.append({"Date": f"{day:02d}/{month:02d}/{yr}", "Fête": name, "Type": T("analyse_cal_nat")})
            for date_str, name in ISLAMIC_HOLIDAYS.get(yr, []):
                holiday_rows.append({"Date": pd.Timestamp(date_str).strftime("%d/%m/%Y"), "Fête": name, "Type": T("analyse_cal_isl")})
            if yr in RAMADAN_DATES:
                s, e = RAMADAN_DATES[yr]
                holiday_rows.append({"Date": f"{pd.Timestamp(s).strftime('%d/%m/%Y')} → {pd.Timestamp(e).strftime('%d/%m/%Y')}",
                                     "Fête": T("analyse_cal_ram"), "Type": T("analyse_cal_ram")})
        cal_df = pd.DataFrame(holiday_rows)

        def _style_cal(row):
            cn, ci, cr = T("analyse_cal_nat"), T("analyse_cal_isl"), T("analyse_cal_ram")
            colors = {cn: "#dbeafe", ci: "#fef3c7", cr: "#fce7f3"}
            return [f"background-color:{colors.get(row['Type'], '#fff')}"] * len(row)

        st.dataframe(cal_df.style.apply(_style_cal, axis=1),
                     use_container_width=True, hide_index=True)
        if ramadan_mode:
            st.info(T("analyse_ram_msg"))

        # Tendance produits (30j vs 30j précédents)
        _sec(T("analyse_trend_caption").format(
            up=T("analyse_trend_up"), stable=T("analyse_trend_stable"), down=T("analyse_trend_down")))
        cut30 = last_date - pd.Timedelta(days=30)
        cut60 = last_date - pd.Timedelta(days=60)
        prod_sums = df.groupby("produit")["quantite_vendue"].sum().reset_index().sort_values("quantite_vendue", ascending=False)
        recent30  = df[df["date"] > cut30].groupby("produit")["quantite_vendue"].sum()
        prev30    = df[(df["date"] > cut60) & (df["date"] <= cut30)].groupby("produit")["quantite_vendue"].sum()
        rows_trend = []
        for _, row in prod_sums.iterrows():
            p   = row["produit"]
            r30 = float(recent30.get(p, 0))
            p30 = float(prev30.get(p, 0))
            t   = T("analyse_trend_up") if r30 > p30 * 1.05 else (T("analyse_trend_down") if r30 < p30 * 0.95 else T("analyse_trend_stable"))
            rows_trend.append({"Produit": p, "Total": int(row["quantite_vendue"]), "30j précédents": int(p30), "30j récents": int(r30), "Tendance": t})
        st.dataframe(pd.DataFrame(rows_trend), use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Tab C — Catalogue produits
    # ══════════════════════════════════════════════════════════════════════════
    with tab_cat:
        _render_catalogue(df, data_hash, catalog, active_prods, active_csv, lang)


def _render_catalogue(
    df: pd.DataFrame,
    data_hash: str,
    catalog: dict,
    active_prods: list[str],
    active_csv: list[str],
    lang: str,
) -> None:
    from constants import RECIPES

    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">{T("products_sub")}</p>',
        unsafe_allow_html=True,
    )

    n_total  = len(catalog)
    n_active = len(active_prods)
    n_custom = sum(1 for v in catalog.values() if v["source"] == "custom")

    k1, k2, k3 = st.columns(3)
    for col, html in zip([k1, k2, k3], [
        _kpi(T("products_kpi_total"),  str(n_total),  "", "📋"),
        _kpi(T("products_kpi_active"), str(n_active), "", "✅"),
        _kpi(T("products_kpi_custom"), str(n_custom), "", "✏️"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

    _sec(T("products_list_title"))
    st.caption(T("products_list_caption"))

    def _has_recipe(name: str) -> bool:
        if name in RECIPES:
            return True
        rkey = f"recipe_{name}_{data_hash}"
        if rkey in st.session_state:
            rdf   = st.session_state[rkey]
            valid = rdf.dropna(subset=["Ingredient"])
            valid = valid[valid["Ingredient"].astype(str).str.strip().ne("")]
            return len(valid) > 0
        return False

    catalog_rows = []
    for name, meta in sorted(catalog.items()):
        share = df[df["produit"] == name]["quantite_vendue"].sum() / max(df["quantite_vendue"].sum(), 1)
        catalog_rows.append({
            T("products_col_dish"):   name,
            T("products_col_source"): T("products_source_csv") if meta["source"] == "csv" else T("products_source_custom"),
            T("products_col_share"):  f"{share*100:.1f}%" if meta["source"] == "csv" else "—",
            T("products_col_recipe"): "✓" if _has_recipe(name) else "—",
            T("products_col_active"): meta["active"],
        })
    cat_df     = pd.DataFrame(catalog_rows)
    active_col = T("products_col_active")
    edited_cat = st.data_editor(
        cat_df,
        column_config={
            T("products_col_dish"):   st.column_config.TextColumn(T("products_col_dish"),   disabled=True),
            T("products_col_source"): st.column_config.TextColumn(T("products_col_source"), disabled=True),
            T("products_col_share"):  st.column_config.TextColumn(T("products_col_share"),  disabled=True),
            T("products_col_recipe"): st.column_config.TextColumn(T("products_col_recipe"), disabled=True),
            active_col:               st.column_config.CheckboxColumn(active_col),
        },
        hide_index=True, use_container_width=True, key=f"catalog_editor_{data_hash}",
    )
    for _, row in edited_cat.iterrows():
        name = row[T("products_col_dish")]
        if name in catalog:
            catalog[name]["active"] = bool(row[active_col])

    # Add custom product
    _sec(T("products_add_title"))
    st.caption(T("products_add_caption"))
    with st.form("add_product_form", clear_on_submit=True):
        add_col, btn_col = st.columns([4, 1])
        with add_col:
            new_name = st.text_input("", placeholder=T("products_add_placeholder"), label_visibility="collapsed")
        with btn_col:
            submitted = st.form_submit_button(T("products_add_btn"), use_container_width=True)
        if submitted:
            name_clean = new_name.strip()
            if not name_clean:
                st.error("—")
            elif name_clean in catalog:
                st.warning(T("products_add_exists").format(name=name_clean))
            else:
                catalog[name_clean] = {"active": True, "source": "custom"}
                st.success(T("products_add_success").format(name=name_clean))
                st.rerun()

    # Delete custom product
    custom_list = [p for p, v in catalog.items() if v["source"] == "custom"]
    if custom_list:
        _sec(T("products_del_title"))
        st.caption(T("products_del_caption"))
        del_c, btn_c = st.columns([4, 1])
        with del_c:
            to_del = st.selectbox("", options=["—"] + sorted(custom_list), label_visibility="collapsed")
        with btn_c:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            if st.button(T("products_del_btn"), type="primary", use_container_width=True):
                if to_del != "—":
                    del catalog[to_del]
                    rkey = f"recipe_{to_del}_{data_hash}"
                    if rkey in st.session_state:
                        del st.session_state[rkey]
                    st.rerun()

    st.divider()
    st.info(T("products_info"))
