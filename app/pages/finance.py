"""
pages/finance.py — Finances & Rentabilité page.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from i18n import T, Tlist
from ui import _fin_kpi, _kpi, _mad, _progress_bar, _sec
from logic.finance import build_fin_df


def render(
    df: pd.DataFrame,
    data_hash: str,
    price_map: dict,
    cost_map: dict,
    all_prods: list[str],
    prices_key: str,
    costs_key: str,
    target_key: str,
    lang: str,
) -> None:
    wrap = 'class="rtl ar-font"' if lang == "ar" else ""

    st.markdown(
        f'<h1 style="color:#1e3a5f;font-weight:800;margin-bottom:.2rem;">{T("fin_title")}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#64748b;font-size:.88rem;margin-bottom:1.2rem;">{T("fin_sub")}</p>',
        unsafe_allow_html=True,
    )

    # ── Config prix / coûts ──────────────────────────────────────────────────
    with st.expander(T("fin_config"), expanded=True):
        st.caption(T("fin_config_caption"))
        if "prix_unitaire" in df.columns:
            st.success(T("fin_auto_price"))

        cp, cc = T("fin_col_price"), T("fin_col_cost")
        fin_rows = [
            {T("fin_col_prod"): p, cp: price_map.get(p, 0.0), cc: cost_map.get(p, 0.0)}
            for p in all_prods
        ]
        edited = st.data_editor(
            pd.DataFrame(fin_rows),
            column_config={
                T("fin_col_prod"): st.column_config.TextColumn(T("fin_col_prod"), disabled=True),
                cp: st.column_config.NumberColumn(cp, min_value=0.0, format="%.2f", help=T("fin_col_price_help")),
                cc: st.column_config.NumberColumn(cc, min_value=0.0, format="%.2f", help=T("fin_col_cost_help")),
            },
            hide_index=True, use_container_width=True, key=f"fin_editor_{data_hash}",
        )
        for _, row in edited.iterrows():
            price_map[row[T("fin_col_prod")]] = float(row[cp])
            cost_map[row[T("fin_col_prod")]]  = float(row[cc])
        st.session_state[prices_key] = price_map
        st.session_state[costs_key]  = cost_map

    prices_set = any(v > 0 for v in price_map.values())
    if not prices_set:
        st.warning(T("fin_warn_no_price"))

    # ── Build financial dataframe ────────────────────────────────────────────
    df_fin = build_fin_df(df, price_map, cost_map)
    daily_fin = (
        df_fin.groupby("date")
        .agg(recette=("recette","sum"), cout_mp=("cout_mp","sum"),
             marge_brute=("marge_brute","sum"), quantite=("quantite_vendue","sum"))
        .reset_index()
    )
    daily_fin["date"] = pd.to_datetime(daily_fin["date"])

    last_day     = daily_fin["date"].max()
    ldr          = daily_fin[daily_fin["date"] == last_day]
    ca_jour      = float(ldr["recette"].sum())
    marge_jour   = float(ldr["marge_brute"].sum())
    wk           = last_day - pd.Timedelta(days=6)
    ca_semaine   = float(daily_fin[daily_fin["date"] >= wk]["recette"].sum())
    marge_semaine= float(daily_fin[daily_fin["date"] >= wk]["marge_brute"].sum())
    mm           = (daily_fin["date"].dt.year == last_day.year) & (daily_fin["date"].dt.month == last_day.month)
    ca_mois      = float(daily_fin[mm]["recette"].sum())
    cout_mois    = float(daily_fin[mm]["cout_mp"].sum())
    marge_mois   = float(daily_fin[mm]["marge_brute"].sum())
    marge_pct    = (marge_mois / ca_mois * 100) if ca_mois > 0 else 0.0

    # ── KPIs CA ──────────────────────────────────────────────────────────────
    _sec(T("fin_ca_section"))
    k1, k2, k3, k4 = st.columns(4)
    for col, html in zip([k1, k2, k3, k4], [
        _fin_kpi(T("fin_kpi_day"),   _mad(ca_jour),    T("fin_margin_sub").format(m=_mad(marge_jour)),    "💶"),
        _fin_kpi(T("fin_kpi_week"),  _mad(ca_semaine), T("fin_margin_sub").format(m=_mad(marge_semaine)), "📅"),
        _fin_kpi(T("fin_kpi_month"), _mad(ca_mois),    last_day.strftime("%B %Y"),                        "📆"),
        _fin_kpi(T("fin_kpi_margin"),_mad(marge_mois), T("fin_margin_pct").format(pct=marge_pct),         "📊"),
    ]):
        col.markdown(html, unsafe_allow_html=True)

    # ── Objectif CA ──────────────────────────────────────────────────────────
    _sec(T("fin_target_section"))
    col_tgt, col_bar = st.columns([1, 2])
    with col_tgt:
        target = st.number_input(
            T("fin_target_input"), min_value=0.0,
            value=float(st.session_state[target_key]),
            step=100.0, format="%.0f", label_visibility="collapsed",
        )
        st.session_state[target_key] = target
    with col_bar:
        if target > 0:
            pct_day = ca_jour / target * 100
            st.markdown(
                _progress_bar(pct_day, T("fin_target_above").format(pct=pct_day, ca=_mad(ca_jour), target=_mad(target))),
                unsafe_allow_html=True,
            )
            days_above = int((daily_fin["recette"] >= target).sum())
            st.caption(T("fin_target_days").format(n=days_above, total=len(daily_fin)))
        else:
            st.caption(T("fin_target_empty"))

    # Revenue trend chart
    fig_ca = go.Figure()
    fig_ca.add_trace(go.Scatter(
        x=daily_fin["date"], y=daily_fin["recette"],
        mode="lines", fill="tozeroy", fillcolor="rgba(34,197,94,.10)",
        line=dict(color="#22c55e", width=1.8), name=T("fin_ca_series"),
    ))
    fig_ca.add_trace(go.Scatter(
        x=daily_fin["date"], y=daily_fin["cout_mp"],
        mode="lines", line=dict(color="#f97316", width=1.4, dash="dot"),
        name=T("fin_cost_series"),
    ))
    if target > 0:
        fig_ca.add_hline(
            y=target, line=dict(color="#6366f1", dash="dash", width=1.5),
            annotation_text=T("fin_target_line").format(t=_mad(target)),
            annotation_position="top left",
        )
    fig_ca.update_layout(
        template="plotly_white", hovermode="x unified",
        xaxis_title="", yaxis_title="MAD",
        height=300, margin=dict(t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ca, use_container_width=True)

    # ── Rentabilité produits ──────────────────────────────────────────────────
    with st.expander(T("fin_prod_section"), expanded=True):
        prod_fin = (
            df_fin.groupby("produit")
            .agg(total_vendu=("quantite_vendue","sum"), total_recette=("recette","sum"),
                 total_cout=("cout_mp","sum"), total_marge=("marge_brute","sum"))
            .reset_index()
        )
        prod_fin["marge_pct"]  = (prod_fin["total_marge"] / prod_fin["total_recette"].replace(0, np.nan) * 100).fillna(0)
        prod_fin["marge_unit"] = (prod_fin["total_marge"] / prod_fin["total_vendu"].replace(0, np.nan)).fillna(0)

        if prices_set and not prod_fin.empty:
            best_seller = prod_fin.loc[prod_fin["total_vendu"].idxmax(), "produit"]
            best_margin = prod_fin.loc[prod_fin["total_marge"].idxmax(), "produit"]
            bm1, bm2 = st.columns(2)
            bm1.markdown(
                _fin_kpi(T("fin_best_seller"), best_seller,
                         f"{int(prod_fin.loc[prod_fin['produit']==best_seller,'total_vendu'].iloc[0]):,} {T('units')}", "🏆"),
                unsafe_allow_html=True,
            )
            bm2.markdown(
                _fin_kpi(T("fin_best_margin"), best_margin,
                         _mad(prod_fin.loc[prod_fin['produit']==best_margin,'total_marge'].iloc[0]), "💰"),
                unsafe_allow_html=True,
            )
            fig_bub = px.scatter(
                prod_fin, x="total_vendu", y="marge_unit",
                size="total_recette", color="produit", text="produit",
                labels={"total_vendu": T("fin_bubble_x"), "marge_unit": T("fin_bubble_y"),
                        "total_recette": T("fin_bubble_size")},
                color_discrete_sequence=["#14b8a6","#1e3a5f","#0ea5e9","#6366f1","#f59e0b","#22c55e"],
                size_max=60,
            )
            fig_bub.update_traces(textposition="top center", textfont_size=11)
            fig_bub.update_layout(template="plotly_white", height=340, margin=dict(t=20, b=20), showlegend=False)
            st.plotly_chart(fig_bub, use_container_width=True)

            disp = prod_fin.copy()
            disp["total_recette"] = disp["total_recette"].map(_mad)
            disp["total_cout"]    = disp["total_cout"].map(_mad)
            disp["total_marge"]   = disp["total_marge"].map(_mad)
            disp["marge_pct"]     = disp["marge_pct"].map(lambda x: f"{x:.1f}%")
            disp["marge_unit"]    = disp["marge_unit"].map(lambda x: f"{x:.2f} MAD")
            disp.columns = Tlist("fin_table_cols") or disp.columns.tolist()
            st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)
        else:
            st.info(T("fin_no_price_info"))

    # ── P&L mensuel ──────────────────────────────────────────────────────────
    with st.expander(T("fin_monthly_section"), expanded=False):
        monthly_fin = (
            daily_fin.copy()
            .assign(mois=lambda d: d["date"].dt.to_period("M"))
            .groupby("mois")
            .agg(recette=("recette","sum"), cout_mp=("cout_mp","sum"), marge=("marge_brute","sum"))
            .reset_index()
        )
        monthly_fin["mois_label"] = monthly_fin["mois"].dt.strftime("%b %Y")

        if prices_set and not monthly_fin.empty:
            fig_mo = go.Figure()
            fig_mo.add_trace(go.Bar(x=monthly_fin["mois_label"], y=monthly_fin["recette"],
                                    name=T("fin_rev_series"), marker_color="#22c55e"))
            fig_mo.add_trace(go.Bar(x=monthly_fin["mois_label"], y=monthly_fin["cout_mp"],
                                    name=T("fin_mp_series"), marker_color="#f97316"))
            fig_mo.add_trace(go.Scatter(x=monthly_fin["mois_label"], y=monthly_fin["marge"],
                                        mode="lines+markers", name=T("fin_margin_series"),
                                        line=dict(color="#1e3a5f", width=2), marker=dict(size=7)))
            fig_mo.update_layout(barmode="group", template="plotly_white", height=340,
                                  margin=dict(t=20, b=20), xaxis_title="", yaxis_title="MAD",
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_mo, use_container_width=True)

            _sec(T("fin_profit_section"))

            def _style_marge(val: str) -> str:
                try:
                    num = float(val.replace("%", ""))
                    if num >= 40: return "color:#166534;font-weight:700"
                    if num >= 20: return "color:#854d0e;font-weight:700"
                    return "color:#991b1b;font-weight:700"
                except Exception:
                    return ""

            fmc = Tlist("fin_month_cols")
            cols = fmc if len(fmc) == 5 else ["Mois","CA (MAD)","MP (MAD)","Marge (MAD)","Marge %"]
            fm_disp = pd.DataFrame({
                cols[0]: monthly_fin["mois_label"],
                cols[1]: monthly_fin["recette"].map(_mad),
                cols[2]: monthly_fin["cout_mp"].map(_mad),
                cols[3]: monthly_fin["marge"].map(_mad),
                cols[4]: (monthly_fin["marge"] / monthly_fin["recette"].replace(0, np.nan) * 100).fillna(0).map(lambda x: f"{x:.1f}%"),
            })
            st.dataframe(
                fm_disp.reset_index(drop=True).style.map(_style_marge, subset=[cols[4]]),
                use_container_width=True, hide_index=True,
            )

            total_ca    = daily_fin["recette"].sum()
            total_cout  = daily_fin["cout_mp"].sum()
            total_marge = daily_fin["marge_brute"].sum()
            total_pct   = (total_marge / total_ca * 100) if total_ca > 0 else 0
            st.markdown(
                f'<div class="sum-box">'
                f'<span class="sum-big">{_mad(total_marge)}</span>'
                f'<span class="sum-sub">{T("fin_total_label")} &nbsp;|&nbsp; '
                f'{T("fin_total_sub").format(ca=_mad(total_ca), cout=_mad(total_cout), pct=total_pct)}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info(T("fin_no_config_info"))

    st.divider()
    st.caption(T("fin_footer"))
