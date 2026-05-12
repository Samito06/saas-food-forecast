"""
logic/insights.py — scoring, promo detection, dormant detection, risk level.
Pure Pandas, no Streamlit.
"""
import pandas as pd


PEAK_THRESH = 1.30
WEEKEND_DAYS = {"Friday", "Saturday"}


def _risk_level(row: pd.Series, avg: float) -> str:
    ratio = row["yhat"] / max(avg, 1)
    if ratio > PEAK_THRESH:
        return "High"
    if ratio > 1.10 or row.get("is_weekend", False):
        return "Medium"
    return "Low"


def score_products_of_day(
    active_csv: list[str],
    prod_daily: pd.DataFrame,
    last_date: pd.Timestamp,
    dow_today: str,
    price_map: dict,
    cost_map: dict,
) -> dict[str, float]:
    """
    Return {product: score} for all active CSV products.
    score = 0.4 × dow_ratio + 0.4 × trend_ratio + 0.2 × margin_ratio  (all in [0,∞))
    """
    prod_global_avg = prod_daily.groupby("produit")["quantite_vendue"].mean()
    cut1 = last_date - pd.Timedelta(days=30)
    cut0 = last_date - pd.Timedelta(days=60)

    max_margin = max(
        [max(price_map.get(p, 0) - cost_map.get(p, 0), 0) for p in active_csv],
        default=1,
    )

    scores: dict[str, float] = {}
    for p in active_csv:
        dow_data = prod_daily[
            (prod_daily["produit"] == p) & (prod_daily["dow"] == dow_today)
        ]["quantite_vendue"]
        dow_avg    = float(dow_data.mean()) if len(dow_data) > 0 else 0.0
        global_p   = float(prod_global_avg.get(p, 1))
        dow_ratio  = dow_avg / max(global_p, 1)

        last30 = float(
            prod_daily[(prod_daily["produit"] == p) & (prod_daily["date"] > cut1)][
                "quantite_vendue"
            ].mean() or 0
        )
        prev30 = float(
            prod_daily[
                (prod_daily["produit"] == p)
                & (prod_daily["date"] > cut0)
                & (prod_daily["date"] <= cut1)
            ]["quantite_vendue"].mean() or 1
        )
        trend_ratio = last30 / max(prev30, 1)

        margin     = max(price_map.get(p, 0.0) - cost_map.get(p, 0.0), 0.0)
        margin_ratio = margin / max(max_margin, 1)

        scores[p] = round(0.4 * dow_ratio + 0.4 * trend_ratio + 0.2 * margin_ratio, 4)
    return scores


def find_declining_products(
    active_csv: list[str],
    prod_daily: pd.DataFrame,
    last_date: pd.Timestamp,
    threshold_pct: float,
) -> list[tuple[str, float, float, float]]:
    """Return [(product, drop_pct, last7_avg, last30_avg)] for products in decline."""
    cut7  = last_date - pd.Timedelta(days=7)
    cut30 = last_date - pd.Timedelta(days=30)
    result = []
    for p in active_csv:
        l7  = float(prod_daily[(prod_daily["produit"] == p) & (prod_daily["date"] > cut7)]["quantite_vendue"].mean() or 0)
        l30 = float(prod_daily[(prod_daily["produit"] == p) & (prod_daily["date"] > cut30) & (prod_daily["date"] <= cut7)]["quantite_vendue"].mean() or 0)
        if l30 > 0:
            drop = (l30 - l7) / l30 * 100
            if drop >= threshold_pct:
                result.append((p, drop, l7, l30))
    return sorted(result, key=lambda x: x[1], reverse=True)


def find_dormant_products(
    active_csv: list[str],
    prod_daily: pd.DataFrame,
    last_date: pd.Timestamp,
    dormant_days: int,
) -> list[tuple[str, int, pd.Timestamp | None]]:
    """Return [(product, days_since_sale, last_sale_date)] for dormant products."""
    last_sold = prod_daily.groupby("produit")["date"].max()
    result = []
    for p in active_csv:
        if p in last_sold.index:
            days_since = (last_date - last_sold[p]).days
            if days_since >= dormant_days:
                result.append((p, days_since, last_sold[p]))
        else:
            result.append((p, 999, None))
    return sorted(result, key=lambda x: x[1], reverse=True)
