import pandas as pd


def compute_stockout_date(
    current_stock: float,
    daily_qtys: list[float],
    dates: list,
) -> str:
    """Day-by-day consumption simulation. Returns first date stock hits ≤ 0,
    or 'OK (>N days)' if stock lasts the full period."""
    stock = float(current_stock)
    for date, qty in zip(dates, daily_qtys):
        stock -= max(float(qty), 0)
        if stock <= 0:
            return pd.Timestamp(date).strftime("%Y-%m-%d")
    return f"OK (>{len(dates)} days)"
