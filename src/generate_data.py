"""
generate_data.py
Generate 2 years of synthetic daily sales data (2023-2024) for a small restaurant.

Effects modelled:
  - Weekend boost: +40% on Friday (weekday=4) and Saturday (weekday=5)
  - Summer peak: +20% in July and August
  - Poisson noise around the daily mean
  - Closed days: ~2 random days per month with zero sales for all products
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
RNG = np.random.default_rng(seed=42)

# ── products and their baseline daily mean quantities ────────────────────────
PRODUCTS = {
    "Pizza Margherita": 30,
    "Chicken Sandwich": 25,
    "Burger": 20,
    "Caesar Salad": 15,
}

WEEKEND_DAYS = {4, 5}   # Friday=4, Saturday=5 (Monday=0)
SUMMER_MONTHS = {7, 8}  # July, August
CLOSED_PER_MONTH = 2    # average number of zero-sales days per month


def pick_closed_days(dates: pd.DatetimeIndex) -> set:
    """Randomly select ~2 closed days per calendar month."""
    closed = set()
    for (year, month), group in dates.to_frame().groupby(
        [dates.year, dates.month]
    ):
        day_indices = group.index.tolist()
        n_closed = min(CLOSED_PER_MONTH, len(day_indices))
        chosen = RNG.choice(day_indices, size=n_closed, replace=False)
        closed.update(chosen)
    return closed


def daily_mean(base: float, date: pd.Timestamp) -> float:
    """Compute the adjusted mean for a given product and date."""
    mean = base
    if date.weekday() in WEEKEND_DAYS:
        mean *= 1.40
    if date.month in SUMMER_MONTHS:
        mean *= 1.20
    return mean


def generate() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    closed_days = pick_closed_days(dates)

    records = []
    for date in dates:
        for product, base_mean in PRODUCTS.items():
            if date in closed_days:
                qty = 0
            else:
                lam = daily_mean(base_mean, date)
                qty = int(RNG.poisson(lam))
            records.append(
                {"date": date.strftime("%Y-%m-%d"), "produit": product, "quantite_vendue": qty}
            )

    return pd.DataFrame(records, columns=["date", "produit", "quantite_vendue"])


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "ventes.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate()
    df.to_csv(output_path, index=False)

    # ── quick sanity check ───────────────────────────────────────────────────
    total_rows = len(df)
    date_range = f"{df['date'].min()} to {df['date'].max()}"
    zero_days = (df["quantite_vendue"] == 0).sum()
    print(f"Saved {total_rows:,} rows to {output_path}")
    print(f"Date range : {date_range}")
    print(f"Zero-sales rows : {zero_days}")
    print("\nMean daily sales by product:")
    print(
        df[df["quantite_vendue"] > 0]
        .groupby("produit")["quantite_vendue"]
        .mean()
        .round(1)
        .to_string()
    )
