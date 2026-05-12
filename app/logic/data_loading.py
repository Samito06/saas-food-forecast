"""
logic/data_loading.py — CSV/Excel loading, validation, aggregation, health checks.
"""
import io
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT        = Path(__file__).parent.parent.parent
DEFAULT_CSV = ROOT / "data" / "boulangerie.csv"
REQUIRED_COLS = {"date", "produit", "quantite_vendue"}


@st.cache_data(show_spinner=False)
def load_and_validate(
    file_bytes: bytes | None, file_ext: str = "csv"
) -> tuple[pd.DataFrame | None, list[str]]:
    """Load CSV or Excel, run validation checks. Returns (df_or_None, error_list)."""
    errors: list[str] = []
    try:
        if file_bytes is None:
            df = pd.read_csv(DEFAULT_CSV, parse_dates=["date"])
        elif file_ext in ("xlsx", "xls"):
            df = pd.read_excel(
                io.BytesIO(file_bytes),
                engine="openpyxl" if file_ext == "xlsx" else None,
            )
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df = pd.read_csv(io.BytesIO(file_bytes), parse_dates=["date"])

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            errors.append(
                f"Your file is missing these required columns: **{', '.join(sorted(missing))}**. "
                "The file must have exactly these column names: `date`, `produit`, `quantite_vendue`. "
                "Download the sample file on the About page to see the correct format."
            )
            return None, errors

        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            errors.append(
                "The `date` column couldn't be read. Please use the format `YYYY-MM-DD` (e.g. 2024-01-15)."
            )
            return None, errors

        if not pd.api.types.is_numeric_dtype(df["quantite_vendue"]):
            errors.append(
                "The `quantite_vendue` column must contain numbers only (e.g. 28, not 'twenty-eight')."
            )
            return None, errors

        n_days = df["date"].nunique()
        if n_days < 30:
            errors.append(
                f"Your file only has **{n_days} days** of data. "
                "At least 30 days are needed to make reliable predictions."
            )
            return None, errors

        return df, errors

    except Exception as exc:
        errors.append(f"Could not read your file. Make sure it is a valid CSV file. Details: {exc}")
        return None, errors


@st.cache_data(show_spinner=False)
def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("date")["quantite_vendue"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "quantite_vendue": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def aggregate_by_product(df: pd.DataFrame, product: str) -> pd.DataFrame:
    return (
        df[df["produit"] == product]
        .groupby("date")["quantite_vendue"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "quantite_vendue": "y"})
        .sort_values("ds")
        .reset_index(drop=True)
    )


def check_data_health(df: pd.DataFrame) -> list[dict]:
    checks = []

    nulls = int(df.isnull().sum().sum())
    checks.append({"label": "Missing values",   "status": "ok" if nulls == 0 else "warn",
                   "detail": "None detected" if nulls == 0 else f"{nulls} null values"})

    neg = int((df["quantite_vendue"] < 0).sum())
    checks.append({"label": "Negative quantities", "status": "ok" if neg == 0 else "warn",
                   "detail": "None" if neg == 0 else f"{neg} rows with qty < 0"})

    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    gaps = len(expected) - df["date"].dt.date.nunique()
    checks.append({"label": "Date continuity", "status": "ok" if gaps == 0 else "warn",
                   "detail": "No gaps" if gaps == 0 else f"{gaps} missing dates"})

    n_prod = df["produit"].nunique()
    checks.append({"label": "Products detected", "status": "ok",
                   "detail": f"{n_prod} unique product(s): " + ", ".join(sorted(df["produit"].unique()))})

    zero = int((df.groupby("date")["quantite_vendue"].sum() == 0).sum())
    checks.append({"label": "Zero-sales days (closed)", "status": "ok",
                   "detail": f"{zero} day(s) with total sales = 0"})

    return checks


def make_sample_csv() -> bytes:
    rows = []
    products = ["Pizza Margherita", "Chicken Sandwich", "Burger", "Caesar Salad"]
    qtys     = [28, 22, 17, 13]
    for i in range(5):
        date_str = f"2024-01-0{i + 1}"
        for p, q in zip(products, qtys):
            rows.append({"date": date_str, "produit": p, "quantite_vendue": q + i})
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
