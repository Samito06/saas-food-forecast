"""
logic/forecasting.py — Prophet training, prediction, MAE, holiday helpers, peak detection.
"""
import numpy as np
import pandas as pd
import streamlit as st
from prophet import Prophet

MAROC_FIXED_HOLIDAYS = [
    (1,  1,  "Jour de l'An"),
    (1,  11, "Manifeste de l'Indépendance"),
    (5,  1,  "Fête du Travail"),
    (7,  30, "Fête du Trône"),
    (8,  14, "Allégeance des Provinces du Sud"),
    (8,  20, "Révolution du Roi et du Peuple"),
    (8,  21, "Fête de la Jeunesse"),
    (11, 6,  "Marche Verte"),
    (11, 18, "Fête de l'Indépendance"),
]

ISLAMIC_HOLIDAYS: dict[int, list[tuple[str, str]]] = {
    2022: [("2022-05-02","Aïd al-Fitr"),("2022-05-03","Aïd al-Fitr J2"),
           ("2022-07-09","Aïd al-Adha"),("2022-07-10","Aïd al-Adha J2"),
           ("2022-07-30","Aïd al-Hijra"),("2022-10-08","Mawlid")],
    2023: [("2023-04-21","Aïd al-Fitr"),("2023-04-22","Aïd al-Fitr J2"),
           ("2023-06-28","Aïd al-Adha"),("2023-06-29","Aïd al-Adha J2"),
           ("2023-07-19","Aïd al-Hijra"),("2023-09-27","Mawlid")],
    2024: [("2024-04-10","Aïd al-Fitr"),("2024-04-11","Aïd al-Fitr J2"),
           ("2024-06-17","Aïd al-Adha"),("2024-06-18","Aïd al-Adha J2"),
           ("2024-07-08","Aïd al-Hijra"),("2024-09-15","Mawlid")],
    2025: [("2025-03-31","Aïd al-Fitr"),("2025-04-01","Aïd al-Fitr J2"),
           ("2025-06-07","Aïd al-Adha"),("2025-06-08","Aïd al-Adha J2"),
           ("2025-06-26","Aïd al-Hijra"),("2025-09-04","Mawlid")],
    2026: [("2026-03-20","Aïd al-Fitr"),("2026-03-21","Aïd al-Fitr J2"),
           ("2026-05-27","Aïd al-Adha"),("2026-05-28","Aïd al-Adha J2"),
           ("2026-06-16","Aïd al-Hijra"),("2026-08-25","Mawlid")],
    2027: [("2027-03-09","Aïd al-Fitr"),("2027-03-10","Aïd al-Fitr J2"),
           ("2027-05-16","Aïd al-Adha"),("2027-05-17","Aïd al-Adha J2"),
           ("2027-06-06","Aïd al-Hijra"),("2027-08-14","Mawlid")],
}

RAMADAN_DATES: dict[int, tuple[str, str]] = {
    2022: ("2022-04-02", "2022-05-01"),
    2023: ("2023-03-23", "2023-04-20"),
    2024: ("2024-03-11", "2024-04-09"),
    2025: ("2025-03-01", "2025-03-29"),
    2026: ("2026-02-18", "2026-03-18"),
    2027: ("2027-02-07", "2027-03-08"),
}

WEEKEND_DAYS  = {"Friday", "Saturday"}
PEAK_THRESH   = 1.30


def get_moroccan_holidays(years: list[int]) -> pd.DataFrame:
    rows: list[dict] = []
    for year in years:
        for month, day, name in MAROC_FIXED_HOLIDAYS:
            try:
                rows.append({"ds": pd.Timestamp(year, month, day), "holiday": name})
            except ValueError:
                pass
        for date_str, name in ISLAMIC_HOLIDAYS.get(year, []):
            rows.append({"ds": pd.Timestamp(date_str), "holiday": name})
    return pd.DataFrame(rows).drop_duplicates("ds").reset_index(drop=True)


def _get_holiday_name(date: pd.Timestamp) -> str | None:
    for month, day, name in MAROC_FIXED_HOLIDAYS:
        if date.month == month and date.day == day:
            return name
    for date_str, name in ISLAMIC_HOLIDAYS.get(date.year, []):
        if pd.Timestamp(date_str).date() == date.date():
            return name
    return None


def is_ramadan_period(date: pd.Timestamp) -> bool:
    for start_str, end_str in RAMADAN_DATES.values():
        if pd.Timestamp(start_str) <= date <= pd.Timestamp(end_str):
            return True
    return False


def get_upcoming_peaks(
    pred_future: pd.DataFrame, avg: float, n_days: int = 7
) -> list[dict]:
    peaks = []
    for _, row in pred_future.head(n_days).iterrows():
        reasons: list[str] = []
        if row["ds"].day_name() in WEEKEND_DAYS:
            reasons.append("week-end")
        ratio = row["yhat"] / max(avg, 1)
        if ratio > PEAK_THRESH:
            reasons.append(f"+{(ratio - 1) * 100:.0f}% vs moyenne")
        holiday = _get_holiday_name(row["ds"])
        if holiday:
            reasons.append(f"Fête : {holiday}")
        if is_ramadan_period(row["ds"]):
            reasons.append("Ramadan")
        if reasons:
            peaks.append({"date": row["ds"], "yhat": int(max(row["yhat"], 0).round()), "reasons": reasons})
    return peaks


@st.cache_resource(show_spinner=False)
def get_model(data_hash: str, _daily: pd.DataFrame) -> Prophet:
    years = list(range(_daily["ds"].dt.year.min(), _daily["ds"].dt.year.max() + 3))
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.95,
        holidays=get_moroccan_holidays(years),
    )
    m.fit(_daily)
    return m


@st.cache_resource(show_spinner=False)
def get_product_model(
    data_hash: str, product: str, _prod_daily: pd.DataFrame
) -> Prophet:
    years = list(range(_prod_daily["ds"].dt.year.min(), _prod_daily["ds"].dt.year.max() + 3))
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.90,
        holidays=get_moroccan_holidays(years),
    )
    m.fit(_prod_daily)
    return m


@st.cache_data(show_spinner=False)
def run_forecast(
    data_hash: str, horizon: int, _model: Prophet, _daily: pd.DataFrame
) -> pd.DataFrame:
    future = pd.DataFrame({
        "ds": pd.date_range(
            _daily["ds"].min(),
            _daily["ds"].max() + pd.Timedelta(days=horizon),
            freq="D",
        )
    })
    pred = _model.predict(future)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(show_spinner=False)
def run_product_forecast(
    data_hash: str, product: str, horizon: int,
    _model: Prophet, _prod_daily: pd.DataFrame,
) -> pd.DataFrame:
    future = pd.DataFrame({
        "ds": pd.date_range(
            _prod_daily["ds"].min(),
            _prod_daily["ds"].max() + pd.Timedelta(days=horizon),
            freq="D",
        )
    })
    pred = _model.predict(future)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


@st.cache_data(show_spinner=False)
def compute_mae(data_hash: str, _daily: pd.DataFrame) -> float | None:
    if len(_daily) < 61:
        return None
    cutoff = len(_daily) - 30
    train  = _daily.iloc[:cutoff].copy()
    test   = _daily.iloc[cutoff:].copy()
    m = Prophet(
        yearly_seasonality=True, weekly_seasonality=True,
        daily_seasonality=False, seasonality_mode="multiplicative",
    )
    m.fit(train)
    pred = m.predict(test[["ds"]])
    return round(float(np.mean(np.abs(pred["yhat"].values - test["y"].values))), 1)
