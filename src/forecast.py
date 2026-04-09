"""
forecast.py
Train a Facebook Prophet model on 2023 daily sales and forecast the
30 days following the last date in the dataset (i.e. beyond 2024-12-31).

Usage:
    python src/forecast.py          # saves data/forecast.csv, shows chart
    from src.forecast import run    # returns (forecast_df, fig) for notebook use
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

DATA_PATH = Path(__file__).parent.parent / "data" / "ventes.csv"
OUT_PATH  = Path(__file__).parent.parent / "data" / "forecast.csv"
FORECAST_HORIZON = 30  # days


def load_and_aggregate(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load ventes.csv and return total daily sales in Prophet's ds/y format."""
    df = pd.read_csv(path, parse_dates=["date"])
    daily = (
        df.groupby("date")["quantite_vendue"]
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "quantite_vendue": "y"})
    )
    return daily.sort_values("ds").reset_index(drop=True)


def train(daily: pd.DataFrame) -> Prophet:
    """Fit Prophet on the 2023 slice only."""
    train_df = daily[daily["ds"].dt.year == 2023].copy()
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # matches the +% effects in the data
        interval_width=0.95,
    )
    model.fit(train_df)
    return model


def forecast(model: Prophet, daily: pd.DataFrame) -> pd.DataFrame:
    """Build a future dataframe starting from the day after the last known date."""
    last_date = daily["ds"].max()
    future = model.make_future_dataframe(
        periods=FORECAST_HORIZON,
        freq="D",
        include_history=True,
    )
    # make_future_dataframe starts from the training end (2023-12-31);
    # extend it manually so it covers all of 2024 + 30 days beyond
    extra_days = pd.date_range(
        start=pd.Timestamp("2024-01-01"),
        end=last_date + pd.Timedelta(days=FORECAST_HORIZON),
        freq="D",
    )
    future = (
        pd.concat([future, pd.DataFrame({"ds": extra_days})])
        .drop_duplicates("ds")
        .sort_values("ds")
        .reset_index(drop=True)
    )
    pred = model.predict(future)
    return pred[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def build_figure(daily: pd.DataFrame, pred: pd.DataFrame) -> go.Figure:
    """Plotly figure: historical actuals + forecast + 95% CI ribbon."""
    last_train = pd.Timestamp("2023-12-31")
    last_actual = daily["ds"].max()
    forecast_start = last_actual + pd.Timedelta(days=1)

    hist = daily.copy()
    pred_future = pred[pred["ds"] >= forecast_start]
    pred_all    = pred.copy()

    fig = go.Figure()

    # 95% confidence interval ribbon (full range)
    fig.add_trace(go.Scatter(
        x=pd.concat([pred_all["ds"], pred_all["ds"][::-1]]),
        y=pd.concat([pred_all["yhat_upper"], pred_all["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% CI",
    ))

    # Prophet fitted line (full period including 2024)
    fig.add_trace(go.Scatter(
        x=pred_all["ds"],
        y=pred_all["yhat"],
        mode="lines",
        line=dict(color="orange", width=1.5, dash="dot"),
        name="Prophet fit / forecast",
    ))

    # Actual historical values
    fig.add_trace(go.Scatter(
        x=hist["ds"],
        y=hist["y"],
        mode="lines",
        line=dict(color="#2196F3", width=1),
        name="Actual sales",
    ))

    # Vertical line marking training / forecast boundary
    for xval, label, color in [
        (last_train,    "Train end (2023-12-31)", "green"),
        (forecast_start, f"Forecast start ({forecast_start.date()})", "red"),
    ]:
        fig.add_vline(
            x=xval.timestamp() * 1000,
            line=dict(color=color, dash="dash", width=1.5),
            annotation_text=label,
            annotation_position="top right",
        )

    fig.update_layout(
        title="Sales Forecast — Prophet model (trained on 2023, forecast +30 days)",
        xaxis_title="Date",
        yaxis_title="Total Units Sold (all products)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def run(show: bool = False):
    """
    Full pipeline: load → train → forecast → save → (optionally) show chart.
    Returns (forecast_df, fig) for use in notebooks.
    """
    daily = load_and_aggregate()
    model = train(daily)
    pred  = forecast(model, daily)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(OUT_PATH, index=False)
    print(f"Forecast saved to {OUT_PATH}  ({len(pred)} rows)")
    print(pred.tail(10).to_string(index=False))

    fig = build_figure(daily, pred)
    if show:
        fig.show()
    return pred, fig


if __name__ == "__main__":
    run(show=True)
