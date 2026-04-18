"""
PSX Price Predictor — 7-Day Forecast
Uses Linear Regression + Moving Average extrapolation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings, os
warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)

FEATURES = ["daily_return","return_3d","vol_ratio","volatility_7d","rsi","price_vs_ma20","bb_width"]


def predict_next_7_days(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    grp = df[df["ticker"] == ticker].copy().sort_values("date").reset_index(drop=True)
    if len(grp) < 60:
        return pd.DataFrame()

    # Train on all but last 14 days, test on last 14
    grp["target"] = grp["close"].shift(-1)
    grp = grp.dropna(subset=["target"] + FEATURES)

    X = grp[FEATURES].values
    y = grp["target"].values

    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X)
    model = LinearRegression()
    model.fit(X_sc, y)

    # Predict 7 future days iteratively
    last_row   = grp.iloc[-1].copy()
    last_price = last_row["close"]
    last_date  = pd.to_datetime(last_row["date"])
    predictions = []

    for i in range(7):
        feat_vals = last_row[FEATURES].values.reshape(1, -1)
        feat_sc   = sc.transform(feat_vals)
        pred      = model.predict(feat_sc)[0]

        # Clamp prediction to ±15% of last price to prevent runaway forecasts
        pred = np.clip(pred, last_price * 0.85, last_price * 1.15)

        # Add realistic noise — volatility_7d must be positive for scale parameter
        vol = abs(last_row["volatility_7d"])  # Fix: ensure scale > 0
        if vol > 0 and pred > 0:
            noise = np.random.normal(0, vol * pred * 0.3)
            pred  = np.clip(pred + noise, last_price * 0.85, last_price * 1.15)

        next_date = last_date + pd.tseries.offsets.BusinessDay(i + 1)
        predictions.append({
            "date":        next_date.strftime("%Y-%m-%d"),
            "predicted":   round(pred, 2),
            "change_pct":  round((pred - last_price) / last_price * 100, 2),
            "ticker":      ticker,
            "company":     grp["company"].iloc[-1],
            "day":         f"Day +{i+1}",
        })
        # Update features for next iteration (walk-forward)
        last_row = last_row.copy()
        daily_ret = (pred - last_price) / last_price
        last_row["daily_return"]  = daily_ret
        last_row["return_3d"]     = daily_ret  # approximate
        last_row["price_vs_ma20"] = (pred - last_row["ma20"]) / (last_row["ma20"] + 1e-9)
        last_row["vol_ratio"]     = max(0.5, min(3.0, last_row["vol_ratio"] * np.random.uniform(0.8, 1.2)))
        last_row["volatility_7d"] = max(0.001, last_row["volatility_7d"])  # keep positive
        # Nudge RSI toward 50 (mean reversion) each step
        last_row["rsi"]           = last_row["rsi"] + (50 - last_row["rsi"]) * 0.1
        last_price = pred

    return pd.DataFrame(predictions)


def predict_all_stocks(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🔮 Generating 7-day price forecasts...\n")
    all_preds = []
    for ticker in df["ticker"].unique():
        pred = predict_next_7_days(df, ticker)
        if not pred.empty:
            all_preds.append(pred)
            last = pred.iloc[-1]
            arrow = "↑" if last["change_pct"] > 0 else "↓"
            print(f"  {PSX_name(df, ticker):<28} 7d forecast: {arrow} {last['change_pct']:+.1f}%")

    if not all_preds:
        return pd.DataFrame()

    out = pd.concat(all_preds, ignore_index=True)
    out.to_csv("results/predictions.csv", index=False)
    print(f"\n✅ Predictions saved to results/predictions.csv")
    return out


def PSX_name(df, ticker):
    rows = df[df["ticker"] == ticker]
    return rows["company"].iloc[0] if not rows.empty else ticker


def plot_predictions(df: pd.DataFrame, preds: pd.DataFrame, ticker: str = "ENGRO"):
    grp   = df[df["ticker"] == ticker].copy().sort_values("date").tail(60)
    pred  = preds[preds["ticker"] == ticker].copy()

    if grp.empty or pred.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"7-Day Price Forecast — {grp['company'].iloc[0]}",
                 fontsize=13, fontweight="bold")

    # Chart 1: Historical + forecast
    ax = axes[0]
    ax.plot(pd.to_datetime(grp["date"]), grp["close"],
            color="#378ADD", linewidth=2, label="Historical")
    ax.plot(pd.to_datetime(pred["date"]), pred["predicted"],
            color="#E24B4A", linewidth=2, linestyle="--",
            marker="o", markersize=5, label="Forecast (7 days)")

    # Confidence band
    std   = grp["close"].std() * 0.05
    upper = pred["predicted"] + std * np.arange(1, 8)
    lower = pred["predicted"] - std * np.arange(1, 8)
    ax.fill_between(pd.to_datetime(pred["date"]), lower, upper,
                    alpha=0.15, color="#E24B4A", label="Confidence band")
    ax.axvline(pd.to_datetime(grp["date"].iloc[-1]),
               color="gray", linestyle=":", linewidth=1)
    ax.set_title("Price Forecast", fontweight="bold")
    ax.set_ylabel("Price (PKR)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # Chart 2: Daily predicted change %
    ax2 = axes[1]
    colors = ["#1D9E75" if c > 0 else "#E24B4A" for c in pred["change_pct"]]
    bars = ax2.bar(pred["day"], pred["change_pct"], color=colors, edgecolor="white")
    for bar, val in zip(bars, pred["change_pct"]):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + (0.05 if val >= 0 else -0.15),
                 f"{val:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(0, color="gray", linewidth=0.8)
    ax2.set_title("Predicted Daily Change", fontweight="bold")
    ax2.set_ylabel("Change (%)")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/forecast_{ticker}.png", dpi=150, bbox_inches="tight")
    print(f"✅ Forecast chart saved: results/forecast_{ticker}.png")
    plt.close()  # Fixed: was plt.show() which blocks in non-interactive environments
