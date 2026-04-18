"""
PSX Visualizer
Generates and saves dashboard charts for each stock.
Uses matplotlib with non-interactive Agg backend (no Tk window pop-ups).
"""

import matplotlib
matplotlib.use("Agg")   # Must be before pyplot import — prevents Tk blocking
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import os

os.makedirs("results", exist_ok=True)

BLUE   = "#378ADD"
RED    = "#E24B4A"
GREEN  = "#1D9E75"
AMBER  = "#BA7517"
PURPLE = "#534AB7"
GRAY   = "#AAAAAA"


def plot_dashboard(df: pd.DataFrame, ticker: str):
    """Save a 4-panel dashboard PNG for one ticker."""
    grp = df[df["ticker"] == ticker].copy().sort_values("date").tail(180)
    grp["date"] = pd.to_datetime(grp["date"])

    if grp.empty:
        return

    company = grp["company"].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{company} ({ticker}) — PSX Anomaly Dashboard",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: Price + signals ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(grp["date"], grp["close"], color=BLUE, lw=2, label="Close")
    if "ma20" in grp.columns:
        ax.plot(grp["date"], grp["ma20"], color=GRAY, lw=1, ls="--",
                alpha=0.8, label="MA-20")
    if "bb_upper" in grp.columns:
        ax.fill_between(grp["date"], grp["bb_lower"], grp["bb_upper"],
                        alpha=0.08, color=BLUE, label="Bollinger Bands")

    if "if_anomaly" in grp.columns:
        an = grp[grp["if_anomaly"] == 1]
        ax.scatter(an["date"], an["close"], color=RED, s=50, zorder=5,
                   marker="v", label="IF Anomaly")
    if "pump_signal" in grp.columns:
        pu = grp[grp["pump_signal"] == 1]
        ax.scatter(pu["date"], pu["close"], color=AMBER, s=70, zorder=6,
                   marker="^", label="Pump")
    if "dump_signal" in grp.columns:
        du = grp[grp["dump_signal"] == 1]
        ax.scatter(du["date"], du["close"], color=PURPLE, s=70, zorder=6,
                   marker="v", label="Dump")

    ax.set_title("Price & Anomaly Signals", fontweight="bold")
    ax.set_ylabel("Price (PKR)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)

    # ── Panel 2: Volume ratio ─────────────────────────────────────────────────
    ax2 = axes[0, 1]
    vol_thresh = 2.5
    colors = [RED if v > vol_thresh else BLUE for v in grp["vol_ratio"]]
    ax2.bar(grp["date"], grp["vol_ratio"], color=colors, width=1, alpha=0.8)
    ax2.axhline(vol_thresh, color=RED, ls="--", lw=1.2,
                label=f"Spike threshold ({vol_thresh}x)")
    ax2.set_title("Volume Ratio (vs 10-day avg)", fontweight="bold")
    ax2.set_ylabel("Volume Ratio")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.tick_params(axis="x", rotation=30)

    # ── Panel 3: RSI ──────────────────────────────────────────────────────────
    ax3 = axes[1, 0]
    ax3.plot(grp["date"], grp["rsi"], color=PURPLE, lw=1.5)
    ax3.axhline(70, color=RED,   ls="--", lw=1, label="Overbought (70)")
    ax3.axhline(30, color=GREEN, ls="--", lw=1, label="Oversold (30)")
    ax3.fill_between(grp["date"], grp["rsi"], 70,
                     where=(grp["rsi"] > 70), alpha=0.2, color=RED)
    ax3.fill_between(grp["date"], grp["rsi"], 30,
                     where=(grp["rsi"] < 30), alpha=0.2, color=GREEN)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI (14-day)", fontweight="bold")
    ax3.set_ylabel("RSI")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    ax3.tick_params(axis="x", rotation=30)

    # ── Panel 4: Returns distribution ─────────────────────────────────────────
    ax4 = axes[1, 1]
    ret_data = grp["daily_return"].dropna() * 100
    ax4.hist(ret_data, bins=30, color=BLUE, alpha=0.75, edgecolor="white")
    ax4.axvline(ret_data.mean(), color=RED, ls="--", lw=1.5,
                label=f"Mean {ret_data.mean():.2f}%")
    ax4.axvline(ret_data.mean() + 2 * ret_data.std(), color=AMBER,
                ls=":", lw=1, label="+2σ")
    ax4.axvline(ret_data.mean() - 2 * ret_data.std(), color=AMBER,
                ls=":", lw=1, label="-2σ")
    ax4.set_title("Daily Returns Distribution", fontweight="bold")
    ax4.set_xlabel("Return (%)")
    ax4.set_ylabel("Frequency")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"results/dashboard_{ticker}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Dashboard saved: {fname}")


def plot_all(df: pd.DataFrame):
    """Generate and save dashboards for every ticker in the dataframe."""
    print("\n📊 Generating visualizations...\n")
    for ticker in sorted(df["ticker"].unique()):
        plot_dashboard(df, ticker)
