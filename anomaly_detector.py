"""
PSX Anomaly Detection Engine
Runs Isolation Forest, Z-Score, and Pump & Dump detection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

FEATURES = ["daily_return", "return_3d", "vol_ratio", "volatility_7d",
            "rsi", "price_vs_ma20", "bb_width"]

PUMP_RETURN_THRESH  = 0.08   # 8% daily return  → pump signal
DUMP_RETURN_THRESH  = -0.08  # -8% daily return → dump signal
VOL_SPIKE_THRESH    = 2.5    # 2.5x average volume → spike
ZSCORE_THRESH       = 3.0    # 3σ from mean → z-score anomaly


# ── 1. Isolation Forest ───────────────────────────────────────────────────────
def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    df["if_anomaly"] = 0
    df["if_score"]   = 0.0

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        sub  = df.loc[mask, FEATURES].dropna()
        if len(sub) < 30:
            continue
        sc   = StandardScaler()
        X    = sc.fit_transform(sub)
        clf  = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        preds = clf.fit_predict(X)
        scores = clf.decision_function(X)
        df.loc[sub.index, "if_anomaly"] = (preds == -1).astype(int)
        df.loc[sub.index, "if_score"]   = -scores   # higher = more anomalous

    n = int(df["if_anomaly"].sum())
    print(f"   Found {n} anomalies across all stocks")
    return df


# ── 2. Z-Score Detector ───────────────────────────────────────────────────────
def run_zscore_detector(df: pd.DataFrame, thresh: float = ZSCORE_THRESH) -> pd.DataFrame:
    df = df.copy()
    df["zscore_anomaly"] = 0
    df["return_zscore"]  = 0.0
    df["vol_zscore"]     = 0.0

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ret  = df.loc[mask, "daily_return"]
        vol  = df.loc[mask, "vol_ratio"]

        ret_z = (ret - ret.mean()) / (ret.std() + 1e-9)
        vol_z = (vol - vol.mean()) / (vol.std() + 1e-9)

        df.loc[mask, "return_zscore"]  = ret_z.round(3)
        df.loc[mask, "vol_zscore"]     = vol_z.round(3)
        df.loc[mask, "zscore_anomaly"] = (
            (ret_z.abs() > thresh) | (vol_z > thresh)
        ).astype(int)

    n = int(df["zscore_anomaly"].sum())
    print(f"   Found {n} Z-score anomalies")
    return df


# ── 3. Pump & Dump Detector ───────────────────────────────────────────────────
def run_pump_dump_detector(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pump_signal"] = 0
    df["dump_signal"] = 0

    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        ret  = df.loc[mask, "daily_return"]
        vol  = df.loc[mask, "vol_ratio"]

        pump = ((ret >= PUMP_RETURN_THRESH) & (vol >= VOL_SPIKE_THRESH))
        dump = ((ret <= DUMP_RETURN_THRESH) & (vol >= VOL_SPIKE_THRESH))

        df.loc[mask, "pump_signal"] = pump.astype(int)
        df.loc[mask, "dump_signal"] = dump.astype(int)

    print(f"   Found {int(df['pump_signal'].sum())} pump signals")
    print(f"   Found {int(df['dump_signal'].sum())} dump signals")
    return df


# ── 4. Alert Scoring ──────────────────────────────────────────────────────────
def compute_alert_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    score = pd.Series(0, index=df.index, dtype=float)

    if "if_anomaly"     in df.columns: score += df["if_anomaly"]
    if "zscore_anomaly" in df.columns: score += df["zscore_anomaly"]
    if "pump_signal"    in df.columns: score += df["pump_signal"]
    if "dump_signal"    in df.columns: score += df["dump_signal"]

    # Extra weight for extreme volume
    if "vol_ratio" in df.columns:
        score += (df["vol_ratio"] > 4.0).astype(float) * 0.5

    df["alert_score"] = score.round(2)
    df["alert_level"] = pd.cut(
        df["alert_score"],
        bins=[-0.1, 0.9, 1.9, 99],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    df.loc[df["alert_score"] == 0, "alert_level"] = "None"
    return df


# ── 5. Master Runner ──────────────────────────────────────────────────────────
def run_all(df: pd.DataFrame) -> tuple:
    print("\n" + "=" * 55)
    print("  PSX ANOMALY DETECTION ENGINE")
    print("=" * 55 + "\n")

    print("🌲 Running Isolation Forest...")
    df = run_isolation_forest(df)

    print("📐 Running Z-Score detector...")
    df = run_zscore_detector(df)

    print("🚨 Running Pump & Dump detector...")
    df = run_pump_dump_detector(df)

    df = compute_alert_score(df)

    # Build alerts dataframe
    alerts = df[df["alert_score"] > 0].copy()
    alerts = alerts.sort_values("alert_score", ascending=False)

    total  = len(alerts)
    high   = len(alerts[alerts["alert_level"] == "High"])
    pumps  = int(df["pump_signal"].sum())

    print(f"\n{'=' * 55}")
    print(f"  ALERT SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Total alerts      : {total}")
    print(f"  High priority     : {high}")
    print(f"  Pump & Dump flags : {pumps}")
    print(f"{'=' * 55}\n")

    import os
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/full_results.csv", index=False)
    alerts.to_csv("results/alerts.csv", index=False)
    print("✅ Full results saved to results/full_results.csv")
    print("✅ Alerts saved to results/alerts.csv")

    return df, alerts
