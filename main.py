"""
PSX Anomaly Detector — Main Runner
Full pipeline: Data → Detection → Prediction → Charts → Alerts
"""

import os, sys
import pandas as pd
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

from psx_scraper       import fetch_all_stocks, add_features
from anomaly_detector  import run_all
from visualizer        import plot_all
from alert_system      import print_alerts, save_alert_report
from predictor         import predict_all_stocks, plot_predictions


def run_pipeline(fetch_fresh: bool = True, days: int = 365):
    print("=" * 60)
    print("   PSX STOCK MARKET ANOMALY DETECTOR")
    print("   Pakistan Stock Exchange | AI-Powered")
    print("=" * 60)

    if fetch_fresh or not os.path.exists("data/psx_features.csv"):
        print("\n📡 STEP 1: Fetching PSX data...\n")
        raw, source = fetch_all_stocks(days=days)
        print(f"   Source: {source}")
        df = add_features(raw)
    else:
        print("\n📂 STEP 1: Loading cached data...")
        df = pd.read_csv("data/psx_features.csv")

    print("\n🔍 STEP 2: Running anomaly detection...\n")
    df, alerts = run_all(df)

    print("\n🔮 STEP 3: Generating 7-day forecasts...\n")
    preds = predict_all_stocks(df)
    if not preds.empty:
        plot_predictions(df, preds, ticker="ENGRO")

    print("\n📊 STEP 4: Generating charts...\n")
    plot_all(df)

    print("\n🚨 STEP 5: Alert summary...\n")
    print_alerts(alerts, days=30)
    save_alert_report(alerts)

    print("\n" + "=" * 60)
    print("  ✅ DONE! Check results/ folder")
    print("  💡 Run web app: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cache", action="store_true", help="Use cached data")
    p.add_argument("--days",  type=int, default=365)
    args = p.parse_args()
    run_pipeline(fetch_fresh=not args.cache, days=args.days)
