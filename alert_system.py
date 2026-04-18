"""
PSX Alert System
Prints and saves human-readable alert reports.
"""

import pandas as pd
import os


def print_alerts(alerts: pd.DataFrame, days: int = 30):
    """Print high-priority alerts from the last N days to console."""
    print(f"\n🚨 HIGH PRIORITY ALERTS (last {days} days):\n")

    if alerts.empty:
        print("  ✅ No alerts detected.")
        return

    alerts = alerts.copy()
    alerts["date"] = pd.to_datetime(alerts["date"])
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
    recent = alerts[alerts["date"] >= cutoff]

    high = recent[recent["alert_level"] == "High"]
    if high.empty:
        print("  ✅ No high-priority alerts in the last 30 days.")
    else:
        for _, row in high.iterrows():
            ticker  = row.get("ticker", "")
            company = row.get("company", ticker)
            date    = row["date"].strftime("%Y-%m-%d")
            ret     = row.get("daily_return", 0) * 100
            vol     = row.get("vol_ratio", 1)
            score   = row.get("alert_score", 0)
            pump    = "🔺PUMP " if row.get("pump_signal", 0) else ""
            dump    = "🔻DUMP " if row.get("dump_signal", 0) else ""
            print(
                f"  🔴 [{date}] {company:<28} {pump}{dump}"
                f"Return: {ret:+.1f}%  Vol: {vol:.1f}x  Score: {score}"
            )

    medium = recent[recent["alert_level"] == "Medium"]
    if not medium.empty:
        print(f"\n  🟡 {len(medium)} medium-priority alerts also detected.")


def save_alert_report(alerts: pd.DataFrame, path: str = "results/alert_report.txt"):
    """Save a plain-text alert report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if alerts.empty:
        with open(path, "w") as f:
            f.write("No alerts detected.\n")
        return

    lines = ["PSX ANOMALY ALERT REPORT", "=" * 60, ""]
    alerts = alerts.copy()
    alerts["date"] = pd.to_datetime(alerts["date"])

    for level in ["High", "Medium", "Low"]:
        sub = alerts[alerts["alert_level"] == level]
        if sub.empty:
            continue
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {level.upper()} PRIORITY — {len(sub)} alerts")
        lines.append(f"{'─'*60}")
        for _, row in sub.iterrows():
            date    = row["date"].strftime("%Y-%m-%d")
            company = row.get("company", row.get("ticker", ""))
            ret     = row.get("daily_return", 0) * 100
            vol     = row.get("vol_ratio", 1)
            score   = row.get("alert_score", 0)
            flags   = []
            if row.get("pump_signal", 0): flags.append("PUMP")
            if row.get("dump_signal", 0): flags.append("DUMP")
            if row.get("if_anomaly",  0): flags.append("IF")
            if row.get("zscore_anomaly", 0): flags.append("ZSCORE")
            flag_str = f"[{','.join(flags)}]" if flags else ""
            lines.append(
                f"  {date}  {company:<26}  {flag_str:<18}"
                f"  Ret:{ret:+.1f}%  Vol:{vol:.1f}x  Score:{score}"
            )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✅ Alert report saved to {path}")
