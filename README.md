# PSX Stock Market Anomaly Detector 🚨📈

An AI-powered system that detects suspicious trading patterns, pump & dump schemes, and price anomalies on the **Pakistan Stock Exchange (PSX)** — with a **live web dashboard** and **7-day price forecasts**.

> 🟢 **Live Demo:** [Deploy free on Streamlit Cloud](#deployment)

---

## What It Detects

| Signal | Method | Trigger |
|--------|--------|---------|
| General anomalies | Isolation Forest (ML) | Top 5% statistical outliers |
| Price/volume spikes | Z-Score (3σ) | 3 standard deviations from mean |
| Pump & dump | Rule-based | Price +8% in 3d + Volume 2.5x + RSI > 70 |

## Features

- **Real-time data** — Tries PSX live API, falls back to realistic synthetic data
- **3 detection models** — Isolation Forest + Z-Score + Pump & Dump rules
- **7-day price forecast** — Linear Regression on 12 technical indicators
- **Web dashboard** — Interactive Streamlit app (free deployment)
- **Email alerts** — Get notified on high-priority anomalies
- **Alert heatmap** — All 15 stocks at a glance

---

## Screenshots

### Web Dashboard
![Dashboard](results/dashboard_ENGRO_KA.png)

### Alert Heatmap
![Heatmap](results/alert_heatmap.png)

### 7-Day Forecast
![Forecast](results/forecast_ENGRO.png)

---

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/psx-anomaly-detector.git
cd psx-anomaly-detector
pip install -r requirements.txt

# Run full pipeline (terminal)
python main.py

# Run web app (browser)
streamlit run app.py
```

---

## Deployment — Free on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Click **Deploy** — live URL in 2 minutes!

---

## PSX Stocks Covered

`ENGRO` `HBL` `LUCK` `PSO` `OGDC` `UBL` `MCB` `HUBC` `PPL` `MARI` `MEBL` `BAFL` `EFERT` `FFC` `KOHC`

---

## Alert Levels

```
🔴 HIGH    — 3/3 detectors flagged
🟡 MEDIUM  — 2/3 detectors flagged
🟢 LOW     — 1/3 detectors flagged
```

---

## CV Description

> Built a real-time AI anomaly detection system for Pakistan Stock Exchange (PSX) monitoring 15 major stocks. Implemented Isolation Forest, Z-Score, and rule-based pump & dump detection with 7-day price forecasting using Linear Regression on 12 technical indicators. Deployed as an interactive Streamlit web application with live alerts and heatmap visualization.

---

## Tech Stack

`Python` `scikit-learn` `pandas` `numpy` `matplotlib` `Streamlit` `requests` `Isolation Forest` `RSI` `Bollinger Bands`

---

⚠️ *Educational purposes only. Not financial advice.*
