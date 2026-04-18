# 🇵🇰 PSX Stock Alerts
### Simple AI-Powered Stock Helper for Pakistan Stock Exchange

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00d48a?style=flat)

> ⚠️ **FOR EDUCATIONAL PURPOSES ONLY — NOT FINANCIAL ADVICE**
> This tool is built for learning and research. Do NOT use it to make real investment decisions. Always consult a licensed financial advisor.

---

## 📸 Screenshots

| Stock Details | Alerts & Warnings | Next Week Prediction |
|---|---|---|
| Live price, RSI status, price chart | High/Medium/Low priority alerts | AI 7-day forecast chart |

---

## 💡 What Does This App Do?

This is a **beginner-friendly** stock monitoring dashboard for the Pakistan Stock Exchange. No finance degree needed — everything is explained in simple language.

- See the **current price** of 15 major PSX stocks
- Get **alerts** when something unusual happens (big price jumps, volume spikes)
- View an **AI prediction** for what might happen next week
- Understand everything with **simple explanations** — no jargon!

---

## 🏦 Stocks Covered

| Ticker | Company | Sector |
|---|---|---|
| ENGRO | Engro Corporation | Chemicals |
| HBL | HBL Bank | Banking |
| LUCK | Lucky Cement | Cement |
| PSO | Pakistan State Oil | Energy |
| OGDC | OGDC (Gas) | Energy |
| UBL | UBL Bank | Banking |
| MCB | MCB Bank | Banking |
| HUBC | Hub Power | Power |
| PPL | PPL (Oil) | Energy |
| MARI | Mari Gas | Energy |
| MEBL | Meezan Bank | Banking |
| BAFL | Bank Alfalah | Banking |
| EFERT | Engro Fertilizer | Fertilizer |
| FFC | Fauji Fertilizer | Fertilizer |
| KOHC | Kohat Cement | Cement |

---

## 🚀 How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/Sidra-009/PSX-Stock-Market-Anomaly-Detector.git
cd PSX-Stock-Market-Anomaly-Detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the simple app
```bash
streamlit run app_simple.py
```

### 4. Or run the advanced terminal app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
PSX-Stock-Market-Anomaly-Detector/
│
├── app_simple.py        # ← Simple app (this README)
├── app.py               # Advanced Bloomberg Terminal UI
├── main.py              # CLI pipeline runner
│
├── psx_scraper.py       # Fetches PSX data (live + synthetic fallback)
├── anomaly_detector.py  # Isolation Forest + Z-Score + Pump & Dump
├── predictor.py         # 7-day price forecast (Linear Regression)
├── visualizer.py        # Chart generator
├── alert_system.py      # Alert printer and report saver
│
└── requirements.txt     # Python dependencies
```

---

## 🤖 How the AI Works

### 📡 Data
The app first tries to fetch **live quotes** from PSX's official API. If unavailable, it generates **realistic synthetic data** using price simulation — so it always works even offline.

### 🔍 Anomaly Detection (3 methods)
| Method | What it does |
|---|---|
| 🌲 Isolation Forest | ML model that finds unusual trading days |
| 📐 Z-Score | Flags returns or volume more than 3× normal |
| 🚨 Pump & Dump | Detects sudden price surge + high volume together |

### 🔮 7-Day Forecast
Uses **Linear Regression** on 7 technical indicators (RSI, Bollinger Bands, volume ratio, moving averages) to predict the next week's price movement.

### 🚦 Alert Levels
| Level | Meaning |
|---|---|
| 🔴 High | Something very unusual happened — check immediately |
| 🟡 Medium | Something a bit unusual — keep an eye on it |
| 🟢 Low | Minor irregularity — just for your information |

---

## 📊 Features Explained Simply

**RSI (Relative Strength Index)**
- Below 30 = Stock fell too much (❄️ Too Cold) — might go up soon
- Above 70 = Stock rose too much (🔥 Too Hot) — might fall soon
- 30–70 = Normal (✅)

**Volume Ratio**
- How much more trading happened today vs the average
- 2.5× or more = Unusual — something big might be happening

**Pump & Dump**
- When someone buys a lot to push the price up (Pump 🔺) then sells everything (Dump 🔻)
- Our AI detects this pattern automatically

---

## ⚙️ Requirements

```
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
scipy==1.11.1
streamlit==1.28.0
requests==2.31.0
beautifulsoup4==4.12.2
plotly
```

---

## ☁️ Live Demo

Deployed on Streamlit Cloud — free and accessible worldwide.

🔗 **[Click here to open the app](https://psx-stock-market-anomaly-detector.streamlit.app/)**

*(Replace this link with your actual Streamlit Cloud URL after deploying)*

---

## ⚠️ Important Disclaimer

> **This project is for EDUCATIONAL and RESEARCH purposes ONLY.**
>
> - It does **NOT** provide financial advice
> - Stock predictions shown are **AI guesses**, not guarantees
> - Past patterns do **NOT** guarantee future results
> - **Never invest real money** based on this tool alone
> - Always consult a **licensed financial advisor** before investing
>
> The creator of this project takes **no responsibility** for any financial decisions made using this tool.

---

## 👩‍💻 Built By

**Sidra** — Pakistani developer building tools for the local community 🇵🇰

If you found this helpful, please ⭐ star the repo!

---

## 📄 License

MIT License — free to use, modify, and share.
