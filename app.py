"""
PSX Anomaly Detector — Enhanced Streamlit Web App
Bloomberg Terminal-style dark UI with advanced features
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import os, sys
from datetime import datetime

st.set_page_config(
    page_title="PSX Anomaly Detector",
    page_icon="🇵🇰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Bloomberg Terminal Dark Theme ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #0f1525;
    --bg-panel:     #111827;
    --border:       #1e2d40;
    --accent-green: #00d48a;
    --accent-red:   #ff4757;
    --accent-blue:  #2196f3;
    --accent-amber: #ffa726;
    --accent-cyan:  #00bcd4;
    --text-primary: #e8eaf0;
    --text-muted:   #6b7a99;
}

.stApp {
    background: linear-gradient(135deg, #060a14 0%, #0a0e1a 50%, #070b16 100%) !important;
    font-family: 'Rajdhani', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c18 0%, #0a0f1e 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    position: relative; overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important; font-size: 10px !important;
    letter-spacing: 1px; text-transform: uppercase;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] { font-family: 'Share Tech Mono', monospace !important; }

[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    color: var(--text-muted) !important; letter-spacing: 0.5px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1a2744, #0f1a35) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-cyan) !important;
    border-radius: 4px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 1px !important;
    text-transform: uppercase !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    background: var(--accent-cyan) !important; color: #000 !important;
    box-shadow: 0 0 20px rgba(0,188,212,0.4) !important;
}

[data-testid="stSelectbox"] > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important; border-radius: 4px !important;
}

.psx-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin-bottom: 12px;
}
.psx-badge-red {
    display:inline-block; background:rgba(255,71,87,0.15); color:#ff4757;
    border:1px solid rgba(255,71,87,0.4); border-radius:3px;
    padding:2px 8px; font-size:11px; font-family:'Share Tech Mono',monospace; letter-spacing:1px;
}
.psx-badge-green {
    display:inline-block; background:rgba(0,212,138,0.15); color:#00d48a;
    border:1px solid rgba(0,212,138,0.4); border-radius:3px;
    padding:2px 8px; font-size:11px; font-family:'Share Tech Mono',monospace; letter-spacing:1px;
}
.psx-badge-amber {
    display:inline-block; background:rgba(255,167,38,0.15); color:#ffa726;
    border:1px solid rgba(255,167,38,0.4); border-radius:3px;
    padding:2px 8px; font-size:11px; font-family:'Share Tech Mono',monospace; letter-spacing:1px;
}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0a0e1a", "axes.facecolor":  "#0f1525",
    "axes.edgecolor":   "#1e2d40", "axes.labelcolor": "#a0aec0",
    "axes.grid": True,             "grid.color":      "#1e2d40",
    "grid.linewidth": 0.6,         "xtick.color":     "#6b7a99",
    "ytick.color":    "#6b7a99",   "text.color":      "#e8eaf0",
    "legend.facecolor":"#111827",  "legend.edgecolor":"#1e2d40",
    "legend.fontsize": 8,          "font.family":     "monospace",
    "figure.dpi": 130,
})

C_GREEN="#00d48a"; C_RED="#ff4757"; C_BLUE="#2196f3"
C_AMBER="#ffa726"; C_CYAN="#00bcd4"; C_PURPLE="#9c27b0"

# ── Import modules ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from psx_scraper      import fetch_all_stocks, add_features, PSX_TICKERS
from anomaly_detector import run_all
from predictor        import predict_all_stocks, predict_next_7_days

SECTOR_MAP = {
    "ENGRO":"Chemicals","HBL":"Banking","LUCK":"Cement","PSO":"Energy",
    "OGDC":"Energy","UBL":"Banking","MCB":"Banking","HUBC":"Power",
    "PPL":"Energy","MARI":"Energy","MEBL":"Banking","BAFL":"Banking",
    "EFERT":"Fertilizer","FFC":"Fertilizer","KOHC":"Cement",
}

# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px;'>
        <div style='font-size:52px; line-height:1;'>🇵🇰</div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:15px;
                    color:#00bcd4; letter-spacing:3px; margin-top:8px;'>PSX TERMINAL</div>
        <div style='font-family:"Rajdhani",sans-serif; font-size:11px;
                    color:#6b7a99; letter-spacing:1px;'>ANOMALY DETECTION SYSTEM</div>
    </div>
    <hr style='border-color:#1e2d40; margin:0 0 16px;'>
    """, unsafe_allow_html=True)

    st.markdown("**📌 STOCK SELECTION**")
    selected_ticker = st.selectbox(
        "Stock", options=list(PSX_TICKERS.keys()),
        format_func=lambda t: f"{t} — {PSX_TICKERS[t]}",
        label_visibility="collapsed"
    )
    sector = SECTOR_MAP.get(selected_ticker, "—")
    st.markdown(f"<span class='psx-badge-amber'>⬡ {sector}</span>", unsafe_allow_html=True)

    st.markdown("<br>**📅 TIME RANGE**", unsafe_allow_html=True)
    days = st.slider("Days", 90, 365, 365, step=30, label_visibility="collapsed")
    st.caption(f"Analysing {days} trading days")

    st.markdown("---")
    st.markdown("**⚙️ ALERT THRESHOLDS**")
    pump_thresh   = st.slider("Pump return threshold (%)", 3, 15, 8)
    vol_thresh    = st.slider("Volume spike threshold (×)", 1.5, 5.0, 2.5, step=0.5)
    zscore_thresh = st.slider("Z-score threshold (σ)", 2.0, 4.0, 3.0, step=0.5)

    st.markdown("---")
    col_r1, col_r2 = st.columns(2)
    refresh = col_r1.button("🔄 Refresh", use_container_width=True)
    export  = col_r2.button("💾 Export",  use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:10px;
                color:#6b7a99; line-height:2;'>
    ● ISOLATION FOREST AI<br>● Z-SCORE DETECTOR<br>
    ● PUMP &amp; DUMP SCANNER<br>● 7-DAY PRICE FORECAST<br>
    ● CORRELATION MATRIX<br>● SECTOR LEADERBOARD
    </div>
    <br>
    <div style='background:rgba(255,71,87,0.08); border:1px solid rgba(255,71,87,0.2);
                border-radius:4px; padding:8px 10px;
                font-size:10px; color:#ff4757; font-family:"Share Tech Mono",monospace;'>
    ⚠ EDUCATIONAL USE ONLY<br>NOT FINANCIAL ADVICE
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def load_data(days):
    raw, source = fetch_all_stocks(days=days)
    feat = add_features(raw)
    return feat, source

@st.cache_data(ttl=1800)
def run_detection(days):
    feat, source = load_data(days)
    df, alerts = run_all(feat)
    preds = predict_all_stocks(df)
    return df, alerts, preds, source

if refresh:
    st.cache_data.clear()
    st.rerun()

with st.spinner("⚡ Running AI anomaly detection..."):
    df, alerts, preds, data_source = run_detection(days)

if export and not alerts.empty:
    csv = alerts.to_csv(index=False)
    st.sidebar.download_button("⬇ Download CSV", csv, "psx_alerts.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
#  TOP HEADER BAR
# ════════════════════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%d %b %Y  %H:%M:%S")
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:center;
            background:#0f1525; border:1px solid #1e2d40; border-radius:8px;
            padding:14px 24px; margin-bottom:20px;'>
    <div>
        <span style='font-family:"Share Tech Mono",monospace; font-size:20px;
                     color:#00bcd4; letter-spacing:2px;'>🇵🇰 PSX ANOMALY TERMINAL</span>
        <span style='font-family:"Rajdhani",sans-serif; font-size:13px;
                     color:#6b7a99; margin-left:16px;'>
            Pakistan Stock Exchange · AI-Powered Surveillance
        </span>
    </div>
    <div style='text-align:right;'>
        <div style='font-family:"Share Tech Mono",monospace; font-size:12px; color:#ffa726;'>
            🕐 {now_str}
        </div>
        <div style='font-size:11px; color:#6b7a99; margin-top:2px;'>
            {"🟢 LIVE+SYNTH" if "Live" in data_source else "🟡 SYNTHETIC DATA"}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  KPI METRICS ROW
# ════════════════════════════════════════════════════════════════════════════
high_count   = len(alerts[alerts["alert_level"]=="High"]) if "alert_level" in alerts.columns else 0
pump_count   = int(df["pump_signal"].sum())   if "pump_signal"    in df.columns else 0
dump_count   = int(df["dump_signal"].sum())   if "dump_signal"    in df.columns else 0
zscore_count = int(df["zscore_anomaly"].sum())if "zscore_anomaly" in df.columns else 0

c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("🏦 Stocks",       f"{df['ticker'].nunique()}")
c2.metric("🚨 Total Alerts", f"{len(alerts)}", delta=f"+{high_count} HIGH", delta_color="inverse")
c3.metric("🔺 Pump Signals", f"{pump_count}",  delta_color="inverse")
c4.metric("🔻 Dump Signals", f"{dump_count}",  delta_color="inverse")
c5.metric("📐 Z-Score Flags",f"{zscore_count}",delta_color="inverse")
c6.metric("📅 Days",         f"{days}")

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  LIVE TICKER STRIP
# ════════════════════════════════════════════════════════════════════════════
df_latest = df.sort_values("date").groupby("ticker").last().reset_index()
df_latest["ret_pct"] = df_latest["daily_return"] * 100

ticker_html = "<div style='display:flex; gap:10px; overflow-x:auto; padding:8px 0; margin-bottom:18px;'>"
for _, row in df_latest.sort_values("ret_pct", ascending=False).iterrows():
    color = "#00d48a" if row["ret_pct"] >= 0 else "#ff4757"
    arrow = "▲" if row["ret_pct"] >= 0 else "▼"
    ticker_html += f"""
    <div style='background:#0f1525; border:1px solid #1e2d40; border-radius:6px;
                padding:8px 14px; white-space:nowrap; min-width:130px; flex-shrink:0;'>
        <div style='font-family:"Share Tech Mono",monospace; font-size:12px; color:#00bcd4;'>{row['ticker']}</div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:14px; color:#e8eaf0; margin-top:2px;'>
            PKR {row['close']:.0f}</div>
        <div style='font-size:11px; color:{color}; margin-top:2px;'>{arrow} {row['ret_pct']:+.2f}%</div>
    </div>"""
ticker_html += "</div>"
st.markdown(ticker_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ════════════════════════════════════════════════════════════════════════════
tab1,tab2,tab3,tab4,tab5,tab6 = st.tabs([
    "📈 Stock Dashboard","🚨 Alerts","🔮 Forecast",
    "🌡 Heatmap","🔗 Correlation","🏆 Leaderboard"
])

# ── TAB 1: STOCK DASHBOARD ──────────────────────────────────────────────────
with tab1:
    grp = df[df["ticker"]==selected_ticker].copy().sort_values("date").tail(180)
    grp["date"] = pd.to_datetime(grp["date"])
    name = PSX_TICKERS[selected_ticker]
    last = grp.iloc[-1]; prev = grp.iloc[-2]
    ret  = (last["close"]-prev["close"])/prev["close"]*100
    ret_color = "#00d48a" if ret >= 0 else "#ff4757"
    rsi_val   = last["rsi"]
    rsi_label = "OVERBOUGHT" if rsi_val>70 else ("OVERSOLD" if rsi_val<30 else "NEUTRAL")
    rsi_color = "#ff4757" if rsi_val>70 else ("#00d48a" if rsi_val<30 else "#ffa726")

    st.markdown(f"""
    <div class='psx-card' style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:16px;'>
        <div>
            <div style='font-family:"Share Tech Mono",monospace; font-size:24px; color:#00bcd4; font-weight:bold;'>{selected_ticker}</div>
            <div style='font-size:13px; color:#6b7a99; margin-top:2px;'>{name} · {sector}</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-family:"Share Tech Mono",monospace; font-size:30px; color:#e8eaf0; font-weight:bold;'>PKR {last['close']:.2f}</div>
            <div style='font-size:15px; color:{ret_color}; font-family:"Share Tech Mono",monospace;'>
                {"▲" if ret>=0 else "▼"} {ret:+.2f}% today
            </div>
        </div>
        <div style='display:flex; gap:24px; text-align:center;'>
            <div>
                <div style='font-size:10px; color:#6b7a99; letter-spacing:1px; font-family:"Share Tech Mono",monospace;'>RSI</div>
                <div style='font-size:22px; color:{rsi_color}; font-family:"Share Tech Mono",monospace;'>{rsi_val:.0f}</div>
                <div style='font-size:9px; color:{rsi_color};'>{rsi_label}</div>
            </div>
            <div>
                <div style='font-size:10px; color:#6b7a99; letter-spacing:1px; font-family:"Share Tech Mono",monospace;'>VOL RATIO</div>
                <div style='font-size:22px; color:#ffa726; font-family:"Share Tech Mono",monospace;'>{last['vol_ratio']:.1f}×</div>
            </div>
            <div>
                <div style='font-size:10px; color:#6b7a99; letter-spacing:1px; font-family:"Share Tech Mono",monospace;'>7D RETURN</div>
                <div style='font-size:22px; color:{"#00d48a" if last.get("return_7d",0)>=0 else "#ff4757"};
                            font-family:"Share Tech Mono",monospace;'>{last.get("return_7d",0)*100:+.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Anomaly badges
    flags = []
    if grp["if_anomaly"].sum()     > 0: flags.append(("🔴 IF ANOMALY",   "red"))
    if grp["pump_signal"].sum()    > 0: flags.append(("🔺 PUMP",         "amber"))
    if grp["dump_signal"].sum()    > 0: flags.append(("🔻 DUMP",         "red"))
    if grp["zscore_anomaly"].sum() > 0: flags.append(("📐 Z-SCORE FLAG", "amber"))
    if flags:
        badge_html = " ".join([f"<span class='psx-badge-{c}'>{f}</span>" for f,c in flags])
        st.markdown(f"<div style='margin:10px 0 16px;'>{badge_html}</div>", unsafe_allow_html=True)

    # 5-panel chart
    fig = plt.figure(figsize=(15, 9), facecolor="#0a0e1a")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.32)

    ax1 = fig.add_subplot(gs[0:2, 0])
    ax1.plot(grp["date"], grp["close"], color=C_CYAN,  lw=2,   label="Close", zorder=3)
    ax1.plot(grp["date"], grp["ma20"],  color="#555",  lw=1,   ls="--", alpha=0.8, label="MA-20")
    ax1.plot(grp["date"], grp["ma50"],  color="#444",  lw=1,   ls=":",  alpha=0.6, label="MA-50")
    if "bb_upper" in grp.columns:
        ax1.fill_between(grp["date"], grp["bb_lower"], grp["bb_upper"], alpha=0.07, color=C_BLUE)
        ax1.plot(grp["date"], grp["bb_upper"], color=C_BLUE, lw=0.6, alpha=0.4)
        ax1.plot(grp["date"], grp["bb_lower"], color=C_BLUE, lw=0.6, alpha=0.4, label="Bollinger")
    an = grp[grp["if_anomaly"]==1]
    ax1.scatter(an["date"], an["close"], color=C_RED,   s=55, zorder=5, marker="v", label="IF Anomaly")
    pu = grp[grp["pump_signal"]==1]
    ax1.scatter(pu["date"], pu["close"], color=C_AMBER, s=70, zorder=6, marker="^", label="Pump")
    du = grp[grp["dump_signal"]==1]
    ax1.scatter(du["date"], du["close"], color=C_PURPLE,s=70, zorder=6, marker="v", label="Dump")
    ax1.set_title(f"{selected_ticker} — Price & Signals", fontsize=11, color="#e8eaf0", pad=8)
    ax1.set_ylabel("Price (PKR)", fontsize=9)
    ax1.legend(fontsize=7, ncol=3)
    ax1.tick_params(axis="x", rotation=30, labelsize=7)

    ax2 = fig.add_subplot(gs[0, 1])
    vol_colors = [C_RED if v>vol_thresh else C_BLUE for v in grp["vol_ratio"]]
    ax2.bar(grp["date"], grp["vol_ratio"], color=vol_colors, width=1, alpha=0.85)
    ax2.axhline(vol_thresh, color=C_RED, ls="--", lw=1.2, label=f"Threshold {vol_thresh}×")
    ax2.set_title("Volume Ratio", fontsize=10, color="#e8eaf0", pad=6)
    ax2.legend(fontsize=7); ax2.tick_params(axis="x", rotation=30, labelsize=7)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(grp["date"], grp["rsi"], color=C_PURPLE, lw=1.5)
    ax3.axhline(70, color=C_RED,   ls="--", lw=1, label="OB 70")
    ax3.axhline(30, color=C_GREEN, ls="--", lw=1, label="OS 30")
    ax3.axhline(50, color="#333",  ls="-",  lw=0.5)
    ax3.fill_between(grp["date"], grp["rsi"], 70, where=(grp["rsi"]>70), alpha=0.2, color=C_RED)
    ax3.fill_between(grp["date"], grp["rsi"], 30, where=(grp["rsi"]<30), alpha=0.2, color=C_GREEN)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI (14-day)", fontsize=10, color="#e8eaf0", pad=6)
    ax3.legend(fontsize=7); ax3.tick_params(axis="x", rotation=30, labelsize=7)

    ax4 = fig.add_subplot(gs[2, 0])
    ret_series = grp["daily_return"].dropna() * 100
    bar_c = [C_GREEN if r>=0 else C_RED for r in ret_series]
    ax4.bar(grp["date"].iloc[-len(ret_series):], ret_series, color=bar_c, width=1, alpha=0.85)
    ax4.axhline(0, color="#444", lw=0.8)
    ax4.set_title("Daily Returns (%)", fontsize=10, color="#e8eaf0", pad=6)
    ax4.set_ylabel("%", fontsize=8)
    ax4.tick_params(axis="x", rotation=30, labelsize=7)

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.hist(ret_series, bins=30, color=C_BLUE, alpha=0.7, edgecolor="#0a0e1a")
    ax5.axvline(ret_series.mean(), color=C_RED,   ls="--", lw=1.5, label=f"μ={ret_series.mean():.2f}%")
    ax5.axvline(ret_series.mean()+2*ret_series.std(), color=C_AMBER, ls=":", lw=1, label="+2σ")
    ax5.axvline(ret_series.mean()-2*ret_series.std(), color=C_AMBER, ls=":", lw=1, label="-2σ")
    ax5.set_title("Returns Distribution", fontsize=10, color="#e8eaf0", pad=6)
    ax5.set_xlabel("Return (%)", fontsize=8)
    ax5.legend(fontsize=7)

    st.pyplot(fig); plt.close()

# ── TAB 2: ALERTS ───────────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🚨 Active Anomaly Alerts")
    if not alerts.empty and "alert_level" in alerts.columns:
        for level, icon, badge_cls in [
            ("High",  "🔴","psx-badge-red"),
            ("Medium","🟡","psx-badge-amber"),
            ("Low",   "🟢","psx-badge-green"),
        ]:
            sub = alerts[alerts["alert_level"]==level].head(25)
            if sub.empty: continue
            st.markdown(f"""
            <div style='display:flex; align-items:center; gap:10px; margin:16px 0 8px;'>
                <span style='font-size:18px;'>{icon}</span>
                <span style='font-family:"Share Tech Mono",monospace; font-size:13px;
                             color:#e8eaf0; letter-spacing:1px;'>{level.upper()} PRIORITY</span>
                <span class='{badge_cls}'>{len(sub)} alerts</span>
            </div>""", unsafe_allow_html=True)
            display = sub[["date","ticker","company","daily_return","vol_ratio",
                            "rsi","alert_score","pump_signal","dump_signal"]].copy()
            display["daily_return"] = (display["daily_return"]*100).round(2).astype(str)+"%"
            display["vol_ratio"]    = display["vol_ratio"].round(2).astype(str)+"×"
            display["rsi"]          = display["rsi"].round(1)
            display["pump_signal"]  = display["pump_signal"].map({1:"🔺 YES",0:"—"})
            display["dump_signal"]  = display["dump_signal"].map({1:"🔻 YES",0:"—"})
            display.columns = ["Date","Ticker","Company","Return","Volume","RSI","Score","Pump","Dump"]
            st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.markdown("""
        <div style='background:rgba(0,212,138,0.08); border:1px solid rgba(0,212,138,0.3);
                    border-radius:6px; padding:20px; text-align:center;
                    font-family:"Share Tech Mono",monospace; color:#00d48a;'>
            ✅ ALL CLEAR — No anomalies detected
        </div>""", unsafe_allow_html=True)

# ── TAB 3: FORECAST ─────────────────────────────────────────────────────────
with tab3:
    st.markdown(f"### 🔮 7-Day Forecast — {PSX_TICKERS[selected_ticker]}")
    st.caption("Linear Regression on 7 technical indicators · Educational only · Not financial advice")

    pred = predict_next_7_days(df, selected_ticker)
    if not pred.empty:
        last_close = df[df["ticker"]==selected_ticker].sort_values("date")["close"].iloc[-1]
        day7_chg   = pred.iloc[-1]["change_pct"]
        day7_pred  = pred.iloc[-1]["predicted"]
        up_days    = int((pred["change_pct"]>0).sum())

        fc1,fc2,fc3,fc4 = st.columns(4)
        fc1.metric("Current Price",   f"PKR {last_close:.2f}")
        fc2.metric("Day +7 Forecast", f"PKR {day7_pred:.2f}", f"{day7_chg:+.1f}%")
        fc3.metric("Up Days (of 7)",  f"{up_days} / 7")
        fc4.metric("Outlook", "📈 BULLISH" if day7_chg>1 else ("📉 BEARISH" if day7_chg<-1 else "➡ NEUTRAL"))

        grp2 = df[df["ticker"]==selected_ticker].sort_values("date").tail(30)
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0a0e1a")

        ax = axes2[0]
        ax.plot(pd.to_datetime(grp2["date"]), grp2["close"],
                color=C_CYAN, lw=2, label="Historical", zorder=3)
        ax.plot(pd.to_datetime(pred["date"]), pred["predicted"],
                color=C_AMBER, lw=2, ls="--", marker="o", ms=6, label="Forecast", zorder=4)
        std   = grp2["close"].std() * 0.04
        upper = pred["predicted"] + std * np.arange(1, 8)
        lower = pred["predicted"] - std * np.arange(1, 8)
        ax.fill_between(pd.to_datetime(pred["date"]), lower, upper,
                        alpha=0.18, color=C_AMBER, label="Confidence")
        ax.axvline(pd.to_datetime(grp2["date"].iloc[-1]), color="#555", ls=":", lw=1.5, label="Today")
        ax.set_title("Price Forecast", fontsize=11, color="#e8eaf0")
        ax.set_ylabel("Price (PKR)", fontsize=9); ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

        ax2 = axes2[1]
        bc = [C_GREEN if c>0 else C_RED for c in pred["change_pct"]]
        bars = ax2.bar(pred["day"], pred["change_pct"], color=bc, edgecolor="#0a0e1a", width=0.6)
        for bar, val in zip(bars, pred["change_pct"]):
            ax2.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+(0.04 if val>=0 else -0.18),
                     f"{val:+.1f}%", ha="center", fontsize=9, color="#e8eaf0", fontweight="bold")
        ax2.axhline(0, color="#555", lw=0.8)
        ax2.set_title("Daily Predicted Change", fontsize=11, color="#e8eaf0")
        ax2.set_ylabel("Change (%)", fontsize=9)

        plt.tight_layout(); st.pyplot(fig2); plt.close()

        display_pred = pred[["day","date","predicted","change_pct"]].copy()
        display_pred["predicted"]  = display_pred["predicted"].round(2)
        display_pred["change_pct"] = display_pred["change_pct"].apply(lambda x: f"{x:+.2f}%")
        display_pred.columns = ["Day","Date","Predicted Price (PKR)","Change (%)"]
        st.dataframe(display_pred, use_container_width=True, hide_index=True)
    else:
        st.warning("Not enough data for forecast.")

# ── TAB 4: HEATMAP ──────────────────────────────────────────────────────────
with tab4:
    st.markdown("### 🌡 Anomaly Heatmap — Last 30 Days")
    if "alert_score" in df.columns:
        df2 = df.copy(); df2["date"] = pd.to_datetime(df2["date"])
        last30 = df2[df2["date"] >= pd.Timestamp.today()-pd.Timedelta(days=30)]
        pivot  = last30.pivot_table(
            index="company", columns="date", values="alert_score", aggfunc="max").fillna(0)

        cmap_psx = LinearSegmentedColormap.from_list(
            "psx", ["#0f1525","#1e3a5f","#ffa726","#ff4757"])
        fig3, ax3 = plt.subplots(figsize=(16, 7), facecolor="#0a0e1a")
        im = ax3.imshow(pivot.values, aspect="auto", cmap=cmap_psx, vmin=0, vmax=3)
        ax3.set_yticks(range(len(pivot.index)))
        ax3.set_yticklabels(pivot.index, fontsize=9, color="#a0aec0")
        ax3.set_xticks(range(0, len(pivot.columns), 5))
        ax3.set_xticklabels([d.strftime("%d %b") for d in pivot.columns[::5]],
                             rotation=45, ha="right", fontsize=8, color="#6b7a99")
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label("Alert Score  (0=Normal → 3=High)", color="#a0aec0", fontsize=9)
        ax3.set_title("PSX Anomaly Heatmap", fontsize=13, color="#e8eaf0", pad=12)
        plt.tight_layout(); st.pyplot(fig3); plt.close()

# ── TAB 5: CORRELATION MATRIX (NEW) ─────────────────────────────────────────
with tab5:
    st.markdown("### 🔗 Stock Correlation Matrix")
    st.caption("Pearson correlation of daily returns — identify co-movement & diversification opportunities")

    pivot_ret = df.pivot_table(index="date", columns="ticker", values="daily_return")
    corr = pivot_ret.corr()
    tickers_list = list(corr.columns)

    cmap_corr = LinearSegmentedColormap.from_list("corr", ["#ff4757","#0f1525","#00d48a"])
    fig4, ax4 = plt.subplots(figsize=(10, 8), facecolor="#0a0e1a")
    im2 = ax4.imshow(corr.values, cmap=cmap_corr, vmin=-1, vmax=1, aspect="auto")
    ax4.set_xticks(range(len(tickers_list))); ax4.set_yticks(range(len(tickers_list)))
    ax4.set_xticklabels(tickers_list, rotation=45, ha="right", fontsize=9, color="#a0aec0")
    ax4.set_yticklabels(tickers_list, fontsize=9, color="#a0aec0")
    for i in range(len(tickers_list)):
        for j in range(len(tickers_list)):
            val = corr.values[i,j]
            ax4.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=7, color="#e8eaf0" if abs(val)<0.7 else "#000")
    cbar2 = plt.colorbar(im2, ax=ax4, shrink=0.8)
    cbar2.set_label("Correlation", color="#a0aec0", fontsize=9)
    ax4.set_title("Returns Correlation Matrix", fontsize=12, color="#e8eaf0", pad=10)
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    pairs = []
    for i in range(len(tickers_list)):
        for j in range(i+1, len(tickers_list)):
            pairs.append({"Stock A":tickers_list[i],"Stock B":tickers_list[j],
                          "Correlation":round(corr.values[i,j],4)})
    pairs_df = pd.DataFrame(pairs).sort_values("Correlation", ascending=False)
    col_a, col_b = st.columns(2)
    col_a.markdown("**🟢 Most Positively Correlated**")
    col_a.dataframe(pairs_df.head(8), use_container_width=True, hide_index=True)
    col_b.markdown("**🔴 Least / Negatively Correlated**")
    col_b.dataframe(pairs_df.tail(8).sort_values("Correlation"), use_container_width=True, hide_index=True)

# ── TAB 6: LEADERBOARD (NEW) ────────────────────────────────────────────────
with tab6:
    st.markdown("### 🏆 Stock Leaderboard")

    latest = df.sort_values("date").groupby("ticker").last().reset_index()
    latest["return_7d_pct"] = latest.get("return_7d", pd.Series(0, index=latest.index)) * 100
    ret30_map = {}
    for tkr, grp_l in df.groupby("ticker"):
        g = grp_l.sort_values("date")
        ret30_map[tkr] = (g["close"].iloc[-1] / g["close"].iloc[max(-30,-len(g))] - 1) * 100
    latest["return_30d_pct"] = latest["ticker"].map(ret30_map)
    latest["sector"]    = latest["ticker"].map(SECTOR_MAP)
    latest["alert_cnt"] = latest["ticker"].map(
        alerts.groupby("ticker").size() if not alerts.empty else {}
    ).fillna(0).astype(int)

    board = latest[["ticker","company","close","daily_return","return_7d_pct",
                     "return_30d_pct","vol_ratio","rsi","alert_cnt","sector"]].copy()
    board["daily_return"]   = (board["daily_return"]*100).round(2)
    board["return_7d_pct"]  = board["return_7d_pct"].round(2)
    board["return_30d_pct"] = board["return_30d_pct"].round(2)
    board["vol_ratio"]      = board["vol_ratio"].round(2)
    board["rsi"]            = board["rsi"].round(1)
    board["close"]          = board["close"].round(2)
    board.columns = ["Ticker","Company","Price","1D %","7D %","30D %","Vol Ratio","RSI","Alerts","Sector"]

    lb1, lb2 = st.columns(2)
    lb1.markdown("#### 🟢 Top Gainers (7D)")
    lb1.dataframe(board.sort_values("7D %", ascending=False).head(7),
                  use_container_width=True, hide_index=True)
    lb2.markdown("#### 🔴 Top Losers (7D)")
    lb2.dataframe(board.sort_values("7D %").head(7),
                  use_container_width=True, hide_index=True)

    st.markdown("#### 📊 Full Leaderboard")
    st.dataframe(board.sort_values("30D %", ascending=False),
                 use_container_width=True, hide_index=True)

    st.markdown("#### ⬡ Sector Performance (30D avg)")
    sector_perf = board.groupby("Sector")["30D %"].mean().sort_values(ascending=False)
    fig5, ax5 = plt.subplots(figsize=(10, 4), facecolor="#0a0e1a")
    sc = [C_GREEN if v>=0 else C_RED for v in sector_perf.values]
    bars5 = ax5.barh(sector_perf.index, sector_perf.values, color=sc, edgecolor="#0a0e1a")
    ax5.axvline(0, color="#555", lw=0.8)
    for bar, val in zip(bars5, sector_perf.values):
        ax5.text(val+(0.1 if val>=0 else -0.1), bar.get_y()+bar.get_height()/2,
                 f"{val:+.1f}%", va="center", ha="left" if val>=0 else "right",
                 fontsize=9, color="#e8eaf0")
    ax5.set_title("Sector Returns (30D avg)", fontsize=11, color="#e8eaf0")
    ax5.set_xlabel("Avg Return (%)", fontsize=9)
    plt.tight_layout(); st.pyplot(fig5); plt.close()

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1e2d40; margin-top:30px;'>
<div style='display:flex; justify-content:space-between; align-items:center; padding:10px 0;
            font-family:"Share Tech Mono",monospace; font-size:11px; color:#6b7a99;'>
    <div>🇵🇰 PSX ANOMALY TERMINAL · AI-POWERED SURVEILLANCE</div>
    <div>⚠ EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE</div>
    <div>Isolation Forest · Z-Score · Pump&amp;Dump · Linear Regression</div>
</div>
""", unsafe_allow_html=True)
