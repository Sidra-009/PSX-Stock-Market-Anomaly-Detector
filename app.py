"""
PSX Stock Alerts - Simple Version
Beautiful Dark UI | Easy to understand
Run: streamlit run app_simple.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os, sys, time
from datetime import datetime

st.set_page_config(
    page_title="PSX Stock Alerts",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  DARK CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');
* { font-family: 'Poppins', sans-serif !important; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
}
[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

h1, h2, h3 { color: #e6edf3 !important; font-weight: 800 !important; }
p, span, div { color: #c9d1d9; }

/* Metric cards */
[data-testid="stMetric"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricValue"] {
    font-size: 28px !important; font-weight: 800 !important; color: #e6edf3 !important;
}
[data-testid="stMetricLabel"] { color: #8b949e !important; font-weight: 600 !important; }

/* Buttons */
.stButton > button {
    background: #238636 !important; color: #ffffff !important;
    border: 1px solid #2ea043 !important; border-radius: 8px !important;
    padding: 10px 20px !important; font-weight: 700 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
    box-shadow: 0 0 12px rgba(46,160,67,0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: #161b22; border-radius: 8px;
    color: #8b949e; font-weight: 600; padding: 10px 18px;
    border: 1px solid #30363d;
}
.stTabs [aria-selected="true"] {
    background: #1f6feb !important; color: #ffffff !important;
    border-color: #1f6feb !important;
}

/* Custom cards */
.dark-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.dark-card h3 { color: #e6edf3 !important; margin: 0 0 8px !important; font-size: 15px !important; }
.dark-card p  { color: #8b949e !important; margin: 4px 0 !important; font-size: 14px !important; }

.card-green  { border-left: 4px solid #2ea043 !important; }
.card-red    { border-left: 4px solid #f85149 !important; }
.card-yellow { border-left: 4px solid #d29922 !important; }
.card-blue   { border-left: 4px solid #1f6feb !important; }

.badge-green  { background:#0f2b14; color:#2ea043; border:1px solid #2ea043;
                border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; display:inline-block; }
.badge-red    { background:#2d1015; color:#f85149; border:1px solid #f85149;
                border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; display:inline-block; }
.badge-yellow { background:#2d2200; color:#d29922; border:1px solid #d29922;
                border-radius:6px; padding:4px 12px; font-size:13px; font-weight:700; display:inline-block; }

.tip-box {
    background: #1c2128; border-left: 3px solid #d29922;
    border-radius: 6px; padding: 12px 16px; margin: 12px 0;
}
.tip-box p { color: #d29922 !important; margin:0 !important; font-size:13px !important; }

.big-signal-green {
    background: #0f2b14; border: 1px solid #2ea043; border-radius: 10px;
    padding: 16px 20px; text-align: center; color: #2ea043 !important;
    font-size: 15px; font-weight: 700;
}
.big-signal-red {
    background: #2d1015; border: 1px solid #f85149; border-radius: 10px;
    padding: 16px 20px; text-align: center; color: #f85149 !important;
    font-size: 15px; font-weight: 700;
}
.big-signal-yellow {
    background: #2d2200; border: 1px solid #d29922; border-radius: 10px;
    padding: 16px 20px; text-align: center; color: #d29922 !important;
    font-size: 15px; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════════════
sys.path.insert(0, os.path.dirname(__file__))
from psx_scraper      import fetch_all_stocks, add_features, PSX_TICKERS
from anomaly_detector import run_all
from predictor        import predict_all_stocks, predict_next_7_days

simple_names = {
    "ENGRO":"Engro Corporation","HBL":"HBL Bank","LUCK":"Lucky Cement",
    "PSO":"Pakistan State Oil","OGDC":"OGDC (Gas)","UBL":"UBL Bank",
    "MCB":"MCB Bank","HUBC":"Hub Power","PPL":"PPL (Oil)","MARI":"Mari Gas",
    "MEBL":"Meezan Bank","BAFL":"Bank Alfalah","EFERT":"Engro Fertilizer",
    "FFC":"Fauji Fertilizer","KOHC":"Kohat Cement",
}

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 20px;'>
        <div style='font-size:44px;'>🇵🇰</div>
        <div style='font-size:17px; font-weight:800; color:#e6edf3; margin-top:6px;'>PSX Stock Alerts</div>
        <div style='font-size:12px; color:#8b949e;'>Pakistan Stock Exchange</div>
    </div>
    <hr style='border-color:#30363d; margin:0 0 16px;'>
    """, unsafe_allow_html=True)

    st.markdown("**💼 Choose a Company**")
    selected = st.selectbox(
        "Stock", options=list(simple_names.keys()),
        format_func=lambda x: simple_names[x],
        label_visibility="collapsed"
    )

    st.markdown("<br>**📅 Time Period**", unsafe_allow_html=True)
    period_map = {"Last Month":30, "Last 3 Months":90, "Last 6 Months":180, "Last Year":365}
    period = st.radio("Period", options=list(period_map.keys()), index=2, label_visibility="collapsed")
    days = period_map[period]

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='background:#1c2128; border-radius:8px; padding:14px; font-size:12px; color:#8b949e; line-height:1.9;'>
    🟢 <b style='color:#2ea043;'>Green</b> = Price going up<br>
    🔴 <b style='color:#f85149;'>Red</b> = Price going down<br>
    🟡 <b style='color:#d29922;'>Yellow</b> = Be careful<br><br>
    ⚠️ <span style='color:#d29922;'>For learning only.<br>Not financial advice.</span>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  DATA — BUG FIX: ensure minimum days for scraper
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800)
def load_data(days):
    safe_days = max(days, 60)   # ← FIX: never pass < 60, prevents low>=high crash
    raw, source = fetch_all_stocks(days=safe_days)
    feat = add_features(raw)
    return feat, source

@st.cache_data(ttl=1800)
def run_detection(days):
    feat, source = load_data(days)
    df, alerts = run_all(feat)
    preds = predict_all_stocks(df)
    return df, alerts, preds, source

with st.spinner("🤖 AI is analysing stocks..."):
    df, alerts, preds, data_source = run_detection(days)

# Filter df to requested days for display only
df_display = df.copy()
df_display["date"] = pd.to_datetime(df_display["date"])
cutoff = pd.Timestamp.today() - pd.Timedelta(days=days)
df_display = df_display[df_display["date"] >= cutoff]

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='background:#161b22; border:1px solid #30363d; border-radius:12px;
            padding:28px 32px; margin-bottom:24px; display:flex;
            justify-content:space-between; align-items:center;'>
    <div>
        <div style='font-size:26px; font-weight:800; color:#e6edf3;'>
            🇵🇰 PSX Stock Alerts
        </div>
        <div style='font-size:14px; color:#8b949e; margin-top:4px;'>
            Your Smart Stock Helper for Pakistan Stock Exchange
        </div>
    </div>
    <div style='text-align:right; font-size:13px; color:#8b949e;'>
        🕐 """ + datetime.now().strftime("%d %b %Y · %H:%M") + """
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TOP METRICS
# ══════════════════════════════════════════════════════════════
high = len(alerts[alerts["alert_level"]=="High"]) if "alert_level" in alerts.columns and not alerts.empty else 0
pump = int(df["pump_signal"].sum()) if "pump_signal" in df.columns else 0

c1,c2,c3,c4 = st.columns(4)
c1.metric("🏦 Stocks Watched", df["ticker"].nunique())
c2.metric("🚨 Total Alerts",   len(alerts) if not alerts.empty else 0)
c3.metric("⚠️ High Priority",  high)
c4.metric("🔺 Pump Signals",   pump)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📈 Stock Details", "🚨 Alerts & Warnings", "🔮 Next Week Prediction"])

# ── TAB 1 ────────────────────────────────────────────────────
with tab1:
    grp = df_display[df_display["ticker"]==selected].copy().sort_values("date")

    if len(grp) >= 2:
        last  = grp.iloc[-1]
        prev  = grp.iloc[-2]
        price = last["close"]
        chg   = price - prev["close"]
        chg_p = (chg / prev["close"] * 100) if prev["close"] != 0 else 0
        rsi   = last.get("rsi", 50)

        # Price card
        arrow = "📈" if chg >= 0 else "📉"
        c_col = "#2ea043" if chg >= 0 else "#f85149"
        border = "card-green" if chg >= 0 else "card-red"

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"""
            <div class='dark-card {border}'>
                <h3>{arrow} {simple_names[selected]}</h3>
                <div style='font-size:44px; font-weight:800; color:{c_col}; margin:12px 0;'>
                    PKR {price:.0f}
                </div>
                <div style='font-size:16px; color:{c_col}; font-weight:700;'>
                    {arrow} {abs(chg_p):.1f}% change today
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            if rsi > 70:
                emoji, label, cls, explain = "🔥","Too Hot!", "card-red", "Price rose too fast. Might cool down soon."
            elif rsi < 30:
                emoji, label, cls, explain = "❄️","Too Cold!", "card-green", "Price fell a lot. Might rise soon."
            else:
                emoji, label, cls, explain = "✅","Normal", "card-yellow", "Everything looks stable."
            st.markdown(f"""
            <div class='dark-card {cls}' style='text-align:center;'>
                <div style='font-size:40px;'>{emoji}</div>
                <h3 style='text-align:center; margin-top:8px !important;'>{label}</h3>
                <p style='text-align:center;'>{explain}</p>
                <p style='text-align:center; font-size:12px; color:#8b949e !important;'>RSI = {rsi:.0f}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class='tip-box'>
            <p>💡 <b>What is RSI?</b> It tells us if the stock is overbought (too hot 🔥) or oversold (too cold ❄️).
            Normal range is 30–70. Outside this = unusual activity.</p>
        </div>
        """, unsafe_allow_html=True)

        # Chart
        st.markdown("### 📊 Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp["close"],
            mode='lines', name='Price',
            line=dict(color='#2ea043', width=3),
            fill='tozeroy', fillcolor='rgba(46,160,67,0.08)',
            hovertemplate='<b>%{x|%b %d}</b><br>PKR %{y:.0f}<extra></extra>'
        ))
        fig.update_layout(
            xaxis_title="Date", yaxis_title="Price (PKR)",
            template="plotly_dark",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            hovermode="x unified", height=400,
            font=dict(family="Poppins, sans-serif", size=12, color="#8b949e"),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signals
        st.markdown("### 🚦 What's Happening?")
        s1, s2 = st.columns(2)
        with s1:
            if grp["pump_signal"].sum() > 0:
                st.markdown("<div class='big-signal-yellow'>⚠️ Price Rising Fast!<br><small>Price jumped up with high volume. Be careful.</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='big-signal-green'>✅ Normal Activity<br><small>No unusual price jumps.</small></div>", unsafe_allow_html=True)
        with s2:
            if grp["dump_signal"].sum() > 0:
                st.markdown("<div class='big-signal-red'>🚨 Price Falling Fast!<br><small>Price is dropping quickly with high volume.</small></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='big-signal-green'>✅ No Big Drops<br><small>Price is holding steady.</small></div>", unsafe_allow_html=True)

    else:
        st.warning("Not enough data for this period. Try 'Last 6 Months' or 'Last Year'.")

# ── TAB 2 ────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🚨 Alerts & Warnings")
    st.markdown("""
    <div class='tip-box'>
        <p>💡 <b>What are alerts?</b> When a stock does something unusual — like a sudden price jump or 
        very high trading — our AI raises an alert. Red = needs attention. Green = just for info.</p>
    </div>
    """, unsafe_allow_html=True)

    if not alerts.empty:
        high_c = len(alerts[alerts["alert_level"]=="High"])   if "alert_level" in alerts.columns else 0
        med_c  = len(alerts[alerts["alert_level"]=="Medium"]) if "alert_level" in alerts.columns else 0
        low_c  = len(alerts[alerts["alert_level"]=="Low"])    if "alert_level" in alerts.columns else 0

        a1,a2,a3 = st.columns(3)
        a1.markdown(f"<div class='big-signal-red'>🔴 {high_c}<br><small>High Priority</small></div>", unsafe_allow_html=True)
        a2.markdown(f"<div class='big-signal-yellow'>🟡 {med_c}<br><small>Medium</small></div>", unsafe_allow_html=True)
        a3.markdown(f"<div class='big-signal-green'>🟢 {low_c}<br><small>Low Priority</small></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        show = alerts.head(25).copy()
        show["What Happened"] = show.apply(
            lambda x: f"{'📈 Up' if x['daily_return']>0 else '📉 Down'} {abs(x['daily_return']*100):.1f}%", axis=1)
        show["Priority"] = show.get("alert_level", "—")
        show = show[["date","company","What Happened","Priority"]]
        show.columns = ["Date","Company","What Happened","Priority"]
        st.dataframe(show, use_container_width=True, hide_index=True, height=400)
    else:
        st.markdown("""
        <div class='big-signal-green' style='padding:30px;'>
            🎉 No alerts right now! Everything looks normal.
        </div>
        """, unsafe_allow_html=True)

# ── TAB 3 ────────────────────────────────────────────────────
with tab3:
    st.markdown("## 🔮 Next Week Prediction")
    st.markdown("""
    <div class='tip-box'>
        <p>⚠️ <b>Important:</b> This is an AI guess based on past patterns. 
        Stock prices can change for many reasons. Do NOT use this to make real money decisions!</p>
    </div>
    """, unsafe_allow_html=True)

    pred = predict_next_7_days(df, selected)

    if not pred.empty:
        last_price = df[df["ticker"]==selected].sort_values("date")["close"].iloc[-1]
        pred_price = pred.iloc[-1]["predicted"]
        pred_chg   = ((pred_price - last_price) / last_price * 100) if last_price != 0 else 0
        up_days    = int((pred["change_pct"] > 0).sum())

        f1,f2,f3 = st.columns(3)
        with f1:
            st.markdown(f"""
            <div class='dark-card card-blue'>
                <h3>📅 Today's Price</h3>
                <div style='font-size:32px; font-weight:800; color:#e6edf3; margin:10px 0;'>PKR {last_price:.0f}</div>
            </div>""", unsafe_allow_html=True)
        with f2:
            c = "#2ea043" if pred_chg>=0 else "#f85149"
            cl = "card-green" if pred_chg>=0 else "card-red"
            a  = "📈" if pred_chg>=0 else "📉"
            st.markdown(f"""
            <div class='dark-card {cl}'>
                <h3>{a} Next Week Estimate</h3>
                <div style='font-size:32px; font-weight:800; color:{c}; margin:10px 0;'>PKR {pred_price:.0f}</div>
                <div style='color:{c}; font-weight:700;'>{a} {abs(pred_chg):.1f}% expected</div>
            </div>""", unsafe_allow_html=True)
        with f3:
            if pred_chg > 2:   outlook, cl2 = "📈 Looks Good!", "card-green"
            elif pred_chg < -2: outlook, cl2 = "📉 Might Drop",  "card-red"
            else:               outlook, cl2 = "➡️ Staying Flat","card-yellow"
            st.markdown(f"""
            <div class='dark-card {cl2}'>
                <h3>🎯 AI Outlook</h3>
                <div style='font-size:22px; font-weight:800; color:#e6edf3; margin:10px 0;'>{outlook}</div>
                <div style='color:#8b949e;'>Up days: {up_days} out of 7</div>
            </div>""", unsafe_allow_html=True)

        grp_f = df[df["ticker"]==selected].sort_values("date").tail(30)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=pd.to_datetime(grp_f["date"]), y=grp_f["close"],
            mode='lines+markers', name='Real Price',
            line=dict(color='#2ea043', width=3), marker=dict(size=5),
            hovertemplate='<b>Real</b> %{x|%b %d} · PKR %{y:.0f}<extra></extra>'
        ))
        fig_f.add_trace(go.Scatter(
            x=pd.to_datetime(pred["date"]), y=pred["predicted"],
            mode='lines+markers', name='AI Prediction',
            line=dict(color='#d29922', width=3, dash='dash'),
            marker=dict(size=8, symbol='star'),
            hovertemplate='<b>Predicted</b> %{x|%b %d} · PKR %{y:.0f}<extra></extra>'
        ))
        fig_f.update_layout(
            xaxis_title="Date", yaxis_title="Price (PKR)",
            template="plotly_dark",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
            hovermode="x unified", height=420,
            font=dict(family="Poppins, sans-serif", size=12, color="#8b949e"),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        )
        st.plotly_chart(fig_f, use_container_width=True)

    else:
        st.warning("🤷 Not enough data. Try 'Last Year' from the sidebar.")

# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<hr style='border-color:#30363d; margin-top:32px;'>
<div style='text-align:center; padding:16px 0; color:#8b949e; font-size:12px;'>
    🇵🇰 PSX Stock Alerts &nbsp;·&nbsp; Made for learning &nbsp;·&nbsp;
    ⚠️ Not financial advice
</div>
""", unsafe_allow_html=True)
