import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

st.set_page_config(page_title="ProfitOne", page_icon="🚀", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0a0a0a, #1a1a2e); }
h1 { color: #00ffcc !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def get_data(symbol, interval='5m'):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        r = requests.get(url, params={'interval': interval, 'range': '5d'}, timeout=10)
        d = r.json()['chart']['result'][0]
        q = d['indicators']['quote'][0]
        df = pd.DataFrame({
            'time': pd.to_datetime(d['timestamp'], unit='s'),
            'open': q['open'], 'high': q['high'], 'low': q['low'], 
            'close': q['close'], 'volume': q['volume']
        })
        return df.dropna()
    except:
        return pd.DataFrame()

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    if len(s) < n+1: return 50
    d = s.diff()
    g = d.where(d > 0, 0)
    l = -d.where(d < 0, 0)
    ag = g.rolling(n, min_periods=1).mean()
    al = l.rolling(n, min_periods=1).mean()
    rs = ag / (al + 1e-10)
    return (100 - (100 / (1 + rs))).iloc[-1]

def score(df):
    if len(df) < 21: return 0
    c = df['close']
    e9 = ema(c, 9).iloc[-1]
    e21 = ema(c, 21).iloc[-1]
    r = rsi(c)
    trend = 50 if e9 > e21 else -50
    return np.clip(trend + (r - 50), -100, 100)

st.title("🚀 ProfitOne")
st.sidebar.title("⚙️ Config")
mode = st.sidebar.selectbox("Modo", ["5min", "15min", "1h"])
interval = {'5min': '5m', '15min': '15m', '1h': '60m'}[mode]

if st.sidebar.button("🔄 Atualizar", type="primary"):
    st.cache_data.clear()
    st.rerun()

df = get_data("^BVSP", interval)

if df.empty or len(df) < 50:
    st.error("❌ Sem dados. Clique 'Atualizar'")
    st.stop()

df['ema9'] = ema(df['close'], 9)
df['ema21'] = ema(df['close'], 21)
s = score(df)
r = rsi(df['close'])
p = df['close'].iloc[-1]

sig = "🟢 COMPRA" if s > 30 else "🔴 VENDA" if s < -30 else "⚪ NEUTRO"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{s:.1f}")
c2.metric("RSI", f"{r:.1f}")
c3.metric("Preço", f"R$ {p:,.2f}")
c4.metric("Sinal", sig)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Preço"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['time'], y=df['ema9'], name="EMA9", line=dict(color='cyan', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['time'], y=df['ema21'], name="EMA21", line=dict(color='orange', width=2)), row=1, col=1)

scores = [score(df.iloc[:i+1]) for i in range(21, len(df))]
fig.add_trace(go.Scatter(x=df['time'].iloc[21:], y=scores, name="Score", line=dict(color='magenta', width=2)), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=-30, line_dash="dash", line_color="red", row=2, col=1)

fig.update_layout(template="plotly_dark", height=600, showlegend=True, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
