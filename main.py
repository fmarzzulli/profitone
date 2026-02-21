import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ProfitOne Quantum",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%); }
    h1, h2, h3 { color: #00ffcc !important; }
    .signal-buy { background: #00ff88; color: #000; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .signal-sell { background: #ff4444; color: #fff; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .signal-neutral { background: #888; color: #fff; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def calculate_ema(prices, period):
    return prices.ewm(span=period, adjust=False).mean()

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50.0

def quantum_score(df):
    if len(df) < 21:
        return 0.0
    close = df['close']
    ema9 = calculate_ema(close, 9).iloc[-1]
    ema21 = calculate_ema(close, 21).iloc[-1]
    rsi = calculate_rsi(close, 14)
    trend = 50 if ema9 > ema21 else -50
    momentum = (rsi - 50)
    score = trend + momentum
    return np.clip(score, -100, 100)

@st.cache_data(ttl=60)
def get_data(symbol, interval='5m'):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {'interval': interval, 'range': '5d'}
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote['volume']
        })
        return df.dropna()
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return pd.DataFrame()

# UI
st.title("🚀 ProfitOne Quantum")
st.markdown("**Motor Anti-Repaint | Análise em Tempo Real**")

# Sidebar
st.sidebar.title("⚙️ Configurações")
mode = st.sidebar.selectbox("Modo", ["Scalp (5min)", "Day Trade (15min)", "Swing (1h)"])
interval_map = {"Scalp (5min)": "5m", "Day Trade (15min)": "15m", "Swing (1h)": "60m"}
interval = interval_map[mode]

if st.sidebar.button("🔄 Atualizar", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Dados
df = get_data("^BVSP", interval)

if df.empty or len(df) < 50:
    st.error("❌ Dados insuficientes. Clique em 'Atualizar'.")
    st.stop()

# Indicadores
df['ema9'] = calculate_ema(df['close'], 9)
df['ema21'] = calculate_ema(df['close'], 21)
score = quantum_score(df)
rsi = calculate_rsi(df['close'], 14)
price = df['close'].iloc[-1]

# Sinal
if score > 30:
    signal, signal_class = "COMPRA 🟢", "buy"
elif score < -30:
    signal, signal_class = "VENDA 🔴", "sell"
else:
    signal, signal_class = "NEUTRO ⚪", "neutral"

# Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Score", f"{score:.1f}")
col2.metric("🎯 RSI", f"{rsi:.1f}")
col3.metric("💰 Preço", f"R$ {price:,.2f}")
col4.markdown(f"<div class='signal-{signal_class}'>{signal}</div>", unsafe_allow_html=True)

# Gráfico
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

fig.add_trace(go.Candlestick(x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Preço"), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema9'], name="EMA 9", line=dict(color='cyan', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ema21'], name="EMA 21", line=dict(color='orange', width=2)), row=1, col=1)

scores = [quantum_score(df.iloc[:i+1]) for i in range(21, len(df))]
fig.add_trace(go.Scatter(x=df['timestamp'].iloc[21:], y=scores, name="Score", line=dict(color='magenta', width=2)), row=2, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
fig.add_hline(y=-30, line_dash="dash", line_color="red", row=2, col=1)

fig.update_layout(template="plotly_dark", height=600, showlegend=True, xaxis_rangeslider_visible=False)
st.plotly_chart(fig, width='stretch')

st.caption(f"⏰ {datetime.now().strftime('%H:%M:%S')} | ⚠️ Apenas educacional")
