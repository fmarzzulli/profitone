import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import entropy, linregress
from scipy.signal import hilbert

st.set_page_config(page_title="ProfitOne Trading", layout="wide", page_icon="📈")

# CSS Tema Escuro
st.markdown("""
<style>
.main {background-color: #0e1117;}
h1, h2, h3 {color: #00d9ff;}
</style>
""", unsafe_allow_html=True)

# Título
st.title("📈 PROFITONE - Sistema Avançado de Trading")
st.markdown("### *Matemática Ultra-Avançada para Trading Profissional*")

# Sidebar
st.sidebar.title("⚙️ Configurações")
symbol = st.sidebar.selectbox("Ativo", ["^BVSP", "BRL=X", "BTC-USD", "ETH-USD"])
period = st.sidebar.selectbox("Período", ["1d", "5d", "1mo", "3mo", "6mo"], index=2)
interval = st.sidebar.selectbox("Intervalo", ["1m", "5m", "15m", "1h", "1d"], index=3)

# Indicadores
show_tema = st.sidebar.checkbox("TEMA + Velocity", value=True)
show_entropy = st.sidebar.checkbox("Entropia Shannon", value=True)
show_fisher = st.sidebar.checkbox("Fisher Transform", value=True)
show_hurst = st.sidebar.checkbox("Hurst Exponent", value=True)

# Download dados
@st.cache_data
def load_data(sym, per, intv):
    try:
        df = yf.Ticker(sym).history(period=per, interval=intv)
        df.columns = [c.lower() for c in df.columns]
        return df
    except:
        return pd.DataFrame()

# Indicadores
def tema(close, period=20):
    ema1 = close.ewm(span=period).mean()
    ema2 = ema1.ewm(span=period).mean()
    ema3 = ema2.ewm(span=period).mean()
    return 3 * ema1 - 3 * ema2 + ema3

def shannon_entropy(close, window=20):
    def calc_entropy(x):
        if len(x) < 2: return 0
        returns = np.diff(x)
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)
    return close.rolling(window).apply(calc_entropy, raw=False)

def fisher_transform(high, low, period=10):
    hl2 = (high + low) / 2
    max_h = hl2.rolling(period).max()
    min_l = hl2.rolling(period).min()
    value = 2 * ((hl2 - min_l) / (max_h - min_l + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)
    return 0.5 * np.log((1 + value) / (1 - value + 1e-10))

def hurst_exponent(data, window=100):
    def calc_hurst(ts):
        if len(ts) < 10: return 0.5
        lags = range(2, min(20, len(ts)//2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        return np.polyfit(np.log(lags), np.log(tau), 1)[0]
    return data.rolling(window).apply(calc_hurst, raw=False)

# Carregar dados
with st.spinner("📥 Carregando dados..."):
    df = load_data(symbol, period, interval)

if df.empty:
    st.error("❌ Erro ao carregar dados")
    st.stop()

# Calcular indicadores
indicators = {}
if show_tema:
    indicators['tema'] = tema(df['close'])
if show_entropy:
    indicators['entropy'] = shannon_entropy(df['close'])
if show_fisher:
    indicators['fisher'] = fisher_transform(df['high'], df['low'])
if show_hurst:
    indicators['hurst'] = hurst_exponent(df['close'])

# Métricas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Preço Atual", f"${df['close'].iloc[-1]:.2f}")
with col2:
    pct = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
    st.metric("Mudança %", f"{pct:+.2f}%")
with col3:
    if show_hurst:
        h = indicators['hurst'].iloc[-1]
        regime = "Tendência 📈" if h > 0.5 else "Lateral 📊"
        st.metric("Regime", regime)
with col4:
    if show_entropy:
        st.metric("Entropia", f"{indicators['entropy'].iloc[-1]:.2f}")

# Gráfico
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=('Price Action', 'Oscillators', 'Entropy & Hurst'))

# Candlestick
fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                             low=df['low'], close=df['close'], name='OHLC',
                             increasing_line_color='#00ff88',
                             decreasing_line_color='#ff4444'), row=1, col=1)

if show_tema:
    fig.add_trace(go.Scatter(x=df.index, y=indicators['tema'], name='TEMA',
                            line=dict(color='#00d9ff', width=2)), row=1, col=1)

# Volume
colors = ['#00ff88' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff4444'
          for i in range(len(df))]
fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume',
                     marker_color=colors, showlegend=False), row=2, col=1)

# Fisher
if show_fisher:
    fig.add_trace(go.Scatter(x=df.index, y=indicators['fisher'], name='Fisher',
                            line=dict(color='#00d9ff')), row=2, col=1)
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)

# Entropy & Hurst
if show_entropy:
    fig.add_trace(go.Scatter(x=df.index, y=indicators['entropy'], name='Entropy',
                            line=dict(color='#ffaa00')), row=3, col=1)
if show_hurst:
    fig.add_trace(go.Scatter(x=df.index, y=indicators['hurst'], name='Hurst',
                            line=dict(color='#00ff88')), row=3, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=3, col=1)

fig.update_layout(template='plotly_dark', height=900, showlegend=True,
                  xaxis_rangeslider_visible=False, hovermode='x unified')

st.plotly_chart(fig, use_container_width=True)

# Dados
with st.expander("📊 Dados Recentes"):
    st.dataframe(df.tail(20))
