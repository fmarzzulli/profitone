# 🚀 PROFITONE QUANTUM V6 - VERSÃO ESTÁVEL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import hashlib
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="ProfitOne Quantum V6",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    h1, h2, h3 { color: #00ffcc !important; }
    [data-testid="stMetricValue"] { color: #00ff88 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# FUNÇÕES DE INDICADORES
# ============================================================

def calculate_ema(prices, period):
    """Exponential Moving Average"""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    return pd.Series(prices).ewm(span=period, adjust=False).mean().values

def calculate_rsi(prices, period=14):
    """RSI"""
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return np.concatenate([np.full(period, 50.0), rsi])

def quantum_score(df):
    """Score Quântico Simplificado"""
    if len(df) < 50:
        return np.full(len(df), 0.0)
    
    closes = df['Close'].values
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    rsi = calculate_rsi(closes, 14)
    
    # Trend (50%)
    trend = np.where(ema9 > ema21, 50, -50)
    
    # RSI (50%)
    rsi_score = (rsi - 50)
    
    score = trend + rsi_score
    return np.clip(score, -100, 100)

# ============================================================
# ANTI-REPAINT
# ============================================================

class AntiRepaint:
    def __init__(self, bars=2):
        self.bars = bars
        self.signals = []
    
    def add(self, ts, sig_type, price, score):
        sig = {
            'ts': ts,
            'type': sig_type,
            'price': price,
            'score': score,
            'bars': 0,
            'hash': hashlib.sha256(json.dumps({'ts': str(ts), 'type': sig_type}).encode()).hexdigest()[:12]
        }
        self.signals.append(sig)
    
    def update(self):
        for s in self.signals:
            s['bars'] += 1
    
    def confirmed(self):
        return [s for s in self.signals if s['bars'] >= self.bars]

# ============================================================
# BUSCAR DADOS
# ============================================================

@st.cache_data(ttl=60)
def get_data(symbol, interval='5m'):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range=1mo"
        r = requests.get(url, timeout=10)
        data = r.json()['chart']['result'][0]
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['timestamp'], unit='s'),
            'Open': data['indicators']['quote'][0]['open'],
            'High': data['indicators']['quote'][0]['high'],
            'Low': data['indicators']['quote'][0]['low'],
            'Close': data['indicators']['quote'][0]['close'],
            'Volume': data['indicators']['quote'][0]['volume']
        }).dropna()
        
        return df
    except:
        return None

# ============================================================
# MAIN
# ============================================================

def main():
    st.title("🚀 PROFITONE QUANTUM V6")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/rocket.png", width=120)
        mode = st.selectbox("Modo", ['Scalp (5m)', 'Day Trade (15m)', 'Swing (1h)'])
        interval = {'Scalp (5m)': '5m', 'Day Trade (15m)': '15m', 'Swing (1h)': '1h'}[mode]
        conf_bars = st.slider("Barras Confirmação", 1, 5, 2)
    
    # Dados
    df = get_data('^BVSP', interval)
    
    if df is None or len(df) < 50:
        st.error("❌ Erro ao carregar dados")
        return
    
    # Indicadores
    closes = df['Close'].values
    score = quantum_score(df)
    rsi = calculate_rsi(closes)
    
    current_score = score[-1]
    current_rsi = rsi[-1]
    
    # Sinal
    if current_score > 30:
        signal = 'BUY'
        sig_color = 'green'
    elif current_score < -30:
        signal = 'SELL'
        sig_color = 'red'
    else:
        signal = 'NEUTRO'
        sig_color = 'gray'
    
    # Anti-repaint
    ar = AntiRepaint(conf_bars)
    ar.add(df['timestamp'].iloc[-1], signal, closes[-1], current_score)
    ar.update()
    
    # Métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Quantum Score", f"{current_score:.1f}")
    
    with col2:
        st.metric("📊 RSI", f"{current_rsi:.1f}")
    
    with col3:
        st.metric("💎 Sinal", signal)
    
    # Sinal visual
    st.markdown(f"""
    <div style='text-align: center; margin: 20px;'>
        <span style='background: {sig_color}; color: white; padding: 15px 30px; 
                     border-radius: 10px; font-size: 24px; font-weight: bold;'>
            {signal}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Gráfico
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Candles
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='IBOV'
    ), row=1, col=1)
    
    # EMAs
    ema9 = calculate_ema(closes, 9)
    ema21 = calculate_ema(closes, 21)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=ema9,
        name='EMA 9', line=dict(color='cyan', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=ema21,
        name='EMA 21', line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    # Score
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=score,
        name='Quantum Score', line=dict(color='magenta', width=2)
    ), row=2, col=1)
    
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-30, line_dash="dash", line_color="red", row=2, col=1)
    
    fig.update_layout(
        height=700,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown(f"""
    <div style='text-align: center; margin-top: 30px; padding: 10px; 
                background: rgba(0,0,0,0.3); border-radius: 10px;'>
        📊 Última atualização: {datetime.now().strftime('%H:%M:%S')} | 
        ⚠️ Sistema educacional - Não usar para trading real
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
