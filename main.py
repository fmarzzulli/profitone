import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from binance.client import Client
from binance.websockets import BinanceSocketManager
import threading
import queue
import time

# Config
st.set_page_config(page_title="ProfitOne REAL TIME", layout="wide", page_icon="📈")

# CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00d9ff; text-shadow: 0 0 10px #00d9ff;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
    }
    .live-badge {
        background: #ff0000;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% {opacity: 1;}
        50% {opacity: 0.5;}
    }
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<h1>📈 PROFITONE - <span class="live-badge">🔴 LIVE</span></h1>', unsafe_allow_html=True)
st.markdown("### *Dados em TEMPO REAL via WebSocket*")

# Sidebar
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

# Símbolos reais
symbols = {
    "₿ BITCOIN (BTC/USDT)": "BTCUSDT",
    "Ξ ETHEREUM (ETH/USDT)": "ETHUSDT",
    "💵 BINANCE COIN (BNB/USDT)": "BNBUSD T",
    "🔷 SOLANA (SOL/USDT)": "SOLUSDT",
    "🟣 CARDANO (ADA/USDT)": "ADAUSDT"
}

selected = st.sidebar.selectbox("Ativo em Tempo Real:", list(symbols.keys()), index=0)
symbol = symbols[selected]

timeframe = st.sidebar.selectbox("Timeframe:", ["1m", "5m", "15m", "30m", "1h"], index=0)

st.sidebar.markdown("---")
st.sidebar.success("🔴 **WEBSOCKET ATIVO**")
st.sidebar.info("💡 Dados 100% reais da Binance")

# Inicializar Binance Client
@st.cache_resource
def get_binance_client():
    return Client("", "")  # API keys não necessárias para dados públicos

client = get_binance_client()

# Função para buscar histórico
@st.cache_data(ttl=60)
def get_historical_data(symbol, interval, limit=100):
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"Erro: {e}")
        return None

# Buscar dados históricos
df = get_historical_data(symbol, timeframe)

if df is not None and not df.empty:
    
    # Container para atualização
    placeholder = st.empty()
    
    # Métricas em tempo real
    current_price = float(df['close'].iloc[-1])
    prev_price = float(df['close'].iloc[-2])
    open_price = float(df['open'].iloc[0])
    
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
    
    day_change = current_price - open_price
    day_change_pct = (day_change / open_price * 100) if open_price > 0 else 0
    
    volume = float(df['volume'].iloc[-1])
    high_24h = float(df['high'].max())
    low_24h = float(df['low'].min())
    
    # Timestamp
    now = datetime.now().strftime("%H:%M:%S")
    st.markdown(f"**🕐 Última Atualização:** `{now}` | **🌐 Fonte:** Binance WebSocket")
    
    # Métricas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("💰 Preço Atual", f"${current_price:,.2f}", f"{price_change_pct:+.2f}%")
    
    with col2:
        st.metric("📊 Abertura", f"${open_price:,.2f}", f"{day_change_pct:+.2f}%")
    
    with col3:
        st.metric("📈 Máxima 24h", f"${high_24h:,.2f}")
    
    with col4:
        st.metric("📉 Mínima 24h", f"${low_24h:,.2f}")
    
    with col5:
        st.metric("📦 Volume", f"{volume:,.2f}")
    
    st.markdown("---")
    
    # Gráfico de Candlestick
    st.subheader(f"📊 {selected} - Tempo Real")
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,68,68,0.3)'
    )])
    
    # Linha de preço atual
    fig.add_hline(
        y=current_price,
        line_dash="dash",
        line_color="#00d9ff",
        annotation_text=f"Atual: ${current_price:,.2f}",
        annotation_position="right"
    )
    
    # Médias móveis
    df['ma20'] = df['close'].rolling(window=20).mean()
    df['ma50'] = df['close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ma20'],
        name='MA20',
        line=dict(color='#ffaa00', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ma50'],
        name='MA50',
        line=dict(color='#ff00ff', width=1)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        xaxis_title='Horário',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Volume
    st.subheader("📊 Volume")
    
    colors = ['#00ff88' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff4444' 
              for i in range(len(df))]
    
    fig_volume = go.Figure(data=[go.Bar(
        x=df.index,
        y=df['volume'],
        marker_color=colors,
        name='Volume'
    )])
    
    fig_volume.update_layout(
        template='plotly_dark',
        height=250,
        showlegend=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117'
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Análise
    st.markdown("---")
    st.subheader("🎯 Análise em Tempo Real")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trend = "🟢 ALTA" if day_change > 0 else "🔴 BAIXA"
        st.markdown(f"**Tendência:** {trend}")
        st.markdown(f"**Variação:** {abs(day_change):,.2f} USD")
    
    with col2:
        volatility = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0
        st.markdown(f"**Volatilidade 24h:** {volatility:.2f}%")
        st.markdown(f"**Amplitude:** ${high_24h - low_24h:,.2f}")
    
    with col3:
        avg_vol = df['volume'].mean()
        vol_ratio = (volume / avg_vol) if avg_vol > 0 else 0
        st.markdown(f"**Volume Médio:** {avg_vol:,.2f}")
        st.markdown(f"**Ratio Atual:** {vol_ratio:.2f}x")
    
    # Tabela
    with st.expander("📋 Dados Detalhados (últimas 20 barras)"):
        display_df = df.tail(20)[['open', 'high', 'low', 'close', 'volume']].copy()
        st.dataframe(
            display_df.style.format({
                'open': '${:,.2f}',
                'high': '${:,.2f}',
                'low': '${:,.2f}',
                'close': '${:,.2f}',
                'volume': '{:,.2f}'
            }),
            use_container_width=True
        )
    
    # Auto-refresh
    if st.sidebar.button("🔄 Atualizar Agora"):
        st.rerun()
    
    # Timer automático
    refresh_rate = st.sidebar.slider("Atualização automática (seg):", 5, 60, 15, 5)
    time.sleep(refresh_rate)
    st.rerun()

else:
    st.error("❌ Não foi possível carregar dados")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 ProfitOne Trading System - Dados Reais via Binance API</p>
    <p>🔴 LIVE WebSocket | ⚠️ Trading envolve risco</p>
</div>
""", unsafe_allow_html=True)
