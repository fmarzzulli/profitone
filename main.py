import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import time

# Config
st.set_page_config(page_title="ProfitOne LIVE", layout="wide", page_icon="📈")

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
    .live {
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
st.markdown('<h1>📈 PROFITONE <span class="live">🔴 LIVE</span></h1>', unsafe_allow_html=True)
st.markdown("### *Dados em TEMPO REAL via API Binance*")

# Sidebar
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

symbols = {
    "₿ BITCOIN": "BTCUSDT",
    "Ξ ETHEREUM": "ETHUSDT",
    "💵 BNB": "BNBUSDT",
    "🔷 SOLANA": "SOLUSDT",
    "🟣 CARDANO": "ADAUSDT",
    "⚡ AVALANCHE": "AVAXUSDT"
}

selected = st.sidebar.selectbox("Ativo:", list(symbols.keys()), index=0)
symbol = symbols[selected]

interval_map = {
    "1 minuto": "1m",
    "5 minutos": "5m",
    "15 minutos": "15m",
    "1 hora": "1h"
}

timeframe_label = st.sidebar.selectbox("Timeframe:", list(interval_map.keys()), index=1)
interval = interval_map[timeframe_label]

refresh = st.sidebar.slider("Auto-refresh (seg):", 5, 60, 10, 5)

st.sidebar.markdown("---")
st.sidebar.success("🔴 **API BINANCE ATIVA**")

# Função para buscar dados
def get_binance_data(symbol, interval, limit=100):
    try:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        response = requests.get(url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
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
        st.error(f"Erro ao buscar dados: {e}")
        return None

# Função para preço atual
def get_current_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price"
        params = {"symbol": symbol}
        response = requests.get(url, params=params)
        data = response.json()
        return float(data['price'])
    except:
        return None

# Loop principal
placeholder = st.empty()

while True:
    with placeholder.container():
        
        # Buscar dados
        df = get_binance_data(symbol, interval)
        current_price = get_current_price(symbol)
        
        if df is not None and current_price is not None:
            
            # Timestamp
            now = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"**🕐 Atualizado:** `{now}` | **🌐 Fonte:** Binance API")
            
            # Cálculos
            prev_price = float(df['close'].iloc[-2])
            open_price = float(df['open'].iloc[0])
            
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0
            
            day_change = current_price - open_price
            day_change_pct = (day_change / open_price * 100) if open_price > 0 else 0
            
            high_24h = float(df['high'].max())
            low_24h = float(df['low'].min())
            volume = float(df['volume'].sum())
            
            # Métricas
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("💰 Preço LIVE", f"${current_price:,.2f}", f"{price_change_pct:+.2f}%")
            
            with col2:
                st.metric("📊 Abertura", f"${open_price:,.2f}", f"{day_change_pct:+.2f}%")
            
            with col3:
                st.metric("📈 Máxima", f"${high_24h:,.2f}")
            
            with col4:
                st.metric("📉 Mínima", f"${low_24h:,.2f}")
            
            with col5:
                st.metric("📦 Volume", f"{volume:,.0f}")
            
            st.markdown("---")
            
            # Gráfico
            st.subheader(f"📊 {selected} - {timeframe_label}")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
            
            # Preço atual
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="#00d9ff",
                annotation_text=f"LIVE: ${current_price:,.2f}",
                annotation_position="right"
            )
            
            # Médias móveis
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma50'] = df['close'].rolling(50).mean()
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ma20'],
                name='MA20', line=dict(color='#ffaa00', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ma50'],
                name='MA50', line=dict(color='#ff00ff', width=1)
            ))
            
            fig.update_layout(
                template='plotly_dark',
                height=600,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volume
            colors = ['#00ff88' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff4444' 
                      for i in range(len(df))]
            
            fig_vol = go.Figure(data=[go.Bar(
                x=df.index, y=df['volume'],
                marker_color=colors
            )])
            
            fig_vol.update_layout(
                template='plotly_dark',
                height=200,
                showlegend=False,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig_vol, use_container_width=True)
            
            # Análise
            st.markdown("---")
            st.subheader("🎯 Análise")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend = "🟢 ALTA" if day_change > 0 else "🔴 BAIXA"
                st.markdown(f"**Tendência:** {trend}")
                st.markdown(f"**Variação:** ${abs(day_change):,.2f}")
            
            with col2:
                volatility = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0
                st.markdown(f"**Volatilidade:** {volatility:.2f}%")
                st.markdown(f"**Amplitude:** ${high_24h - low_24h:,.2f}")
            
            with col3:
                trades = int(df['trades'].sum())
                st.markdown(f"**Total Trades:** {trades:,}")
                avg_price = df['close'].mean()
                st.markdown(f"**Preço Médio:** ${avg_price:,.2f}")
            
            # Dados
            with st.expander("📋 Últimos 20 Candles"):
                st.dataframe(
                    df.tail(20)[['open', 'high', 'low', 'close', 'volume']],
                    use_container_width=True
                )
        
        else:
            st.error("❌ Erro ao conectar com Binance API")
    
    # Auto-refresh
    time.sleep(refresh)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>🚀 ProfitOne | 🔴 Dados Reais Binance | ⚠️ Risco</div>", unsafe_allow_html=True)
