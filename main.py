import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Configuração da página
st.set_page_config(page_title="ProfitOne - WINFUT Real Time", layout="wide", page_icon="📈")

# CSS Tema Escuro
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00d9ff; text-shadow: 0 0 10px #00d9ff;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stMetric label {color: #888;}
    .stMetric [data-testid="stMetricValue"] {
        color: #00d9ff;
        font-size: 2rem;
        font-weight: bold;
    }
    .price-up {color: #00ff88 !important;}
    .price-down {color: #ff4444 !important;}
</style>
""", unsafe_allow_html=True)

# Título
st.title("📈 PROFITONE - WINFUT TEMPO REAL")
st.markdown("### *🔴 LIVE - Índice Futuro Bovespa (WIN)*")

# Sidebar
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

# Seleção de ativo com WINFUT
symbol_map = {
    "🏆 WINFUT (Mini Índice)": "^BVSP",  # Proxy: Ibovespa spot
    "📊 IBOVESPA": "^BVSP",
    "💵 DÓLAR (BRL)": "BRL=X",
    "₿ BITCOIN": "BTC-USD",
    "Ξ ETHEREUM": "ETH-USD",
    "🍎 APPLE": "AAPL",
    "⚡ TESLA": "TSLA"
}

selected = st.sidebar.selectbox(
    "Selecione o Ativo:",
    list(symbol_map.keys()),
    index=0
)
symbol = symbol_map[selected]

# Intervalo de atualização
refresh_rate = st.sidebar.slider(
    "Atualização (segundos):",
    min_value=5,
    max_value=60,
    value=10,
    step=5
)

# Timeframe
timeframe = st.sidebar.selectbox(
    "Timeframe:",
    ["1m", "5m", "15m", "30m", "1h"],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.info("🔴 **LIVE**: Atualização automática ativa!")

# Botão de controle
auto_refresh = st.sidebar.checkbox("🔄 Auto-Refresh", value=True)

if st.sidebar.button("🔄 Atualizar Agora"):
    st.rerun()

# Container para atualização em tempo real
placeholder = st.empty()

# Função para buscar dados em tempo real
@st.cache_data(ttl=refresh_rate)
def get_realtime_data(ticker, period="1d", interval="5m"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

# Loop de atualização
while True:
    with placeholder.container():
        
        # Buscar dados
        df = get_realtime_data(symbol, period="1d", interval=timeframe)
        
        if df is not None and not df.empty:
            
            # Timestamp
            now = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"**🕐 Última Atualização:** `{now}`")
            
            # Métricas principais
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
            price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            # Abrir do dia
            open_price = df['Open'].iloc[0]
            day_change = current_price - open_price
            day_change_pct = (day_change / open_price) * 100 if open_price != 0 else 0
            
            volume = df['Volume'].iloc[-1]
            high_24h = df['High'].max()
            low_24h = df['Low'].min()
            
            # Display de métricas
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                delta_class = "price-up" if price_change >= 0 else "price-down"
                st.metric(
                    label="💰 Cotação Atual",
                    value=f"{current_price:,.2f}",
                    delta=f"{price_change_pct:+.2f}%"
                )
            
            with col2:
                st.metric(
                    label="📊 Abertura",
                    value=f"{open_price:,.2f}",
                    delta=f"{day_change_pct:+.2f}% (dia)"
                )
            
            with col3:
                st.metric(
                    label="📈 Máxima",
                    value=f"{high_24h:,.2f}"
                )
            
            with col4:
                st.metric(
                    label="📉 Mínima",
                    value=f"{low_24h:,.2f}"
                )
            
            with col5:
                st.metric(
                    label="📦 Volume",
                    value=f"{volume:,.0f}"
                )
            
            st.markdown("---")
            
            # Gráfico de Candlestick em tempo real
            st.subheader(f"📊 {selected} - Gráfico em Tempo Real")
            
            fig = go.Figure()
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='rgba(0,255,136,0.3)',
                decreasing_fillcolor='rgba(255,68,68,0.3)'
            ))
            
            # Linha de preço atual
            fig.add_hline(
                y=current_price,
                line_dash="dash",
                line_color="#00d9ff",
                annotation_text=f"Atual: {current_price:,.2f}",
                annotation_position="right"
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=600,
                xaxis_title='Horário',
                yaxis_title='Preço',
                hovermode='x unified',
                xaxis_rangeslider_visible=False,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de Volume
            st.subheader("📊 Volume")
            
            colors = ['#00ff88' if df['Close'].iloc[i] > df['Open'].iloc[i] else '#ff4444' 
                      for i in range(len(df))]
            
            fig_volume = go.Figure(data=[go.Bar(
                x=df.index,
                y=df['Volume'],
                marker_color=colors,
                name='Volume'
            )])
            
            fig_volume.update_layout(
                template='plotly_dark',
                height=250,
                xaxis_title='Horário',
                yaxis_title='Volume',
                showlegend=False,
                plot_bgcolor='#0e1117',
                paper_bgcolor='#0e1117'
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
            # Análise rápida
            st.markdown("---")
            st.subheader("🎯 Análise Rápida")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trend = "🟢 ALTA" if day_change > 0 else "🔴 BAIXA" if day_change < 0 else "⚪ LATERAL"
                st.markdown(f"**Tendência do Dia:** {trend}")
                
            with col2:
                volatility = ((high_24h - low_24h) / low_24h * 100)
                st.markdown(f"**Volatilidade:** {volatility:.2f}%")
                
            with col3:
                amplitude = high_24h - low_24h
                st.markdown(f"**Amplitude:** {amplitude:,.2f} pontos")
            
            # Tabela de dados
            with st.expander("📋 Dados Detalhados (últimas 20 barras)"):
                display_df = df.tail(20).copy()
                display_df.index = display_df.index.strftime("%H:%M")
                st.dataframe(
                    display_df.style.format({
                        'Open': '{:,.2f}',
                        'High': '{:,.2f}',
                        'Low': '{:,.2f}',
                        'Close': '{:,.2f}',
                        'Volume': '{:,.0f}'
                    }),
                    use_container_width=True
                )
        
        else:
            st.error("❌ Não foi possível carregar os dados em tempo real.")
            st.info("💡 Tente outro ativo ou aguarde alguns segundos.")
    
    # Controle de auto-refresh
    if not auto_refresh:
        break
    
    time.sleep(refresh_rate)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 ProfitOne Trading System - WINFUT Real Time</p>
    <p>📊 Dados: Yahoo Finance | ⚠️ Trading envolve risco</p>
</div>
""", unsafe_allow_html=True)
