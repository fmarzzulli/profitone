import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go

# Configuração da página
st.set_page_config(page_title="ProfitOne Trading", layout="wide", page_icon="📈")

# CSS Tema Escuro
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2, h3 {color: #00d9ff;}
    .stMetric {background-color: #1a1d29; padding: 15px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Título
st.title("📈 PROFITONE - Sistema de Trading")
st.markdown("### *Análise Profissional de Mercado*")

# Sidebar
st.sidebar.title("⚙️ Configurações")
st.sidebar.markdown("---")

symbol = st.sidebar.selectbox(
    "Selecione o Ativo:",
    ["^BVSP", "BRL=X", "BTC-USD", "ETH-USD", "AAPL", "TSLA"],
    index=0
)

period = st.sidebar.selectbox(
    "Período:",
    ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    index=2
)

interval = st.sidebar.selectbox(
    "Intervalo:",
    ["1m", "5m", "15m", "30m", "1h", "1d"],
    index=4
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Selecione diferentes ativos e períodos para análise")

# Função para carregar dados
@st.cache_data(ttl=300)
def load_data(ticker, per, intv):
    try:
        data = yf.Ticker(ticker)
        df = data.history(period=per, interval=intv)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

# Carregar dados
with st.spinner("📥 Carregando dados do mercado..."):
    df = load_data(symbol, period, interval)

if df is not None and not df.empty:
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    volume = df['Volume'].iloc[-1]
    high_24h = df['High'].max()
    low_24h = df['Low'].min()
    
    with col1:
        st.metric(
            label="💰 Preço Atual",
            value=f"${current_price:.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="📊 Volume",
            value=f"{volume:,.0f}"
        )
    
    with col3:
        st.metric(
            label="📈 Máxima",
            value=f"${high_24h:.2f}"
        )
    
    with col4:
        st.metric(
            label="📉 Mínima",
            value=f"${low_24h:.2f}"
        )
    
    st.markdown("---")
    
    # Gráfico de Candlestick
    st.subheader("📊 Gráfico de Candlestick")
    
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    )])
    
    fig.update_layout(
        template='plotly_dark',
        height=600,
        xaxis_title='Data/Hora',
        yaxis_title='Preço (USD)',
        hovermode='x unified',
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráfico de Volume
    st.subheader("📊 Volume de Negociação")
    
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
        height=300,
        xaxis_title='Data/Hora',
        yaxis_title='Volume',
        showlegend=False
    )
    
    st.plotly_chart(fig_volume, use_container_width=True)
    
    # Estatísticas
    st.markdown("---")
    st.subheader("📈 Estatísticas do Período")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**📊 Variação:**")
        st.write(f"Abertura: ${df['Open'].iloc[0]:.2f}")
        st.write(f"Fechamento: ${df['Close'].iloc[-1]:.2f}")
        total_change = ((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0]) * 100
        st.write(f"Variação Total: {total_change:+.2f}%")
    
    with col2:
        st.write("**📉 Volatilidade:**")
        volatility = df['Close'].pct_change().std() * 100
        st.write(f"Desvio Padrão: {volatility:.2f}%")
        st.write(f"Amplitude: ${high_24h - low_24h:.2f}")
    
    with col3:
        st.write("**📊 Volume:**")
        avg_volume = df['Volume'].mean()
        st.write(f"Média: {avg_volume:,.0f}")
        st.write(f"Total: {df['Volume'].sum():,.0f}")
    
    # Tabela de dados recentes
    with st.expander("📋 Ver Dados Detalhados"):
        st.dataframe(
            df.tail(20).style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True
        )

else:
    st.error("❌ Não foi possível carregar os dados. Tente outro ativo ou período.")
    st.info("💡 Dica: Alguns ativos podem não ter dados para intervalos muito curtos (como 1m)")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚀 ProfitOne Trading System | Dados fornecidos por Yahoo Finance</p>
    <p>⚠️ Este sistema é para fins educacionais. Trading envolve risco.</p>
</div>
""", unsafe_allow_html=True)
