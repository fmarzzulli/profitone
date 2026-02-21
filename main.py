"""
ProfitOne - Sistema de Análise Técnica Simplificado
Versão Minimalista que FUNCIONA 100%
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="ProfitOne - Análise Técnica",
    page_icon="📈",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    h1, h2, h3 {
        color: #00ff88 !important;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data(ttl=60)
def get_data(symbol, period="5d", interval="15m"):
    """Busca dados do Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return pd.DataFrame()
        
        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index()
        return df
    except Exception as e:
        st.error(f"Erro ao buscar dados: {e}")
        return pd.DataFrame()


def calculate_ema(data, period):
    """Calcula EMA"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data, period=14):
    """Calcula RSI"""
    delta = data.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_score(df):
    """Calcula score simples baseado em EMA e RSI"""
    prices = df['close']
    
    # EMAs
    ema9 = calculate_ema(prices, 9)
    ema21 = calculate_ema(prices, 21)
    
    # RSI
    rsi = calculate_rsi(prices, 14)
    
    # Score
    current_price = prices.iloc[-1]
    current_ema9 = ema9.iloc[-1]
    current_ema21 = ema21.iloc[-1]
    current_rsi = rsi.iloc[-1]
    
    # Trend score (EMA9 vs EMA21)
    if current_ema9 > current_ema21:
        trend_score = 50
    else:
        trend_score = -50
    
    # RSI score
    rsi_score = (current_rsi - 50)
    
    # Combined
    total_score = trend_score + rsi_score
    total_score = np.clip(total_score, -100, 100)
    
    # Signal
    if total_score > 30:
        signal = 'BUY'
    elif total_score < -30:
        signal = 'SELL'
    else:
        signal = 'NEUTRAL'
    
    return {
        'score': total_score,
        'signal': signal,
        'ema9': current_ema9,
        'ema21': current_ema21,
        'rsi': current_rsi
    }


def create_chart(df):
    """Cria gráfico"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=('Preço & Indicadores', 'RSI')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Preço',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # EMAs
    ema9 = calculate_ema(df['close'], 9)
    ema21 = calculate_ema(df['close'], 21)
    
    fig.add_trace(
        go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='cyan', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.index, y=ema21, name='EMA 21', line=dict(color='orange', width=2)),
        row=1, col=1
    )
    
    # RSI
    rsi = calculate_rsi(df['close'], 14)
    
    fig.add_trace(
        go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple', width=2)),
        row=2, col=1
    )
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Layout
    fig.update_layout(
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    fig.update_xaxes(title_text="Data/Hora", row=2, col=1)
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    
    return fig


# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>📈 PROFITONE - ANÁLISE TÉCNICA 📈</h1>
        <p style='font-size: 18px; color: #00ff88;'>Sistema Simplificado de Monitoramento</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        symbol = st.text_input("📊 Símbolo", value="^BVSP")
        
        timeframe = st.selectbox(
            "⏱️ Timeframe",
            ["5 min", "15 min", "1 hora", "1 dia"],
            index=1
        )
        
        interval_map = {
            "5 min": "5m",
            "15 min": "15m",
            "1 hora": "1h",
            "1 dia": "1d"
        }
        interval = interval_map[timeframe]
        
        period_map = {
            "5m": "1d",
            "15m": "5d",
            "1h": "1mo",
            "1d": "6mo"
        }
        period = period_map.get(interval, "5d")
        
        if st.button("🔄 Atualizar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Buscar dados
    with st.spinner("📊 Carregando dados..."):
        df = get_data(symbol, period, interval)
    
    if df.empty:
        st.error("❌ Sem dados disponíveis. Verifique o símbolo.")
        return
    
    # Garantir que o índice seja datetime
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df = df.set_index('date')
    
    # Calcular score
    result = calculate_score(df)
    
    # Métricas
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Score", f"{result['score']:.1f}")
    
    with col2:
        st.metric("📊 RSI", f"{result['rsi']:.1f}")
    
    with col3:
        st.metric("💰 Preço", f"R$ {df['close'].iloc[-1]:.2f}")
    
    with col4:
        signal_colors = {
            'BUY': '🟢',
            'SELL': '🔴',
            'NEUTRAL': '🟡'
        }
        st.metric("🎪 Sinal", f"{signal_colors[result['signal']]} {result['signal']}")
    
    # Gráfico
    st.markdown("---")
    st.markdown("## 📈 Gráfico de Análise")
    
    fig = create_chart(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Info adicional
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Indicadores")
        st.write(f"**EMA 9:** R$ {result['ema9']:.2f}")
        st.write(f"**EMA 21:** R$ {result['ema21']:.2f}")
        st.write(f"**RSI 14:** {result['rsi']:.1f}")
    
    with col2:
        st.markdown("### 📌 Informações")
        st.write(f"**Símbolo:** {symbol}")
        st.write(f"**Timeframe:** {timeframe}")
        st.write(f"**Candles:** {len(df)}")
        st.write(f"**Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Footer
    st.markdown("---")
    st.caption("📈 ProfitOne - Sistema de Análise Técnica")
    st.caption("⚠️ Este sistema é apenas para fins educacionais. Não constitui recomendação de investimento.")


if __name__ == "__main__":
    main()
