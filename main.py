import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.stats import entropy
from datetime import datetime
import time

st.set_page_config(page_title="ProfitOne Ultimate V2", layout="wide", page_icon="🚀")

# CSS com fontes brancas
st.markdown("""
<style>
    .main {background-color: #000000;}
    * {color: #ffffff !important;}
    h1, h2, h3, h4, h5, h6 {color: #00d9ff !important; text-shadow: 0 0 20px #00d9ff;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
    }
    .stMetric label {color: #ffffff !important;}
    .stMetric [data-testid="stMetricValue"] {color: #00d9ff !important;}
    .signal-board {
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        font-size: 60px;
        font-weight: bold;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    .signal-up {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #000 !important;
    }
    .signal-down {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #fff !important;
    }
    .asset-card {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
        margin: 10px 0;
    }
    @keyframes pulse {
        0%, 100% {transform: scale(1);}
        50% {transform: scale(1.02);}
    }
    .stTabs [data-baseweb="tab-list"] {background-color: #1a1d29;}
    .stTabs [data-baseweb="tab"] {color: #ffffff !important;}
    .stDataFrame {color: #ffffff !important;}
    table {color: #ffffff !important;}
    .css-1d391kg {color: #ffffff !important;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# INDICADORES AVANÇADOS
# ==========================================

def quantum_hunter_v13(df, modo=2, sensibilidade=1.5):
    """
    Quantum Hunter V13 - Institutional Trend
    Retorna Score (-100 a +100) e sinal de cor
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Calibragem por modo
    if modo == 1:  # 5min Scalp
        periodo_lento, periodo_rapido, periodo_rsi = 21, 8, 9
    elif modo == 2:  # 15min Day Trade
        periodo_lento, periodo_rapido, periodo_rsi = 34, 13, 14
    else:  # 60min+ Swing
        periodo_lento, periodo_rapido, periodo_rsi = 72, 21, 21
    
    # Médias
    media_lenta = close.ewm(span=periodo_lento, adjust=False).mean()
    media_rapida = close.ewm(span=periodo_rapido, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periodo_rsi).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periodo_rsi).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # VWAP Intraday
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    
    # Cálculo do Score
    trend_score = pd.Series(0, index=close.index, dtype=float)
    
    # A) Força da Tendência (40%)
    trend_score += (close > media_rapida).astype(float) * 20
    trend_score += (media_rapida > media_lenta).astype(float) * 20
    trend_score -= (close < media_rapida).astype(float) * 20
    trend_score -= (media_rapida < media_lenta).astype(float) * 20
    
    # B) Contexto VWAP (30%)
    trend_score += (close > vwap).astype(float) * 30
    trend_score -= (close <= vwap).astype(float) * 30
    
    # C) Momentum RSI (30%)
    trend_score += ((rsi > 55).astype(float) * 15)
    trend_score += ((rsi > 50) & (rsi <= 55)).astype(float) * 5
    trend_score -= ((rsi < 45).astype(float) * 15)
    trend_score -= ((rsi < 50) & (rsi >= 45)).astype(float) * 5
    
    # Suavização (Cobra Sólida)
    score_suavizado = trend_score.ewm(span=3, adjust=False).mean()
    score_suavizado = score_suavizado.clip(-100, 100)
    
    return score_suavizado, media_lenta, media_rapida, vwap

def turbo_stoch_win(df):
    """
    TurboStoch WIN - Schaff + Fisher + Hurst
    Retorna STC, Fisher, Hurst e cor
    """
    close = df['close']
    high = df['high']
    low = df['low']
    
    # Schaff Trend Cycle
    periodo_curto, periodo_longo, periodo_ciclo = 24, 52, 20
    
    macd_line = close.ewm(span=periodo_curto, adjust=False).mean() - close.ewm(span=periodo_longo, adjust=False).mean()
    
    min_macd = macd_line.rolling(periodo_ciclo).min()
    max_macd = macd_line.rolling(periodo_ciclo).max()
    
    st1 = ((macd_line - min_macd) / (max_macd - min_macd + 1e-10)) * 100
    st1 = st1.ewm(span=3, adjust=False).mean()
    
    min_st1 = st1.rolling(periodo_ciclo).min()
    max_st1 = st1.rolling(periodo_ciclo).max()
    
    stc = ((st1 - min_st1) / (max_st1 - min_st1 + 1e-10)) * 100
    stc = stc.ewm(span=3, adjust=False).mean()
    
    # Fisher Transform
    periodo_fisher = 10
    hl2 = (high + low) / 2
    max_h = hl2.rolling(periodo_fisher).max()
    min_l = hl2.rolling(periodo_fisher).min()
    
    val = 2 * ((hl2 - min_l) / (max_h - min_l + 1e-10) - 0.5)
    val = val.clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + val) / (1 - val + 1e-10))
    fisher_norm = ((fisher + 4) / 8) * 100
    fisher_norm = fisher_norm.clip(0, 100)
    
    # Hurst Exponent (simplificado)
    periodo_hurst = 30
    range_price = high.rolling(periodo_hurst).max() - low.rolling(periodo_hurst).min()
    mean_range = close.diff().abs().rolling(periodo_hurst).mean()
    
    hurst = np.log(range_price / (mean_range + 1e-10)) / np.log(periodo_hurst)
    hurst = hurst.fillna(0.5)
    
    return stc, fisher_norm, hurst

def calcular_forca_win(df):
    """
    Força do WIN baseada em múltiplos indicadores
    Retorna valor de 0-100
    """
    score, _, _, _ = quantum_hunter_v13(df)
    stc, fisher, hurst = turbo_stoch_win(df)
    
    # Combinar indicadores
    forca = (abs(score.iloc[-1]) * 0.4 + stc.iloc[-1] * 0.3 + fisher.iloc[-1] * 0.3)
    
    # Ajustar por Hurst (reduz força se lateral)
    if not pd.isna(hurst.iloc[-1]):
        if hurst.iloc[-1] < 0.45:
            forca *= 0.5  # Mercado lateral, reduz confiança
    
    return min(forca, 100)

# ==========================================
# BUSCAR DADOS
# ==========================================

@st.cache_data(ttl=10)
def get_market_data(symbol, interval="5m"):
    """Buscar dados de múltiplas fontes"""
    
    # Mapeamento de símbolos
    symbol_map = {
        "IBOV": "%5EBVSP",
        "DOLAR": "USDBRL%3DX",
        "SP500": "%5EGSPC",
        "NASDAQ": "%5EIXIC",
        "DOW": "%5EDJI"
    }
    
    yahoo_symbol = symbol_map.get(symbol, symbol)
    
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = {"interval": interval, "range": "1d"}
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' in data and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]
                
                df = pd.DataFrame({
                    'timestamp': pd.to_datetime(timestamps, unit='s'),
                    'open': quotes['open'],
                    'high': quotes['high'],
                    'low': quotes['low'],
                    'close': quotes['close'],
                    'volume': quotes['volume']
                })
                
                df.set_index('timestamp', inplace=True)
                df = df.dropna()
                
                return df if len(df) > 0 else None
    
    except Exception as e:
        st.warning(f"Erro {symbol}: {str(e)}")
    
    return None

# ==========================================
# INTERFACE
# ==========================================

st.markdown("<h1 style='text-align: center;'>🚀 PROFITONE ULTIMATE V2</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #ffffff;'>Sistema Institucional com Quantum Hunter & TurboStoch</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Config")

modo_operacao = st.sidebar.selectbox(
    "Modo de Operação:",
    ["Scalp (5min)", "Day Trade (15min)", "Swing (60min)"],
    index=1
)

modo_map = {"Scalp (5min)": 1, "Day Trade (15min)": 2, "Swing (60min)": 3}
modo = modo_map[modo_operacao]

timeframe = st.sidebar.selectbox("Timeframe:", ["5m", "15m", "1h"], index=0)

# Abas
tab1, tab2, tab3 = st.tabs(["🎯 WIN Principal", "📊 Monitor Multi-Ativos", "⚡ Força em Tempo Real"])

# ==========================================
# ABA 1: WIN PRINCIPAL
# ==========================================

with tab1:
    
    placeholder1 = st.empty()
    
    with placeholder1.container():
        
        df_win = get_market_data("IBOV", timeframe)
        
        if df_win is not None and len(df_win) > 100:
            
            now = datetime.now().strftime("%H:%M:%S")
            st.success(f"✅ IBOVESPA: {len(df_win)} candles | {now}")
            
            # Quantum Hunter
            score_qh, ml, mr, vwap = quantum_hunter_v13(df_win, modo=modo)
            
            # TurboStoch
            stc, fisher, hurst = turbo_stoch_win(df_win)
            
            # Score atual
            score_atual = score_qh.iloc[-1]
            
            # SIGNAL BOARD
            if score_atual > 15:
                st.markdown(f"""
                <div class="signal-board signal-up">
                    <div>🚀 COMPRA INSTITUCIONAL</div>
                    <div style="font-size: 80px; margin-top: 15px;">+{score_atual:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            elif score_atual < -15:
                st.markdown(f"""
                <div class="signal-board signal-down">
                    <div>📉 VENDA INSTITUCIONAL</div>
                    <div style="font-size: 80px; margin-top: 15px;">{score_atual:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"⚖️ Neutro | Score: {score_atual:.1f}")
            
            st.markdown("---")
            
            # Métricas
            col1, col2, col3, col4, col5 = st.columns(5)
            
            current = df_win['close'].iloc[-1]
            prev = df_win['close'].iloc[-2]
            change = ((current - prev) / prev * 100) if prev > 0 else 0
            
            with col1:
                st.metric("💰 IBOVESPA", f"{current:,.0f}", f"{change:+.2f}%")
            
            with col2:
                st.metric("🎯 Quantum Score", f"{score_atual:.1f}")
            
            with col3:
                st.metric("📊 TurboStoch", f"{stc.iloc[-1]:.1f}")
            
            with col4:
                h = hurst.iloc[-1]
                regime = "Trend" if h > 0.5 else "Lateral"
                st.metric("🌀 Hurst", regime, f"{h:.2f}")
            
            with col5:
                forca = calcular_forca_win(df_win)
                st.metric("⚡ Força WIN", f"{forca:.0f}/100")
            
            st.markdown("---")
            
            # GRÁFICO
            st.subheader("📊 Análise Completa")
            
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.4, 0.2, 0.2, 0.2],
                subplot_titles=('IBOVESPA + Quantum', 'Quantum Hunter Score', 'TurboStoch', 'Hurst')
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_win.index,
                open=df_win['open'],
                high=df_win['high'],
                low=df_win['low'],
                close=df_win['close'],
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ), row=1, col=1)
            
            # Médias Quantum
            fig.add_trace(go.Scatter(
                x=df_win.index, y=ml,
                name='Média Lenta', line=dict(color='#ff00ff', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_win.index, y=mr,
                name='Média Rápida', line=dict(color='#00d9ff', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_win.index, y=vwap,
                name='VWAP', line=dict(color='#ffaa00', width=2, dash='dash')
            ), row=1, col=1)
            
            # Quantum Score (histograma colorido)
            colors_qh = ['#00ff88' if s > 0 else '#ff4444' for s in score_qh]
            
            fig.add_trace(go.Bar(
                x=df_win.index, y=score_qh,
                marker_color=colors_qh,
                name='Quantum Score',
                showlegend=False
            ), row=2, col=1)
            
            fig.add_hline(y=60, line_dash="dash", line_color="white", row=2, col=1)
            fig.add_hline(y=-60, line_dash="dash", line_color="white", row=2, col=1)
            
            # TurboStoch
            fig.add_trace(go.Scatter(
                x=df_win.index, y=stc,
                name='Schaff', line=dict(color='#00ff88', width=3)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=df_win.index, y=fisher,
                name='Fisher', line=dict(color='#ffffff', width=1, dash='dot')
            ), row=3, col=1)
            
            fig.add_hline(y=90, line_dash="dot", line_color="gray", row=3, col=1)
            fig.add_hline(y=10, line_dash="dot", line_color="gray", row=3, col=1)
            
            # Hurst
            fig.add_trace(go.Scatter(
                x=df_win.index, y=hurst,
                name='Hurst', line=dict(color='#ffaa00', width=2),
                fill='tozeroy'
            ), row=4, col=1)
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=4, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                height=1100,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#ffffff')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Erro ao carregar IBOVESPA")

# ==========================================
# ABA 2: MONITOR MULTI-ATIVOS
# ==========================================

with tab2:
    
    st.subheader("📊 Monitor de Ativos em Tempo Real")
    
    ativos = ["IBOV", "DOLAR", "SP500", "NASDAQ", "DOW"]
    
    placeholder2 = st.empty()
    
    with placeholder2.container():
        
        cols = st.columns(len(ativos))
        
        for idx, ativo in enumerate(ativos):
            with cols[idx]:
                
                df_ativo = get_market_data(ativo, timeframe)
                
                if df_ativo is not None and len(df_ativo) > 50:
                    
                    score, _, _, _ = quantum_hunter_v13(df_ativo, modo=modo)
                    forca = calcular_forca_win(df_ativo)
                    
                    current = df_ativo['close'].iloc[-1]
                    prev = df_ativo['close'].iloc[-2]
                    change = ((current - prev) / prev * 100) if prev > 0 else 0
                    
                    score_atual = score.iloc[-1]
                    
                    # Card colorido
                    if score_atual > 15:
                        card_color = "#00ff88"
                        signal = "🚀 COMPRA"
                    elif score_atual < -15:
                        card_color = "#ff4444"
                        signal = "📉 VENDA"
                    else:
                        card_color = "#ffaa00"
                        signal = "⚖️ NEUTRO"
                    
                    st.markdown(f"""
                    <div class="asset-card" style="border-left-color: {card_color};">
                        <h3 style="color: {card_color};">{ativo}</h3>
                        <p style="font-size: 24px; color: #ffffff;">{current:,.2f}</p>
                        <p style="color: {'#00ff88' if change > 0 else '#ff4444'};">{change:+.2f}%</p>
                        <hr style="border-color: {card_color};">
                        <p><strong>{signal}</strong></p>
                        <p>Score: {score_atual:.1f}</p>
                        <p>Força: {forca:.0f}/100</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error(f"❌ {ativo}")

# ==========================================
# ABA 3: FORÇA EM TEMPO REAL
# ==========================================

with tab3:
    
    st.subheader("⚡ Força do WIN em Tempo Real")
    
    placeholder3 = st.empty()
    
    with placeholder3.container():
        
        df_forca = get_market_data("IBOV", timeframe)
        
        if df_forca is not None and len(df_forca) > 100:
            
            forca_historica = []
            
            for i in range(max(50, len(df_forca) - 200), len(df_forca)):
                df_slice = df_forca.iloc[:i+1]
                if len(df_slice) > 50:
                    forca = calcular_forca_win(df_slice)
                    forca_historica.append({'timestamp': df_slice.index[-1], 'forca': forca})
            
            df_forca_hist = pd.DataFrame(forca_historica).set_index('timestamp')
            
            # Gauge atual
            forca_atual = df_forca_hist['forca'].iloc[-1]
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=forca_atual,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Força Institucional WIN", 'font': {'size': 24, 'color': '#ffffff'}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': '#ffffff'},
                    'bar': {'color': "#00d9ff"},
                    'bgcolor': "black",
                    'borderwidth': 2,
                    'bordercolor': "#ffffff",
                    'steps': [
                        {'range': [0, 30], 'color': '#ff4444'},
                        {'range': [30, 70], 'color': '#ffaa00'},
                        {'range': [70, 100], 'color': '#00ff88'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor="black",
                font={'color': "#ffffff", 'family': "Arial"},
                height=400
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            st.markdown("---")
            
            # Histórico de força
            st.subheader("📈 Evolução da Força")
            
            fig_linha = go.Figure()
            
            fig_linha.add_trace(go.Scatter(
                x=df_forca_hist.index,
                y=df_forca_hist['forca'],
                mode='lines+markers',
                line=dict(color='#00d9ff', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 217, 255, 0.2)'
            ))
            
            fig_linha.add_hline(y=70, line_dash="dash", line_color="green", annotation_text="Força Alta")
            fig_linha.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Força Baixa")
            
            fig_linha.update_layout(
                template='plotly_dark',
                height=400,
                yaxis_title="Força (%)",
                plot_bgcolor='#000000',
                paper_bgcolor='#000000',
                font=dict(color='#ffffff')
            )
            
            st.plotly_chart(fig_linha, use_container_width=True)
        
        else:
            st.error("❌ Erro ao calcular força")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #ffffff;'>🚀 ProfitOne Ultimate V2 | Quantum Hunter + TurboStoch</div>", unsafe_allow_html=True)
