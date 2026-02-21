"""
ProfitOne V2.0 - Sistema Avançado de Análise Técnica
Sistema Quântico com Indicadores de Engenharia de Mercado

Módulos:
1. Cinemática & Velocidade (TEMA, Kalman, JMA, Vortex)
2. Física & Termodinâmica (Entropy, Reynolds, FVG)
3. Estatística & Probabilidade (Fisher, Hurst, Z-Score, VPIN)
4. Fluxo & Microestrutura (VPIN, Wicks, Trapped Traders, Delta)
5. Caos & Geometria (Hurst, Laguerre, COG, Weis Wave)
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
    page_title="ProfitOne V2.0 - Sistema Quântico",
    page_icon="🚀",
    layout="wide"
)

# CSS Avançado
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
    .indicator-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #00ff88;
    }
    .signal-buy {
        background: rgba(0, 255, 136, 0.2);
        border-left: 4px solid #00ff88;
    }
    .signal-sell {
        background: rgba(255, 68, 68, 0.2);
        border-left: 4px solid #ff4444;
    }
    .signal-neutral {
        background: rgba(255, 170, 0, 0.2);
        border-left: 4px solid #ffaa00;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MÓDULO 1: CINEMÁTICA & VELOCIDADE
# ============================================================================

def calculate_tema(data, period=21):
    """Triple Exponential Moving Average"""
    ema1 = data.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema


def calculate_velocity(prices, period=3):
    """Velocidade de mudança de preço"""
    return prices.diff(period) / period


def shannon_entropy(data, bins=10):
    """Entropia de Shannon - Medidor de Caos"""
    counts, _ = np.histogram(data.dropna(), bins=bins)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def kalman_filter(prices, q=0.01, r=0.1):
    """Filtro de Kalman - Preço Justo"""
    n = len(prices)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = prices[0]
    P[0] = 1.0
    
    for k in range(1, n):
        # Predição
        xhatminus = xhat[k-1]
        Pminus = P[k-1] + q
        
        # Atualização
        K = Pminus / (Pminus + r)
        xhat[k] = xhatminus + K * (prices[k] - xhatminus)
        P[k] = (1 - K) * Pminus
    
    return xhat


def vortex_indicator(high, low, close, period=14):
    """Vortex Indicator - Medidor de Saúde da Tendência"""
    vm_plus = np.abs(high - low.shift(1))
    vm_minus = np.abs(low - high.shift(1))
    
    true_high = np.maximum(high, close.shift(1))
    true_low = np.minimum(low, close.shift(1))
    true_range = true_high - true_low
    
    vi_plus = vm_plus.rolling(window=period).sum() / true_range.rolling(window=period).sum()
    vi_minus = vm_minus.rolling(window=period).sum() / true_range.rolling(window=period).sum()
    
    return vi_plus, vi_minus


# ============================================================================
# MÓDULO 2: FÍSICA & TERMODINÂMICA
# ============================================================================

def reynolds_number(close, volume, window=14):
    """Número de Reynolds - Fluxo Laminar vs Turbulento"""
    returns = close.pct_change()
    volatility = returns.rolling(window=window).std()
    avg_volatility = volatility.rolling(window=window).mean()
    
    avg_volume = volume.rolling(window=window).mean()
    normalized_volume = volume / avg_volume
    
    reynolds = (volatility * normalized_volume) / (avg_volatility + 1e-10)
    reynolds = reynolds * 1000
    
    return reynolds


def detect_fvg(high, low, close):
    """Fair Value Gaps (Imãs)"""
    bullish_fvg = low.iloc[-1] > high.iloc[-3]
    bearish_fvg = high.iloc[-1] < low.iloc[-3]
    
    if bullish_fvg:
        return 'BULLISH', low.iloc[-1], high.iloc[-3]
    elif bearish_fvg:
        return 'BEARISH', high.iloc[-1], low.iloc[-3]
    else:
        return 'NONE', None, None


# ============================================================================
# MÓDULO 3: ESTATÍSTICA & PROBABILIDADE
# ============================================================================

def fisher_transform(high, low, period=10):
    """Fisher Transform - Sniper de Topos e Fundos"""
    hl_range = high.rolling(window=period).max() - low.rolling(window=period).min()
    hl_range = hl_range.replace(0, 1e-10)
    
    value = 2 * ((high - low.rolling(window=period).min()) / hl_range - 0.5)
    value = value.clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    fisher = fisher.fillna(0)
    
    return fisher.ewm(span=3).mean()


def hurst_exponent(ts, max_lag=20):
    """Expoente de Hurst - Classificador de Mercado"""
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        ts_split = [ts[i:i+lag] for i in range(0, len(ts), lag)]
        rs_values = []
        
        for subset in ts_split:
            if len(subset) < lag:
                continue
            
            mean = np.mean(subset)
            deviations = subset - mean
            cumulative_deviations = np.cumsum(deviations)
            R = np.max(cumulative_deviations) - np.min(cumulative_deviations)
            S = np.std(subset)
            
            if S > 0:
                rs_values.append(R / S)
        
        if rs_values:
            tau.append(np.mean(rs_values))
    
    if len(tau) < 2:
        return 0.5
    
    lags_log = np.log(list(lags[:len(tau)]))
    tau_log = np.log(tau)
    
    coeffs = np.polyfit(lags_log, tau_log, 1)
    return coeffs[0]


def calculate_zscore(data, window=20):
    """Z-Score - Detector de Anomalias"""
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    zscore = (data - mean) / (std + 1e-10)
    return zscore


def vpin_indicator(close, volume, window=50):
    """VPIN - Volume Toxicity"""
    price_change = close.diff()
    
    buy_volume = volume.where(price_change > 0, 0)
    sell_volume = volume.where(price_change < 0, 0)
    
    volume_imbalance = np.abs(buy_volume - sell_volume)
    total_volume = volume
    
    vpin = (volume_imbalance.rolling(window=window).sum() / 
            total_volume.rolling(window=window).sum())
    
    return vpin


# ============================================================================
# MÓDULO 4: FLUXO & MICROESTRUTURA
# ============================================================================

def analyze_wicks(open_price, high, low, close):
    """Análise de Pavios (Wicks)"""
    body = np.abs(close - open_price)
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    total_range = high - low
    
    upper_wick_ratio = upper_wick / (total_range + 1e-10)
    lower_wick_ratio = lower_wick / (total_range + 1e-10)
    
    return upper_wick_ratio, lower_wick_ratio


def detect_trapped_traders(high, low, close, lookback=20):
    """Teoria da Dor Máxima - Trapped Traders"""
    recent_high = high.rolling(window=lookback).max()
    recent_low = low.rolling(window=lookback).min()
    
    current_high = high.iloc[-1]
    current_low = low.iloc[-1]
    current_close = close.iloc[-1]
    
    prev_recent_high = recent_high.iloc[-2]
    prev_recent_low = recent_low.iloc[-2]
    
    breakout_up = current_high > prev_recent_high
    breakout_down = current_low < prev_recent_low
    
    trapped_long = breakout_up and (current_close < prev_recent_high)
    trapped_short = breakout_down and (current_close > prev_recent_low)
    
    return trapped_long, trapped_short


def synthetic_delta(open_price, close, volume):
    """Delta Sintético - Raio-X da Vela"""
    delta = (close - open_price) * volume
    cumulative_delta = delta.rolling(window=14).sum()
    
    avg_volume = volume.rolling(window=14).mean()
    normalized_delta = cumulative_delta / (avg_volume * 14 + 1e-10)
    
    return normalized_delta


# ============================================================================
# MÓDULO 5: CAOS, CIBERNÉTICA & GEOMETRIA
# ============================================================================

def laguerre_rsi(close, gamma=0.5):
    """Laguerre RSI - Zero Lag"""
    prices = close.values
    n = len(prices)
    
    L0 = np.zeros(n)
    L1 = np.zeros(n)
    L2 = np.zeros(n)
    L3 = np.zeros(n)
    
    L0[0] = L1[0] = L2[0] = L3[0] = prices[0]
    
    for i in range(1, n):
        L0[i] = (1 - gamma) * prices[i] + gamma * L0[i-1]
        L1[i] = -gamma * L0[i] + L0[i-1] + gamma * L1[i-1]
        L2[i] = -gamma * L1[i] + L1[i-1] + gamma * L2[i-1]
        L3[i] = -gamma * L2[i] + L2[i-1] + gamma * L3[i-1]
    
    lrsi = np.zeros(n)
    
    for i in range(n):
        cu = 0
        cd = 0
        
        if L0[i] >= L1[i]:
            cu += L0[i] - L1[i]
        else:
            cd += L1[i] - L0[i]
        
        if L1[i] >= L2[i]:
            cu += L1[i] - L2[i]
        else:
            cd += L2[i] - L1[i]
        
        if L2[i] >= L3[i]:
            cu += L2[i] - L3[i]
        else:
            cd += L3[i] - L2[i]
        
        if cu + cd != 0:
            lrsi[i] = cu / (cu + cd)
        else:
            lrsi[i] = 0
    
    return pd.Series(lrsi, index=close.index)


def center_of_gravity(close, period=10):
    """Center of Gravity (Ehlers) - GPS Zero-Lag"""
    cog_values = []
    
    for i in range(period - 1, len(close)):
        window = close.iloc[i - period + 1:i + 1].values
        weights = np.arange(1, period + 1)
        
        numerator = -np.sum(weights * window)
        denominator = np.sum(window)
        
        if denominator != 0:
            cog = numerator / denominator
        else:
            cog = 0
        
        cog_values.append(cog)
    
    result = [np.nan] * (period - 1) + cog_values
    return pd.Series(result, index=close.index)


# ============================================================================
# SISTEMA DE PONTUAÇÃO MULTI-INDICADOR
# ============================================================================

def calculate_quantum_score(df):
    """Score Quântico Mestre - Combina Todos os Indicadores"""
    
    scores = {}
    
    # 1. CINEMÁTICA
    tema = calculate_tema(df['close'], 21)
    velocity = calculate_velocity(df['close'], 3)
    entropy = shannon_entropy(df['close'].tail(20), bins=10)
    kalman = kalman_filter(df['close'].values)
    vi_plus, vi_minus = vortex_indicator(df['high'], df['low'], df['close'], 14)
    
    tema_score = 50 if df['close'].iloc[-1] > tema.iloc[-1] else -50
    velocity_score = np.clip(velocity.iloc[-1] * 100, -50, 50)
    entropy_score = -50 if entropy < 2.5 else 0  # Baixa entropia = tendência
    vortex_score = 50 if vi_plus.iloc[-1] > vi_minus.iloc[-1] else -50
    
    scores['cinemática'] = (tema_score + velocity_score + entropy_score + vortex_score) / 4
    
    # 2. FÍSICA & TERMODINÂMICA
    reynolds = reynolds_number(df['close'], df['volume'], 14)
    fvg_type, fvg_top, fvg_bottom = detect_fvg(df['high'], df['low'], df['close'])
    
    reynolds_score = 50 if reynolds.iloc[-1] < 2300 else -30  # Laminar = bom
    fvg_score = 40 if fvg_type == 'BULLISH' else (-40 if fvg_type == 'BEARISH' else 0)
    
    scores['física'] = (reynolds_score + fvg_score) / 2
    
    # 3. ESTATÍSTICA
    fisher = fisher_transform(df['high'], df['low'], 10)
    hurst = hurst_exponent(df['close'].tail(100).values, max_lag=20)
    zscore = calculate_zscore(df['close'], 20)
    vpin = vpin_indicator(df['close'], df['volume'], 50)
    
    fisher_score = np.clip(fisher.iloc[-1] * 25, -50, 50)
    hurst_score = 50 if hurst > 0.55 else (-50 if hurst < 0.45 else 0)
    zscore_score = -50 if zscore.iloc[-1] > 2 else (50 if zscore.iloc[-1] < -2 else 0)
    vpin_score = -30 if vpin.iloc[-1] > 0.5 else 0
    
    scores['estatística'] = (fisher_score + hurst_score + zscore_score + vpin_score) / 4
    
    # 4. FLUXO & MICROESTRUTURA
    upper_wick, lower_wick = analyze_wicks(df['open'], df['high'], df['low'], df['close'])
    trapped_long, trapped_short = detect_trapped_traders(df['high'], df['low'], df['close'], 20)
    delta = synthetic_delta(df['open'], df['close'], df['volume'])
    
    wick_score = 50 if lower_wick.iloc[-1] > 0.5 else (-50 if upper_wick.iloc[-1] > 0.5 else 0)
    trapped_score = 60 if trapped_short else (-60 if trapped_long else 0)
    delta_score = np.clip(delta.iloc[-1] * 50, -50, 50)
    
    scores['microestrutura'] = (wick_score + trapped_score + delta_score) / 3
    
    # 5. CAOS & GEOMETRIA
    lrsi = laguerre_rsi(df['close'], gamma=0.5)
    cog = center_of_gravity(df['close'], period=10)
    
    lrsi_score = 50 if lrsi.iloc[-1] < 0.2 else (-50 if lrsi.iloc[-1] > 0.8 else 0)
    cog_score = np.clip(-cog.iloc[-1] * 20, -50, 50)
    
    scores['caos'] = (lrsi_score + cog_score + hurst_score) / 3
    
    # SCORE MESTRE (ponderado)
    weights = {
        'cinemática': 1.2,
        'física': 1.0,
        'estatística': 1.3,
        'microestrutura': 1.5,
        'caos': 0.8
    }
    
    weighted_scores = [scores[key] * weights[key] for key in scores.keys()]
    master_score = np.mean(weighted_scores)
    master_score = np.clip(master_score, -100, 100)
    
    # SINAL
    if master_score > 40:
        signal = 'BUY'
    elif master_score < -40:
        signal = 'SELL'
    else:
        signal = 'NEUTRAL'
    
    return {
        'master_score': master_score,
        'signal': signal,
        'module_scores': scores,
        'indicators': {
            'tema': tema.iloc[-1],
            'entropy': entropy,
            'hurst': hurst,
            'fisher': fisher.iloc[-1],
            'vpin': vpin.iloc[-1],
            'lrsi': lrsi.iloc[-1],
            'fvg': fvg_type,
            'trapped_long': trapped_long,
            'trapped_short': trapped_short
        }
    }


# ============================================================================
# BUSCA DE DADOS
# ============================================================================

@st.cache_data(ttl=60)
def get_data(symbol, period="5d", interval="15m"):
    """Busca dados do Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            df = ticker.history(period="1mo", interval="1h")
        
        if df.empty:
            return pd.DataFrame(), "Sem dados disponíveis"
        
        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index()
        
        return df, None
        
    except Exception as e:
        return pd.DataFrame(), str(e)


# ============================================================================
# GRÁFICO AVANÇADO
# ============================================================================

def create_advanced_chart(df, indicators):
    """Cria gráfico com múltiplos indicadores"""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('Preço & TEMA & Kalman', 'Fisher Transform', 'Laguerre RSI', 'VPIN & Hurst')
    )
    
    # ROW 1: CANDLESTICK
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
    
    # TEMA
    tema = calculate_tema(df['close'], 21)
    fig.add_trace(
        go.Scatter(x=df.index, y=tema, name='TEMA 21', line=dict(color='cyan', width=2)),
        row=1, col=1
    )
    
    # Kalman
    kalman = kalman_filter(df['close'].values)
    fig.add_trace(
        go.Scatter(x=df.index, y=kalman, name='Kalman Fair Price', 
                  line=dict(color='yellow', width=2, dash='dot')),
        row=1, col=1
    )
    
    # ROW 2: FISHER TRANSFORM
    fisher = fisher_transform(df['high'], df['low'], 10)
    fig.add_trace(
        go.Scatter(x=df.index, y=fisher, name='Fisher', 
                  line=dict(color='purple', width=2), fill='tozeroy'),
        row=2, col=1
    )
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
    
    # ROW 3: LAGUERRE RSI
    lrsi = laguerre_rsi(df['close'], gamma=0.5)
    fig.add_trace(
        go.Scatter(x=df.index, y=lrsi, name='Laguerre RSI', 
                  line=dict(color='orange', width=2)),
        row=3, col=1
    )
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=0.2, line_dash="dash", line_color="green", row=3, col=1)
    
    # ROW 4: VPIN
    vpin = vpin_indicator(df['close'], df['volume'], 50)
    fig.add_trace(
        go.Scatter(x=df.index, y=vpin, name='VPIN (Toxicity)', 
                  line=dict(color='red', width=2), fill='tozeroy'),
        row=4, col=1
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", row=4, col=1)
    
    # LAYOUT
    fig.update_layout(
        height=1000,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    fig.update_xaxes(title_text="Data/Hora", row=4, col=1)
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Fisher", row=2, col=1)
    fig.update_yaxes(title_text="LRSI", row=3, col=1)
    fig.update_yaxes(title_text="VPIN", row=4, col=1)
    
    return fig


# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

def main():
    # HEADER
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🚀 PROFITONE V2.0 - SISTEMA QUÂNTICO 🚀</h1>
        <p style='font-size: 18px; color: #00ff88;'>Engenharia de Mercado | Multi-Indicador</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        symbols_preset = {
            "Ibovespa (^BVSP)": "^BVSP",
            "PETR4.SA": "PETR4.SA",
            "VALE3.SA": "VALE3.SA",
            "ITUB4.SA": "ITUB4.SA",
            "S&P 500 (^GSPC)": "^GSPC",
            "Bitcoin (BTC-USD)": "BTC-USD",
            "Ethereum (ETH-USD)": "ETH-USD",
            "Custom": "CUSTOM"
        }
        
        selected_preset = st.selectbox("📊 Ativo", list(symbols_preset.keys()), index=0)
        
        if symbols_preset[selected_preset] == "CUSTOM":
            symbol = st.text_input("Digite o símbolo:", value="^BVSP")
        else:
            symbol = symbols_preset[selected_preset]
        
        timeframe = st.selectbox("⏱️ Timeframe", ["15 min", "1 hora", "1 dia"], index=0)
        
        interval_map = {"15 min": "15m", "1 hora": "1h", "1 dia": "1d"}
        interval = interval_map[timeframe]
        
        period_map = {"15m": "5d", "1h": "1mo", "1d": "6mo"}
        period = period_map.get(interval, "5d")
        
        st.markdown("---")
        
        if st.button("🔄 Atualizar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📚 Módulos Ativos")
        st.caption("🏎️ Cinemática & Velocidade")
        st.caption("⚛️ Física & Termodinâmica")
        st.caption("🎲 Estatística & Probabilidade")
        st.caption("🐋 Fluxo & Microestrutura")
        st.caption("🌀 Caos & Geometria")
    
    # BUSCAR DADOS
    with st.spinner(f"📊 Carregando {symbol}..."):
        df, error = get_data(symbol, period, interval)
    
    if error or df.empty:
        st.error(f"❌ {error if error else 'Sem dados'}")
        st.info("💡 Tente outro ativo (ex: PETR4.SA, BTC-USD)")
        return
    
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df = df.set_index('date')
    
    # CALCULAR SCORE QUÂNTICO
    with st.spinner("🧮 Calculando Score Quântico..."):
        result = calculate_quantum_score(df)
    
    # MÉTRICAS PRINCIPAIS
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Score Quântico", f"{result['master_score']:.1f}", 
                 delta=f"{result['master_score']:.1f}")
    
    with col2:
        st.metric("📊 Hurst Exp", f"{result['indicators']['hurst']:.3f}")
    
    with col3:
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
        price_change = current_price - prev_price
        st.metric("💰 Preço", f"R$ {current_price:.2f}", delta=f"{price_change:+.2f}")
    
    with col4:
        signal_colors = {'BUY': '🟢', 'SELL': '🔴', 'NEUTRAL': '🟡'}
        signal_class = f"signal-{result['signal'].lower()}"
        st.markdown(f"<div class='indicator-card {signal_class}' style='text-align: center;'>"
                   f"<h3>{signal_colors[result['signal']]} {result['signal']}</h3></div>",
                   unsafe_allow_html=True)
    
    # SCORES POR MÓDULO
    st.markdown("---")
    st.markdown("## 📊 Scores por Módulo")
    
    cols = st.columns(5)
    module_names = ['🏎️ Cinemática', '⚛️ Física', '🎲 Estatística', '🐋 Microestrutura', '🌀 Caos']
    
    for col, (name, (key, score)) in zip(cols, zip(module_names, result['module_scores'].items())):
        with col:
            signal_class = "signal-buy" if score > 20 else ("signal-sell" if score < -20 else "signal-neutral")
            col.markdown(f"<div class='indicator-card {signal_class}'>"
                        f"<h4>{name}</h4><h2>{score:.1f}</h2></div>",
                        unsafe_allow_html=True)
    
    # GRÁFICO AVANÇADO
    st.markdown("---")
    st.markdown("## 📈 Gráfico Quântico Multi-Indicador")
    
    fig = create_advanced_chart(df, result['indicators'])
    st.plotly_chart(fig, use_container_width=True)
    
    # INDICADORES DETALHADOS
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏎️ Cinemática")
        st.write(f"**TEMA:** R$ {result['indicators']['tema']:.2f}")
        st.write(f"**Entropia:** {result['indicators']['entropy']:.3f}")
        if result['indicators']['entropy'] < 2.5:
            st.success("✅ Mercado Ordenado (Trending)")
        else:
            st.warning("⚠️ Mercado Caótico (Lateral)")
    
    with col2:
        st.markdown("### 🎲 Estatística")
        st.write(f"**Fisher:** {result['indicators']['fisher']:.3f}")
        st.write(f"**Hurst:** {result['indicators']['hurst']:.3f}")
        
        if result['indicators']['hurst'] > 0.55:
            st.success("✅ Tendência Persistente")
        elif result['indicators']['hurst'] < 0.45:
            st.info("ℹ️ Mean Reverting")
        else:
            st.warning("⚠️ Random Walk")
    
    with col3:
        st.markdown("### 🐋 Microestrutura")
        st.write(f"**VPIN:** {result['indicators']['vpin']:.3f}")
        st.write(f"**Laguerre RSI:** {result['indicators']['lrsi']:.3f}")
        st.write(f"**FVG:** {result['indicators']['fvg']}")
        
        if result['indicators']['trapped_long']:
            st.error("🔴 LONGS TRAPPED!")
        elif result['indicators']['trapped_short']:
            st.success("🟢 SHORTS TRAPPED!")
    
    # FOOTER
    st.markdown("---")
    st.caption(f"📊 {symbol} | {timeframe} | {len(df)} candles | {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    st.caption("🚀 ProfitOne V2.0 - Sistema Quântico de Análise Técnica")
    st.caption("⚠️ Apenas fins educacionais. Não constitui recomendação de investimento.")


if __name__ == "__main__":
    main()
