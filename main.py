import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.stats import entropy, linregress
from scipy.signal import hilbert
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime

st.set_page_config(page_title="ProfitOne Ultimate", layout="wide", page_icon="🚀")

# ==========================================
# CSS PROFISSIONAL
# ==========================================
st.markdown("""
<style>
    .main {background-color: #000000;}
    h1, h2, h3 {color: #00d9ff; text-shadow: 0 0 20px #00d9ff;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
    }
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
        color: #000;
        border: 5px solid #00ff88;
    }
    .signal-down {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #fff;
        border: 5px solid #ff4444;
    }
    @keyframes pulse {
        0%, 100% {transform: scale(1);}
        50% {transform: scale(1.02);}
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# FUNÇÕES DE INDICADORES - CINEMÁTICA
# ==========================================

def tema(close, period=20):
    """Triple Exponential Moving Average + Velocity"""
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    velocity = tema.diff()
    return tema, velocity

def kalman_filter(data, Q=1e-5, R=1e-2):
    """Filtro de Kalman - Preço Justo vs Ruído"""
    n = len(data)
    x_hat = np.zeros(n)
    P = np.zeros(n)
    
    x_hat[0] = data.iloc[0]
    P[0] = 1.0
    
    for k in range(1, n):
        x_hat_minus = x_hat[k-1]
        P_minus = P[k-1] + Q
        K = P_minus / (P_minus + R)
        x_hat[k] = x_hat_minus + K * (data.iloc[k] - x_hat_minus)
        P[k] = (1 - K) * P_minus
    
    return pd.Series(x_hat, index=data.index)

def jurik_moving_average(data, period=14, phase=0):
    """Jurik Moving Average (JMA) - Espinha Dorsal"""
    alpha = 2 / (period + 1)
    beta = alpha / 2
    
    jma = data.copy()
    e0, e1, e2 = data.copy(), data.copy(), data.copy()
    
    for i in range(1, len(data)):
        e0.iloc[i] = (1 - alpha) * e0.iloc[i-1] + alpha * data.iloc[i]
        e1.iloc[i] = (1 - beta) * e1.iloc[i-1] + beta * e0.iloc[i]
        e2.iloc[i] = (1 - beta) * e2.iloc[i-1] + beta * e1.iloc[i]
        jma.iloc[i] = e2.iloc[i] + phase * (e2.iloc[i] - e1.iloc[i])
    
    return jma

# ==========================================
# FÍSICA E TERMODINÂMICA
# ==========================================

def shannon_entropy(data, window=20):
    """Entropia de Shannon - O Guardião"""
    def calc_entropy(window_data):
        if len(window_data) < 2:
            return 0
        returns = np.diff(window_data)
        hist, _ = np.histogram(returns, bins=10, density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)
    
    return data.rolling(window=window).apply(calc_entropy, raw=False)

def vortex_indicator(high, low, close, period=14):
    """Vortex Indicator - Saúde da Tendência"""
    vm_plus = abs(high - low.shift(1))
    vm_minus = abs(low - high.shift(1))
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    vi_plus = vm_plus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    vi_minus = vm_minus.rolling(window=period).sum() / tr.rolling(window=period).sum()
    
    return vi_plus, vi_minus

def reynolds_number(close, volume, period=20):
    """Número de Reynolds - Fluxo Laminar vs Turbulento"""
    velocity = close.diff().abs()
    density = volume / volume.rolling(window=period).mean()
    viscosity = close.rolling(window=period).std()
    viscosity = viscosity.replace(0, 0.0001)
    
    reynolds = (velocity * density) / viscosity
    return reynolds

# ==========================================
# ESTATÍSTICA E PROBABILIDADE
# ==========================================

def fisher_transform(high, low, period=10):
    """Fisher Transform - Sniper de Topos e Fundos"""
    hl2 = (high + low) / 2
    max_high = hl2.rolling(window=period).max()
    min_low = hl2.rolling(window=period).min()
    
    value = 2 * ((hl2 - min_low) / (max_high - min_low + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
    return fisher

def z_score(data, window=20):
    """Z-Score - Detector de Anomalias"""
    mean = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    return (data - mean) / (std + 1e-10)

def linear_regression_channel(data, period=100):
    """Regressão Linear + Erro Padrão"""
    def calc_regression(window_data):
        if len(window_data) < 2:
            return np.nan
        x = np.arange(len(window_data)).reshape(-1, 1)
        y = window_data.values
        model = LinearRegression().fit(x, y)
        return model.predict(x)[-1]
    
    regression = data.rolling(window=period).apply(calc_regression, raw=False)
    std = data.rolling(window=period).std()
    
    return regression, regression + 2*std, regression - 2*std

# ==========================================
# FLUXO E MICROESTRUTURA
# ==========================================

def vpin(close, volume, window=50):
    """VPIN - Volume Toxicity"""
    price_change = close.diff()
    buy_volume = volume.where(price_change > 0, 0)
    sell_volume = volume.where(price_change < 0, 0)
    
    vpin = abs(buy_volume - sell_volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vpin

def atomic_density(close, volume, window=20):
    """Densidade Atômica - Icebergs"""
    price_range = close.rolling(window=window).max() - close.rolling(window=window).min()
    price_range = price_range.replace(0, 0.0001)
    total_volume = volume.rolling(window=window).sum()
    
    return total_volume / price_range

def fair_value_gaps(high, low, close):
    """FVG - Fair Value Gaps (Ímãs)"""
    fvg_signals = pd.Series(0, index=close.index)
    
    for i in range(2, len(close)):
        if low.iloc[i] > high.iloc[i-2]:
            fvg_signals.iloc[i] = 1  # Bullish FVG
        elif high.iloc[i] < low.iloc[i-2]:
            fvg_signals.iloc[i] = -1  # Bearish FVG
    
    return fvg_signals

def synthetic_delta(close, volume):
    """Delta Sintético - Raio-X da Vela"""
    price_change = close.diff()
    buy_volume = volume.where(price_change > 0, 0)
    sell_volume = volume.where(price_change < 0, 0)
    
    delta = buy_volume - sell_volume
    return delta.cumsum()

# ==========================================
# CAOS E GEOMETRIA
# ==========================================

def hurst_exponent(data, window=100):
    """Expoente de Hurst - Classificador de Mercado"""
    def calc_hurst(ts):
        if len(ts) < 10:
            return 0.5
        lags = range(2, min(20, len(ts)//2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    return data.rolling(window=window).apply(calc_hurst, raw=False)

def laguerre_filter(close, gamma=0.8):
    """Laguerre Filter - Timing Zero-Lag"""
    L0, L1, L2, L3 = close.copy(), close.copy(), close.copy(), close.copy()
    
    for i in range(1, len(close)):
        L0.iloc[i] = (1 - gamma) * close.iloc[i] + gamma * L0.iloc[i-1]
        L1.iloc[i] = -gamma * L0.iloc[i] + L0.iloc[i-1] + gamma * L1.iloc[i-1]
        L2.iloc[i] = -gamma * L1.iloc[i] + L1.iloc[i-1] + gamma * L2.iloc[i-1]
        L3.iloc[i] = -gamma * L2.iloc[i] + L2.iloc[i-1] + gamma * L3.iloc[i-1]
    
    cu = (L0 > L1).astype(int) * (L0 - L1) + (L1 > L2).astype(int) * (L1 - L2) + (L2 > L3).astype(int) * (L2 - L3)
    cd = (L0 < L1).astype(int) * (L1 - L0) + (L1 < L2).astype(int) * (L2 - L1) + (L2 < L3).astype(int) * (L3 - L2)
    
    laguerre_rsi = cu / (cu + cd + 1e-10) * 100
    return laguerre_rsi

def hilbert_transform_analysis(data):
    """Análise Espectral de Hilbert"""
    analytic_signal = hilbert(data.values)
    amplitude = np.abs(analytic_signal)
    phase = np.angle(analytic_signal)
    
    return pd.Series(amplitude, index=data.index), pd.Series(phase, index=data.index)

def center_of_gravity(close, period=10):
    """Center of Gravity (Ehlers) - GPS Zero-Lag"""
    cog = pd.Series(index=close.index, dtype=float)
    
    for i in range(period, len(close)):
        num = sum([(j+1) * close.iloc[i-period+j+1] for j in range(period)])
        den = sum([close.iloc[i-period+j+1] for j in range(period)])
        cog.iloc[i] = -num / (den + 1e-10)
    
    return cog

# ==========================================
# FUNÇÃO PRINCIPAL - BUSCAR DADOS
# ==========================================

@st.cache_data(ttl=10)
def get_market_data(symbol, interval="5m"):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": interval, "range": "1d"}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        
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
        
        return df
    except Exception as e:
        st.error(f"Erro: {e}")
        return None

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================

st.markdown("<h1 style='text-align: center;'>🚀 PROFITONE ULTIMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Sistema Completo com 25+ Indicadores Avançados</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configurações")

symbols = {
    "₿ BITCOIN": "BTC-USD",
    "^BVSP IBOVESPA": "^BVSP",
    "💵 DÓLAR": "USDBRL=X",
    "S&P 500": "^GSPC"
}

selected = st.sidebar.selectbox("Ativo:", list(symbols.keys()))
symbol = symbols[selected]

timeframe = st.sidebar.selectbox("Timeframe:", ["1m", "5m", "15m", "1h"], index=1)

# Indicadores para ativar
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Indicadores Ativos")

show_tema = st.sidebar.checkbox("TEMA + Velocity", value=True)
show_kalman = st.sidebar.checkbox("Kalman Filter", value=True)
show_jma = st.sidebar.checkbox("JMA", value=False)
show_entropy = st.sidebar.checkbox("Entropia Shannon", value=True)
show_vortex = st.sidebar.checkbox("Vortex", value=False)
show_fisher = st.sidebar.checkbox("Fisher Transform", value=True)
show_hurst = st.sidebar.checkbox("Hurst Exponent", value=True)
show_vpin = st.sidebar.checkbox("VPIN", value=False)
show_cog = st.sidebar.checkbox("Center of Gravity", value=False)

# Main
placeholder = st.empty()

while True:
    with placeholder.container():
        
        df = get_market_data(symbol, timeframe)
        
        if df is not None and len(df) > 50:
            
            # Timestamp
            now = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"<p style='text-align: center; color: #888;'>🕐 {now}</p>", unsafe_allow_html=True)
            
            # Calcular TODOS os indicadores
            indicators = {}
            
            if show_tema:
                indicators['tema'], indicators['tema_velocity'] = tema(df['close'])
            
            if show_kalman:
                indicators['kalman'] = kalman_filter(df['close'])
            
            if show_jma:
                indicators['jma'] = jurik_moving_average(df['close'])
            
            if show_entropy:
                indicators['entropy'] = shannon_entropy(df['close'])
            
            if show_vortex:
                indicators['vi_plus'], indicators['vi_minus'] = vortex_indicator(df['high'], df['low'], df['close'])
            
            if show_fisher:
                indicators['fisher'] = fisher_transform(df['high'], df['low'])
            
            if show_hurst:
                indicators['hurst'] = hurst_exponent(df['close'])
            
            if show_vpin:
                indicators['vpin'] = vpin(df['close'], df['volume'])
            
            if show_cog:
                indicators['cog'] = center_of_gravity(df['close'])
            
            # Calcular Z-Score e Reynolds sempre
            indicators['z_score'] = z_score(df['close'])
            indicators['reynolds'] = reynolds_number(df['close'], df['volume'])
            
            # SCORE MESTRE
            master_score = 0
            count = 0
            
            # TEMA
            if 'tema_velocity' in indicators:
                if indicators['tema_velocity'].iloc[-1] > 0:
                    master_score += 15
                else:
                    master_score -= 15
                count += 1
            
            # Hurst
            if 'hurst' in indicators:
                hurst_val = indicators['hurst'].iloc[-1]
                if hurst_val > 0.5:
                    master_score += 20  # Tendência
                elif hurst_val < 0.5:
                    master_score -= 10  # Lateral
                count += 1
            
            # Fisher
            if 'fisher' in indicators:
                fisher_val = indicators['fisher'].iloc[-1]
                if fisher_val > 2:
                    master_score -= 15  # Overbought
                elif fisher_val < -2:
                    master_score += 15  # Oversold
                count += 1
            
            # Entropy
            if 'entropy' in indicators:
                if indicators['entropy'].iloc[-1] < 1.5:
                    master_score += 10  # Ordem (bom para tendência)
                count += 1
            
            # Normalizar
            if count > 0:
                master_score = master_score / count
            
            # SIGNAL BOARD
            if master_score > 5:
                signal_class = "signal-up"
                signal_text = "🚀 COMPRA FORTE"
            elif master_score < -5:
                signal_class = "signal-down"
                signal_text = "📉 VENDA FORTE"
            else:
                signal_text = "⚖️ NEUTRO"
                signal_class = ""
            
            if signal_class:
                st.markdown(f"""
                <div class="signal-board {signal_class}">
                    <div>{signal_text}</div>
                    <div style="font-size: 80px; margin-top: 15px;">Score: {master_score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"⚖️ Mercado Neutro | Score: {master_score:.1f}")
            
            st.markdown("---")
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            price_change_pct = ((current_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
            
            with col1:
                st.metric("💰 Preço", f"${current_price:,.2f}", f"{price_change_pct:+.2f}%")
            
            with col2:
                if 'hurst' in indicators:
                    h = indicators['hurst'].iloc[-1]
                    regime = "Tendência" if h > 0.5 else "Lateral"
                    st.metric("📊 Regime (Hurst)", regime, f"{h:.2f}")
            
            with col3:
                if 'entropy' in indicators:
                    ent = indicators['entropy'].iloc[-1]
                    st.metric("⚛️ Entropia", f"{ent:.2f}")
            
            with col4:
                vol = df['volume'].iloc[-1]
                st.metric("📦 Volume", f"{vol/1e6:.1f}M" if vol > 1e6 else f"{vol:,.0f}")
            
            st.markdown("---")
            
            # GRÁFICO PRINCIPAL
            st.subheader(f"📊 {selected} - Análise Completa")
            
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.2, 0.15, 0.15],
                subplot_titles=('Price + Indicators', 'Oscillators', 'Hurst + Entropy', 'Volume')
            )
            
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
            ), row=1, col=1)
            
            # TEMA
            if 'tema' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['tema'],
                    name='TEMA', line=dict(color='#00d9ff', width=2)
                ), row=1, col=1)
            
            # Kalman
            if 'kalman' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['kalman'],
                    name='Kalman', line=dict(color='#ff00ff', width=2)
                ), row=1, col=1)
            
            # JMA
            if 'jma' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['jma'],
                    name='JMA', line=dict(color='#ffaa00', width=2)
                ), row=1, col=1)
            
            # Fisher
            if 'fisher' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['fisher'],
                    name='Fisher', line=dict(color='#00d9ff')
                ), row=2, col=1)
                fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
            
            # Z-Score
            fig.add_trace(go.Scatter(
                x=df.index, y=indicators['z_score'],
                name='Z-Score', line=dict(color='#ff00ff')
            ), row=2, col=1)
            
            # Hurst
            if 'hurst' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['hurst'],
                    name='Hurst', line=dict(color='#00ff88')
                ), row=3, col=1)
                fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=3, col=1)
            
            # Entropy
            if 'entropy' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['entropy'],
                    name='Entropy', line=dict(color='#ffaa00')
                ), row=3, col=1)
            
            # Volume
            colors = ['#00ff88' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff4444' 
                      for i in range(len(df))]
            
            fig.add_trace(go.Bar(
                x=df.index, y=df['volume'],
                marker_color=colors,
                showlegend=False
            ), row=4, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                height=1000,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                plot_bgcolor='#000000',
                paper_bgcolor='#000000'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Análise detalhada
            with st.expander("🔬 Análise Detalhada dos Indicadores"):
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Indicadores de Tendência")
                    
                    if 'tema_velocity' in indicators:
                        vel = indicators['tema_velocity'].iloc[-1]
                        if vel > 0:
                            st.success(f"✅ TEMA Velocity: +{vel:.2f} (Momentum de Alta)")
                        else:
                            st.error(f"❌ TEMA Velocity: {vel:.2f} (Momentum de Baixa)")
                    
                    if 'hurst' in indicators:
                        h = indicators['hurst'].iloc[-1]
                        if h > 0.6:
                            st.success(f"✅ Hurst: {h:.2f} (Tendência Forte)")
                        elif h > 0.5:
                            st.info(f"ℹ️ Hurst: {h:.2f} (Tendência Fraca)")
                        else:
                            st.warning(f"⚠️ Hurst: {h:.2f} (Mercado Lateral)")
                
                with col2:
                    st.markdown("### 🎯 Indicadores de Reversão")
                    
                    if 'fisher' in indicators:
                        f = indicators['fisher'].iloc[-1]
                        if f > 2:
                            st.warning(f"⚠️ Fisher: {f:.2f} (Overbought - Possível reversão)")
                        elif f < -2:
                            st.success(f"✅ Fisher: {f:.2f} (Oversold - Possível alta)")
                        else:
                            st.info(f"ℹ️ Fisher: {f:.2f} (Neutro)")
                    
                    z = indicators['z_score'].iloc[-1]
                    if abs(z) > 2:
                        st.warning(f"⚠️ Z-Score: {z:.2f} (Anomalia detectada!)")
                    else:
                        st.info(f"ℹ️ Z-Score: {z:.2f} (Normal)")
        
        else:
            st.error("❌ Erro ao carregar dados")
    
    time.sleep(15)
    st.rerun()
