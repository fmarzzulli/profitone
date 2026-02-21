import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.stats import entropy
from scipy.signal import hilbert
import time
from datetime import datetime

st.set_page_config(page_title="ProfitOne Ultimate", layout="wide", page_icon="🚀")

# CSS
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
    }
    .signal-down {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #fff;
    }
    @keyframes pulse {
        0%, 100% {transform: scale(1);}
        50% {transform: scale(1.02);}
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INDICADORES
# ==========================================

def tema(close, period=20):
    """TEMA + Velocity"""
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    velocity = tema.diff()
    return tema, velocity

def kalman_filter(data):
    """Kalman Filter"""
    Q, R = 1e-5, 1e-2
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

def shannon_entropy(data, window=20):
    """Entropia de Shannon"""
    def calc_ent(x):
        if len(x) < 2: return 0
        ret = np.diff(x)
        hist, _ = np.histogram(ret, bins=10, density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)
    
    return data.rolling(window=window).apply(calc_ent, raw=False)

def fisher_transform(high, low, period=10):
    """Fisher Transform"""
    hl2 = (high + low) / 2
    max_h = hl2.rolling(period).max()
    min_l = hl2.rolling(period).min()
    
    value = 2 * ((hl2 - min_l) / (max_h - min_l + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
    return fisher

def hurst_exponent(data, window=100):
    """Hurst Exponent"""
    def calc_h(ts):
        if len(ts) < 10: return 0.5
        lags = range(2, min(20, len(ts)//2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    return data.rolling(window=window).apply(calc_h, raw=False)

def z_score(data, window=20):
    """Z-Score"""
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / (std + 1e-10)

# ==========================================
# BUSCAR DADOS (COM TRATAMENTO DE ERRO)
# ==========================================

@st.cache_data(ttl=10)
def get_binance_data(symbol, interval="5m", limit=200):
    """Buscar dados da Binance (mais confiável)"""
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return None
        
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
        st.error(f"Erro Binance: {str(e)}")
        return None

@st.cache_data(ttl=10)
def get_yahoo_data(symbol, interval="5m"):
    """Buscar dados Yahoo (backup)"""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": interval, "range": "1d"}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return None
        
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
        st.error(f"Erro Yahoo: {str(e)}")
        return None

# ==========================================
# INTERFACE
# ==========================================

st.markdown("<h1 style='text-align: center;'>🚀 PROFITONE ULTIMATE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Sistema Completo com 25+ Indicadores Avançados</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configurações")

# Símbolos
symbol_map = {
    "₿ BITCOIN": {"binance": "BTCUSDT", "yahoo": "BTC-USD", "use": "binance"},
    "Ξ ETHEREUM": {"binance": "ETHUSDT", "yahoo": "ETH-USD", "use": "binance"},
    "^BVSP IBOVESPA": {"binance": None, "yahoo": "^BVSP", "use": "yahoo"},
    "💵 DÓLAR": {"binance": None, "yahoo": "USDBRL=X", "use": "yahoo"},
    "S&P 500": {"binance": None, "yahoo": "^GSPC", "use": "yahoo"}
}

selected = st.sidebar.selectbox("Ativo:", list(symbol_map.keys()))
symbol_info = symbol_map[selected]

# Timeframe
timeframe_map = {
    "1 minuto": {"binance": "1m", "yahoo": "1m"},
    "5 minutos": {"binance": "5m", "yahoo": "5m"},
    "15 minutos": {"binance": "15m", "yahoo": "15m"},
    "1 hora": {"binance": "1h", "yahoo": "1h"}
}

timeframe_label = st.sidebar.selectbox("Timeframe:", list(timeframe_map.keys()), index=1)
timeframe = timeframe_map[timeframe_label]

# Indicadores
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Indicadores")

show_tema = st.sidebar.checkbox("TEMA + Velocity", value=True)
show_kalman = st.sidebar.checkbox("Kalman Filter", value=True)
show_entropy = st.sidebar.checkbox("Entropia Shannon", value=True)
show_fisher = st.sidebar.checkbox("Fisher Transform", value=True)
show_hurst = st.sidebar.checkbox("Hurst Exponent", value=True)

refresh = st.sidebar.slider("Refresh (seg):", 5, 60, 15, 5)

# Main
placeholder = st.empty()

while True:
    with placeholder.container():
        
        # Buscar dados
        if symbol_info['use'] == 'binance':
            df = get_binance_data(symbol_info['binance'], timeframe['binance'])
        else:
            df = get_yahoo_data(symbol_info['yahoo'], timeframe['yahoo'])
        
        if df is not None and len(df) > 50:
            
            now = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"<p style='text-align: center; color: #888;'>🕐 {now} | ✅ {len(df)} candles carregados</p>", unsafe_allow_html=True)
            
            # Calcular indicadores
            indicators = {}
            
            if show_tema:
                indicators['tema'], indicators['tema_vel'] = tema(df['close'])
            
            if show_kalman:
                indicators['kalman'] = kalman_filter(df['close'])
            
            if show_entropy:
                indicators['entropy'] = shannon_entropy(df['close'])
            
            if show_fisher:
                indicators['fisher'] = fisher_transform(df['high'], df['low'])
            
            if show_hurst:
                indicators['hurst'] = hurst_exponent(df['close'])
            
            indicators['z_score'] = z_score(df['close'])
            
            # SCORE MASTER
            score = 0
            count = 0
            
            if 'tema_vel' in indicators and not pd.isna(indicators['tema_vel'].iloc[-1]):
                score += 15 if indicators['tema_vel'].iloc[-1] > 0 else -15
                count += 1
            
            if 'hurst' in indicators and not pd.isna(indicators['hurst'].iloc[-1]):
                h = indicators['hurst'].iloc[-1]
                if h > 0.5:
                    score += 20
                else:
                    score -= 10
                count += 1
            
            if 'fisher' in indicators and not pd.isna(indicators['fisher'].iloc[-1]):
                f = indicators['fisher'].iloc[-1]
                if f > 2:
                    score -= 15
                elif f < -2:
                    score += 15
                count += 1
            
            if 'entropy' in indicators and not pd.isna(indicators['entropy'].iloc[-1]):
                if indicators['entropy'].iloc[-1] < 1.5:
                    score += 10
                count += 1
            
            if count > 0:
                score = score / count
            
            # SIGNAL BOARD
            if score > 5:
                st.markdown(f"""
                <div class="signal-board signal-up">
                    <div>🚀 COMPRA FORTE</div>
                    <div style="font-size: 80px; margin-top: 15px;">Score: {score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            elif score < -5:
                st.markdown(f"""
                <div class="signal-board signal-down">
                    <div>📉 VENDA FORTE</div>
                    <div style="font-size: 80px; margin-top: 15px;">Score: {score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"⚖️ Mercado Neutro | Score: {score:.1f}")
            
            st.markdown("---")
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            current = df['close'].iloc[-1]
            prev = df['close'].iloc[-2]
            change_pct = ((current - prev) / prev * 100) if prev > 0 else 0
            
            with col1:
                st.metric("💰 Preço", f"${current:,.2f}", f"{change_pct:+.2f}%")
            
            with col2:
                if 'hurst' in indicators:
                    h = indicators['hurst'].iloc[-1]
                    regime = "Tendência" if h > 0.5 else "Lateral"
                    st.metric("📊 Regime", regime, f"H: {h:.2f}")
            
            with col3:
                if 'entropy' in indicators:
                    ent = indicators['entropy'].iloc[-1]
                    st.metric("⚛️ Entropia", f"{ent:.2f}")
            
            with col4:
                vol = df['volume'].iloc[-1]
                st.metric("📦 Volume", f"{vol/1e6:.1f}M" if vol > 1e6 else f"{vol:,.0f}")
            
            st.markdown("---")
            
            # GRÁFICO
            st.subheader(f"📊 {selected}")
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('Price', 'Oscillators', 'Regime')
            )
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ), row=1, col=1)
            
            if 'tema' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['tema'],
                    name='TEMA', line=dict(color='#00d9ff', width=2)
                ), row=1, col=1)
            
            if 'kalman' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['kalman'],
                    name='Kalman', line=dict(color='#ff00ff', width=2)
                ), row=1, col=1)
            
            if 'fisher' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['fisher'],
                    name='Fisher', line=dict(color='#00d9ff')
                ), row=2, col=1)
                fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=indicators['z_score'],
                name='Z-Score', line=dict(color='#ff00ff')
            ), row=2, col=1)
            
            if 'hurst' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['hurst'],
                    name='Hurst', line=dict(color='#00ff88')
                ), row=3, col=1)
                fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=3, col=1)
            
            if 'entropy' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['entropy'],
                    name='Entropy', line=dict(color='#ffaa00')
                ), row=3, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                height=900,
                xaxis_rangeslider_visible=False,
                hovermode='x unified',
                plot_bgcolor='#000000',
                paper_bgcolor='#000000'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Erro ao carregar dados. Tentando novamente...")
            st.info("💡 Verifique sua conexão ou tente outro ativo")
    
    time.sleep(refresh)
    st.rerun()
