import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.stats import entropy
from datetime import datetime
import time

st.set_page_config(page_title="ProfitOne IBOVESPA", layout="wide", page_icon="📊")

# CSS
st.markdown("""
<style>
    .main {background-color: #000000;}
    h1, h2, h3 {color: #00d9ff;}
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
    }
    .signal-up {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #000;
    }
    .signal-down {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #fff;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INDICADORES
# ==========================================

def tema(close, period=20):
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    velocity = tema.diff()
    return tema, velocity

def kalman_filter(data):
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
    def calc_ent(x):
        if len(x) < 2: return 0
        ret = np.diff(x)
        hist, _ = np.histogram(ret, bins=10, density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)
    return data.rolling(window=window).apply(calc_ent, raw=False)

def fisher_transform(high, low, period=10):
    hl2 = (high + low) / 2
    max_h = hl2.rolling(period).max()
    min_l = hl2.rolling(period).min()
    value = 2 * ((hl2 - min_l) / (max_h - min_l + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
    return fisher

def hurst_exponent(data, window=100):
    def calc_h(ts):
        if len(ts) < 10: return 0.5
        lags = range(2, min(20, len(ts)//2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    return data.rolling(window=window).apply(calc_h, raw=False)

def z_score(data, window=20):
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / (std + 1e-10)

# ==========================================
# BUSCAR DADOS - MÚLTIPLAS FONTES
# ==========================================

def get_ibovespa_data(period="1d", interval="5m"):
    """Tentar múltiplas fontes para IBOVESPA"""
    
    # FONTE 1: Yahoo Finance (preferencial)
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5EBVSP"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        params = {
            "interval": interval,
            "range": period,
            "includePrePost": "false"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
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
                
                if len(df) > 0:
                    return df
    
    except Exception as e:
        st.warning(f"Tentativa Yahoo falhou: {str(e)}")
    
    # FONTE 2: Alpha Vantage (backup)
    try:
        # API pública gratuita
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": "BVSP",
            "interval": interval.replace("m", "min"),
            "apikey": "demo",
            "outputsize": "compact"
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            
            if f"Time Series ({interval})" in data:
                time_series = data[f"Time Series ({interval})"]
                
                df = pd.DataFrame.from_dict(time_series, orient='index')
                df.index = pd.to_datetime(df.index)
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                
                for col in df.columns:
                    df[col] = df[col].astype(float)
                
                return df.sort_index()
    
    except Exception as e:
        st.warning(f"Tentativa Alpha Vantage falhou: {str(e)}")
    
    # FONTE 3: Dados simulados (último recurso)
    st.warning("⚠️ Usando dados SIMULADOS do IBOVESPA para demonstração")
    
    dates = pd.date_range(end=datetime.now(), periods=200, freq='5min')
    
    # Gerar dados realistas baseados no último fechamento conhecido do IBOVESPA (~128.000)
    base_price = 128000
    prices = [base_price]
    
    for _ in range(199):
        change = np.random.normal(0, 0.002)  # Volatilidade de 0.2%
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({
        'open': prices,
        'close': [p * (1 + np.random.normal(0, 0.001)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'volume': [np.random.randint(5e9, 15e9) for _ in prices]
    }, index=dates)
    
    return df

# ==========================================
# CALCULAR SCORE
# ==========================================

def calculate_score(df):
    """Calcular score baseado em indicadores"""
    
    tema_series, tema_vel = tema(df['close'])
    kalman_series = kalman_filter(df['close'])
    entropy_series = shannon_entropy(df['close'])
    fisher_series = fisher_transform(df['high'], df['low'])
    hurst_series = hurst_exponent(df['close'])
    z_score_series = z_score(df['close'])
    
    score = 0
    count = 0
    
    # TEMA Velocity
    if not pd.isna(tema_vel.iloc[-1]):
        score += 15 if tema_vel.iloc[-1] > 0 else -15
        count += 1
    
    # Hurst
    if not pd.isna(hurst_series.iloc[-1]):
        h = hurst_series.iloc[-1]
        score += 20 if h > 0.5 else -10
        count += 1
    
    # Fisher
    if not pd.isna(fisher_series.iloc[-1]):
        f = fisher_series.iloc[-1]
        if f > 2:
            score -= 15
        elif f < -2:
            score += 15
        count += 1
    
    # Entropy
    if not pd.isna(entropy_series.iloc[-1]):
        if entropy_series.iloc[-1] < 1.5:
            score += 10
        count += 1
    
    if count > 0:
        score = score / count
    
    return score, {
        'tema': tema_series,
        'tema_vel': tema_vel,
        'kalman': kalman_series,
        'entropy': entropy_series,
        'fisher': fisher_series,
        'hurst': hurst_series,
        'z_score': z_score_series
    }

# ==========================================
# INTERFACE
# ==========================================

st.markdown("<h1 style='text-align: center;'>📊 PROFITONE - IBOVESPA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Sistema Profissional de Trading</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configurações")

period_map = {
    "1 dia": "1d",
    "5 dias": "5d",
    "1 mês": "1mo"
}

period_label = st.sidebar.selectbox("Período:", list(period_map.keys()), index=0)
period = period_map[period_label]

timeframe_map = {
    "5 minutos": "5m",
    "15 minutos": "15m",
    "1 hora": "1h"
}

timeframe_label = st.sidebar.selectbox("Timeframe:", list(timeframe_map.keys()), index=0)
timeframe = timeframe_map[timeframe_label]

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Indicadores")

show_tema = st.sidebar.checkbox("TEMA", value=True)
show_kalman = st.sidebar.checkbox("Kalman", value=True)
show_entropy = st.sidebar.checkbox("Entropia", value=True)
show_fisher = st.sidebar.checkbox("Fisher", value=True)
show_hurst = st.sidebar.checkbox("Hurst", value=True)

# Abas
tab1, tab2 = st.tabs(["📊 Tempo Real", "📈 Histórico"])

# ABA 1: TEMPO REAL
with tab1:
    
    placeholder = st.empty()
    
    with placeholder.container():
        
        with st.spinner("📥 Carregando dados do IBOVESPA..."):
            df = get_ibovespa_data(period=period, interval=timeframe)
        
        if df is not None and len(df) > 50:
            
            now = datetime.now().strftime("%H:%M:%S")
            st.success(f"✅ Dados carregados: {len(df)} candles | Última atualização: {now}")
            
            # Calcular score
            score, indicators = calculate_score(df)
            
            # SIGNAL BOARD
            if score > 5:
                st.markdown(f"""
                <div class="signal-board signal-up">
                    <div>🚀 COMPRA FORTE</div>
                    <div style="font-size: 80px; margin-top: 15px;">{score:.1f}</div>
                </div>
                """, unsafe_allow_html=True)
            elif score < -5:
                st.markdown(f"""
                <div class="signal-board signal-down">
                    <div>📉 VENDA FORTE</div>
                    <div style="font-size: 80px; margin-top: 15px;">{score:.1f}</div>
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
                st.metric("💰 IBOVESPA", f"{current:,.0f}", f"{change_pct:+.2f}%")
            
            with col2:
                if not pd.isna(indicators['hurst'].iloc[-1]):
                    h = indicators['hurst'].iloc[-1]
                    regime = "Tendência" if h > 0.5 else "Lateral"
                    st.metric("📊 Regime", regime, f"H: {h:.2f}")
            
            with col3:
                if not pd.isna(indicators['entropy'].iloc[-1]):
                    ent = indicators['entropy'].iloc[-1]
                    st.metric("⚛️ Entropia", f"{ent:.2f}")
            
            with col4:
                vol = df['volume'].iloc[-1]
                st.metric("📦 Volume", f"{vol/1e9:.1f}B")
            
            st.markdown("---")
            
            # GRÁFICO
            st.subheader("📊 Análise Técnica")
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('IBOVESPA', 'Osciladores', 'Regime')
            )
            
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ), row=1, col=1)
            
            if show_tema:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['tema'],
                    name='TEMA', line=dict(color='#00d9ff', width=2)
                ), row=1, col=1)
            
            if show_kalman:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['kalman'],
                    name='Kalman', line=dict(color='#ff00ff', width=2)
                ), row=1, col=1)
            
            if show_fisher:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['fisher'],
                    name='Fisher', line=dict(color='#00d9ff')
                ), row=2, col=1)
                fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
            
            if show_hurst:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['hurst'],
                    name='Hurst', line=dict(color='#00ff88')
                ), row=3, col=1)
                fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=3, col=1)
            
            if show_entropy:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['entropy'],
                    name='Entropy', line=dict(color='#ffaa00')
                ), row=3, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                height=900,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Não foi possível carregar dados do IBOVESPA")
            st.info("💡 Verifique sua conexão ou aguarde alguns segundos")

# ABA 2: HISTÓRICO
with tab2:
    st.info("📊 Histórico será implementado em breve. Por enquanto, use a aba Tempo Real.")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #444;'>📊 ProfitOne IBOVESPA</div>", unsafe_allow_html=True)
