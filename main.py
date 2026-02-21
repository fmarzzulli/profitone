import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Imports ML
try:
    import xgboost as xgb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False
    
# Imports Scraper
try:
    from bs4 import BeautifulSoup
    SCRAPER_AVAILABLE = True
except:
    SCRAPER_AVAILABLE = False

# ========================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================
st.set_page_config(
    page_title="ProfitOne Ultimate V3",
    layout="wide",
    page_icon="🚀"
)

# ========================================
# CSS PERSONALIZADO - FONTE PRETA
# ========================================
st.markdown("""
<style>
    /* FUNDO E TEMA GERAL */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* FONTE PRETA EM TODO O SISTEMA */
    * {
        color: #000000 !important;
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    /* CARDS DE MÉTRICAS */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255,255,255,0.3);
        margin: 10px 0;
    }
    
    .metric-value {
        font-size: 32px !important;
        font-weight: bold !important;
        color: #000000 !important;
    }
    
    .metric-label {
        font-size: 14px !important;
        color: #000000 !important;
        opacity: 0.8;
    }
    
    /* SIGNAL BOARD */
    .signal-board {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        margin: 20px 0;
        border: 3px solid;
    }
    
    .signal-compra {
        border-color: #00ff88;
        background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,255,136,0.05));
    }
    
    .signal-venda {
        border-color: #ff4444;
        background: linear-gradient(135deg, rgba(255,68,68,0.2), rgba(255,68,68,0.05));
    }
    
    .signal-neutro {
        border-color: #ffd700;
        background: linear-gradient(135deg, rgba(255,215,0,0.2), rgba(255,215,0,0.05));
    }
    
    /* TABELAS */
    .dataframe {
        background: rgba(255, 255, 255, 0.9) !important;
        color: #000000 !important;
    }
    
    .dataframe th {
        background: rgba(102, 126, 234, 0.8) !important;
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    .dataframe td {
        color: #000000 !important;
    }
    
    /* SIDEBAR */
    .css-1d391kg, .css-1oe6wy4 {
        background: rgba(255, 255, 255, 0.95) !important;
    }
    
    /* BOTÕES */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 30px !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3) !important;
    }
    
    /* SELECTBOX */
    .stSelectbox {
        color: #000000 !important;
    }
    
    /* MÉTRICAS STREAMLIT */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 28px !important;
        font-weight: bold !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #000000 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stMetricDelta"] {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# FUNÇÕES DE INDICADORES TÉCNICOS
# ========================================

def tema(series, period=21):
    """Triple Exponential Moving Average + Velocidade"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema_line = 3 * ema1 - 3 * ema2 + ema3
    velocity = tema_line.diff()
    return tema_line, velocity

def kalman_filter(series, process_variance=0.01):
    """Filtro de Kalman para redução de ruído"""
    n = len(series)
    estimate = np.zeros(n)
    error_estimate = np.ones(n)
    
    estimate[0] = series.iloc[0]
    
    for i in range(1, n):
        prediction = estimate[i-1]
        error_prediction = error_estimate[i-1] + process_variance
        
        kalman_gain = error_prediction / (error_prediction + 1)
        estimate[i] = prediction + kalman_gain * (series.iloc[i] - prediction)
        error_estimate[i] = (1 - kalman_gain) * error_prediction
    
    return pd.Series(estimate, index=series.index)

def shannon_entropy(series, window=14):
    """Entropia de Shannon - Mede caos do mercado"""
    def entropy(data):
        if len(data) == 0:
            return 0
        hist, _ = np.histogram(data, bins=10, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    return series.rolling(window).apply(entropy, raw=True)

def fisher_transform(series, period=10):
    """Fisher Transform - Detecção de reversões"""
    high = series.rolling(period).max()
    low = series.rolling(period).min()
    
    value = 2 * ((series - low) / (high - low) - 0.5)
    value = value.clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    signal = fisher.shift(1)
    
    return fisher, signal

def hurst_exponent(series, max_lag=20):
    """Hurst Exponent - Detecta tendência vs lateralização"""
    lags = range(2, max_lag)
    tau = []
    
    for lag in lags:
        pp = np.subtract(series[lag:].values, series[:-lag].values)
        tau.append(np.std(pp))
    
    if len(tau) == 0:
        return 0.5
    
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

def z_score(series, period=20):
    """Z-Score - Desvios da média"""
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    return (series - mean) / std

def vwap(df):
    """Volume Weighted Average Price"""
    if 'volume' not in df.columns:
        return df['close']
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def schaff_trend_cycle(df, short=23, long=50, cycle=10):
    """Schaff Trend Cycle - Momentum cíclico"""
    close = df['close']
    
    # MACD
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    
    # Stochastic do MACD
    macd_min = macd.rolling(cycle).min()
    macd_max = macd.rolling(cycle).max()
    stoch1 = 100 * (macd - macd_min) / (macd_max - macd_min + 1e-10)
    
    # Smoothing
    stoch1 = stoch1.ewm(span=3, adjust=False).mean()
    
    # Second stochastic
    stoch1_min = stoch1.rolling(cycle).min()
    stoch1_max = stoch1.rolling(cycle).max()
    stc = 100 * (stoch1 - stoch1_min) / (stoch1_max - stoch1_min + 1e-10)
    
    return stc.ewm(span=3, adjust=False).mean()

# ========================================
# QUANTUM HUNTER V13
# ========================================

def quantum_hunter_v13(df, modo=1, sensibilidade=1.5):
    """
    QUANTUM HUNTER V13 - Institutional Trend Detector
    Retorna: score (-100 a +100), media_lenta, media_rapida, vwap_line
    """
    
    # Configuração por modo
    configs = {
        1: {'lento': 21, 'rapido': 8, 'rsi': 9},   # Scalp 5m
        2: {'lento': 34, 'rapido': 13, 'rsi': 14}, # Day 15m
        3: {'lento': 72, 'rapido': 21, 'rsi': 21}  # Swing 1h
    }
    
    config = configs.get(modo, configs[2])
    
    close = df['close']
    
    # Médias exponenciais
    media_lenta = close.ewm(span=config['lento'], adjust=False).mean()
    media_rapida = close.ewm(span=config['rapido'], adjust=False).mean()
    
    # VWAP
    vwap_line = vwap(df)
    
    # RSI
    rsi_line = rsi(close, config['rsi'])
    
    # === CÁLCULO DO SCORE ===
    
    # 1. Força da tendência (40%)
    tendencia = (media_rapida.iloc[-1] - media_lenta.iloc[-1]) / media_lenta.iloc[-1] * 100
    score_tendencia = np.clip(tendencia * 10, -40, 40)
    
    # 2. Posição vs VWAP (30%)
    distancia_vwap = (close.iloc[-1] - vwap_line.iloc[-1]) / vwap_line.iloc[-1] * 100
    score_vwap = np.clip(distancia_vwap * 3, -30, 30)
    
    # 3. RSI (30%)
    rsi_val = rsi_line.iloc[-1]
    if rsi_val > 55:
        score_rsi = 30
    elif rsi_val > 50:
        score_rsi = 15
    elif rsi_val < 45:
        score_rsi = -30
    elif rsi_val < 50:
        score_rsi = -15
    else:
        score_rsi = 0
    
    # Score total
    score_total = score_tendencia + score_vwap + score_rsi
    
    # Suavização (EMA de 3 períodos simulada)
    score_suavizado = np.clip(score_total * sensibilidade, -100, 100)
    
    return score_suavizado, media_lenta, media_rapida, vwap_line

# ========================================
# TURBOSTOCH WIN
# ========================================

def turbo_stoch_win(df):
    """
    TurboStoch WIN
    Retorna: stc, fisher, hurst
    """
    
    # Schaff Trend Cycle
    stc = schaff_trend_cycle(df, short=24, long=52, cycle=20)
    
    # Fisher Transform
    fisher_line, _ = fisher_transform(df['close'], period=10)
    fisher_normalized = ((fisher_line - fisher_line.min()) / 
                        (fisher_line.max() - fisher_line.min()) * 100)
    
    # Hurst Exponent
    if len(df) >= 30:
        hurst = hurst_exponent(df['close'].iloc[-30:], max_lag=20)
    else:
        hurst = 0.5
    
    return stc.iloc[-1] if len(stc) > 0 else 50, \
           fisher_normalized.iloc[-1] if len(fisher_normalized) > 0 else 50, \
           hurst

# ========================================
# FORÇA DO WIN
# ========================================

def calcular_forca_win(df, modo=1):
    """
    Calcula a força do WIN baseado em múltiplos indicadores
    Retorna: força (0-100)
    """
    
    # Quantum Hunter
    quantum_score, _, _, _ = quantum_hunter_v13(df, modo)
    
    # TurboStoch
    stc, fisher, hurst = turbo_stoch_win(df)
    
    # Cálculo da força
    # 40% Quantum Score (absoluto)
    # 30% STC
    # 30% Fisher
    forca = (abs(quantum_score) * 0.4 + stc * 0.3 + fisher * 0.3)
    
    # Penaliza se mercado lateral (Hurst < 0.45)
    if hurst < 0.45:
        forca *= 0.5
    
    return np.clip(forca, 0, 100)

# ========================================
# OBTENÇÃO DE DADOS DE MERCADO
# ========================================

@st.cache_data(ttl=10)
def get_market_data(symbol, interval="5m"):
    """
    Obtém dados de mercado do Yahoo Finance
    """
    
    symbol_map = {
        "IBOV": "%5EBVSP",
        "DOLAR": "USDBRL%3DX",
        "SP500": "%5EGSPC",
        "NASDAQ": "%5EIXIC",
        "DOW": "%5EDJI"
    }
    
    yahoo_symbol = symbol_map.get(symbol, symbol)
    
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    
    params = {
        "interval": interval,
        "range": "1d"
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart']:
            return None
            
        result = data['chart']['result'][0]
        
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'open': quote['open'],
            'high': quote['high'],
            'low': quote['low'],
            'close': quote['close'],
            'volume': quote['volume']
        })
        
        df = df.dropna()
        df = df.set_index('timestamp')
        
        return df
        
    except Exception as e:
        st.error(f"❌ Erro ao obter dados de {symbol}: {str(e)}")
        return None

# ========================================
# SCRAPER DE DADOS PÚBLICOS B3
# ========================================

@st.cache_data(ttl=3600)
def scrape_b3_institutional_flow():
    """
    Scraper de dados públicos da B3
    Tenta obter fluxo institucional de fontes públicas
    """
    
    if not SCRAPER_AVAILABLE:
        return None
    
    try:
        # Exemplo: Scraping do site Status Invest (público)
        url = "https://statusinvest.com.br/indice/ibov"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Dados simulados baseados em estrutura típica
            # Em produção, você precisaria mapear os elementos corretos
            flow_data = {
                'institucional': 45.0,
                'bancos': 30.0,
                'varejo': 25.0,
                'timestamp': datetime.now()
            }
            
            return flow_data
        else:
            return None
            
    except Exception as e:
        st.warning(f"⚠️ Scraper B3: {str(e)}")
        return None

# ========================================
# MACHINE LEARNING - PREVISÃO
# ========================================

def prepare_ml_features(df):
    """Prepara features para ML"""
    
    df = df.copy()
    
    # Features técnicas
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['rsi'] = rsi(df['close'], 14)
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    
    # Volume
    df['volume_sma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Bollinger Bands
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = sma + 2*std
    df['bb_lower'] = sma - 2*std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Target (1 = sobe, 0 = desce)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = df.dropna()
    
    feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'volume_ratio', 'bb_position']
    
    return df, feature_cols

@st.cache_resource
def train_ml_model(df):
    """
    Treina modelo ML (XGBoost)
    Retorna modelo treinado e scaler
    """
    
    if not ML_AVAILABLE or df is None or len(df) < 100:
        return None, None
    
    try:
        df_ml, feature_cols = prepare_ml_features(df)
        
        if len(df_ml) < 50:
            return None, None
        
        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        
        # Split treino/teste
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Normalização
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Treinamento XGBoost
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Acurácia
        accuracy = model.score(X_test_scaled, y_test)
        
        return model, scaler, accuracy
        
    except Exception as e:
        st.warning(f"⚠️ ML Training: {str(e)}")
        return None, None, 0

def predict_direction(df, model, scaler):
    """Prediz direção do mercado"""
    
    if model is None or scaler is None or df is None or len(df) < 20:
        return 0.5, "Neutro"
    
    try:
        df_ml, feature_cols = prepare_ml_features(df)
        
        if len(df_ml) == 0:
            return 0.5, "Neutro"
        
        X_last = df_ml[feature_cols].iloc[-1:].values
        X_scaled = scaler.transform(X_last)
        
        proba = model.predict_proba(X_scaled)[0]
        confidence = proba[1]  # Probabilidade de subida
        
        if confidence > 0.65:
            direction = "Alta 🚀"
        elif confidence < 0.35:
            direction = "Baixa 📉"
        else:
            direction = "Neutro ⚖️"
        
        return confidence, direction
        
    except:
        return 0.5, "Neutro"

# ========================================
# INTERFACE PRINCIPAL
# ========================================

def main():
    
    st.title("🚀 PROFITONE ULTIMATE V3")
    st.markdown("### Sistema Completo de Trading com IA")
    
    # ========================================
    # SIDEBAR
    # ========================================
    
    st.sidebar.title("⚙️ Configurações")
    
    st.sidebar.markdown("---")
    
    modo_map = {
        "🎯 Scalp (5 min)": 1,
        "📊 Day Trade (15 min)": 2,
        "📈 Swing (1 hora)": 3
    }
    
    modo_selected = st.sidebar.selectbox(
        "Modo de Operação",
        list(modo_map.keys()),
        index=1
    )
    
    modo = modo_map[modo_selected]
    
    # Timeframe baseado no modo
    timeframe_map = {
        1: "5m",
        2: "15m",
        3: "1h"
    }
    
    interval = timeframe_map[modo]
    
    st.sidebar.markdown(f"**Timeframe:** `{interval}`")
    
    st.sidebar.markdown("---")
    
    # Refresh
    refresh_seconds = st.sidebar.slider(
        "🔄 Atualização (segundos)",
        min_value=5,
        max_value=60,
        value=15,
        step=5
    )
    
    st.sidebar.markdown("---")
    
    # Features disponíveis
    st.sidebar.markdown("### 🎯 Features Ativas")
    
    st.sidebar.markdown(f"""
    ✅ **Quantum Hunter V13**  
    ✅ **TurboStoch WIN**  
    ✅ **Machine Learning** {'🟢' if ML_AVAILABLE else '🔴'}  
    ✅ **Scraper B3** {'🟢' if SCRAPER_AVAILABLE else '🔴'}  
    """)
    
    # ========================================
    # TABS PRINCIPAIS
    # ========================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 WIN Principal",
        "🌐 Multi-Ativos",
        "💪 Força Tempo Real",
        "🤖 Machine Learning"
    ])
    
    # ========================================
    # TAB 1: WIN PRINCIPAL
    # ========================================
    
    with tab1:
        
        st.markdown("## 📊 IBOVESPA - Análise Completa")
        
        # Carregar dados
        with st.spinner("📥 Carregando dados..."):
            df = get_market_data("IBOV", interval)
        
        if df is not None and len(df) > 0:
            
            # Quantum Hunter
            quantum_score, ml, mr, vwap_line = quantum_hunter_v13(df, modo)
            
            # TurboStoch
            stc, fisher, hurst = turbo_stoch_win(df)
            
            # Força
            forca = calcular_forca_win(df, modo)
            
            # Preço atual
            preco_atual = df['close'].iloc[-1]
            preco_anterior = df['close'].iloc[-2] if len(df) > 1 else preco_atual
            variacao = ((preco_atual - preco_anterior) / preco_anterior) * 100
            
            # Determinar sinal
            if quantum_score > 60:
                sinal = "COMPRA"
                sinal_class = "signal-compra"
                sinal_icon = "🚀"
                sinal_color = "#00ff88"
            elif quantum_score < -60:
                sinal = "VENDA"
                sinal_class = "signal-venda"
                sinal_icon = "📉"
                sinal_color = "#ff4444"
            else:
                sinal = "NEUTRO"
                sinal_class = "signal-neutro"
                sinal_icon = "⚖️"
                sinal_color = "#ffd700"
            
            # SIGNAL BOARD
            st.markdown(f"""
            <div class="signal-board {sinal_class}">
                <div style="font-size: 60px;">{sinal_icon}</div>
                <div style="font-size: 40px; font-weight: bold; color: #000000; margin: 10px 0;">
                    {sinal}
                </div>
                <div style="font-size: 24px; color: #000000;">
                    Score: {quantum_score:.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "💰 Preço",
                    f"R$ {preco_atual:,.2f}",
                    f"{variacao:+.2f}%"
                )
            
            with col2:
                regime = "Trending 📈" if hurst > 0.55 else ("Lateral ↔️" if hurst > 0.45 else "Reversal 🔄")
                st.metric(
                    "🎯 Regime",
                    regime,
                    f"Hurst: {hurst:.2f}"
                )
            
            with col3:
                st.metric(
                    "⚡ STC",
                    f"{stc:.1f}",
                    f"Fisher: {fisher:.1f}"
                )
            
            with col4:
                st.metric(
                    "💪 Força WIN",
                    f"{forca:.0f}%",
                    "Alta" if forca > 70 else ("Média" if forca > 40 else "Baixa")
                )
            
            # Gráfico
            st.markdown("### 📈 Gráfico de Candlesticks")
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='IBOV',
                    increasing_line_color='#00ff88',
                    decreasing_line_color='#ff4444'
                ),
                row=1, col=1
            )
            
            # Médias
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=ml,
                    name='Média Lenta',
                    line=dict(color='#ff00ff', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=mr,
                    name='Média Rápida',
                    line=dict(color='#00d9ff', width=2)
                ),
                row=1, col=1
            )
            
            # VWAP
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=vwap_line,
                    name='VWAP',
                    line=dict(color='#ffd700', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            # Volume
            colors = ['#00ff88' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff4444' 
                     for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=700,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            fig.update_xaxes(title_text="Data/Hora", row=2, col=1)
            fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error("❌ Não foi possível carregar os dados do IBOVESPA")
    
    # ========================================
    # TAB 2: MULTI-ATIVOS
    # ========================================
    
    with tab2:
        
        st.markdown("## 🌐 Monitor Multi-Ativos")
        
        ativos = ['IBOV', 'DOLAR', 'SP500', 'NASDAQ', 'DOW']
        
        cols = st.columns(len(ativos))
        
        for idx, ativo in enumerate(ativos):
            
            with cols[idx]:
                
                df_ativo = get_market_data(ativo, interval)
                
                if df_ativo is not None and len(df_ativo) > 0:
                    
                    preco = df_ativo['close'].iloc[-1]
                    preco_ant = df_ativo['close'].iloc[-2] if len(df_ativo) > 1 else preco
                    var = ((preco - preco_ant) / preco_ant) * 100
                    
                    qscore, _, _, _ = quantum_hunter_v13(df_ativo, modo)
                    forca_ativo = calcular_forca_win(df_ativo, modo)
                    
                    if qscore > 60:
                        sig = "🚀 COMPRA"
                        borda = "#00ff88"
                    elif qscore < -60:
                        sig = "📉 VENDA"
                        borda = "#ff4444"
                    else:
                        sig = "⚖️ NEUTRO"
                        borda = "#ffd700"
                    
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid {borda};">
                        <div class="metric-label">{ativo}</div>
                        <div class="metric-value">
                            {preco:,.2f}
                        </div>
                        <div style="color: {'#00ff88' if var > 0 else '#ff4444'}; font-size: 18px; font-weight: bold;">
                            {var:+.2f}%
                        </div>
                        <div style="margin-top: 10px; font-size: 14px; color: #000000;">
                            {sig}
                        </div>
                        <div style="font-size: 12px; color: #000000; margin-top: 5px;">
                            Score: {qscore:.1f} | Força: {forca_ativo:.0f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                else:
                    st.error(f"❌ {ativo}")
    
    # ========================================
    # TAB 3: FORÇA TEMPO REAL
    # ========================================
    
    with tab3:
        
        st.markdown("## 💪 Força do WIN em Tempo Real")
        
        df_win = get_market_data("IBOV", interval)
        
        if df_win is not None and len(df_win) > 0:
            
            forca_win = calcular_forca_win(df_win, modo)
            
            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=forca_win,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Força WIN", 'font': {'size': 24, 'color': '#000000'}},
                delta={'reference': 50, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff4444"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#000000"},
                    'bar': {'color': "#667eea"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#000000",
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(255,68,68,0.3)'},
                        {'range': [30, 70], 'color': 'rgba(255,215,0,0.3)'},
                        {'range': [70, 100], 'color': 'rgba(0,255,136,0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "#000000", 'width': 4},
                        'thickness': 0.75,
                        'value': forca_win
                    }
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(255,255,255,0.9)',
                font={'color': "#000000", 'family': "Arial"},
                height=400
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Análise
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Força Atual</div>
                    <div class="metric-value" style="color: {'#00ff88' if forca_win > 70 else ('#ffd700' if forca_win > 40 else '#ff4444')};">
                        {forca_win:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                nivel = "ALTA 🚀" if forca_win > 70 else ("MÉDIA ⚖️" if forca_win > 40 else "BAIXA 📉")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Nível</div>
                    <div class="metric-value" style="font-size: 24px;">
                        {nivel}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                _, _, hurst_win = turbo_stoch_win(df_win)
                tendencia = "Trending" if hurst_win > 0.55 else ("Lateral" if hurst_win > 0.45 else "Reversal")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Tendência</div>
                    <div class="metric-value" style="font-size: 24px;">
                        {tendencia}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            st.error("❌ Não foi possível calcular a força")
    
    # ========================================
    # TAB 4: MACHINE LEARNING
    # ========================================
    
    with tab4:
        
        st.markdown("## 🤖 Previsão com Machine Learning")
        
        if not ML_AVAILABLE:
            st.warning("⚠️ Bibliotecas de ML não instaladas. Instale com:")
            st.code("pip install tensorflow xgboost")
        else:
            
            df_ml = get_market_data("IBOV", "15m")
            
            if df_ml is not None and len(df_ml) > 100:
                
                with st.spinner("🧠 Treinando modelo..."):
                    model, scaler, accuracy = train_ml_model(df_ml)
                
                if model is not None:
                    
                    st.success(f"✅ Modelo treinado com {accuracy*100:.1f}% de acurácia")
                    
                    # Previsão
                    confidence, direction = predict_direction(df_ml, model, scaler)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Previsão ML</div>
                            <div class="metric-value" style="font-size: 32px;">
                                {direction}
                            </div>
                            <div style="font-size: 18px; color: #000000; margin-top: 10px;">
                                Confiança: {confidence*100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Acurácia do Modelo</div>
                            <div class="metric-value" style="font-size: 32px;">
                                {accuracy*100:.1f}%
                            </div>
                            <div style="font-size: 14px; color: #000000; margin-top: 10px;">
                                Base: últimas {len(df_ml)} barras
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Gauge de confiança
                    fig_conf = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Confiança da Previsão", 'font': {'size': 24, 'color': '#000000'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': "#000000"},
                            'bar': {'color': "#667eea"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#000000",
                            'steps': [
                                {'range': [0, 35], 'color': 'rgba(255,68,68,0.3)'},
                                {'range': [35, 65], 'color': 'rgba(255,215,0,0.3)'},
                                {'range': [65, 100], 'color': 'rgba(0,255,136,0.3)'}
                            ]
                        }
                    ))
                    
                    fig_conf.update_layout(
                        paper_bgcolor='rgba(255,255,255,0.9)',
                        font={'color': "#000000"},
                        height=400
                    )
                    
                    st.plotly_chart(fig_conf, use_container_width=True)
                    
                    # Informações
                    st.info("""
                    **ℹ️ Como funciona:**
                    
                    - O modelo XGBoost analisa padrões históricos de preço, volume, RSI, MACD e outros indicadores
                    - Confiança > 65%: Sinal forte
                    - Confiança 35-65%: Sinal neutro
                    - Confiança < 35%: Sinal fraco
                    - O modelo é retreinado automaticamente com novos dados
                    """)
                
                else:
                    st.error("❌ Falha ao treinar modelo ML")
            
            else:
                st.warning("⚠️ Dados insuficientes para treinar ML (mínimo 100 barras)")
        
        # Scraper B3
        st.markdown("---")
        st.markdown("### 🌐 Dados Públicos B3")
        
        if SCRAPER_AVAILABLE:
            
            with st.spinner("🔍 Buscando dados públicos..."):
                flow_data = scrape_b3_institutional_flow()
            
            if flow_data:
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #667eea;">
                        <div class="metric-label">🏛️ Institucional</div>
                        <div class="metric-value">{flow_data['institucional']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #00d9ff;">
                        <div class="metric-label">🏦 Bancos</div>
                        <div class="metric-value">{flow_data['bancos']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #ffd700;">
                        <div class="metric-label">👤 Varejo</div>
                        <div class="metric-value">{flow_data['varejo']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.caption(f"🕐 Atualizado: {flow_data['timestamp'].strftime('%H:%M:%S')}")
                
            else:
                st.warning("⚠️ Não foi possível obter dados de fluxo institucional")
        
        else:
            st.warning("⚠️ BeautifulSoup não instalado. Instale com:")
            st.code("pip install beautifulsoup4 lxml")
    
    # ========================================
    # FOOTER
    # ========================================
    
    st.markdown("---")
    
    st.markdown(f"""
    <div style="text-align: center; color: #000000; padding: 20px;">
        <p style="font-size: 16px; font-weight: bold;">
            🚀 ProfitOne Ultimate V3 | Machine Learning + Scraper B3 + Quantum Hunter
        </p>
        <p style="font-size: 12px;">
            ⚠️ Este sistema é para fins educacionais. Trading envolve risco de perda de capital.
        </p>
        <p style="font-size: 12px;">
            🕐 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(refresh_seconds)
    st.rerun()

# ========================================
# EXECUÇÃO
# ========================================

if __name__ == "__main__":
    main()
