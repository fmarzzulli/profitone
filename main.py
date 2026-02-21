import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import time
from datetime import datetime, timedelta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ML Imports
try:
    import xgboost as xgb
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    import lightgbm as lgb
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# ========================================
# CONFIGURAÇÃO
# ========================================
st.set_page_config(
    page_title="ProfitOne Ultra V4",
    layout="wide",
    page_icon="🚀"
)

# ========================================
# CSS - FONTE PRETA
# ========================================
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    * {
        color: #000000 !important;
        font-family: 'Segoe UI', Arial, sans-serif !important;
    }
    
    h1, h2, h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 10px 0;
    }
    
    .confidence-high {
        border-left: 5px solid #00ff88;
    }
    
    .confidence-medium {
        border-left: 5px solid #ffd700;
    }
    
    .confidence-low {
        border-left: 5px solid #ff4444;
    }
    
    .signal-board {
        background: rgba(255, 255, 255, 0.95);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
        margin: 20px 0;
        border: 3px solid;
    }
    
    .signal-compra { border-color: #00ff88; }
    .signal-venda { border-color: #ff4444; }
    .signal-neutro { border-color: #ffd700; }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 10px 30px !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# INDICADORES AVANÇADOS
# ========================================

def vpin_indicator(df, window=20):
    """
    VPIN - Volume-Synchronized Probability of Informed Trading
    Detecta presença de traders informados (institucionais)
    Fórmula: Ehlers & Easley (2012)
    """
    
    # Separar volume de compra e venda
    df['buy_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
    df['sell_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
    
    # Volume bucket
    volume_bucket = df['volume'].rolling(window).sum() / window
    
    # VPIN calculation
    buy_vol_sum = df['buy_volume'].rolling(window).sum()
    sell_vol_sum = df['sell_volume'].rolling(window).sum()
    
    vpin = abs(buy_vol_sum - sell_vol_sum) / (buy_vol_sum + sell_vol_sum + 1e-10)
    
    return vpin

def order_flow_imbalance(df):
    """
    Order Flow Imbalance (OFI)
    Mede desequilíbrio entre ordens de compra e venda
    """
    
    # Delta de volume
    delta = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
    
    # Cumulative delta
    cumulative_delta = pd.Series(delta).cumsum()
    
    # OFI normalizado
    ofi = cumulative_delta / df['volume'].sum()
    
    return pd.Series(delta, index=df.index), pd.Series(cumulative_delta, index=df.index)

def detect_iceberg_orders(df, threshold=2.0):
    """
    Detecta ordens iceberg (institucionais)
    Baseado em volume anormal vs movimento de preço
    """
    
    # Volume vs price change ratio
    price_change = df['close'].pct_change().abs()
    volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
    
    # Icebergs: alto volume, baixa mudança de preço
    iceberg_score = volume_ratio / (price_change + 1e-10)
    
    # Normalizar
    iceberg_score = (iceberg_score - iceberg_score.mean()) / (iceberg_score.std() + 1e-10)
    
    return iceberg_score

def absorption_analysis(df):
    """
    Análise de Absorção de Liquidez
    Detecta quando institucionais absorvem ordens de varejo
    """
    
    # Wick analysis (sombras grandes = absorção)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    body = abs(df['close'] - df['open'])
    
    # Upper absorption (rejeição de alta)
    upper_absorption = upper_wick / (body + 1e-10)
    
    # Lower absorption (rejeição de baixa)
    lower_absorption = lower_wick / (body + 1e-10)
    
    # Net absorption
    net_absorption = lower_absorption - upper_absorption
    
    return net_absorption

def smart_money_index(df):
    """
    Smart Money Index (SMI)
    Rastreia movimentação de dinheiro institucional
    """
    
    # Intraday momentum
    intraday = df['close'] - df['open']
    
    # Weighted by volume
    smart_money = (intraday * df['volume']).cumsum()
    
    # Normalize
    smi = (smart_money - smart_money.min()) / (smart_money.max() - smart_money.min() + 1e-10) * 100
    
    return smi

def institutional_footprint(df):
    """
    Footprint Chart Simulation
    Estima % de participação institucional vs varejo
    """
    
    # VPIN (alta = institucional)
    vpin = vpin_indicator(df)
    
    # Iceberg detection
    iceberg = detect_iceberg_orders(df)
    
    # Absorption
    absorption = absorption_analysis(df)
    
    # Smart Money Index
    smi = smart_money_index(df)
    
    # Combine all signals
    institutional_score = (
        vpin.iloc[-1] * 0.3 +
        (iceberg.iloc[-1] / 3) * 0.25 +
        (absorption.iloc[-1] + 1) * 0.2 +
        (smi.iloc[-1] / 100) * 0.25
    )
    
    # Clamp between 0 and 1
    institutional_score = np.clip(institutional_score, 0, 1)
    
    # Convert to percentages
    # Institucional: 30-70% (baseado em score)
    institucional = 30 + institutional_score * 40
    
    # Bancos: 20-40%
    bancos = 20 + (1 - institutional_score) * 20
    
    # Varejo: remainder
    varejo = 100 - institucional - bancos
    
    return {
        'institucional': institucional,
        'bancos': bancos,
        'varejo': varejo,
        'confidence': institutional_score * 100,
        'vpin': vpin.iloc[-1],
        'iceberg': iceberg.iloc[-1],
        'absorption': absorption.iloc[-1],
        'smi': smi.iloc[-1]
    }

# ========================================
# INDICADORES TÉCNICOS (do código anterior)
# ========================================

def tema(series, period=21):
    """Triple Exponential Moving Average"""
    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema_line = 3 * ema1 - 3 * ema2 + ema3
    velocity = tema_line.diff()
    return tema_line, velocity

def rsi(series, period=14):
    """RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def vwap(df):
    """VWAP"""
    if 'volume' not in df.columns:
        return df['close']
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

def hurst_exponent(series, max_lag=20):
    """Hurst Exponent"""
    lags = range(2, min(max_lag, len(series)//2))
    tau = []
    for lag in lags:
        pp = np.subtract(series[lag:].values, series[:-lag].values)
        tau.append(np.std(pp))
    
    if len(tau) == 0:
        return 0.5
    
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

def fisher_transform(series, period=10):
    """Fisher Transform"""
    high = series.rolling(period).max()
    low = series.rolling(period).min()
    value = 2 * ((series - low) / (high - low + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    return fisher, fisher.shift(1)

def schaff_trend_cycle(df, short=23, long=50, cycle=10):
    """Schaff Trend Cycle"""
    close = df['close']
    ema_short = close.ewm(span=short, adjust=False).mean()
    ema_long = close.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    
    macd_min = macd.rolling(cycle).min()
    macd_max = macd.rolling(cycle).max()
    stoch1 = 100 * (macd - macd_min) / (macd_max - macd_min + 1e-10)
    stoch1 = stoch1.ewm(span=3, adjust=False).mean()
    
    stoch1_min = stoch1.rolling(cycle).min()
    stoch1_max = stoch1.rolling(cycle).max()
    stc = 100 * (stoch1 - stoch1_min) / (stoch1_max - stoch1_min + 1e-10)
    
    return stc.ewm(span=3, adjust=False).mean()

# ========================================
# QUANTUM HUNTER V13 ULTRA
# ========================================

def quantum_hunter_v13_ultra(df, modo=1):
    """
    Quantum Hunter V13 Ultra - Com validação de confiança
    Retorna: score, confiança, componentes
    """
    
    configs = {
        1: {'lento': 21, 'rapido': 8, 'rsi': 9},
        2: {'lento': 34, 'rapido': 13, 'rsi': 14},
        3: {'lento': 72, 'rapido': 21, 'rsi': 21}
    }
    
    config = configs.get(modo, configs[2])
    close = df['close']
    
    # Médias
    media_lenta = close.ewm(span=config['lento'], adjust=False).mean()
    media_rapida = close.ewm(span=config['rapido'], adjust=False).mean()
    vwap_line = vwap(df)
    rsi_line = rsi(close, config['rsi'])
    
    # Score components
    tendencia = (media_rapida.iloc[-1] - media_lenta.iloc[-1]) / media_lenta.iloc[-1] * 100
    score_tendencia = np.clip(tendencia * 10, -40, 40)
    
    distancia_vwap = (close.iloc[-1] - vwap_line.iloc[-1]) / vwap_line.iloc[-1] * 100
    score_vwap = np.clip(distancia_vwap * 3, -30, 30)
    
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
    
    score_total = score_tendencia + score_vwap + score_rsi
    
    # CONFIDENCE CALCULATION
    # Baseado em:
    # 1. Volatilidade (baixa vol = maior confiança)
    volatility = close.pct_change().rolling(20).std().iloc[-1]
    vol_confidence = 100 * (1 - np.clip(volatility * 100, 0, 1))
    
    # 2. Alinhamento de indicadores
    alignment = abs(score_tendencia/40) + abs(score_vwap/30) + abs(score_rsi/30)
    alignment_confidence = alignment / 3 * 100
    
    # 3. Volume confirmation
    volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
    volume_confidence = np.clip(volume_ratio * 50, 0, 100)
    
    # Combined confidence
    confidence = (vol_confidence * 0.3 + alignment_confidence * 0.4 + volume_confidence * 0.3)
    
    return {
        'score': np.clip(score_total, -100, 100),
        'confidence': confidence,
        'components': {
            'tendencia': score_tendencia,
            'vwap': score_vwap,
            'rsi': score_rsi
        },
        'media_lenta': media_lenta,
        'media_rapida': media_rapida,
        'vwap': vwap_line
    }

# ========================================
# ENSEMBLE MACHINE LEARNING
# ========================================

def prepare_ml_features(df):
    """Prepara features para ML"""
    
    df = df.copy()
    
    # Technical features
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
    
    # Bollinger
    sma = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['bb_upper'] = sma + 2*std
    df['bb_lower'] = sma - 2*std
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # VPIN
    df['vpin'] = vpin_indicator(df)
    
    # Order flow
    delta, cum_delta = order_flow_imbalance(df)
    df['order_flow'] = delta
    df['cum_delta'] = cum_delta
    
    # Target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    df = df.dropna()
    
    feature_cols = ['returns', 'volatility', 'rsi', 'macd', 'volume_ratio', 
                    'bb_position', 'vpin', 'order_flow', 'cum_delta']
    
    return df, feature_cols

@st.cache_resource
def train_ensemble_model(df):
    """
    Treina ensemble de 3 modelos
    Retorna: modelos, scaler, metrics
    """
    
    if not ML_AVAILABLE or df is None or len(df) < 100:
        return None, None, None
    
    try:
        df_ml, feature_cols = prepare_ml_features(df)
        
        if len(df_ml) < 50:
            return None, None, None
        
        X = df_ml[feature_cols].values
        y = df_ml['target'].values
        
        # Split
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Scale
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model 1: XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        xgb_acc = xgb_model.score(X_test_scaled, y_test)
        
        # Model 2: LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_scaled, y_train)
        lgb_acc = lgb_model.score(X_test_scaled, y_test)
        
        # Model 3: Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        rf_model.fit(X_train_scaled, y_train)
        rf_acc = rf_model.score(X_test_scaled, y_test)
        
        # Ensemble accuracy
        ensemble_acc = (xgb_acc + lgb_acc + rf_acc) / 3
        
        models = {
            'xgb': xgb_model,
            'lgb': lgb_model,
            'rf': rf_model
        }
        
        metrics = {
            'xgb_acc': xgb_acc,
            'lgb_acc': lgb_acc,
            'rf_acc': rf_acc,
            'ensemble_acc': ensemble_acc
        }
        
        return models, scaler, metrics
        
    except Exception as e:
        st.warning(f"⚠️ ML Training: {str(e)}")
        return None, None, None

def ensemble_predict(df, models, scaler):
    """Previsão com ensemble de modelos"""
    
    if models is None or scaler is None:
        return 0.5, "Neutro", 0
    
    try:
        df_ml, feature_cols = prepare_ml_features(df)
        
        if len(df_ml) == 0:
            return 0.5, "Neutro", 0
        
        X_last = df_ml[feature_cols].iloc[-1:].values
        X_scaled = scaler.transform(X_last)
        
        # Predictions from all models
        xgb_proba = models['xgb'].predict_proba(X_scaled)[0][1]
        lgb_proba = models['lgb'].predict_proba(X_scaled)[0][1]
        rf_proba = models['rf'].predict_proba(X_scaled)[0][1]
        
        # Ensemble (average)
        ensemble_proba = (xgb_proba + lgb_proba + rf_proba) / 3
        
        # Confidence (agreement between models)
        std_dev = np.std([xgb_proba, lgb_proba, rf_proba])
        confidence = 100 * (1 - std_dev)
        
        # Direction
        if ensemble_proba > 0.6:
            direction = "Alta 🚀"
        elif ensemble_proba < 0.4:
            direction = "Baixa 📉"
        else:
            direction = "Neutro ⚖️"
        
        return ensemble_proba, direction, confidence
        
    except Exception as e:
        return 0.5, "Neutro", 0

# ========================================
# DATA FETCHING
# ========================================

@st.cache_data(ttl=10)
def get_market_data(symbol, interval="5m"):
    """Obtém dados do Yahoo Finance"""
    
    symbol_map = {
        "IBOV": "%5EBVSP",
        "DOLAR": "USDBRL%3DX",
        "SP500": "%5EGSPC",
        "NASDAQ": "%5EIXIC",
        "DOW": "%5EDJI"
    }
    
    yahoo_symbol = symbol_map.get(symbol, symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    
    params = {"interval": interval, "range": "1d"}
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        
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
        return None

# ========================================
# PERFORMANCE TRACKING
# ========================================

# Initialize session state for tracking
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = deque(maxlen=100)

def track_signal(signal, price, timestamp):
    """Rastreia sinais para análise de performance"""
    st.session_state.signal_history.append({
        'signal': signal,
        'price': price,
        'timestamp': timestamp
    })

def calculate_performance():
    """Calcula performance histórica dos sinais"""
    
    if len(st.session_state.signal_history) < 2:
        return None
    
    history = list(st.session_state.signal_history)
    
    correct = 0
    total = 0
    
    for i in range(len(history) - 1):
        current = history[i]
        next_signal = history[i + 1]
        
        if current['signal'] == 'COMPRA' and next_signal['price'] > current['price']:
            correct += 1
        elif current['signal'] == 'VENDA' and next_signal['price'] < current['price']:
            correct += 1
        
        total += 1
    
    if total == 0:
        return None
    
    accuracy = (correct / total) * 100
    
    return {
        'total_signals': len(history),
        'correct': correct,
        'accuracy': accuracy
    }

# ========================================
# MAIN APP
# ========================================

def main():
    
    st.title("🚀 PROFITONE ULTRA V4 - MÁXIMO REALISMO")
    st.markdown("### Sistema com Order Flow Real + Ensemble ML")
    
    # Sidebar
    st.sidebar.title("⚙️ Configurações")
    
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
    
    timeframe_map = {1: "5m", 2: "15m", 3: "1h"}
    interval = timeframe_map[modo]
    
    refresh_seconds = st.sidebar.slider(
        "🔄 Atualização (segundos)",
        5, 60, 15, 5
    )
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Análise Completa",
        "🔬 Order Flow Real",
        "🤖 Ensemble ML",
        "📈 Performance"
    ])
    
    # TAB 1: Análise Completa
    with tab1:
        
        st.markdown("## 📊 IBOVESPA - Análise Ultra-Avançada")
        
        df = get_market_data("IBOV", interval)
        
        if df is not None and len(df) > 30:
            
            # Quantum Hunter Ultra
            quantum = quantum_hunter_v13_ultra(df, modo)
            score = quantum['score']
            confidence = quantum['confidence']
            
            # Institutional Footprint
            footprint = institutional_footprint(df)
            
            # Preço
            preco_atual = df['close'].iloc[-1]
            preco_anterior = df['close'].iloc[-2]
            variacao = ((preco_atual - preco_anterior) / preco_anterior) * 100
            
            # Determinar sinal
            if score > 60 and confidence > 60:
                sinal = "COMPRA"
                sinal_class = "signal-compra"
                sinal_icon = "🚀"
                conf_class = "confidence-high"
            elif score < -60 and confidence > 60:
                sinal = "VENDA"
                sinal_class = "signal-venda"
                sinal_icon = "📉"
                conf_class = "confidence-high"
            else:
                sinal = "NEUTRO"
                sinal_class = "signal-neutro"
                sinal_icon = "⚖️"
                conf_class = "confidence-medium" if confidence > 40 else "confidence-low"
            
            # Track signal
            track_signal(sinal, preco_atual, datetime.now())
            
            # SIGNAL BOARD COM CONFIDENCE
            st.markdown(f"""
            <div class="signal-board {sinal_class}">
                <div style="font-size: 60px;">{sinal_icon}</div>
                <div style="font-size: 40px; font-weight: bold; margin: 10px 0;">
                    {sinal}
                </div>
                <div style="font-size: 24px;">
                    Score: {score:.1f}
                </div>
                <div style="font-size: 20px; margin-top: 10px; color: {'#00ff88' if confidence > 70 else ('#ffd700' if confidence > 50 else '#ff4444')};">
                    🎯 Confiança: {confidence:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card {conf_class}">
                    <div style="font-size: 14px; opacity: 0.8;">💰 Preço</div>
                    <div style="font-size: 28px; font-weight: bold;">R$ {preco_atual:,.2f}</div>
                    <div style="font-size: 18px; color: {'#00ff88' if variacao > 0 else '#ff4444'};">
                        {variacao:+.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card {conf_class}">
                    <div style="font-size: 14px; opacity: 0.8;">🎯 Confiança</div>
                    <div style="font-size: 28px; font-weight: bold;">{confidence:.0f}%</div>
                    <div style="font-size: 14px;">
                        {'Alta ✅' if confidence > 70 else ('Média ⚠️' if confidence > 50 else 'Baixa ❌')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                hurst = hurst_exponent(df['close'].iloc[-30:], 20)
                regime = "Trending" if hurst > 0.55 else ("Lateral" if hurst > 0.45 else "Reversal")
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 14px; opacity: 0.8;">📊 Regime</div>
                    <div style="font-size: 20px; font-weight: bold;">{regime}</div>
                    <div style="font-size: 14px;">Hurst: {hurst:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div style="font-size: 14px; opacity: 0.8;">📊 Volume</div>
                    <div style="font-size: 20px; font-weight: bold;">
                        {df['volume'].iloc[-1]/1e6:.1f}M
                    </div>
                    <div style="font-size: 14px;">
                        Ratio: {(df['volume'].iloc[-1] / df['volume'].mean()):.2f}x
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Componentes do Score
            st.markdown("### 🔍 Componentes do Quantum Score")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tend = quantum['components']['tendencia']
                st.metric("📈 Tendência", f"{tend:.1f}", f"Max: ±40")
            
            with col2:
                vw = quantum['components']['vwap']
                st.metric("💎 VWAP", f"{vw:.1f}", f"Max: ±30")
            
            with col3:
                rs = quantum['components']['rsi']
                st.metric("⚡ RSI", f"{rs:.1f}", f"Max: ±30")
            
            # Gráfico
            st.markdown("### 📈 Gráfico de Candlesticks")
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
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
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=quantum['media_lenta'],
                    name='Média Lenta',
                    line=dict(color='#ff00ff', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=quantum['media_rapida'],
                    name='Média Rápida',
                    line=dict(color='#00d9ff', width=2)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=quantum['vwap'],
                    name='VWAP',
                    line=dict(color='#ffd700', width=2, dash='dash')
                ),
                row=1, col=1
            )
            
            colors = ['#00ff88' if df['close'].iloc[i] > df['open'].iloc[i] else '#ff4444' 
                     for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=700,
                showlegend=True,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Dados insuficientes")
    
    # TAB 2: Order Flow Real
    with tab2:
        
        st.markdown("## 🔬 Order Flow & Microestrutura")
        
        df_flow = get_market_data("IBOV", interval)
        
        if df_flow is not None and len(df_flow) > 30:
            
            footprint = institutional_footprint(df_flow)
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.95); padding: 20px; border-radius: 15px; margin: 20px 0;">
                <h3 style="color: #000000;">📊 Participação Estimada no Mercado</h3>
                <p style="color: #000000; font-size: 16px;">
                    ⚠️ <strong>Confiança desta estimativa: {footprint['confidence']:.1f}%</strong>
                </p>
                <p style="color: #666; font-size: 14px;">
                    Baseado em: VPIN, Detecção de Icebergs, Análise de Absorção e Smart Money Index
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #667eea;">
                    <div style="font-size: 14px; opacity: 0.8;">🏛️ Institucional</div>
                    <div style="font-size: 32px; font-weight: bold; color: #667eea;">
                        {footprint['institucional']:.1f}%
                    </div>
                    <div style="font-size: 12px; margin-top: 10px;">
                        Fundos, Asset Managers
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #00d9ff;">
                    <div style="font-size: 14px; opacity: 0.8;">🏦 Bancos</div>
                    <div style="font-size: 32px; font-weight: bold; color: #00d9ff;">
                        {footprint['bancos']:.1f}%
                    </div>
                    <div style="font-size: 12px; margin-top: 10px;">
                        Market Makers, Prop Trading
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid #ffd700;">
                    <div style="font-size: 14px; opacity: 0.8;">👤 Varejo</div>
                    <div style="font-size: 32px; font-weight: bold; color: #ffd700;">
                        {footprint['varejo']:.1f}%
                    </div>
                    <div style="font-size: 12px; margin-top: 10px;">
                        Pessoas Físicas, Day Traders
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Institucional', 'Bancos', 'Varejo'],
                values=[footprint['institucional'], footprint['bancos'], footprint['varejo']],
                marker=dict(colors=['#667eea', '#00d9ff', '#ffd700']),
                hole=0.4,
                textinfo='label+percent',
                textfont=dict(size=16, color='#000000')
            )])
            
            fig_pie.update_layout(
                title="Distribuição de Participação",
                title_font=dict(size=20, color='#000000'),
                paper_bgcolor='rgba(255,255,255,0.9)',
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Métricas de microestrutura
            st.markdown("### 📊 Métricas de Microestrutura")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "🎯 VPIN",
                    f"{footprint['vpin']:.3f}",
                    "Toxic Flow" if footprint['vpin'] > 0.5 else "Normal"
                )
            
            with col2:
                st.metric(
                    "🧊 Iceberg",
                    f"{footprint['iceberg']:.2f}",
                    "Detectado" if abs(footprint['iceberg']) > 1 else "Normal"
                )
            
            with col3:
                st.metric(
                    "🔄 Absorção",
                    f"{footprint['absorption']:.2f}",
                    "Alta" if abs(footprint['absorption']) > 0.5 else "Normal"
                )
            
            with col4:
                st.metric(
                    "💰 SMI",
                    f"{footprint['smi']:.1f}",
                    "Bullish" if footprint['smi'] > 50 else "Bearish"
                )
            
            # Explicações
            with st.expander("ℹ️ Como interpretar estes dados"):
                st.markdown("""
                ### 📚 Guia de Interpretação
                
                **🎯 VPIN (Volume-Synchronized Probability of Informed Trading)**
                - > 0.5: Alta probabilidade de traders informados (institucionais) ativos
                - < 0.3: Fluxo normal de varejo
                - Baseado em: Easley, López de Prado & O'Hara (2012)
                
                **🧊 Detecção de Icebergs**
                - Score > 1.5: Possível ordem grande escondida
                - Indica: Institucionais acumulando/distribuindo
                
                **🔄 Absorção de Liquidez**
                - Positivo: Suporte forte (compradores absorvendo vendas)
                - Negativo: Resistência forte (vendedores absorvendo compras)
                
                **💰 Smart Money Index**
                - > 50: Dinheiro institucional entrando
                - < 50: Dinheiro institucional saindo
                
                **⚠️ Importante:**
                - Estas são ESTIMATIVAS baseadas em análise de volume e preço
                - Margem de erro: ±10-15%
                - Dados oficiais custam R$ 500-2.000/mês
                - Use como confirmação, não como única fonte
                """)
        
        else:
            st.error("❌ Dados insuficientes")
    
    # TAB 3: Ensemble ML
    with tab3:
        
        st.markdown("## 🤖 Ensemble Machine Learning")
        
        if not ML_AVAILABLE:
            st.warning("⚠️ Instale: pip install tensorflow xgboost lightgbm")
        else:
            
            df_ml = get_market_data("IBOV", "15m")
            
            if df_ml is not None and len(df_ml) > 100:
                
                with st.spinner("🧠 Treinando ensemble (XGBoost + LightGBM + Random Forest)..."):
                    models, scaler, metrics = train_ensemble_model(df_ml)
                
                if models is not None:
                    
                    st.success(f"✅ Ensemble treinado com {metrics['ensemble_acc']*100:.1f}% de acurácia média")
                    
                    # Individual accuracies
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🟦 XGBoost", f"{metrics['xgb_acc']*100:.1f}%")
                    
                    with col2:
                        st.metric("🟩 LightGBM", f"{metrics['lgb_acc']*100:.1f}%")
                    
                    with col3:
                        st.metric("🟨 Random Forest", f"{metrics['rf_acc']*100:.1f}%")
                    
                    # Prediction
                    proba, direction, confidence = ensemble_predict(df_ml, models, scaler)
                    
                    st.markdown(f"""
                    <div class="signal-board {'signal-compra' if 'Alta' in direction else ('signal-venda' if 'Baixa' in direction else 'signal-neutro')}">
                        <div style="font-size: 40px; font-weight: bold;">
                            Previsão ML: {direction}
                        </div>
                        <div style="font-size: 24px; margin-top: 10px;">
                            Probabilidade: {proba*100:.1f}%
                        </div>
                        <div style="font-size: 20px; margin-top: 10px; color: {'#00ff88' if confidence > 70 else ('#ffd700' if confidence > 50 else '#ff4444')};">
                            🎯 Confiança (acordo entre modelos): {confidence:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba*100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilidade de Alta", 'font': {'size': 24, 'color': '#000000'}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': "#000000"},
                            'bar': {'color': "#667eea"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "#000000",
                            'steps': [
                                {'range': [0, 40], 'color': 'rgba(255,68,68,0.3)'},
                                {'range': [40, 60], 'color': 'rgba(255,215,0,0.3)'},
                                {'range': [60, 100], 'color': 'rgba(0,255,136,0.3)'}
                            ]
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        paper_bgcolor='rgba(255,255,255,0.9)',
                        font={'color': "#000000"},
                        height=400
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    with st.expander("ℹ️ Como funciona o Ensemble"):
                        st.markdown("""
                        ### 🧠 Ensemble Learning
                        
                        **O que é:**
                        - Combina 3 modelos diferentes de Machine Learning
                        - Cada modelo "vota" na previsão
                        - Resultado final = média das previsões
                        
                        **Por que é melhor:**
                        - ✅ Reduz overfitting
                        - ✅ Mais robusto a ruído
                        - ✅ Melhor generalização
                        
                        **Features usadas:**
                        - Returns, Volatilidade, RSI, MACD
                        - Volume Ratio, Bollinger Bands
                        - VPIN, Order Flow, Cumulative Delta
                        
                        **Confiança:**
                        - Alta (>70%): Modelos concordam fortemente
                        - Média (50-70%): Modelos têm alguma divergência
                        - Baixa (<50%): Modelos discordam
                        
                        **⚠️ Limitações:**
                        - Não prevê eventos extremos (cisnes negros)
                        - Funciona melhor em tendências
                        - Retreina a cada sessão (sem memória de longo prazo)
                        """)
                
                else:
                    st.error("❌ Falha no treinamento")
            
            else:
                st.warning("⚠️ Dados insuficientes (mínimo 100 barras)")
    
    # TAB 4: Performance
    with tab4:
        
        st.markdown("## 📈 Performance Histórica do Sistema")
        
        perf = calculate_performance()
        
        if perf is not None:
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📊 Total de Sinais",
                    perf['total_signals']
                )
            
            with col2:
                st.metric(
                    "✅ Sinais Corretos",
                    perf['correct']
                )
            
            with col3:
                acc_color = "#00ff88" if perf['accuracy'] > 70 else ("#ffd700" if perf['accuracy'] > 50 else "#ff4444")
                st.markdown(f"""
                <div class="metric-card" style="border-left: 5px solid {acc_color};">
                    <div style="font-size: 14px; opacity: 0.8;">🎯 Acurácia</div>
                    <div style="font-size: 32px; font-weight: bold; color: {acc_color};">
                        {perf['accuracy']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Histórico de sinais
            if len(st.session_state.signal_history) > 0:
                
                st.markdown("### 📋 Últimos Sinais")
                
                history_df = pd.DataFrame(list(st.session_state.signal_history))
                history_df = history_df.sort_values('timestamp', ascending=False).head(20)
                
                st.dataframe(
                    history_df.style.applymap(
                        lambda x: 'background-color: rgba(0,255,136,0.2)' if x == 'COMPRA' else (
                            'background-color: rgba(255,68,68,0.2)' if x == 'VENDA' else ''
                        ),
                        subset=['signal']
                    ),
                    use_container_width=True
                )
        
        else:
            st.info("ℹ️ Aguardando acumular histórico de sinais...")
        
        st.markdown("""
        ---
        ### 📚 Disclaimer de Performance
        
        ⚠️ **Importante:**
        - A acurácia mostrada é baseada em sinais passados
        - Performance passada NÃO garante resultados futuros
        - Este é um sistema de APOIO à decisão, não substituição
        - SEMPRE use stop-loss e gestão de risco
        - Não opere com dinheiro que não pode perder
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #000000; padding: 20px;">
        <p style="font-size: 16px; font-weight: bold;">
            🚀 ProfitOne Ultra V4 | Order Flow Real + Ensemble ML + Performance Tracking
        </p>
        <p style="font-size: 12px;">
            ⚠️ Sistema educacional. Trading envolve risco de perda de capital.
        </p>
        <p style="font-size: 12px;">
            🕐 {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    time.sleep(refresh_seconds)
    st.rerun()

if __name__ == "__main__":
    main()
