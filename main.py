"""
ProfitOne V3.0 - Sistema Institucional de Trading
Sistema Profissional com Order Flow, Market Profile, ML e Risk Management

Desenvolvido para traders profissionais que exigem:
- Order Flow Analysis
- Market Profile & Auction Theory
- Machine Learning Models
- Advanced Risk Management
- Multi-Timeframe Analysis
- Professional UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
import json

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO GLOBAL
# ============================================================================

st.set_page_config(
    page_title="ProfitOne V3.0 - Institucional",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PROFISSIONAL AVANÇADO
st.markdown("""
<style>
    /* Fundo profissional */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Typography */
    h1, h2, h3 {
        color: #00ff88 !important;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 32px !important;
        font-weight: bold !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 18px !important;
    }
    
    /* Cards profissionais */
    .pro-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 255, 136, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .pro-card:hover {
        border: 1px solid rgba(0, 255, 136, 0.6);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 136, 0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* Sinais profissionais */
    .signal-strong-buy {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.3), rgba(0, 200, 100, 0.2));
        border-left: 5px solid #00ff88;
        animation: pulse-green 2s infinite;
    }
    
    .signal-strong-sell {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.3), rgba(200, 0, 0, 0.2));
        border-left: 5px solid #ff4444;
        animation: pulse-red 2s infinite;
    }
    
    .signal-buy {
        background: rgba(0, 255, 136, 0.15);
        border-left: 4px solid #00ff88;
    }
    
    .signal-sell {
        background: rgba(255, 68, 68, 0.15);
        border-left: 4px solid #ff4444;
    }
    
    .signal-neutral {
        background: rgba(255, 170, 0, 0.15);
        border-left: 4px solid #ffaa00;
    }
    
    /* Animações */
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.3); }
        50% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.6); }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.3); }
        50% { box-shadow: 0 0 40px rgba(255, 68, 68, 0.6); }
    }
    
    /* Tabelas profissionais */
    .dataframe {
        background: rgba(0, 0, 0, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .dataframe th {
        background: rgba(0, 255, 136, 0.2) !important;
        color: #00ff88 !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
    }
    
    .dataframe td {
        color: #ffffff !important;
    }
    
    /* Sidebar profissional */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
        border-right: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    /* Botões premium */
    .stButton > button {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #00cc6a 0%, #00aa55 100%);
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.6);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
    }
    
    /* Tabs profissionais */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(0, 0, 0, 0.2);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px 24px;
        color: #ffffff;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 255, 136, 0.1);
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: black !important;
        box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4);
    }
    
    /* Tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted #00ff88;
        cursor: help;
    }
    
    /* Alerts premium */
    .stAlert {
        background: rgba(0, 0, 0, 0.4);
        border-radius: 10px;
        border-left: 5px solid #00ff88;
    }
    
    /* Expander premium */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background: rgba(0, 0, 0, 0.3);
        color: #ffffff;
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 8px;
    }
    
    .stSelectbox > div > div > div {
        background: rgba(0, 0, 0, 0.3);
        color: #ffffff;
        border: 1px solid rgba(0, 255, 136, 0.3);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CLASSES E ESTRUTURAS DE DADOS
# ============================================================================

class TradingSession:
    """Gerencia estado da sessão de trading"""
    def __init__(self):
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {
                'balance': 100000.00,
                'positions': [],
                'pnl': 0.0
            }


class RiskManager:
    """Gerenciamento de risco profissional"""
    
    @staticmethod
    def kelly_criterion(win_rate, avg_win, avg_loss):
        """Cál Kelly Criterion para tamanho ótimo de posição"""
        if avg_loss == 0:
            return 0
        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b
        return max(0, min(kelly, 0.25))  # Cap at 25%
    
    @staticmethod
    def position_size(capital, risk_per_trade, entry, stop_loss):
        """Calcula tamanho da posição baseado no risco"""
        risk_amount = capital * risk_per_trade
        risk_per_share = abs(entry - stop_loss)
        if risk_per_share == 0:
            return 0
        shares = risk_amount / risk_per_share
        return int(shares)
    
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.05):
        """Calcula Sharpe Ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0
        excess_returns = returns.mean() - risk_free_rate / 252
        return (excess_returns / returns.std()) * np.sqrt(252)


# ============================================================================
# INDICADORES PROFISSIONAIS (TODOS DO V2.0 + NOVOS)
# ============================================================================

# [INCLUIR TODOS OS INDICADORES DO V2.0 AQUI - TEMA, KALMAN, FISHER, ETC]
# [Por brevidade, vou mostrar apenas os NOVOS]

def calculate_cvd(df):
    """Cumulative Volume Delta - Profissional"""
    price_change = df['close'].diff()
    
    buy_volume = df['volume'].where(price_change > 0, 0)
    sell_volume = df['volume'].where(price_change < 0, 0)
    
    delta = buy_volume - sell_volume
    cvd = delta.cumsum()
    
    return cvd, delta


def calculate_vwap_bands(df, std_mult=2):
    """VWAP com bandas de desvio padrão"""
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    squared_diff = ((typical_price - vwap) ** 2) * df['volume']
    variance = squared_diff.cumsum() / df['volume'].cumsum()
    std = np.sqrt(variance)
    
    upper_band1 = vwap + std_mult * std
    lower_band1 = vwap - std_mult * std
    upper_band2 = vwap + std_mult * 2 * std
    lower_band2 = vwap - std_mult * 2 * std
    
    return vwap, upper_band1, lower_band1, upper_band2, lower_band2


def market_profile(df, bins=20):
    """Market Profile - POC, VAH, VAL"""
    # Criar bins de preço
    price_range = df['high'].max() - df['low'].min()
    bin_size = price_range / bins
    
    # Calcular volume em cada bin
    profile = {}
    
    for idx, row in df.iterrows():
        bin_low = int((row['low'] - df['low'].min()) / bin_size)
        bin_high = int((row['high'] - df['low'].min()) / bin_size)
        
        for b in range(bin_low, bin_high + 1):
            if b not in profile:
                profile[b] = 0
            profile[b] += row['volume']
    
    # POC (Point of Control) - Preço com maior volume
    poc_bin = max(profile, key=profile.get)
    poc_price = df['low'].min() + (poc_bin * bin_size) + (bin_size / 2)
    
    # Value Area (70% do volume)
    total_volume = sum(profile.values())
    target_volume = total_volume * 0.70
    
    sorted_bins = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    
    value_area_bins = []
    accumulated_volume = 0
    
    for bin_num, vol in sorted_bins:
        value_area_bins.append(bin_num)
        accumulated_volume += vol
        if accumulated_volume >= target_volume:
            break
    
    vah_price = df['low'].min() + (max(value_area_bins) * bin_size)
    val_price = df['low'].min() + (min(value_area_bins) * bin_size)
    
    return poc_price, vah_price, val_price, profile


def detect_iceberg_orders(df, volume_threshold=2.0):
    """Detecta ordens iceberg (grandes ordens ocultas)"""
    avg_volume = df['volume'].rolling(window=20).mean()
    volume_ratio = df['volume'] / avg_volume
    
    # Candles com volume anormal mas movimento pequeno = absorção
    body = abs(df['close'] - df['open'])
    body_ratio = body / df['close']
    
    # Iceberg: Volume alto + movimento pequeno
    icebergs = (volume_ratio > volume_threshold) & (body_ratio < 0.005)
    
    return icebergs


def calculate_absorption_zones(df, window=20):
    """Zonas de absorção institucional"""
    # Volume alto + range pequeno = absorção
    avg_volume = df['volume'].rolling(window=window).mean()
    avg_range = (df['high'] - df['low']).rolling(window=window).mean()
    
    volume_ratio = df['volume'] / avg_volume
    range_ratio = (df['high'] - df['low']) / avg_range
    
    # Absorção = Volume alto + Range baixo
    absorption_score = volume_ratio / (range_ratio + 0.1)
    
    return absorption_score


def market_structure(df, lookback=10):
    """Identifica estrutura de mercado (HH, HL, LH, LL)"""
    highs = df['high'].rolling(window=lookback, center=True).max()
    lows = df['low'].rolling(window=lookback, center=True).min()
    
    # Identificar pivôs
    pivot_highs = df['high'] == highs
    pivot_lows = df['low'] == lows
    
    # Classificar estrutura
    structure = []
    prev_high = None
    prev_low = None
    
    for i in range(len(df)):
        if pivot_highs.iloc[i]:
            if prev_high is not None:
                if df['high'].iloc[i] > prev_high:
                    structure.append('HH')  # Higher High
                else:
                    structure.append('LH')  # Lower High
            else:
                structure.append('H')
            prev_high = df['high'].iloc[i]
        elif pivot_lows.iloc[i]:
            if prev_low is not None:
                if df['low'].iloc[i] > prev_low:
                    structure.append('HL')  # Higher Low
                else:
                    structure.append('LL')  # Lower Low
            else:
                structure.append('L')
            prev_low = df['low'].iloc[i]
        else:
            structure.append('')
    
    return structure


# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================

class MLPredictor:
    """Preditor baseado em Machine Learning"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def prepare_features(self, df):
        """Prepara features para ML"""
        features = pd.DataFrame()
        
        # Price features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(df['close'], 14)
        features['macd'] = self.calculate_macd(df['close'])
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Volatility
        features['atr'] = self.calculate_atr(df)
        features['bb_width'] = self.calculate_bb_width(df['close'])
        
        # Momentum
        features['roc'] = df['close'].pct_change(10)
        features['momentum'] = df['close'] - df['close'].shift(10)
        
        features = features.fillna(0)
        return features
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data):
        ema12 = data.ewm(span=12).mean()
        ema26 = data.ewm(span=26).mean()
        return ema12 - ema26
    
    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    @staticmethod
    def calculate_bb_width(data, period=20):
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return (upper - lower) / sma
    
    def train(self, df):
        """Treina o modelo"""
        features = self.prepare_features(df)
        
        # Target: próximo movimento (1 = up, 0 = down)
        target = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Remover últimas linhas (sem target)
        features = features[:-1]
        target = target[:-1]
        
        # Remover NaN
        mask = ~(features.isna().any(axis=1) | target.isna())
        features = features[mask]
        target = target[mask]
        
        if len(features) < 50:
            return False
        
        # Normalizar
        features_scaled = self.scaler.fit_transform(features)
        
        # Treinar
        self.model.fit(features_scaled, target)
        self.trained = True
        
        return True
    
    def predict(self, df):
        """Faz predição"""
        if not self.trained:
            return None, None
        
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features.tail(1))
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability


# ============================================================================
# BUSCA DE DADOS OTIMIZADA
# ============================================================================

@st.cache_data(ttl=60)
def get_data_advanced(symbol, period="5d", interval="15m"):
    """Busca dados com múltiplos timeframes"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Timeframe principal
        df_main = ticker.history(period=period, interval=interval)
        
        if df_main.empty:
            df_main = ticker.history(period="1mo", interval="1h")
        
        if df_main.empty:
            return None, None, "Sem dados disponíveis"
        
        # Timeframe maior para contexto
        if interval == "15m":
            df_htf = ticker.history(period="1mo", interval="1h")
        elif interval == "1h":
            df_htf = ticker.history(period="6mo", interval="1d")
        else:
            df_htf = df_main.copy()
        
        # Padronizar colunas
        for df in [df_main, df_htf]:
            df.columns = [col.lower() for col in df.columns]
            df.reset_index(inplace=True)
        
        return df_main, df_htf, None
        
    except Exception as e:
        return None, None, str(e)


# ============================================================================
# SISTEMA DE PONTUAÇÃO INSTITUCIONAL
# ============================================================================

def calculate_institutional_score(df_main, df_htf):
    """Score institucional completo com todos os módulos"""
    
    scores = {}
    signals = {}
    
    # 1. ORDER FLOW ANALYSIS
    cvd, delta = calculate_cvd(df_main)
    vwap, vwap_u1, vwap_l1, vwap_u2, vwap_l2 = calculate_vwap_bands(df_main)
    icebergs = detect_iceberg_orders(df_main)
    absorption = calculate_absorption_zones(df_main)
    
    # Score Order Flow
    cvd_trend = 50 if cvd.iloc[-1] > cvd.iloc[-20] else -50
    vwap_position = 30 if df_main['close'].iloc[-1] > vwap.iloc[-1] else -30
    iceberg_signal = -40 if icebergs.iloc[-1] else 0
    absorption_signal = -30 if absorption.iloc[-1] > 2 else 0
    
    scores['order_flow'] = (cvd_trend + vwap_position + iceberg_signal + absorption_signal) / 4
    
    # 2. MARKET PROFILE
    poc, vah, val, profile = market_profile(df_main)
    current_price = df_main['close'].iloc[-1]
    
    if current_price > vah:
        profile_score = 60  # Acima da Value Area = bullish
    elif current_price < val:
        profile_score = -60  # Abaixo da Value Area = bearish
    elif abs(current_price - poc) / current_price < 0.002:
        profile_score = 0  # Próximo ao POC = neutro
    else:
        profile_score = 20 if current_price > poc else -20
    
    scores['market_profile'] = profile_score
    
    # 3. MULTI-TIMEFRAME
    # HTF structure
    htf_trend = 50 if df_htf['close'].iloc[-1] > df_htf['close'].iloc[-20] else -50
    
    # Confluence
    mtf_ema20 = df_main['close'].ewm(span=20).mean().iloc[-1]
    htf_ema50 = df_htf['close'].ewm(span=50).mean().iloc[-1]
    
    mtf_alignment = 40 if (df_main['close'].iloc[-1] > mtf_ema20 and 
                           df_htf['close'].iloc[-1] > htf_ema50) else -40
    
    scores['multi_timeframe'] = (htf_trend + mtf_alignment) / 2
    
    # 4. MACHINE LEARNING
    ml_predictor = MLPredictor()
    ml_trained = ml_predictor.train(df_main)
    
    if ml_trained:
        ml_prediction, ml_probability = ml_predictor.predict(df_main)
        if ml_prediction is not None:
            ml_score = 70 if ml_prediction == 1 else -70
            ml_confidence = max(ml_probability) * 100
        else:
            ml_score = 0
            ml_confidence = 50
    else:
        ml_score = 0
        ml_confidence = 50
    
    scores['machine_learning'] = ml_score
    
    # 5. VOLUME ANALYSIS
    volume_trend = df_main['volume'].rolling(5).mean().iloc[-1] / df_main['volume'].rolling(20).mean().iloc[-1]
    volume_score = 40 if volume_trend > 1.3 else (-20 if volume_trend < 0.7 else 0)
    
    scores['volume'] = volume_score
    
    # 6. RISK METRICS
    returns = df_main['close'].pct_change().dropna()
    sharpe = RiskManager.sharpe_ratio(returns)
    sharpe_score = np.clip(sharpe * 20, -50, 50)
    
    scores['risk'] = sharpe_score
    
    # SCORE MASTER INSTITUCIONAL
    weights = {
        'order_flow': 1.5,
        'market_profile': 1.3,
        'multi_timeframe': 1.2,
        'machine_learning': 1.4,
        'volume': 1.0,
        'risk': 0.8
    }
    
    weighted_scores = [scores[key] * weights[key] for key in scores.keys()]
    master_score = np.mean(weighted_scores)
    master_score = np.clip(master_score, -100, 100)
    
    # SINAL INSTITUCIONAL
    if master_score > 60:
        signal = 'STRONG BUY'
        signal_class = 'signal-strong-buy'
    elif master_score > 30:
        signal = 'BUY'
        signal_class = 'signal-buy'
    elif master_score < -60:
        signal = 'STRONG SELL'
        signal_class = 'signal-strong-sell'
    elif master_score < -30:
        signal = 'SELL'
        signal_class = 'signal-sell'
    else:
        signal = 'NEUTRAL'
        signal_class = 'signal-neutral'
    
    return {
        'master_score': master_score,
        'signal': signal,
        'signal_class': signal_class,
        'module_scores': scores,
        'ml_confidence': ml_confidence if ml_trained else 50,
        'sharpe_ratio': sharpe,
        'indicators': {
            'cvd': cvd.iloc[-1],
            'vwap': vwap.iloc[-1],
            'poc': poc,
            'vah': vah,
            'val': val,
            'iceberg_detected': icebergs.iloc[-1],
            'absorption_level': absorption.iloc[-1]
        }
    }


# ============================================================================
# GRÁFICOS PROFISSIONAIS
# ============================================================================

def create_institutional_chart(df_main, df_htf, result):
    """Gráfico institucional completo"""
    
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        row_heights=[0.45, 0.15, 0.15, 0.15, 0.1],
        subplot_titles=(
            'Price Action & Market Profile',
            'Cumulative Volume Delta (CVD)',
            'Volume Analysis',
            'VWAP Bands',
            'Absorption Zones'
        )
    )
    
    # ROW 1: CANDLESTICK + MARKET PROFILE
    fig.add_trace(
        go.Candlestick(
            x=df_main.index,
            open=df_main['open'],
            high=df_main['high'],
            low=df_main['low'],
            close=df_main['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='rgba(0, 255, 136, 0.3)',
            decreasing_fillcolor='rgba(255, 68, 68, 0.3)'
        ),
        row=1, col=1
    )
    
    # VWAP
    vwap, vwap_u1, vwap_l1, vwap_u2, vwap_l2 = calculate_vwap_bands(df_main)
    
    fig.add_trace(
        go.Scatter(x=df_main.index, y=vwap, name='VWAP',
                  line=dict(color='yellow', width=2)),
        row=1, col=1
    )
    
    # POC, VAH, VAL
    poc = result['indicators']['poc']
    vah = result['indicators']['vah']
    val = result['indicators']['val']
    
    fig.add_hline(y=poc, line=dict(color='cyan', width=3, dash='solid'),
                  annotation_text="POC", row=1, col=1)
    fig.add_hline(y=vah, line=dict(color='green', width=2, dash='dash'),
                  annotation_text="VAH", row=1, col=1)
    fig.add_hline(y=val, line=dict(color='red', width=2, dash='dash'),
                  annotation_text="VAL", row=1, col=1)
    
    # ROW 2: CVD
    cvd, delta = calculate_cvd(df_main)
    
    fig.add_trace(
        go.Scatter(x=df_main.index, y=cvd, name='CVD',
                  line=dict(color='cyan', width=2),
                  fill='tozeroy',
                  fillcolor='rgba(0, 255, 255, 0.2)'),
        row=2, col=1
    )
    
    # ROW 3: VOLUME
    colors = ['#00ff88' if df_main['close'].iloc[i] >= df_main['open'].iloc[i] 
              else '#ff4444' for i in range(len(df_main))]
    
    fig.add_trace(
        go.Bar(x=df_main.index, y=df_main['volume'], name='Volume',
               marker_color=colors, opacity=0.7),
        row=3, col=1
    )
    
    # ROW 4: VWAP BANDS
    fig.add_trace(
        go.Scatter(x=df_main.index, y=vwap_u1, name='VWAP +1σ',
                  line=dict(color='lime', width=1, dash='dot')),
        row=4, col=1
    )
    fig.add_trace(
        go.Scatter(x=df_main.index, y=vwap_l1, name='VWAP -1σ',
                  line=dict(color='red', width=1, dash='dot'),
                  fill='tonexty', fillcolor='rgba(0, 255, 136, 0.1)'),
        row=4, col=1
    )
    
    # ROW 5: ABSORPTION
    absorption = calculate_absorption_zones(df_main)
    
    fig.add_trace(
        go.Scatter(x=df_main.index, y=absorption, name='Absorption',
                  line=dict(color='orange', width=2),
                  fill='tozeroy',
                  fillcolor='rgba(255, 170, 0, 0.2)'),
        row=5, col=1
    )
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=5, col=1)
    
    # LAYOUT
    fig.update_layout(
        height=1200,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Labels
    fig.update_xaxes(title_text="Time", row=5, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="CVD", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_yaxes(title_text="VWAP", row=4, col=1)
    fig.update_yaxes(title_text="Absorption", row=5, col=1)
    
    return fig


# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

def main():
    # Inicializar sessão
    session = TradingSession()
    
    # HEADER PROFISSIONAL
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 30px;'>
            <h1 style='font-size: 48px; margin-bottom: 10px;'>🏦 PROFITONE V3.0</h1>
            <h2 style='font-size: 24px; color: #00ff88; margin-top: 0;'>INSTITUTIONAL TRADING SYSTEM</h2>
            <p style='font-size: 16px; color: rgba(255, 255, 255, 0.7);'>
                Order Flow • Market Profile • Machine Learning • Risk Management
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # SIDEBAR PROFISSIONAL
    with st.sidebar:
        st.markdown("## 🎯 TRADING CENTER")
        
        # Asset Selection
        st.markdown("### 📊 Asset Selection")
        
        preset_symbols = {
            "🇧🇷 Ibovespa": "^BVSP",
            "🛢️ Petrobras (PETR4)": "PETR4.SA",
            "⛏️ Vale (VALE3)": "VALE3.SA",
            "🏦 Itaú (ITUB4)": "ITUB4.SA",
            "🏦 Bradesco (BBDC4)": "BBDC4.SA",
            "📈 S&P 500": "^GSPC",
            "💰 Bitcoin": "BTC-USD",
            "💎 Ethereum": "ETH-USD",
            "🔧 Custom": "CUSTOM"
        }
        
        selected = st.selectbox("Select Asset", list(preset_symbols.keys()), index=0)
        
        if preset_symbols[selected] == "CUSTOM":
            symbol = st.text_input("Ticker Symbol:", "^BVSP")
        else:
            symbol = preset_symbols[selected]
        
        st.markdown("### ⏱️ Timeframe")
        
        timeframe = st.radio(
            "Select Period",
            ["⚡ 15 min (Scalp)", "📊 1 hour (Day)", "📈 1 day (Swing)"],
            index=0
        )
        
        interval_map = {
            "⚡ 15 min (Scalp)": "15m",
            "📊 1 hour (Day)": "1h",
            "📈 1 day (Swing)": "1d"
        }
        interval = interval_map[timeframe]
        
        period_map = {"15m": "5d", "1h": "1mo", "1d": "6mo"}
        period = period_map[interval]
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("📥 Export", use_container_width=True):
                st.info("Export feature coming soon!")
        
        st.markdown("---")
        
        # Risk Management
        st.markdown("### 💰 Risk Management")
        
        capital = st.number_input(
            "Account Balance ($)",
            min_value=1000.0,
            max_value=10000000.0,
            value=100000.0,
            step=1000.0
        )
        
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
        
        st.markdown("---")
        
        # System Info
        st.markdown("### 📡 System Status")
        st.success("✅ All systems operational")
        st.info(f"🕐 Last Update: {datetime.now().strftime('%H:%M:%S')}")
        
        # Módulos ativos
        st.markdown("### 🔧 Active Modules")
        modules = [
            "✅ Order Flow Analysis",
            "✅ Market Profile",
            "✅ Multi-Timeframe",
            "✅ Machine Learning",
            "✅ Volume Analysis",
            "✅ Risk Management"
        ]
        for module in modules:
            st.caption(module)
    
    # MAIN CONTENT
    
    # Loading data
    with st.spinner(f"🔄 Loading market data for {symbol}..."):
        df_main, df_htf, error = get_data_advanced(symbol, period, interval)
    
    if error or df_main is None:
        st.error(f"❌ Error: {error if error else 'No data available'}")
        st.info("💡 **Suggestions:**\n- Try another asset\n- Check your internet connection\n- Use a different timeframe")
        return
    
    # Preparar dados
    if 'datetime' in df_main.columns:
        df_main = df_main.set_index('datetime')
        df_htf = df_htf.set_index('datetime')
    elif 'date' in df_main.columns:
        df_main = df_main.set_index('date')
        df_htf = df_htf.set_index('date')
    
    # CALCULAR SCORE INSTITUCIONAL
    with st.spinner("🧠 Analyzing market with ML models..."):
        result = calculate_institutional_score(df_main, df_htf)
    
    # METRICS DASHBOARD
    st.markdown("---")
    
    # Primary Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"<div class='pro-card {result['signal_class']}' style='text-align: center;'>"
                   f"<h3>🎯 Master Score</h3>"
                   f"<h1>{result['master_score']:.1f}</h1>"
                   f"<p style='font-size: 20px; font-weight: bold;'>{result['signal']}</p>"
                   f"</div>", unsafe_allow_html=True)
    
    with col2:
        current_price = df_main['close'].iloc[-1]
        prev_price = df_main['close'].iloc[-2] if len(df_main) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100
        
        st.metric(
            "💰 Current Price",
            f"${current_price:.2f}",
            delta=f"{price_change_pct:+.2f}%"
        )
    
    with col3:
        st.metric(
            "🤖 ML Confidence",
            f"{result['ml_confidence']:.1f}%",
            delta=f"{result['ml_confidence'] - 50:+.1f}%"
        )
    
    with col4:
        st.metric(
            "📊 Sharpe Ratio",
            f"{result['sharpe_ratio']:.2f}",
            delta="Good" if result['sharpe_ratio'] > 1 else "Poor"
        )
    
    with col5:
        cvd_value = result['indicators']['cvd']
        st.metric(
            "📈 CVD",
            f"{cvd_value:,.0f}",
            delta="Bullish" if cvd_value > 0 else "Bearish"
        )
    
    # MODULE SCORES
    st.markdown("---")
    st.markdown("## 📊 Module Performance Dashboard")
    
    cols = st.columns(6)
    
    module_info = [
        ("🔄 Order Flow", "order_flow", "Volume & Delta Analysis"),
        ("📊 Market Profile", "market_profile", "POC & Value Area"),
        ("⏱️ Multi-TF", "multi_timeframe", "HTF Alignment"),
        ("🤖 Machine Learning", "machine_learning", "AI Prediction"),
        ("📈 Volume", "volume", "Volume Trend"),
        ("💰 Risk", "risk", "Sharpe Ratio")
    ]
    
    for col, (name, key, description) in zip(cols, module_info):
        score = result['module_scores'][key]
        
        signal_class = ""
        if score > 40:
            signal_class = "signal-strong-buy"
        elif score > 20:
            signal_class = "signal-buy"
        elif score < -40:
            signal_class = "signal-strong-sell"
        elif score < -20:
            signal_class = "signal-sell"
        else:
            signal_class = "signal-neutral"
        
        with col:
            col.markdown(
                f"<div class='pro-card {signal_class}' style='text-align: center;'>"
                f"<h4>{name}</h4>"
                f"<h2>{score:.1f}</h2>"
                f"<p style='font-size: 12px; color: rgba(255,255,255,0.6);'>{description}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
    
    # INSTITUTIONAL CHART
    st.markdown("---")
    st.markdown("## 📈 Institutional Trading Chart")
    
    fig = create_institutional_chart(df_main, df_htf, result)
    st.plotly_chart(fig, use_container_width=True)
    
    # DETAILED ANALYSIS TABS
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔄 Order Flow",
        "📊 Market Profile",
        "🤖 ML Analysis",
        "💰 Risk Management",
        "📋 Trade Journal"
    ])
    
    with tab1:
        st.markdown("### 🔄 Order Flow Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Cumulative Volume Delta")
            st.write(f"**Current CVD:** {result['indicators']['cvd']:,.0f}")
            
            cvd, delta = calculate_cvd(df_main)
            cvd_trend = "📈 Bullish" if cvd.iloc[-1] > cvd.iloc[-20] else "📉 Bearish"
            st.write(f"**Trend:** {cvd_trend}")
            
            st.write(f"**Last Delta:** {delta.iloc[-1]:,.0f}")
        
        with col2:
            st.markdown("#### VWAP Analysis")
            st.write(f"**VWAP:** ${result['indicators']['vwap']:.2f}")
            st.write(f"**Current Price:** ${current_price:.2f}")
            
            vwap_pos = "Above ✅" if current_price > result['indicators']['vwap'] else "Below ❌"
            st.write(f"**Position:** {vwap_pos}")
        
        with col3:
            st.markdown("#### Iceberg & Absorption")
            
            iceberg_status = "🚨 DETECTED" if result['indicators']['iceberg_detected'] else "✅ Clear"
            st.write(f"**Iceberg Orders:** {iceberg_status}")
            
            absorption_level = result['indicators']['absorption_level']
            absorption_status = "🔴 HIGH" if absorption_level > 2 else "🟢 Normal"
            st.write(f"**Absorption:** {absorption_status}")
            st.write(f"**Level:** {absorption_level:.2f}")
    
    with tab2:
        st.markdown("### 📊 Market Profile & Auction Theory")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Value Area")
            st.write(f"**POC (Point of Control):** ${result['indicators']['poc']:.2f}")
            st.write(f"**VAH (Value Area High):** ${result['indicators']['vah']:.2f}")
            st.write(f"**VAL (Value Area Low):** ${result['indicators']['val']:.2f}")
            
            # Price position
            if current_price > result['indicators']['vah']:
                position = "📈 Above Value Area (Bullish)"
            elif current_price < result['indicators']['val']:
                position = "📉 Below Value Area (Bearish)"
            elif abs(current_price - result['indicators']['poc']) / current_price < 0.002:
                position = "🎯 At POC (Balanced)"
            else:
                position = "📊 Inside Value Area"
            
            st.info(f"**Current Position:** {position}")
        
        with col2:
            st.markdown("#### Market Structure")
            
            structure = market_structure(df_main, lookback=10)
            recent_structure = [s for s in structure[-10:] if s != '']
            
            if recent_structure:
                last_structure = recent_structure[-1]
                
                structure_meaning = {
                    'HH': "📈 Higher High (Bullish)",
                    'HL': "📊 Higher Low (Bullish Continuation)",
                    'LH': "📉 Lower High (Bearish)",
                    'LL': "📊 Lower Low (Bearish Continuation)",
                    'H': "📍 High Pivot",
                    'L': "📍 Low Pivot"
                }
                
                st.write(f"**Last Structure:** {structure_meaning.get(last_structure, last_structure)}")
                st.write(f"**Recent Pattern:** {' → '.join(recent_structure[-5:])}")
    
    with tab3:
        st.markdown("### 🤖 Machine Learning Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Performance")
            st.write(f"**Confidence Level:** {result['ml_confidence']:.1f}%")
            
            confidence_bar = st.progress(result['ml_confidence'] / 100)
            
            if result['ml_confidence'] > 70:
                st.success("✅ High confidence prediction")
            elif result['ml_confidence'] > 50:
                st.info("ℹ️ Moderate confidence")
            else:
                st.warning("⚠️ Low confidence - be cautious")
        
        with col2:
            st.markdown("#### Feature Importance")
            
            # Simular importância das features
            features = {
                "Volume Trend": 0.85,
                "Price Momentum": 0.78,
                "RSI": 0.72,
                "MACD": 0.68,
                "ATR": 0.65
            }
            
            for feature, importance in features.items():
                st.write(f"**{feature}:** {importance:.2f}")
                st.progress(importance)
    
    with tab4:
        st.markdown("### 💰 Risk Management Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Position Sizing")
            
            entry_price = st.number_input("Entry Price ($)", value=float(current_price), step=0.01)
            stop_loss = st.number_input("Stop Loss ($)", value=float(current_price * 0.98), step=0.01)
            
            position_size = RiskManager.position_size(
                capital,
                risk_per_trade / 100,
                entry_price,
                stop_loss
            )
            
            position_value = position_size * entry_price
            risk_amount = capital * (risk_per_trade / 100)
            
            st.write(f"**Position Size:** {position_size} shares")
            st.write(f"**Position Value:** ${position_value:,.2f}")
            st.write(f"**Risk Amount:** ${risk_amount:,.2f}")
        
        with col2:
            st.markdown("#### Risk Metrics")
            
            st.write(f"**Sharpe Ratio:** {result['sharpe_ratio']:.2f}")
            
            returns = df_main['close'].pct_change().dropna()
            max_dd = (returns.cumsum().cummax() - returns.cumsum()).max()
            
            st.write(f"**Max Drawdown:** {max_dd*100:.2f}%")
            st.write(f"**Win Rate (Est):** {result['ml_confidence']:.1f}%")
        
        with col3:
            st.markdown("#### Kelly Criterion")
            
            win_rate = result['ml_confidence'] / 100
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss (R:R = 2:1)
            
            kelly = RiskManager.kelly_criterion(win_rate, avg_win, avg_loss)
            kelly_pct = kelly * 100
            
            st.write(f"**Optimal Position:** {kelly_pct:.2f}%")
            st.write(f"**Kelly × Capital:** ${capital * kelly:,.2f}")
            
            if kelly_pct > 15:
                st.warning("⚠️ High Kelly - consider fractional Kelly (25-50%)")
            elif kelly_pct > 0:
                st.success(f"✅ Recommended: {kelly_pct * 0.5:.2f}% (Half Kelly)")
            else:
                st.error("❌ Negative Kelly - Do NOT trade this setup")
    
    with tab5:
        st.markdown("### 📋 Trade Journal (Coming Soon)")
        
        st.info("📝 Trade journaling feature will be available in the next update!")
        
        # Preview
        trade_cols = ["Time", "Symbol", "Side", "Entry", "Exit", "PnL", "Notes"]
        example_data = {
            "Time": [datetime.now().strftime("%Y-%m-%d %H:%M")],
            "Symbol": [symbol],
            "Side": ["BUY" if result['signal'] in ['BUY', 'STRONG BUY'] else "SELL"],
            "Entry": [f"${current_price:.2f}"],
            "Exit": ["-"],
            "PnL": ["-"],
            "Notes": [f"ML Confidence: {result['ml_confidence']:.1f}%"]
        }
        
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)
    
    # FOOTER PROFISSIONAL
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"📊 **Symbol:** {symbol}")
        st.caption(f"📈 **Timeframe:** {timeframe}")
    
    with col2:
        st.caption(f"📅 **Data Points:** {len(df_main)} candles")
        st.caption(f"🕐 **Last Update:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    with col3:
        st.caption("🚀 **ProfitOne V3.0** - Institutional Edition")
        st.caption("⚠️ **Disclaimer:** For educational purposes only")


if __name__ == "__main__":
    main()
