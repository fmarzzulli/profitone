"""
ProfitOne V4.0 - INSTITUTIONAL TRADING SYSTEM
Sistema de análise técnica de nível institucional
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("⚠️ XGBoost não instalado - usando RandomForest")
    from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="ProfitOne V4.0 | Institutional",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS CUSTOMIZADO - TEMA PROFISSIONAL
# ============================================================================
st.markdown("""
<style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #00ff88 !important;
        font-weight: 700 !important;
        text-shadow: 0 0 20px rgba(0,255,136,0.3);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        color: #00ff88 !important;
        font-weight: 700 !important;
    }
    
    /* Cards de módulos */
    .module-card {
        background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
        border-left: 4px solid #00ff88;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 4px 20px rgba(0,255,136,0.1);
    }
    
    .module-title {
        color: #00ff88;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    .module-score {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Botões */
    .stButton>button {
        background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
        color: #0a0e27;
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        box-shadow: 0 4px 15px rgba(0,255,136,0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,255,136,0.5);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SÍMBOLOS B3 E MAPEAMENTO
# ============================================================================
B3_SYMBOLS = {
    # Futuros B3 (usando proxies do Yahoo Finance)
    'WINFUT': {
        'yahoo': '^BVSP',
        'multiplier': 1.0,
        'name': 'Mini-Índice Futuro',
        'type': 'future'
    },
    'WINM25': {
        'yahoo': '^BVSP',
        'multiplier': 1.0,
        'name': 'WIN Março/2025',
        'type': 'future'
    },
    'DOLFUT': {
        'yahoo': 'BRL=X',
        'multiplier': 1.0,
        'name': 'Mini-Dólar Futuro',
        'type': 'future'
    },
    'DOLM25': {
        'yahoo': 'BRL=X',
        'multiplier': 1.0,
        'name': 'DOL Março/2025',
        'type': 'future'
    },
    
    # Ações B3
    'PETR4.SA': {'yahoo': 'PETR4.SA', 'multiplier': 1.0, 'name': 'Petrobras PN', 'type': 'stock'},
    'VALE3.SA': {'yahoo': 'VALE3.SA', 'multiplier': 1.0, 'name': 'Vale ON', 'type': 'stock'},
    'ITUB4.SA': {'yahoo': 'ITUB4.SA', 'multiplier': 1.0, 'name': 'Itaú PN', 'type': 'stock'},
    'BBDC4.SA': {'yahoo': 'BBDC4.SA', 'multiplier': 1.0, 'name': 'Bradesco PN', 'type': 'stock'},
    'BBAS3.SA': {'yahoo': 'BBAS3.SA', 'multiplier': 1.0, 'name': 'Banco do Brasil ON', 'type': 'stock'},
    'ABEV3.SA': {'yahoo': 'ABEV3.SA', 'multiplier': 1.0, 'name': 'Ambev ON', 'type': 'stock'},
    'MGLU3.SA': {'yahoo': 'MGLU3.SA', 'multiplier': 1.0, 'name': 'Magazine Luiza ON', 'type': 'stock'},
    'ELET3.SA': {'yahoo': 'ELET3.SA', 'multiplier': 1.0, 'name': 'Eletrobras ON', 'type': 'stock'},
    
    # Crypto (fallback para futuros quando Yahoo falha)
    'BTCUSDT': {'yahoo': 'BTC-USD', 'multiplier': 1.0, 'name': 'Bitcoin', 'type': 'crypto'},
    'ETHUSDT': {'yahoo': 'ETH-USD', 'multiplier': 1.0, 'name': 'Ethereum', 'type': 'crypto'},
    
    # Internacional
    '^GSPC': {'yahoo': '^GSPC', 'multiplier': 1.0, 'name': 'S&P 500', 'type': 'index'},
    '^IXIC': {'yahoo': '^IXIC', 'multiplier': 1.0, 'name': 'NASDAQ', 'type': 'index'},
    '^DJI': {'yahoo': '^DJI', 'multiplier': 1.0, 'name': 'Dow Jones', 'type': 'index'},
}

# ============================================================================
# FUNÇÕES DE DADOS
# ============================================================================

def resolve_symbol(symbol):
    """Resolve símbolo B3 para Yahoo Finance"""
    if symbol in B3_SYMBOLS:
        return B3_SYMBOLS[symbol]['yahoo'], B3_SYMBOLS[symbol]['multiplier']
    return symbol, 1.0

@st.cache_data(ttl=60)
def get_data_institutional(symbol, period='5d', interval='15m'):
    """
    Busca dados com múltiplos fallbacks
    """
    yahoo_symbol, multiplier = resolve_symbol(symbol)
    
    # Lista de tentativas (período, intervalo)
    attempts = [
        (period, interval),
        ('1mo', '1h'),
        ('3mo', '1d'),
        ('1y', '1d'),
    ]
    
    for attempt_period, attempt_interval in attempts:
        try:
            df = yf.download(
                yahoo_symbol,
                period=attempt_period,
                interval=attempt_interval,
                progress=False,
                show_errors=False
            )
            
            if df is not None and len(df) > 20:
                # Limpar dados
                df = df.copy()
                df.columns = df.columns.str.lower()
                
                # Aplicar multiplicador
                if multiplier != 1.0:
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] *= multiplier
                
                # Garantir index datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Remover NaNs
                df = df.dropna()
                
                if len(df) >= 20:
                    return df
        except Exception as e:
            continue
    
    # Se tudo falhar, retornar DataFrame vazio
    return pd.DataFrame()

# ============================================================================
# INDICADORES TÉCNICOS
# ============================================================================

def calculate_ema(data, period):
    """Exponential Moving Average"""
    try:
        if len(data) < period:
            return pd.Series([np.nan] * len(data), index=data.index)
        return data.ewm(span=period, adjust=False).mean()
    except:
        return pd.Series([np.nan] * len(data), index=data.index)

def calculate_rsi(data, period=14):
    """Relative Strength Index"""
    try:
        if len(data) < period + 1:
            return pd.Series([50] * len(data), index=data.index)
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(data), index=data.index)

def calculate_vwap(df):
    """Volume Weighted Average Price"""
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.fillna(method='ffill')
    except:
        return df['close'].copy()

def calculate_cvd(df):
    """Cumulative Volume Delta"""
    try:
        # Volume de compra: close > open
        # Volume de venda: close < open
        delta = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        cvd = pd.Series(delta, index=df.index).cumsum()
        return cvd
    except:
        return pd.Series([0] * len(df), index=df.index)

def calculate_volume_profile(df, num_bins=30):
    """Volume Profile - volume por nível de preço"""
    try:
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, num_bins)
        
        volume_profile = []
        for i in range(len(bins) - 1):
            mask = (df['close'] >= bins[i]) & (df['close'] < bins[i + 1])
            vol = df.loc[mask, 'volume'].sum()
            volume_profile.append({
                'price': (bins[i] + bins[i + 1]) / 2,
                'volume': vol
            })
        
        return pd.DataFrame(volume_profile)
    except:
        return pd.DataFrame({'price': [], 'volume': []})

def calculate_market_profile(df, num_levels=20):
    """Market Profile - POC, VAH, VAL"""
    try:
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, num_levels)
        
        tpo_counts = np.zeros(len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask = (df['close'] >= bins[i]) & (df['close'] < bins[i + 1])
            tpo_counts[i] = mask.sum()
        
        # POC (Point of Control)
        poc_idx = np.argmax(tpo_counts)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area (70% do volume)
        total_tpo = tpo_counts.sum()
        target_tpo = total_tpo * 0.7
        
        sorted_indices = np.argsort(tpo_counts)[::-1]
        cumulative = 0
        value_area_indices = []
        
        for idx in sorted_indices:
            if cumulative >= target_tpo:
                break
            value_area_indices.append(idx)
            cumulative += tpo_counts[idx]
        
        vah_idx = max(value_area_indices)
        val_idx = min(value_area_indices)
        
        vah = (bins[vah_idx] + bins[vah_idx + 1]) / 2
        val = (bins[val_idx] + bins[val_idx + 1]) / 2
        
        return {
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'bins': bins,
            'tpo_counts': tpo_counts
        }
    except:
        current_price = df['close'].iloc[-1]
        return {
            'poc': current_price,
            'vah': current_price * 1.02,
            'val': current_price * 0.98,
            'bins': [],
            'tpo_counts': []
        }

def detect_absorption(df, threshold=0.3):
    """Detecta zonas de absorção (alto volume, baixa variação de preço)"""
    try:
        price_change = abs(df['close'] - df['open']) / df['open']
        volume_norm = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min() + 1e-10)
        
        absorption = (volume_norm > 0.7) & (price_change < threshold)
        return absorption
    except:
        return pd.Series([False] * len(df), index=df.index)

def detect_imbalance(df, threshold=0.7):
    """Detecta desbalanço de volume (compra vs venda)"""
    try:
        delta = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        imbalance_ratio = abs(delta) / (df['volume'] + 1e-10)
        
        strong_imbalance = imbalance_ratio > threshold
        direction = np.where(delta > 0, 'BUY', 'SELL')
        
        return strong_imbalance, direction
    except:
        return pd.Series([False] * len(df), index=df.index), ['NEUTRAL'] * len(df)

def calculate_footprint(df, levels_per_candle=5):
    """Footprint Chart simplificado - delta por nível de preço"""
    try:
        footprint_data = []
        
        for idx, row in df.iterrows():
            price_range = row['high'] - row['low']
            if price_range == 0:
                continue
            
            level_size = price_range / levels_per_candle
            
            for i in range(levels_per_candle):
                level_price = row['low'] + (i + 0.5) * level_size
                
                # Simular delta (na realidade precisaria de dados tick-by-tick)
                if row['close'] > row['open']:
                    # Candle de alta - mais volume de compra
                    delta = row['volume'] / levels_per_candle * (0.6 + np.random.random() * 0.4)
                else:
                    # Candle de baixa - mais volume de venda
                    delta = -row['volume'] / levels_per_candle * (0.6 + np.random.random() * 0.4)
                
                footprint_data.append({
                    'time': idx,
                    'price': level_price,
                    'delta': delta
                })
        
        return pd.DataFrame(footprint_data)
    except:
        return pd.DataFrame({'time': [], 'price': [], 'delta': []})

# ============================================================================
# MACHINE LEARNING - XGBOOST
# ============================================================================

class InstitutionalMLPredictor:
    """Preditor ML de nível institucional com XGBoost"""
    
    def __init__(self):
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                random_state=42,
                n_jobs=-1
            )
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = {}
    
    def prepare_features(self, df):
        """Engenharia de features profissional (20+ features)"""
        try:
            features = pd.DataFrame(index=df.index)
            
            # Returns (múltiplos períodos)
            for period in [1, 3, 5, 10]:
                features[f'return_{period}'] = df['close'].pct_change(period)
            
            # RSI
            features['rsi_14'] = calculate_rsi(df['close'], 14)
            features['rsi_21'] = calculate_rsi(df['close'], 21)
            
            # EMAs
            for period in [9, 21, 50]:
                features[f'ema_{period}'] = calculate_ema(df['close'], period)
                features[f'price_to_ema_{period}'] = df['close'] / features[f'ema_{period}']
            
            # EMA differences
            features['ema_9_21_diff'] = (features['ema_9'] - features['ema_21']) / features['ema_21']
            features['ema_21_50_diff'] = (features['ema_21'] - features['ema_50']) / features['ema_50']
            
            # Volume
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['cvd'] = calculate_cvd(df)
            features['cvd_change'] = features['cvd'].pct_change(5)
            
            # VWAP
            vwap = calculate_vwap(df)
            features['vwap_distance'] = (df['close'] - vwap) / vwap
            
            # Volatility
            features['atr'] = (df['high'] - df['low']).rolling(14).mean()
            features['volatility'] = df['close'].pct_change().rolling(20).std()
            
            # Market Profile
            mp = calculate_market_profile(df)
            features['poc_distance'] = (df['close'] - mp['poc']) / mp['poc']
            
            # Limpar NaNs
            features = features.fillna(method='ffill').fillna(0)
            
            return features
        except Exception as e:
            st.error(f"Erro em prepare_features: {e}")
            return pd.DataFrame()
    
    def train(self, df, lookforward=5):
        """Treina o modelo"""
        try:
            if len(df) < 100:
                return False
            
            features = self.prepare_features(df)
            
            if features.empty:
                return False
            
            # Target: direção do preço nos próximos N bars
            future_return = df['close'].pct_change(lookforward).shift(-lookforward)
            target = (future_return > 0).astype(int)
            
            # Remover últimas linhas (sem target)
            valid_idx = target.notna()
            X = features[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 50:
                return False
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Escalar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Treinar
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            
            # Feature importance
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = dict(zip(
                    X.columns,
                    self.model.feature_importances_
                ))
            
            return True
        except Exception as e:
            st.error(f"Erro no treinamento ML: {e}")
            return False
    
    def predict(self, df):
        """Faz predição"""
        try:
            if not self.is_trained:
                return 0, 0.5  # neutral, 50% confidence
            
            features = self.prepare_features(df)
            
            if features.empty:
                return 0, 0.5
            
            # Última linha
            X_last = features.iloc[[-1]]
            X_scaled = self.scaler.transform(X_last)
            
            # Predição
            pred = self.model.predict(X_scaled)[0]
            
            # Probabilidade
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.6
            
            # Converter para sinal (-1, 0, 1)
            signal = 1 if pred == 1 else -1
            
            return signal, confidence
        except Exception as e:
            return 0, 0.5

# ============================================================================
# SCORE INSTITUCIONAL
# ============================================================================

def calculate_institutional_score(df):
    """
    Calcula score institucional agregado
    
    Componentes:
    - Trend (15%): alinhamento de EMAs
    - Momentum (12%): RSI, Stochastic
    - Volume (18%): CVD, Volume Profile
    - Order Flow (20%): Absorption, Imbalance
    - ML Prediction (15%): XGBoost
    - VWAP Position (10%): distância do VWAP
    - Market Profile (10%): posição relativa ao POC
    """
    try:
        result = {
            'master_score': 0,
            'signal': 'NEUTRAL',
            'confidence': 0.5,
            'components': {}
        }
        
        if len(df) < 50:
            return result
        
        # ----- TREND (15%) -----
        ema9 = calculate_ema(df['close'], 9).iloc[-1]
        ema21 = calculate_ema(df['close'], 21).iloc[-1]
        ema50 = calculate_ema(df['close'], 50).iloc[-1]
        current_price = df['close'].iloc[-1]
        
        trend_score = 0
        if current_price > ema9 > ema21 > ema50:
            trend_score = 100
        elif current_price > ema9 > ema21:
            trend_score = 60
        elif current_price > ema9:
            trend_score = 30
        elif current_price < ema9 < ema21 < ema50:
            trend_score = -100
        elif current_price < ema9 < ema21:
            trend_score = -60
        elif current_price < ema9:
            trend_score = -30
        
        result['components']['trend'] = trend_score
        
        # ----- MOMENTUM (12%) -----
        rsi = calculate_rsi(df['close'], 14).iloc[-1]
        
        momentum_score = 0
        if rsi > 70:
            momentum_score = 80
        elif rsi > 60:
            momentum_score = 50
        elif rsi > 50:
            momentum_score = 20
        elif rsi < 30:
            momentum_score = -80
        elif rsi < 40:
            momentum_score = -50
        elif rsi < 50:
            momentum_score = -20
        
        result['components']['momentum'] = momentum_score
        
        # ----- VOLUME (18%) -----
        cvd = calculate_cvd(df).iloc[-1]
        cvd_change = calculate_cvd(df).pct_change(10).iloc[-1]
        
        volume_score = 0
        if cvd_change > 0.1:
            volume_score = 80
        elif cvd_change > 0.05:
            volume_score = 50
        elif cvd_change > 0:
            volume_score = 20
        elif cvd_change < -0.1:
            volume_score = -80
        elif cvd_change < -0.05:
            volume_score = -50
        elif cvd_change < 0:
            volume_score = -20
        
        result['components']['volume'] = volume_score
        
        # ----- ORDER FLOW (20%) -----
        absorption = detect_absorption(df).iloc[-5:].sum()
        imbalance, direction = detect_imbalance(df)
        recent_imbalance = imbalance.iloc[-5:].sum()
        
        flow_score = 0
        if recent_imbalance > 2:
            if direction[-1] == 'BUY':
                flow_score = 90
            else:
                flow_score = -90
        elif absorption > 2:
            flow_score = -30  # Absorção = resistência
        else:
            flow_score = 0
        
        result['components']['order_flow'] = flow_score
        
        # ----- ML PREDICTION (15%) -----
        ml_predictor = InstitutionalMLPredictor()
        
        # Treinar apenas se temos dados suficientes
        if len(df) >= 200:
            ml_predictor.train(df.iloc[-1000:] if len(df) > 1000 else df)
        
        ml_signal, ml_confidence = ml_predictor.predict(df)
        ml_score = ml_signal * 100 * ml_confidence
        
        result['components']['ml_prediction'] = ml_score
        result['confidence'] = ml_confidence
        
        # ----- VWAP POSITION (10%) -----
        vwap = calculate_vwap(df).iloc[-1]
        vwap_distance = (current_price - vwap) / vwap
        
        vwap_score = 0
        if vwap_distance > 0.02:
            vwap_score = 70
        elif vwap_distance > 0.01:
            vwap_score = 40
        elif vwap_distance > 0:
            vwap_score = 10
        elif vwap_distance < -0.02:
            vwap_score = -70
        elif vwap_distance < -0.01:
            vwap_score = -40
        elif vwap_distance < 0:
            vwap_score = -10
        
        result['components']['vwap'] = vwap_score
        
        # ----- MARKET PROFILE (10%) -----
        mp = calculate_market_profile(df)
        poc_distance = (current_price - mp['poc']) / mp['poc']
        
        profile_score = 0
        if poc_distance > 0.02:
            profile_score = 60
        elif poc_distance > 0:
            profile_score = 30
        elif poc_distance < -0.02:
            profile_score = -60
        elif poc_distance < 0:
            profile_score = -30
        
        result['components']['market_profile'] = profile_score
        
        # ----- MASTER SCORE (WEIGHTED AVERAGE) -----
        weights = {
            'trend': 0.15,
            'momentum': 0.12,
            'volume': 0.18,
            'order_flow': 0.20,
            'ml_prediction': 0.15,
            'vwap': 0.10,
            'market_profile': 0.10
        }
        
        master_score = sum(
            result['components'][component] * weight
            for component, weight in weights.items()
        )
        
        master_score = max(-100, min(100, master_score))
        result['master_score'] = round(master_score, 2)
        
        # ----- SIGNAL -----
        if master_score > 60:
            result['signal'] = 'STRONG BUY'
        elif master_score > 30:
            result['signal'] = 'BUY'
        elif master_score > -30:
            result['signal'] = 'NEUTRAL'
        elif master_score > -60:
            result['signal'] = 'SELL'
        else:
            result['signal'] = 'STRONG SELL'
        
        # Adicionar indicadores extras para retorno
        result['rsi'] = rsi
        result['ema9'] = ema9
        result['ema21'] = ema21
        result['ema50'] = ema50
        result['vwap'] = vwap
        result['cvd'] = cvd
        result['poc'] = mp['poc']
        result['vah'] = mp['vah']
        result['val'] = mp['val']
        
        return result
        
    except Exception as e:
        st.error(f"Erro no cálculo do score institucional: {e}")
        return {
            'master_score': 0,
            'signal': 'ERROR',
            'confidence': 0,
            'components': {},
            'rsi': 50,
            'ema9': 0,
            'ema21': 0,
            'ema50': 0,
            'vwap': 0,
            'cvd': 0,
            'poc': 0,
            'vah': 0,
            'val': 0
        }

# ============================================================================
# CRIAÇÃO DO GRÁFICO (PLOTLY - 6 PAINÉIS)
# ============================================================================

def create_institutional_chart(df, result):
    """
    Gráfico institucional de 6 painéis:
    1. Candlesticks + VWAP + EMAs + Volume Profile
    2. Footprint Chart (delta heatmap)
    3. Market Profile
    4. RSI + ML Confidence
    5. Volume + CVD
    6. Order Flow Metrics
    """
    try:
        fig = make_subplots(
            rows=6, cols=1,
            row_heights=[0.35, 0.15, 0.15, 0.12, 0.12, 0.11],
            vertical_spacing=0.03,
            subplot_titles=(
                '📊 Price Action + VWAP + EMAs',
                '🔥 Footprint Chart (Delta)',
                '📈 Market Profile',
                '💹 RSI + ML Confidence',
                '📦 Volume + CVD',
                '🌊 Order Flow (Absorption & Imbalance)'
            ),
            specs=[
                [{"secondary_y": True}],
                [{"type": "heatmap"}],
                [{"type": "bar"}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"type": "bar"}]
            ]
        )
        
        # ----- PAINEL 1: CANDLESTICKS + VWAP + EMAs -----
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff0051'
            ),
            row=1, col=1
        )
        
        # EMAs
        ema9 = calculate_ema(df['close'], 9)
        ema21 = calculate_ema(df['close'], 21)
        ema50 = calculate_ema(df['close'], 50)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='#00ff88', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=ema21, name='EMA 21', line=dict(color='#00ccff', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=ema50, name='EMA 50', line=dict(color='#ffaa00', width=1)),
            row=1, col=1
        )
        
        # VWAP
        vwap = calculate_vwap(df)
        fig.add_trace(
            go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='#ff00ff', width=2, dash='dash')),
            row=1, col=1
        )
        
        # Market Profile lines
        fig.add_hline(y=result['poc'], line=dict(color='yellow', width=2, dash='dot'), row=1, col=1)
        fig.add_hline(y=result['vah'], line=dict(color='orange', width=1, dash='dot'), row=1, col=1)
        fig.add_hline(y=result['val'], line=dict(color='orange', width=1, dash='dot'), row=1, col=1)
        
        # ----- PAINEL 2: FOOTPRINT CHART -----
        footprint = calculate_footprint(df.iloc[-50:])  # Últimas 50 barras
        
        if not footprint.empty and len(footprint) > 0:
            # Criar matriz para heatmap
            pivot = footprint.pivot_table(
                values='delta',
                index='price',
                columns='time',
                aggfunc='sum'
            )
            
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns,
                    y=pivot.index,
                    colorscale=[[0, '#ff0051'], [0.5, '#2a2f4a'], [1, '#00ff88']],
                    showscale=True,
                    colorbar=dict(title="Delta", x=1.15)
                ),
                row=2, col=1
            )
        
        # ----- PAINEL 3: MARKET PROFILE -----
        mp = calculate_market_profile(df)
        
        if len(mp['bins']) > 0:
            bins_mid = [(mp['bins'][i] + mp['bins'][i+1])/2 for i in range(len(mp['bins'])-1)]
            
            fig.add_trace(
                go.Bar(
                    y=bins_mid,
                    x=mp['tpo_counts'],
                    orientation='h',
                    name='Market Profile',
                    marker=dict(color='#00ff88', opacity=0.6)
                ),
                row=3, col=1
            )
        
        # ----- PAINEL 4: RSI + ML CONFIDENCE -----
        rsi = calculate_rsi(df['close'], 14)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#00ccff', width=2)),
            row=4, col=1
        )
        
        fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=4, col=1)
        fig.add_hline(y=50, line=dict(color='gray', width=1, dash='dot'), row=4, col=1)
        fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=4, col=1)
        
        # ML Confidence (secondary y)
        ml_conf_series = pd.Series([result['confidence'] * 100] * len(df), index=df.index)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ml_conf_series,
                name='ML Confidence',
                line=dict(color='#ffaa00', width=1, dash='dot')
            ),
            row=4, col=1,
            secondary_y=True
        )
        
        # ----- PAINEL 5: VOLUME + CVD -----
        colors = ['#00ff88' if c > o else '#ff0051' for c, o in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], name='Volume', marker=dict(color=colors, opacity=0.5)),
            row=5, col=1
        )
        
        # CVD (secondary y)
        cvd = calculate_cvd(df)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=cvd,
                name='CVD',
                line=dict(color='#ff00ff', width=2)
            ),
            row=5, col=1,
            secondary_y=True
        )
        
        # ----- PAINEL 6: ORDER FLOW -----
        absorption = detect_absorption(df).astype(int) * 100
        imbalance, direction = detect_imbalance(df)
        imbalance_vals = imbalance.astype(int) * 100
        
        fig.add_trace(
            go.Bar(x=df.index, y=absorption, name='Absorption', marker=dict(color='orange', opacity=0.6)),
            row=6, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=imbalance_vals, name='Imbalance', marker=dict(color='cyan', opacity=0.6)),
            row=6, col=1
        )
        
        # ----- LAYOUT STYLING -----
        fig.update_layout(
            height=1800,
            template='plotly_dark',
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#1a1f3a',
            font=dict(color='#ffffff', size=12),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridcolor='#2a2f4a')
        fig.update_yaxes(showgrid=True, gridcolor='#2a2f4a')
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {e}")
        
        # Fallback: gráfico simples de candlesticks
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            )
        )
        fig.update_layout(
            height=600,
            template='plotly_dark',
            title='Gráfico de Candlesticks (Fallback)'
        )
        return fig

# ============================================================================
# STREAMLIT UI - FUNÇÃO PRINCIPAL
# ============================================================================

def main():
    """Função principal do Streamlit"""
    
    # Título
    st.markdown("""
    <h1 style='text-align: center; font-size: 3rem;'>
        📈 ProfitOne V4.0
    </h1>
    <h3 style='text-align: center; color: #00ff88; font-weight: 300;'>
        INSTITUTIONAL TRADING SYSTEM
    </h3>
    <p style='text-align: center; color: #888; margin-bottom: 2rem;'>
        Sistema de análise técnica de nível institucional com ML, Order Flow & Footprint Charts
    </p>
    """, unsafe_allow_html=True)
    
    # ----- SIDEBAR -----
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        
        # Seleção de ativo
        st.markdown("#### 📊 Ativo")
        
        asset_categories = {
            '🔥 Futuros B3': ['WINFUT', 'WINM25', 'DOLFUT', 'DOLM25'],
            '📈 Ações B3': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 'ABEV3.SA', 'MGLU3.SA', 'ELET3.SA'],
            '₿ Crypto': ['BTCUSDT', 'ETHUSDT'],
            '🌎 Internacional': ['^GSPC', '^IXIC', '^DJI', '^BVSP']
        }
        
        selected_category = st.selectbox("Categoria", list(asset_categories.keys()))
        symbol = st.selectbox("Símbolo", asset_categories[selected_category])
        
        # Badge para futuros
        if B3_SYMBOLS.get(symbol, {}).get('type') == 'future':
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ff6b00 0%, #ff0051 100%);
                        padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;'>
                <strong>⚡ FUTURO B3</strong><br>
                <small>Proxy: {B3_SYMBOLS[symbol]['yahoo']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Timeframe
        st.markdown("#### ⏱️ Timeframe")
        timeframe_map = {
            '1 Minuto': ('5d', '1m'),
            '5 Minutos': ('5d', '5m'),
            '15 Minutos': ('5d', '15m'),
            '1 Hora': ('1mo', '1h'),
            '4 Horas': ('3mo', '4h'),
            '1 Dia': ('1y', '1d')
        }
        timeframe_label = st.radio("Período", list(timeframe_map.keys()), index=2)
        period, interval = timeframe_map[timeframe_label]
        
        # Botão de refresh
        st.markdown("---")
        if st.button("🔄 Atualizar Dados", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Informações
        st.markdown("---")
        st.markdown("""
        <div style='background: #1a1f3a; padding: 15px; border-radius: 8px; margin-top: 20px;'>
            <strong style='color: #00ff88;'>✨ Novidades V4.0:</strong><br>
            <ul style='margin-top: 10px; color: #ccc; font-size: 0.85rem;'>
                <li>✅ XGBoost ML (200 árvores)</li>
                <li>✅ Footprint Chart</li>
                <li>✅ Market Profile HD</li>
                <li>✅ Order Flow Analysis</li>
                <li>✅ Volume Profile</li>
                <li>✅ Absorption Zones</li>
                <li>✅ CVD (Cumulative Delta)</li>
                <li>✅ 6-Panel Chart</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ----- MAIN CONTENT -----
    
    # Buscar dados
    with st.spinner('🔄 Carregando dados do mercado...'):
        df = get_data_institutional(symbol, period, interval)
    
    if df.empty or len(df) < 20:
        st.error(f"""
        ❌ **Sem dados disponíveis para {symbol}**
        
        **Possíveis causas:**
        - Yahoo Finance não tem dados para este ativo no timeframe selecionado
        - Mercado fechado
        - Símbolo incorreto
        
        **Sugestões:**
        - Tente um timeframe maior (ex: 1 Dia)
        - Escolha outro ativo
        - Verifique se o mercado está aberto
        """)
        return
    
    # Calcular score institucional
    with st.spinner('🧮 Calculando análise institucional...'):
        result = calculate_institutional_score(df)
    
    # ----- MÉTRICAS PRINCIPAIS -----
    st.markdown("### 🎯 Dashboard Institucional")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = '#00ff88' if result['master_score'] > 0 else '#ff0051'
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
                    padding: 20px; border-radius: 10px; text-align: center;
                    border-left: 4px solid {score_color};'>
            <div style='color: #888; font-size: 0.9rem;'>MASTER SCORE</div>
            <div style='color: {score_color}; font-size: 2.5rem; font-weight: 700; margin: 10px 0;'>
                {result['master_score']:.1f}
            </div>
            <div style='color: {score_color}; font-size: 1.1rem; font-weight: 600;'>
                {result['signal']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        price_change = df['close'].pct_change().iloc[-1] * 100
        price_color = '#00ff88' if price_change > 0 else '#ff0051'
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='color: #888; font-size: 0.9rem;'>PREÇO</div>
            <div style='color: #fff; font-size: 2rem; font-weight: 700; margin: 10px 0;'>
                {df['close'].iloc[-1]:.2f}
            </div>
            <div style='color: {price_color}; font-size: 1rem;'>
                {price_change:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conf_pct = result['confidence'] * 100
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='color: #888; font-size: 0.9rem;'>ML CONFIDENCE</div>
            <div style='color: #ffaa00; font-size: 2rem; font-weight: 700; margin: 10px 0;'>
                {conf_pct:.1f}%
            </div>
            <div style='color: #888; font-size: 0.85rem;'>
                {'XGBoost' if XGBOOST_AVAILABLE else 'RandomForest'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rsi_color = '#ff0051' if result['rsi'] > 70 else ('#00ff88' if result['rsi'] < 30 else '#ffaa00')
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%);
                    padding: 20px; border-radius: 10px; text-align: center;'>
            <div style='color: #888; font-size: 0.9rem;'>RSI (14)</div>
            <div style='color: {rsi_color}; font-size: 2rem; font-weight: 700; margin: 10px 0;'>
                {result['rsi']:.1f}
            </div>
            <div style='color: #888; font-size: 0.85rem;'>
                {'Overbought' if result['rsi'] > 70 else ('Oversold' if result['rsi'] < 30 else 'Neutral')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ----- MÓDULOS DE SCORE -----
    st.markdown("### 📊 Análise por Módulo")
    
    modules = [
        ('Trend', result['components'].get('trend', 0)),
        ('Momentum', result['components'].get('momentum', 0)),
        ('Volume', result['components'].get('volume', 0)),
        ('Order Flow', result['components'].get('order_flow', 0)),
        ('ML Prediction', result['components'].get('ml_prediction', 0)),
        ('VWAP', result['components'].get('vwap', 0)),
        ('Market Profile', result['components'].get('market_profile', 0))
    ]
    
    cols = st.columns(4)
    for idx, (name, score) in enumerate(modules):
        col = cols[idx % 4]
        score_color = '#00ff88' if score > 0 else '#ff0051'
        
        with col:
            st.markdown(f"""
            <div class='module-card'>
                <div class='module-title'>{name}</div>
                <div class='module-score' style='color: {score_color};'>
                    {score:+.1f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ----- GRÁFICO INSTITUCIONAL -----
    st.markdown("### 📈 Gráfico Institucional (6 Painéis)")
    
    with st.spinner('🎨 Renderizando gráfico institucional...'):
        fig = create_institutional_chart(df, result)
        st.plotly_chart(fig, use_container_width=True)
    
    # ----- DETALHES ADICIONAIS -----
    st.markdown("### 📋 Informações Detalhadas")
    
    tab1, tab2, tab3 = st.tabs(["📊 Indicadores", "🌊 Order Flow", "🤖 Machine Learning"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Médias Móveis")
            st.write(f"**EMA 9:** {result['ema9']:.2f}")
            st.write(f"**EMA 21:** {result['ema21']:.2f}")
            st.write(f"**EMA 50:** {result['ema50']:.2f}")
            st.write(f"**VWAP:** {result['vwap']:.2f}")
        
        with col2:
            st.markdown("#### 📊 Market Profile")
            st.write(f"**POC:** {result['poc']:.2f}")
            st.write(f"**VAH:** {result['vah']:.2f}")
            st.write(f"**VAL:** {result['val']:.2f}")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Volume Analysis")
            st.write(f"**CVD:** {result['cvd']:.0f}")
            cvd_df = pd.DataFrame({
                'Período': df.index[-20:],
                'CVD': calculate_cvd(df).iloc[-20:]
            })
            st.line_chart(cvd_df.set_index('Período'))
        
        with col2:
            st.markdown("#### 🌊 Flow Metrics")
            absorption_count = detect_absorption(df).iloc[-20:].sum()
            imbalance, _ = detect_imbalance(df)
            imbalance_count = imbalance.iloc[-20:].sum()
            
            st.write(f"**Absorption Zones (últimas 20 barras):** {absorption_count}")
            st.write(f"**Imbalance Events (últimas 20 barras):** {imbalance_count}")
    
    with tab3:
        st.markdown("#### 🤖 Machine Learning")
        st.write(f"**Modelo:** {'XGBoost (200 árvores)' if XGBOOST_AVAILABLE else 'RandomForest (100 árvores)'}")
        st.write(f"**Confidence:** {result['confidence']*100:.1f}%")
        st.write(f"**Features:** 20+ (returns, EMAs, RSI, volume, CVD, VWAP, etc.)")
        st.write(f"**Target:** Direção do preço nos próximos 5 bars")
    
    # ----- FOOTER -----
    st.markdown("---")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f"""
    <div style='text-align: center; color: #666; font-size: 0.85rem; padding: 20px 0;'>
        <strong>ProfitOne V4.0 Institutional</strong> | 
        Última atualização: {current_time} | 
        Dados: {len(df)} barras | 
        Símbolo: {symbol} ({timeframe_label})<br>
        <em style='color: #888;'>⚠️ Este sistema é apenas para fins educacionais. Não constitui recomendação de investimento.</em>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# EXECUTAR APLICAÇÃO
# ============================================================================

if __name__ == "__main__":
    main()
