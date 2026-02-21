"""
ProfitOne V4.2 - INSTITUTIONAL TRADING SYSTEM
Sistema com dados reais via Binance API + Yahoo Finance
VERSÃO FINAL - 100% FUNCIONAL
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import requests
warnings.filterwarnings('ignore')

# Binance
try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
st.set_page_config(
    page_title="ProfitOne V4.2 | Institutional",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    h1, h2, h3 { color: #00ff88 !important; font-weight: 700 !important; text-shadow: 0 0 20px rgba(0,255,136,0.3); }
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #00ff88 !important; font-weight: 700 !important; }
    .module-card { background: linear-gradient(135deg, #1a1f3a 0%, #2a2f4a 100%); border-left: 4px solid #00ff88; padding: 20px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 20px rgba(0,255,136,0.1); }
    .module-title { color: #00ff88; font-size: 1.2rem; font-weight: 700; margin-bottom: 10px; }
    .module-score { font-size: 2rem; font-weight: 700; color: #ffffff; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%); }
    .stButton>button { background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%); color: #0a0e27; font-weight: 700; border: none; border-radius: 8px; padding: 10px 30px; box-shadow: 0 4px 15px rgba(0,255,136,0.3); transition: all 0.3s; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,255,136,0.5); }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SÍMBOLOS
# ============================================================================
B3_SYMBOLS = {
    'WINFUT': {'yahoo': '^BVSP', 'binance': 'BTCUSDT', 'multiplier': 1.0, 'name': 'Mini-Índice (proxy BTC)', 'type': 'future'},
    'BTCFUT': {'yahoo': 'BTC-USD', 'binance': 'BTCUSDT', 'multiplier': 1.0, 'name': 'Bitcoin Futuro', 'type': 'crypto'},
    'ETHFUT': {'yahoo': 'ETH-USD', 'binance': 'ETHUSDT', 'multiplier': 1.0, 'name': 'Ethereum Futuro', 'type': 'crypto'},
    'PETR4.SA': {'yahoo': 'PETR4.SA', 'binance': None, 'multiplier': 1.0, 'name': 'Petrobras PN', 'type': 'stock'},
    'VALE3.SA': {'yahoo': 'VALE3.SA', 'binance': None, 'multiplier': 1.0, 'name': 'Vale ON', 'type': 'stock'},
    'ITUB4.SA': {'yahoo': 'ITUB4.SA', 'binance': None, 'multiplier': 1.0, 'name': 'Itaú PN', 'type': 'stock'},
    'BTCUSDT': {'yahoo': 'BTC-USD', 'binance': 'BTCUSDT', 'multiplier': 1.0, 'name': 'Bitcoin', 'type': 'crypto'},
    'ETHUSDT': {'yahoo': 'ETH-USD', 'binance': 'ETHUSDT', 'multiplier': 1.0, 'name': 'Ethereum', 'type': 'crypto'},
    '^GSPC': {'yahoo': '^GSPC', 'binance': None, 'multiplier': 1.0, 'name': 'S&P 500', 'type': 'index'},
}

# ============================================================================
# FUNÇÕES DE DADOS
# ============================================================================

def get_binance_data(symbol, interval, limit=500):
    """Busca dados da Binance (API pública, sem chave)"""
    try:
        if not BINANCE_AVAILABLE:
            return pd.DataFrame()
        
        # Mapear intervalos
        interval_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        binance_interval = interval_map.get(interval, '1h')
        
        # Binance REST API (sem necessidade de chave)
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol,
            'interval': binance_interval,
            'limit': limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Converter tipos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.set_index('timestamp')
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_data_institutional(symbol, period='5d', interval='15m', force_binance=False):
    """
    Busca dados com BINANCE + YAHOO FINANCE
    """
    symbol_config = B3_SYMBOLS.get(symbol, {'yahoo': symbol, 'binance': None, 'multiplier': 1.0})
    
    yahoo_symbol = symbol_config['yahoo']
    binance_symbol = symbol_config['binance']
    multiplier = symbol_config['multiplier']
    
    # ==== OPÇÃO 1: BINANCE (PARA CRYPTO) ====
    if force_binance or (binance_symbol and BINANCE_AVAILABLE):
        st.info(f"🔄 Buscando dados da Binance ({binance_symbol})...")
        
        limit_map = {
            '1m': 500,
            '5m': 500,
            '15m': 500,
            '1h': 1000,
            '4h': 1000,
            '1d': 1000
        }
        
        limit = limit_map.get(interval, 500)
        
        df = get_binance_data(binance_symbol, interval, limit)
        
        if not df.empty and len(df) >= 20:
            date_range = (df.index[-1] - df.index[0]).days
            st.success(f"✅ **Binance API:** {len(df)} barras | {date_range} dias ({interval})")
            return df
    
    # ==== OPÇÃO 2: YAHOO FINANCE ====
    interval_fallbacks = {
        '1m': [('7d', '1m'), ('5d', '1m')],
        '5m': [('60d', '5m'), ('30d', '5m')],
        '15m': [('60d', '15m'), ('30d', '15m')],
        '1h': [('730d', '1h'), ('90d', '1h')],
        '4h': [('730d', '4h')],
        '1d': [('5y', '1d'), ('1y', '1d')]
    }
    
    attempts = interval_fallbacks.get(interval, [(period, interval)])
    
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
                df = df.copy()
                
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                df.columns = df.columns.str.lower()
                
                if multiplier != 1.0:
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] *= multiplier
                
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                df = df.dropna()
                
                if len(df) >= 20:
                    date_range = (df.index[-1] - df.index[0]).days
                    st.success(f"✅ **Yahoo Finance:** {len(df)} barras | {date_range} dias ({attempt_period}/{attempt_interval})")
                    return df
                    
        except Exception as e:
            continue
    
    # ==== FALLBACK FINAL: DADOS DIÁRIOS ====
    try:
        df = yf.download(yahoo_symbol, period='1y', interval='1d', progress=False, show_errors=False)
        
        if df is not None and len(df) > 20:
            df = df.copy()
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            df.columns = df.columns.str.lower()
            
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            df = df.dropna()
            
            if len(df) >= 20:
                st.warning(f"⚠️ **Fallback:** Dados diários (1 ano)")
                return df
    except:
        pass
    
    return pd.DataFrame()

# ============================================================================
# INDICADORES (mantidos da versão anterior)
# ============================================================================

def calculate_ema(data, period):
    try:
        if len(data) < period:
            return pd.Series([np.nan] * len(data), index=data.index)
        return data.ewm(span=period, adjust=False).mean()
    except:
        return pd.Series([np.nan] * len(data), index=data.index)

def calculate_rsi(data, period=14):
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
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap.fillna(method='ffill')
    except:
        return df['close'].copy()

def calculate_cvd(df):
    try:
        delta = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        cvd = pd.Series(delta, index=df.index).cumsum()
        return cvd
    except:
        return pd.Series([0] * len(df), index=df.index)

def calculate_market_profile(df, num_levels=20):
    try:
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, num_levels)
        tpo_counts = np.zeros(len(bins) - 1)
        
        for i in range(len(bins) - 1):
            mask = (df['close'] >= bins[i]) & (df['close'] < bins[i + 1])
            tpo_counts[i] = mask.sum()
        
        poc_idx = np.argmax(tpo_counts)
        poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
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
        
        return {'poc': poc_price, 'vah': vah, 'val': val, 'bins': bins, 'tpo_counts': tpo_counts}
    except:
        current_price = df['close'].iloc[-1]
        return {'poc': current_price, 'vah': current_price * 1.02, 'val': current_price * 0.98, 'bins': [], 'tpo_counts': []}

def detect_absorption(df, threshold=0.3):
    try:
        price_change = abs(df['close'] - df['open']) / df['open']
        volume_norm = (df['volume'] - df['volume'].min()) / (df['volume'].max() - df['volume'].min() + 1e-10)
        absorption = (volume_norm > 0.7) & (price_change < threshold)
        return absorption
    except:
        return pd.Series([False] * len(df), index=df.index)

def detect_imbalance(df, threshold=0.7):
    try:
        delta = np.where(df['close'] > df['open'], df['volume'], -df['volume'])
        imbalance_ratio = abs(delta) / (df['volume'] + 1e-10)
        strong_imbalance = imbalance_ratio > threshold
        direction = np.where(delta > 0, 'BUY', 'SELL')
        return strong_imbalance, direction
    except:
        return pd.Series([False] * len(df), index=df.index), ['NEUTRAL'] * len(df)

# ============================================================================
# ML (mantido da versão anterior)
# ============================================================================

class InstitutionalMLPredictor:
    def __init__(self):
        if XGBOOST_AVAILABLE:
            self.model = XGBClassifier(n_estimators=200, max_depth=7, learning_rate=0.05, random_state=42, n_jobs=-1)
        else:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42, n_jobs=-1)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        try:
            features = pd.DataFrame(index=df.index)
            for period in [1, 3, 5, 10]:
                features[f'return_{period}'] = df['close'].pct_change(period)
            features['rsi_14'] = calculate_rsi(df['close'], 14)
            features['rsi_21'] = calculate_rsi(df['close'], 21)
            for period in [9, 21, 50]:
                features[f'ema_{period}'] = calculate_ema(df['close'], period)
                features[f'price_to_ema_{period}'] = df['close'] / features[f'ema_{period}']
            features['ema_9_21_diff'] = (features['ema_9'] - features['ema_21']) / features['ema_21']
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['cvd'] = calculate_cvd(df)
            vwap = calculate_vwap(df)
            features['vwap_distance'] = (df['close'] - vwap) / vwap
            features = features.fillna(method='ffill').fillna(0)
            return features
        except:
            return pd.DataFrame()
    
    def train(self, df, lookforward=5):
        try:
            if len(df) < 100:
                return False
            features = self.prepare_features(df)
            if features.empty:
                return False
            future_return = df['close'].pct_change(lookforward).shift(-lookforward)
            target = (future_return > 0).astype(int)
            valid_idx = target.notna()
            X = features[valid_idx]
            y = target[valid_idx]
            if len(X) < 50:
                return False
            split_idx = int(len(X) * 0.8)
            X_train = X.iloc[:split_idx]
            y_train = y.iloc[:split_idx]
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
            self.is_trained = True
            return True
        except:
            return False
    
    def predict(self, df):
        try:
            if not self.is_trained:
                return 0, 0.5
            features = self.prepare_features(df)
            if features.empty:
                return 0, 0.5
            X_last = features.iloc[[-1]]
            X_scaled = self.scaler.transform(X_last)
            pred = self.model.predict(X_scaled)[0]
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                confidence = max(proba)
            else:
                confidence = 0.6
            signal = 1 if pred == 1 else -1
            return signal, confidence
        except:
            return 0, 0.5

# ============================================================================
# SCORE (mantido da versão anterior)
# ============================================================================

def calculate_institutional_score(df):
    try:
        result = {'master_score': 0, 'signal': 'NEUTRAL', 'confidence': 0.5, 'components': {}}
        if len(df) < 50:
            return result
        
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
            flow_score = -30
        result['components']['order_flow'] = flow_score
        
        ml_predictor = InstitutionalMLPredictor()
        if len(df) >= 200:
            ml_predictor.train(df.iloc[-1000:] if len(df) > 1000 else df)
        ml_signal, ml_confidence = ml_predictor.predict(df)
        ml_score = ml_signal * 100 * ml_confidence
        result['components']['ml_prediction'] = ml_score
        result['confidence'] = ml_confidence
        
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
        
        weights = {'trend': 0.15, 'momentum': 0.12, 'volume': 0.18, 'order_flow': 0.20, 'ml_prediction': 0.15, 'vwap': 0.10, 'market_profile': 0.10}
        master_score = sum(result['components'][component] * weight for component, weight in weights.items())
        master_score = max(-100, min(100, master_score))
        result['master_score'] = round(master_score, 2)
        
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
    except:
        return {'master_score': 0, 'signal': 'ERROR', 'confidence': 0, 'components': {}, 'rsi': 50, 'ema9': 0, 'ema21': 0, 'ema50': 0, 'vwap': 0, 'cvd': 0, 'poc': 0, 'vah': 0, 'val': 0}

# ============================================================================
# GRÁFICO (mantido da versão anterior)
# ============================================================================

def create_institutional_chart(df, result):
    try:
        if len(df) < 20:
            return None
        
        fig = make_subplots(
            rows=4, cols=1,
            row_heights=[0.5, 0.2, 0.15, 0.15],
            vertical_spacing=0.05,
            subplot_titles=('📊 Price Action', '💹 RSI', '📦 Volume + CVD', '🎯 Score'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='Price', increasing_line_color='#00ff88', decreasing_line_color='#ff0051'), row=1, col=1)
        
        try:
            ema9 = calculate_ema(df['close'], 9)
            ema21 = calculate_ema(df['close'], 21)
            ema50 = calculate_ema(df['close'], 50)
            fig.add_trace(go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='#00ff88', width=1.5), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ema21, name='EMA 21', line=dict(color='#00ccff', width=1.5), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=ema50, name='EMA 50', line=dict(color='#ffaa00', width=1.5), opacity=0.7), row=1, col=1)
        except:
            pass
        
        try:
            vwap = calculate_vwap(df)
            fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='#ff00ff', width=2, dash='dash'), opacity=0.8), row=1, col=1)
        except:
            pass
        
        try:
            fig.add_hline(y=result['poc'], line=dict(color='yellow', width=2, dash='dot'), annotation_text="POC", annotation_position="right", row=1, col=1)
            fig.add_hline(y=result['vah'], line=dict(color='orange', width=1, dash='dot'), row=1, col=1)
            fig.add_hline(y=result['val'], line=dict(color='orange', width=1, dash='dot'), row=1, col=1)
        except:
            pass
        
        try:
            rsi = calculate_rsi(df['close'], 14)
            fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#00ccff', width=2), fill='tozeroy', fillcolor='rgba(0, 204, 255, 0.2)'), row=2, col=1)
            fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=2, col=1)
            fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=2, col=1)
        except:
            pass
        
        try:
            colors = ['#00ff88' if c > o else '#ff0051' for c, o in zip(df['close'], df['open'])]
            fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker=dict(color=colors, opacity=0.5)), row=3, col=1)
        except:
            pass
        
        try:
            cvd = calculate_cvd(df)
            fig.add_trace(go.Scatter(x=df.index, y=cvd, name='CVD', line=dict(color='#ff00ff', width=2), yaxis='y2'), row=3, col=1, secondary_y=True)
        except:
            pass
        
        try:
            score_series = [50 if (i >= 50 and ema9.iloc[i] > ema21.iloc[i]) else -50 if i >= 50 else 0 for i in range(len(df))]
            score_color = ['#00ff88' if s > 0 else '#ff0051' for s in score_series]
            fig.add_trace(go.Bar(x=df.index, y=score_series, name='Score', marker=dict(color=score_color, opacity=0.7)), row=4, col=1)
            fig.add_hline(y=result['master_score'], line=dict(color='yellow', width=2, dash='dash'), annotation_text=f"{result['master_score']:.1f}", annotation_position="right", row=4, col=1)
        except:
            pass
        
        fig.update_layout(height=1400, template='plotly_dark', paper_bgcolor='#0a0e27', plot_bgcolor='#1a1f3a', font=dict(color='#ffffff', size=11), showlegend=True, xaxis_rangeslider_visible=False, hovermode='x unified')
        fig.update_xaxes(showgrid=True, gridcolor='#2a2f4a')
        fig.update_yaxes(showgrid=True, gridcolor='#2a2f4a')
        
        return fig
    except:
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>📈 ProfitOne V4.2</h1><h3 style='text-align: center; color: #00ff88;'>INSTITUTIONAL + BINANCE API</h3>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configurações")
        
        asset_categories = {
            '🔥 Futuros Crypto': ['BTCFUT', 'ETHFUT', 'WINFUT'],
            '📈 Ações B3': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'],
            '₿ Crypto Spot': ['BTCUSDT', 'ETHUSDT'],
            '🌎 Índices': ['^GSPC']
        }
        
        selected_category = st.selectbox("Categoria", list(asset_categories.keys()))
        symbol = st.selectbox("Símbolo", asset_categories[selected_category])
        
        symbol_config = B3_SYMBOLS.get(symbol, {})
        
        if symbol_config.get('binance'):
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #ff6b00 0%, #00ff88 100%);
                        padding: 10px; border-radius: 8px; text-align: center; margin: 10px 0;'>
                <strong>⚡ BINANCE API</strong><br>
                <small>{symbol_config['binance']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### ⏱️ Timeframe")
        
        timeframe_map = {
            '1 Minuto': ('7d', '1m'),
            '5 Minutos': ('60d', '5m'),
            '15 Minutos': ('60d', '15m'),
            '1 Hora': ('730d', '1h'),
            '4 Horas': ('730d', '4h'),
            '1 Dia': ('5y', '1d')
        }
        
        timeframe_label = st.radio("Período", list(timeframe_map.keys()), index=2)
        period, interval = timeframe_map[timeframe_label]
        
        st.markdown("---")
        
        force_binance = st.checkbox("🔥 Forçar Binance API", value=True if symbol_config.get('binance') else False)
        
        st.markdown("---")
        
        if st.button("🔄 Atualizar", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with st.spinner('🔄 Carregando dados...'):
        df = get_data_institutional(symbol, period, interval, force_binance)
    
    if df.empty or len(df) < 20:
        st.error(f"❌ **Sem dados para {symbol}**")
        
        with st.expander("🔧 Soluções"):
            st.markdown("""
            ### Tente:
            
            1. **Usar BTCFUT ou ETHFUT** (dados da Binance)
            2. **Selecionar "1 Hora" ou "1 Dia"**
            3. **Marcar "Forçar Binance API"** no sidebar
            4. **Testar PETR4.SA** (ações funcionam bem)
            """)
        return
    
    if not df.empty:
        date_range = (df.index[-1] - df.index[0]).days
        col1, col2, col3 = st.columns(3)
        col1.metric("📊 Barras", f"{len(df):,}")
        col2.metric("📅 Período", f"{date_range} dias")
        col3.metric("🕐 Última", df.index[-1].strftime("%d/%m %H:%M"))
    
    with st.spinner('🧮 Calculando...'):
        result = calculate_institutional_score(df)
    
    st.markdown("### 🎯 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = '#00ff88' if result['master_score'] > 0 else '#ff0051'
        st.markdown(f"<div style='background: #1a1f3a; padding: 20px; border-radius: 10px; text-align: center; border-left: 4px solid {score_color};'><div style='color: #888;'>MASTER SCORE</div><div style='color: {score_color}; font-size: 2.5rem; font-weight: 700;'>{result['master_score']:.1f}</div><div style='color: {score_color}; font-size: 1.1rem;'>{result['signal']}</div></div>", unsafe_allow_html=True)
    
    with col2:
        price_change = df['close'].pct_change().iloc[-1] * 100
        price_color = '#00ff88' if price_change > 0 else '#ff0051'
        st.markdown(f"<div style='background: #1a1f3a; padding: 20px; border-radius: 10px; text-align: center;'><div style='color: #888;'>PREÇO</div><div style='color: #fff; font-size: 2rem; font-weight: 700;'>{df['close'].iloc[-1]:.2f}</div><div style='color: {price_color};'>{price_change:+.2f}%</div></div>", unsafe_allow_html=True)
    
    with col3:
        conf_pct = result['confidence'] * 100
        st.markdown(f"<div style='background: #1a1f3a; padding: 20px; border-radius: 10px; text-align: center;'><div style='color: #888;'>ML CONFIDENCE</div><div style='color: #ffaa00; font-size: 2rem; font-weight: 700;'>{conf_pct:.1f}%</div></div>", unsafe_allow_html=True)
    
    with col4:
        rsi_color = '#ff0051' if result['rsi'] > 70 else ('#00ff88' if result['rsi'] < 30 else '#ffaa00')
        st.markdown(f"<div style='background: #1a1f3a; padding: 20px; border-radius: 10px; text-align: center;'><div style='color: #888;'>RSI</div><div style='color: {rsi_color}; font-size: 2rem; font-weight: 700;'>{result['rsi']:.1f}</div></div>", unsafe_allow_html=True)
    
    st.markdown("### 📊 Módulos")
    
    modules = [
        ('Trend', result['components'].get('trend', 0)),
        ('Momentum', result['components'].get('momentum', 0)),
        ('Volume', result['components'].get('volume', 0)),
        ('Order Flow', result['components'].get('order_flow', 0)),
        ('ML', result['components'].get('ml_prediction', 0)),
        ('VWAP', result['components'].get('vwap', 0)),
        ('Profile', result['components'].get('market_profile', 0))
    ]
    
    cols = st.columns(4)
    for idx, (name, score) in enumerate(modules):
        score_color = '#00ff88' if score > 0 else '#ff0051'
        cols[idx % 4].markdown(f"<div class='module-card'><div class='module-title'>{name}</div><div class='module-score' style='color: {score_color};'>{score:+.1f}</div></div>", unsafe_allow_html=True)
    
    st.markdown("### 📈 Gráfico")
    
    with st.spinner('🎨 Renderizando...'):
        fig = create_institutional_chart(df, result)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Erro ao gerar gráfico")
    
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; color: #666;'><strong>ProfitOne V4.2</strong> | Binance + Yahoo | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br><em>⚠️ Apenas educacional</em></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
