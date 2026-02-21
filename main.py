"""
ProfitOne V3.0 - Sistema Institucional OTIMIZADO
Versão estável com todos os recursos profissionais
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="ProfitOne V3.0 - Institucional",
    page_icon="🏦",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%); }
    h1, h2, h3 { color: #00ff88 !important; text-shadow: 0 0 20px rgba(0, 255, 136, 0.5); }
    [data-testid="stMetricValue"] { font-size: 32px !important; font-weight: bold !important; }
    
    .pro-card {
        background: linear-gradient(145deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 255, 136, 0.1);
    }
    
    .signal-strong-buy {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.3), rgba(0, 200, 100, 0.2));
        border-left: 5px solid #00ff88;
    }
    
    .signal-buy { background: rgba(0, 255, 136, 0.15); border-left: 4px solid #00ff88; }
    .signal-sell { background: rgba(255, 68, 68, 0.15); border-left: 4px solid #ff4444; }
    .signal-neutral { background: rgba(255, 170, 0, 0.15); border-left: 4px solid #ffaa00; }
    
    .stButton > button {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INDICADORES CORE
# ============================================================================

def safe_calculation(func, *args, default=0, **kwargs):
    """Wrapper seguro para cálculos"""
    try:
        result = func(*args, **kwargs)
        return result if not pd.isna(result).all() else default
    except Exception as e:
        st.warning(f"⚠️ Calculation warning: {str(e)[:50]}")
        return default


def calculate_ema(data, period):
    """EMA seguro"""
    try:
        return data.ewm(span=period, adjust=False).mean()
    except:
        return data


def calculate_rsi(data, period=14):
    """RSI seguro"""
    try:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series([50] * len(data), index=data.index)


def calculate_cvd(df):
    """Cumulative Volume Delta"""
    try:
        price_change = df['close'].diff()
        buy_volume = df['volume'].where(price_change > 0, 0)
        sell_volume = df['volume'].where(price_change < 0, 0)
        delta = buy_volume - sell_volume
        cvd = delta.cumsum()
        return cvd, delta
    except:
        return pd.Series([0] * len(df), index=df.index), pd.Series([0] * len(df), index=df.index)


def calculate_vwap(df):
    """VWAP simples e seguro"""
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    except:
        return df['close']


def market_profile_simple(df):
    """Market Profile simplificado"""
    try:
        # Calcular quartis como proxy para VAH, POC, VAL
        prices = []
        for _, row in df.iterrows():
            prices.extend([row['close']] * int(row['volume'] / 1000))  # Simplificado
        
        if len(prices) < 10:
            prices = df['close'].tolist()
        
        poc = np.median(prices)
        vah = np.percentile(prices, 85)
        val = np.percentile(prices, 15)
        
        return poc, vah, val
    except:
        current = df['close'].iloc[-1]
        return current, current * 1.02, current * 0.98


def hurst_exponent_simple(ts):
    """Hurst Exponent simplificado"""
    try:
        if len(ts) < 20:
            return 0.5
        
        lags = range(2, min(20, len(ts) // 2))
        tau = []
        
        for lag in lags:
            std_lag = ts.rolling(window=lag).std().dropna()
            if len(std_lag) > 0:
                tau.append(np.mean(std_lag))
        
        if len(tau) < 2:
            return 0.5
        
        # Aproximação simples
        return np.polyfit(np.log(list(lags[:len(tau)])), np.log(tau), 1)[0]
    except:
        return 0.5


# ============================================================================
# MACHINE LEARNING SIMPLIFICADO
# ============================================================================

class SimpleMLPredictor:
    """ML simplificado e rápido"""
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def prepare_features(self, df):
        """Features básicas"""
        try:
            features = pd.DataFrame()
            
            features['returns'] = df['close'].pct_change()
            features['rsi'] = calculate_rsi(df['close'], 14)
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['ema_fast'] = calculate_ema(df['close'], 9)
            features['ema_slow'] = calculate_ema(df['close'], 21)
            features['ema_diff'] = (features['ema_fast'] - features['ema_slow']) / features['ema_slow']
            
            features = features.fillna(0)
            features = features.replace([np.inf, -np.inf], 0)
            
            return features
        except:
            # Fallback: features vazias
            return pd.DataFrame(np.zeros((len(df), 6)), columns=['returns', 'rsi', 'volume_ratio', 'ema_fast', 'ema_slow', 'ema_diff'])
    
    def train(self, df):
        """Treina modelo"""
        try:
            if len(df) < 50:
                return False
            
            features = self.prepare_features(df)
            target = (df['close'].shift(-1) > df['close']).astype(int)
            
            features = features[:-1]
            target = target[:-1]
            
            mask = ~(features.isna().any(axis=1) | target.isna())
            features = features[mask]
            target = target[mask]
            
            if len(features) < 30:
                return False
            
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, target)
            self.trained = True
            
            return True
        except Exception as e:
            st.warning(f"⚠️ ML training skipped: {str(e)[:50]}")
            return False
    
    def predict(self, df):
        """Predição"""
        try:
            if not self.trained:
                return None, [0.5, 0.5]
            
            features = self.prepare_features(df)
            features_scaled = self.scaler.transform(features.tail(1))
            
            prediction = self.model.predict(features_scaled)[0]
            probability = self.model.predict_proba(features_scaled)[0]
            
            return prediction, probability
        except:
            return None, [0.5, 0.5]


# ============================================================================
# SCORE MASTER SIMPLIFICADO
# ============================================================================

def calculate_master_score(df):
    """Score mestre otimizado"""
    
    scores = {}
    
    try:
        # 1. TENDÊNCIA (EMA)
        ema9 = calculate_ema(df['close'], 9)
        ema21 = calculate_ema(df['close'], 21)
        
        trend_score = 50 if df['close'].iloc[-1] > ema9.iloc[-1] else -50
        ema_alignment = 40 if ema9.iloc[-1] > ema21.iloc[-1] else -40
        
        scores['trend'] = (trend_score + ema_alignment) / 2
        
        # 2. MOMENTUM (RSI)
        rsi = calculate_rsi(df['close'], 14)
        rsi_val = rsi.iloc[-1]
        
        if rsi_val > 70:
            rsi_score = -40
        elif rsi_val < 30:
            rsi_score = 40
        else:
            rsi_score = (rsi_val - 50)
        
        scores['momentum'] = rsi_score
        
        # 3. VOLUME
        volume_trend = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        volume_score = 30 if volume_trend > 1.2 else (-20 if volume_trend < 0.8 else 0)
        
        scores['volume'] = volume_score
        
        # 4. CVD (Order Flow)
        cvd, delta = calculate_cvd(df)
        cvd_trend = 40 if cvd.iloc[-1] > cvd.iloc[-20] else -40
        
        scores['order_flow'] = cvd_trend
        
        # 5. VWAP
        vwap = calculate_vwap(df)
        vwap_score = 30 if df['close'].iloc[-1] > vwap.iloc[-1] else -30
        
        scores['vwap'] = vwap_score
        
        # 6. MARKET PROFILE
        poc, vah, val = market_profile_simple(df)
        current_price = df['close'].iloc[-1]
        
        if current_price > vah:
            profile_score = 50
        elif current_price < val:
            profile_score = -50
        else:
            profile_score = 20 if current_price > poc else -20
        
        scores['market_profile'] = profile_score
        
        # 7. MACHINE LEARNING
        ml_predictor = SimpleMLPredictor()
        ml_trained = ml_predictor.train(df)
        
        if ml_trained:
            ml_prediction, ml_probability = ml_predictor.predict(df)
            if ml_prediction is not None:
                ml_score = 60 if ml_prediction == 1 else -60
                ml_confidence = max(ml_probability) * 100
            else:
                ml_score = 0
                ml_confidence = 50
        else:
            ml_score = 0
            ml_confidence = 50
        
        scores['machine_learning'] = ml_score
        
        # 8. HURST (Regime)
        hurst = hurst_exponent_simple(df['close'].tail(100))
        hurst_score = 40 if hurst > 0.55 else (-40 if hurst < 0.45 else 0)
        
        scores['regime'] = hurst_score
        
    except Exception as e:
        st.error(f"❌ Error calculating scores: {e}")
        # Scores padrão em caso de erro
        scores = {
            'trend': 0,
            'momentum': 0,
            'volume': 0,
            'order_flow': 0,
            'vwap': 0,
            'market_profile': 0,
            'machine_learning': 0,
            'regime': 0
        }
        ml_confidence = 50
        poc, vah, val = df['close'].iloc[-1], df['close'].iloc[-1] * 1.02, df['close'].iloc[-1] * 0.98
        cvd, delta = pd.Series([0] * len(df)), pd.Series([0] * len(df))
        vwap = df['close']
    
    # SCORE MASTER
    weights = {
        'trend': 1.2,
        'momentum': 1.0,
        'volume': 0.9,
        'order_flow': 1.3,
        'vwap': 1.0,
        'market_profile': 1.1,
        'machine_learning': 1.2,
        'regime': 0.8
    }
    
    weighted_scores = [scores[key] * weights[key] for key in scores.keys()]
    master_score = np.mean(weighted_scores)
    master_score = np.clip(master_score, -100, 100)
    
    # SINAL
    if master_score > 50:
        signal = 'STRONG BUY'
        signal_class = 'signal-strong-buy'
    elif master_score > 25:
        signal = 'BUY'
        signal_class = 'signal-buy'
    elif master_score < -50:
        signal = 'STRONG SELL'
        signal_class = 'signal-sell'
    elif master_score < -25:
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
        'ml_confidence': ml_confidence,
        'indicators': {
            'cvd': cvd.iloc[-1],
            'vwap': vwap.iloc[-1],
            'poc': poc,
            'vah': vah,
            'val': val,
            'rsi': rsi.iloc[-1],
            'hurst': hurst
        }
    }


# ============================================================================
# BUSCA DE DADOS OTIMIZADA
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_data_safe(symbol, period="5d", interval="15m"):
    """Busca dados com fallback robusto"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Tentar timeframe solicitado
        df = ticker.history(period=period, interval=interval)
        
        # Fallback 1: período maior
        if df.empty:
            df = ticker.history(period="1mo", interval="1h")
        
        # Fallback 2: dados diários
        if df.empty:
            df = ticker.history(period="6mo", interval="1d")
        
        # Fallback 3: últimos 30 dias
        if df.empty:
            df = ticker.history(period="1mo")
        
        if df.empty:
            return None, f"No data available for {symbol}"
        
        # Padronizar
        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index()
        
        # Remover linhas com NaN críticos
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        if len(df) < 10:
            return None, f"Insufficient data ({len(df)} candles)"
        
        return df, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"


# ============================================================================
# GRÁFICO OTIMIZADO
# ============================================================================

def create_chart_optimized(df, result):
    """Gráfico simplificado e rápido"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Indicators', 'RSI', 'Volume')
    )
    
    # CANDLESTICK
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # EMA
    ema9 = calculate_ema(df['close'], 9)
    ema21 = calculate_ema(df['close'], 21)
    
    fig.add_trace(go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='cyan', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name='EMA 21', line=dict(color='orange', width=2)), row=1, col=1)
    
    # VWAP
    vwap = calculate_vwap(df)
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='yellow', width=2, dash='dot')), row=1, col=1)
    
    # Market Profile
    fig.add_hline(y=result['indicators']['poc'], line=dict(color='cyan', width=2, dash='solid'), row=1, col=1)
    fig.add_hline(y=result['indicators']['vah'], line=dict(color='green', width=1, dash='dash'), row=1, col=1)
    fig.add_hline(y=result['indicators']['val'], line=dict(color='red', width=1, dash='dash'), row=1, col=1)
    
    # RSI
    rsi = calculate_rsi(df['close'], 14)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # VOLUME
    colors = ['#00ff88' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff4444' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors, opacity=0.7), row=3, col=1)
    
    # LAYOUT
    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    return fig


# ============================================================================
# APP PRINCIPAL
# ============================================================================

def main():
    # HEADER
    st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <h1 style='font-size: 48px;'>🏦 PROFITONE V3.0</h1>
        <h2 style='font-size: 24px; color: #00ff88;'>INSTITUTIONAL TRADING SYSTEM</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # SIDEBAR
    with st.sidebar:
        st.markdown("## 🎯 TRADING CENTER")
        
        symbols = {
            "🇧🇷 Ibovespa": "^BVSP",
            "🛢️ Petrobras": "PETR4.SA",
            "⛏️ Vale": "VALE3.SA",
            "🏦 Itaú": "ITUB4.SA",
            "📈 S&P 500": "^GSPC",
            "💰 Bitcoin": "BTC-USD",
            "💎 Ethereum": "ETH-USD"
        }
        
        selected = st.selectbox("Asset", list(symbols.keys()), index=1)  # Padrão: PETR4
        symbol = symbols[selected]
        
        timeframe = st.radio("Timeframe", ["15 min", "1 hour", "1 day"], index=1)  # Padrão: 1 hour
        
        interval_map = {"15 min": "15m", "1 hour": "1h", "1 day": "1d"}
        interval = interval_map[timeframe]
        
        period_map = {"15m": "5d", "1h": "1mo", "1d": "6mo"}
        period = period_map[interval]
        
        st.markdown("---")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # LOADING
    with st.spinner(f"📊 Loading {symbol}..."):
        df, error = get_data_safe(symbol, period, interval)
    
    if error or df is None:
        st.error(f"❌ {error}")
        st.info("💡 **Try:**\n- PETR4.SA\n- BTC-USD\n- Different timeframe")
        return
    
    # Preparar dados
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df = df.set_index('date')
    
    st.success(f"✅ Loaded {len(df)} candles")
    
    # CALCULAR SCORE
    with st.spinner("🧠 Analyzing..."):
        result = calculate_master_score(df)
    
    # METRICS
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"<div class='pro-card {result['signal_class']}' style='text-align: center;'>"
                   f"<h3>Master Score</h3><h1>{result['master_score']:.1f}</h1>"
                   f"<p style='font-size: 20px;'><b>{result['signal']}</b></p></div>",
                   unsafe_allow_html=True)
    
    with col2:
        price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        change_pct = ((price - prev_price) / prev_price) * 100
        st.metric("💰 Price", f"${price:.2f}", delta=f"{change_pct:+.2f}%")
    
    with col3:
        st.metric("🤖 ML Confidence", f"{result['ml_confidence']:.1f}%")
    
    with col4:
        st.metric("📊 RSI", f"{result['indicators']['rsi']:.1f}")
    
    # MODULE SCORES
    st.markdown("---")
    st.markdown("## 📊 Module Scores")
    
    cols = st.columns(len(result['module_scores']))
    
    for col, (name, score) in zip(cols, result['module_scores'].items()):
        signal_class = "signal-buy" if score > 20 else ("signal-sell" if score < -20 else "signal-neutral")
        
        with col:
            col.markdown(f"<div class='pro-card {signal_class}' style='text-align: center;'>"
                        f"<h4>{name.title()}</h4><h2>{score:.1f}</h2></div>",
                        unsafe_allow_html=True)
    
    # GRÁFICO
    st.markdown("---")
    st.markdown("## 📈 Chart")
    
    fig = create_chart_optimized(df, result)
    st.plotly_chart(fig, use_container_width=True)
    
    # DETAILED INFO
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Indicators")
        st.write(f"**VWAP:** ${result['indicators']['vwap']:.2f}")
        st.write(f"**CVD:** {result['indicators']['cvd']:,.0f}")
        st.write(f"**Hurst:** {result['indicators']['hurst']:.3f}")
    
    with col2:
        st.markdown("### 🎯 Market Profile")
        st.write(f"**POC:** ${result['indicators']['poc']:.2f}")
        st.write(f"**VAH:** ${result['indicators']['vah']:.2f}")
        st.write(f"**VAL:** ${result['indicators']['val']:.2f}")
    
    with col3:
        st.markdown("### 📈 Status")
        price_pos = "Above" if price > result['indicators']['vwap'] else "Below"
        st.write(f"**Price vs VWAP:** {price_pos}")
        
        regime = "Trending" if result['indicators']['hurst'] > 0.55 else ("Ranging" if result['indicators']['hurst'] < 0.45 else "Random")
        st.write(f"**Market Regime:** {regime}")
    
    # FOOTER
    st.markdown("---")
    st.caption(f"🏦 ProfitOne V3.0 | {symbol} | {timeframe} | {datetime.now().strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
