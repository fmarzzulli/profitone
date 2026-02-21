"""
ProfitOne V3.0 - Sistema Institucional com Suporte B3
Inclui: WINFUT, WINM24, DOLM24 e outros futuros B3
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
import requests

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

st.set_page_config(
    page_title="ProfitOne V3.0 - B3 Edition",
    page_icon="🇧🇷",
    layout="wide"
)

# CSS (mesmo do anterior)
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
    
    .b3-badge {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: black;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MAPEAMENTO DE SÍMBOLOS B3
# ============================================================================

B3_SYMBOLS = {
    # Futuros B3 (usar Ibovespa como proxy)
    "WINFUT": {"yahoo": "^BVSP", "multiplier": 0.2, "name": "Mini Índice Futuro", "type": "future"},
    "WINM24": {"yahoo": "^BVSP", "multiplier": 0.2, "name": "Mini Índice Mar/24", "type": "future"},
    "WINH24": {"yahoo": "^BVSP", "multiplier": 0.2, "name": "Mini Índice Ago/24", "type": "future"},
    "DOLM24": {"yahoo": "BRL=X", "multiplier": 50, "name": "Mini Dólar Mar/24", "type": "future"},
    "DOLFUT": {"yahoo": "BRL=X", "multiplier": 50, "name": "Mini Dólar Futuro", "type": "future"},
    
    # Ações B3 (nativas)
    "PETR4": {"yahoo": "PETR4.SA", "multiplier": 1, "name": "Petrobras PN", "type": "stock"},
    "VALE3": {"yahoo": "VALE3.SA", "multiplier": 1, "name": "Vale ON", "type": "stock"},
    "ITUB4": {"yahoo": "ITUB4.SA", "multiplier": 1, "name": "Itaú Unibanco PN", "type": "stock"},
    "BBDC4": {"yahoo": "BBDC4.SA", "multiplier": 1, "name": "Bradesco PN", "type": "stock"},
    "BBAS3": {"yahoo": "BBAS3.SA", "multiplier": 1, "name": "Banco do Brasil ON", "type": "stock"},
    "MGLU3": {"yahoo": "MGLU3.SA", "multiplier": 1, "name": "Magazine Luiza ON", "type": "stock"},
    "WEGE3": {"yahoo": "WEGE3.SA", "multiplier": 1, "name": "WEG ON", "type": "stock"},
}


def resolve_symbol(symbol_input):
    """Resolve símbolo B3 para Yahoo Finance"""
    symbol_upper = symbol_input.upper().replace(".SA", "")
    
    # Check if it's a B3 symbol
    if symbol_upper in B3_SYMBOLS:
        info = B3_SYMBOLS[symbol_upper]
        return info['yahoo'], info['multiplier'], info['name'], info['type']
    
    # Check if it ends with a number (B3 stock)
    if symbol_upper[-1].isdigit():
        return f"{symbol_upper}.SA", 1, symbol_upper, "stock"
    
    # Default: use as is
    return symbol_input, 1, symbol_input, "unknown"


# ============================================================================
# INDICADORES (mesmos do código anterior)
# ============================================================================

def calculate_ema(data, period):
    try:
        return data.ewm(span=period, adjust=False).mean()
    except:
        return data


def calculate_rsi(data, period=14):
    try:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series([50] * len(data), index=data.index)


def calculate_cvd(df):
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
    try:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    except:
        return df['close']


def market_profile_simple(df):
    try:
        prices = []
        for _, row in df.iterrows():
            prices.extend([row['close']] * int(row['volume'] / 1000))
        
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
        
        return np.polyfit(np.log(list(lags[:len(tau)])), np.log(tau), 1)[0]
    except:
        return 0.5


# ============================================================================
# MACHINE LEARNING (mesmo do anterior)
# ============================================================================

class SimpleMLPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
    
    def prepare_features(self, df):
        try:
            features = pd.DataFrame()
            features['returns'] = df['close'].pct_change()
            features['rsi'] = calculate_rsi(df['close'], 14)
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            features['ema_fast'] = calculate_ema(df['close'], 9)
            features['ema_slow'] = calculate_ema(df['close'], 21)
            features['ema_diff'] = (features['ema_fast'] - features['ema_slow']) / features['ema_slow']
            features = features.fillna(0).replace([np.inf, -np.inf], 0)
            return features
        except:
            return pd.DataFrame(np.zeros((len(df), 6)), columns=['returns', 'rsi', 'volume_ratio', 'ema_fast', 'ema_slow', 'ema_diff'])
    
    def train(self, df):
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
        except:
            return False
    
    def predict(self, df):
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
# SCORE MASTER (mesmo do anterior)
# ============================================================================

def calculate_master_score(df):
    scores = {}
    
    try:
        # Trend
        ema9 = calculate_ema(df['close'], 9)
        ema21 = calculate_ema(df['close'], 21)
        trend_score = 50 if df['close'].iloc[-1] > ema9.iloc[-1] else -50
        ema_alignment = 40 if ema9.iloc[-1] > ema21.iloc[-1] else -40
        scores['trend'] = (trend_score + ema_alignment) / 2
        
        # Momentum
        rsi = calculate_rsi(df['close'], 14)
        rsi_val = rsi.iloc[-1]
        if rsi_val > 70:
            rsi_score = -40
        elif rsi_val < 30:
            rsi_score = 40
        else:
            rsi_score = (rsi_val - 50)
        scores['momentum'] = rsi_score
        
        # Volume
        volume_trend = df['volume'].rolling(5).mean().iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
        volume_score = 30 if volume_trend > 1.2 else (-20 if volume_trend < 0.8 else 0)
        scores['volume'] = volume_score
        
        # Order Flow
        cvd, delta = calculate_cvd(df)
        cvd_trend = 40 if cvd.iloc[-1] > cvd.iloc[-20] else -40
        scores['order_flow'] = cvd_trend
        
        # VWAP
        vwap = calculate_vwap(df)
        vwap_score = 30 if df['close'].iloc[-1] > vwap.iloc[-1] else -30
        scores['vwap'] = vwap_score
        
        # Market Profile
        poc, vah, val = market_profile_simple(df)
        current_price = df['close'].iloc[-1]
        if current_price > vah:
            profile_score = 50
        elif current_price < val:
            profile_score = -50
        else:
            profile_score = 20 if current_price > poc else -20
        scores['market_profile'] = profile_score
        
        # ML
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
        
        # Regime
        hurst = hurst_exponent_simple(df['close'].tail(100))
        hurst_score = 40 if hurst > 0.55 else (-40 if hurst < 0.45 else 0)
        scores['regime'] = hurst_score
        
    except Exception as e:
        scores = {'trend': 0, 'momentum': 0, 'volume': 0, 'order_flow': 0, 'vwap': 0, 'market_profile': 0, 'machine_learning': 0, 'regime': 0}
        ml_confidence = 50
        poc, vah, val = df['close'].iloc[-1], df['close'].iloc[-1] * 1.02, df['close'].iloc[-1] * 0.98
        cvd, delta = pd.Series([0] * len(df)), pd.Series([0] * len(df))
        vwap = df['close']
        rsi = pd.Series([50] * len(df))
        hurst = 0.5
    
    # Master Score
    weights = {'trend': 1.2, 'momentum': 1.0, 'volume': 0.9, 'order_flow': 1.3, 'vwap': 1.0, 'market_profile': 1.1, 'machine_learning': 1.2, 'regime': 0.8}
    weighted_scores = [scores[key] * weights[key] for key in scores.keys()]
    master_score = np.mean(weighted_scores)
    master_score = np.clip(master_score, -100, 100)
    
    # Signal
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
# BUSCA DE DADOS COM SUPORTE B3
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_data_b3(symbol_input, period="5d", interval="15m"):
    """Busca dados com suporte B3"""
    try:
        # Resolve symbol
        yahoo_symbol, multiplier, name, asset_type = resolve_symbol(symbol_input)
        
        ticker = yf.Ticker(yahoo_symbol)
        
        # Tentar timeframe solicitado
        df = ticker.history(period=period, interval=interval)
        
        # Fallbacks
        if df.empty:
            df = ticker.history(period="1mo", interval="1h")
        if df.empty:
            df = ticker.history(period="6mo", interval="1d")
        if df.empty:
            df = ticker.history(period="1mo")
        
        if df.empty:
            return None, None, f"No data for {symbol_input}", None
        
        # Padronizar
        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index()
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        if len(df) < 10:
            return None, None, f"Insufficient data ({len(df)} candles)", None
        
        # Aplicar multiplicador para futuros
        if multiplier != 1:
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * multiplier
        
        symbol_info = {
            'original': symbol_input,
            'yahoo': yahoo_symbol,
            'name': name,
            'type': asset_type,
            'multiplier': multiplier
        }
        
        return df, symbol_info, None, asset_type
        
    except Exception as e:
        return None, None, f"Error: {str(e)}", None


# ============================================================================
# GRÁFICO (mesmo do anterior)
# ============================================================================

def create_chart_optimized(df, result):
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=('Price & Indicators', 'RSI', 'Volume')
    )
    
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
    
    ema9 = calculate_ema(df['close'], 9)
    ema21 = calculate_ema(df['close'], 21)
    
    fig.add_trace(go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='cyan', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name='EMA 21', line=dict(color='orange', width=2)), row=1, col=1)
    
    vwap = calculate_vwap(df)
    fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='yellow', width=2, dash='dot')), row=1, col=1)
    
    fig.add_hline(y=result['indicators']['poc'], line=dict(color='cyan', width=2, dash='solid'), annotation_text="POC", row=1, col=1)
    fig.add_hline(y=result['indicators']['vah'], line=dict(color='green', width=1, dash='dash'), annotation_text="VAH", row=1, col=1)
    fig.add_hline(y=result['indicators']['val'], line=dict(color='red', width=1, dash='dash'), annotation_text="VAL", row=1, col=1)
    
    rsi = calculate_rsi(df['close'], 14)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    colors = ['#00ff88' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff4444' for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors, opacity=0.7), row=3, col=1)
    
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
    st.markdown("""
    <div style='text-align: center; padding: 30px;'>
        <h1 style='font-size: 48px;'>🇧🇷 PROFITONE V3.0 - B3 EDITION</h1>
        <h2 style='font-size: 24px; color: #00ff88;'>Suporte para WINFUT, DOLM24 e Ações B3</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("## 🎯 TRADING CENTER")
        
        # B3 Assets
        st.markdown("### 🇧🇷 Futuros B3")
        b3_futures = ["WINFUT", "WINM24", "DOLFUT", "DOLM24"]
        
        st.markdown("### 📈 Ações B3")
        b3_stocks = ["PETR4", "VALE3", "ITUB4", "BBDC4", "BBAS3", "MGLU3", "WEGE3"]
        
        st.markdown("### 🌎 Internacional")
        intl = ["^BVSP (Ibovespa)", "^GSPC (S&P 500)", "BTC-USD (Bitcoin)", "ETH-USD (Ethereum)"]
        
        all_symbols = b3_futures + b3_stocks + intl + ["Custom"]
        
        selected = st.selectbox("Select Asset", all_symbols, index=0)  # Default: WINFUT
        
        if selected == "Custom":
            symbol = st.text_input("Ticker:", "WINFUT")
        else:
            symbol = selected.split(" ")[0]
        
        timeframe = st.radio("Timeframe", ["5 min", "15 min", "1 hour", "1 day"], index=2)
        
        interval_map = {"5 min": "5m", "15 min": "15m", "1 hour": "1h", "1 day": "1d"}
        interval = interval_map[timeframe]
        
        period_map = {"5m": "1d", "15m": "5d", "1h": "1mo", "1d": "6mo"}
        period = period_map[interval]
        
        st.markdown("---")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        st.info("💡 **Nota:** WINFUT usa dados do Ibovespa (^BVSP) como proxy em tempo real.")
    
    # LOADING
    with st.spinner(f"📊 Loading {symbol}..."):
        df, symbol_info, error, asset_type = get_data_b3(symbol, period, interval)
    
    if error or df is None:
        st.error(f"❌ {error}")
        st.info("💡 **Try:**\n- WINFUT\n- PETR4\n- BTC-USD")
        return
    
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'date' in df.columns:
        df = df.set_index('date')
    
    # Display symbol info
    if symbol_info and asset_type == "future":
        st.markdown(f"<div style='text-align: center;'>"
                   f"<span class='b3-badge'>🇧🇷 B3 FUTURE</span> "
                   f"<span style='color: #00ff88; font-size: 20px;'>{symbol_info['name']}</span>"
                   f"</div>", unsafe_allow_html=True)
        st.caption(f"📊 Proxy: {symbol_info['yahoo']} | Multiplier: {symbol_info['multiplier']}")
    
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
        st.metric("💰 Price", f"{price:,.2f} pts" if asset_type == "future" else f"${price:.2f}", delta=f"{change_pct:+.2f}%")
    
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
        st.write(f"**VWAP:** {result['indicators']['vwap']:,.2f}")
        st.write(f"**CVD:** {result['indicators']['cvd']:,.0f}")
        st.write(f"**Hurst:** {result['indicators']['hurst']:.3f}")
    
    with col2:
        st.markdown("### 🎯 Market Profile")
        st.write(f"**POC:** {result['indicators']['poc']:,.2f}")
        st.write(f"**VAH:** {result['indicators']['vah']:,.2f}")
        st.write(f"**VAL:** {result['indicators']['val']:,.2f}")
    
    with col3:
        st.markdown("### 📈 Status")
        price_pos = "Above" if price > result['indicators']['vwap'] else "Below"
        st.write(f"**Price vs VWAP:** {price_pos}")
        
        regime = "Trending" if result['indicators']['hurst'] > 0.55 else ("Ranging" if result['indicators']['hurst'] < 0.45 else "Random")
        st.write(f"**Market Regime:** {regime}")
    
    # FOOTER
    st.markdown("---")
    st.caption(f"🇧🇷 ProfitOne V3.0 B3 Edition | {symbol} ({symbol_info['name'] if symbol_info else symbol}) | {timeframe} | {datetime.now().strftime('%H:%M:%S')}")
    st.caption("⚠️ WINFUT e outros futuros B3 usam dados proxy do Ibovespa para fins educacionais")


if __name__ == "__main__":
    main()
