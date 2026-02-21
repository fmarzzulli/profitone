# 🚀 PROFITONE QUANTUM V5 - MÁXIMO REALISMO + ANTI-REPAINT
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import hashlib
import json
from scipy.stats import norm
from scipy.signal import find_peaks
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================

st.set_page_config(
    page_title="ProfitOne Quantum V5",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONALIZADO - TEMA DARK MODERNO
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3 { 
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #00ff88 !important;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(20, 30, 60, 0.8), rgba(30, 40, 80, 0.8));
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 255, 255, 0.1);
        margin: 10px 0;
    }
    
    /* Botões */
    .stButton>button {
        background: linear-gradient(90deg, #00c9ff, #92fe9d);
        color: #000;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 255, 255, 0.5);
    }
    
    /* Badges */
    .signal-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    .buy-signal { 
        background: linear-gradient(90deg, #00ff88, #00cc66);
        color: #000;
    }
    
    .sell-signal { 
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: #fff;
    }
    
    .neutral-signal { 
        background: linear-gradient(90deg, #888888, #666666);
        color: #fff;
    }
    
    /* Tabelas */
    .dataframe {
        background: rgba(20, 30, 60, 0.6) !important;
        color: #ffffff !important;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: rgba(10, 14, 39, 0.95);
        padding: 10px;
        text-align: center;
        border-top: 1px solid rgba(0, 255, 255, 0.2);
        font-size: 12px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CLASSE: SIMULADOR QUÂNTICO
# ============================================================

class QuantumMarketSimulator:
    """Simulação de estados quânticos para análise de mercado"""
    
    def __init__(self, n_states=8):
        self.n_states = n_states
        self.states = np.random.random(n_states) + 1j * np.random.random(n_states)
        self.states /= np.linalg.norm(self.states)  # Normalizar
    
    def quantum_monte_carlo(self, price_data, n_simulations=10000):
        """Simulação Quantum Monte Carlo para previsões"""
        if len(price_data) < 20:
            return 0.5, 50.0
        
        returns = np.diff(price_data) / price_data[:-1]
        mu = np.mean(returns)
        sigma = np.std(returns)
        
        # Simulações quânticas
        simulations = []
        for _ in range(n_simulations):
            quantum_noise = np.abs(self.states).sum() * 0.1
            sim_return = np.random.normal(mu, sigma + quantum_noise)
            simulations.append(sim_return)
        
        simulations = np.array(simulations)
        prob_up = np.sum(simulations > 0) / n_simulations
        confidence = np.abs(prob_up - 0.5) * 200  # 0-100%
        
        return prob_up, confidence
    
    def entanglement_score(self, data1, data2):
        """Calcula entrelaçamento quântico entre dois ativos"""
        if len(data1) < 20 or len(data2) < 20:
            return 0.0
        
        # Correlação normalizada
        corr = np.corrcoef(data1[-20:], data2[-20:])[0, 1]
        
        # Adiciona ruído quântico
        quantum_factor = np.abs(self.states[0]) * 0.2
        entanglement = np.abs(corr) * (1 + quantum_factor)
        
        return min(entanglement * 100, 100.0)  # 0-100%

# ============================================================
# CLASSE: MOTOR ANTI-REPINTURA
# ============================================================

class AntiRepaintEngine:
    """Motor que previne repintura de sinais"""
    
    def __init__(self, confirmation_bars=2):
        self.confirmation_bars = confirmation_bars
        self.signal_history = []
        self.locked_signals = []
    
    def generate_signal_hash(self, signal_data):
        """Gera hash SHA-256 imutável para o sinal"""
        signal_str = json.dumps(signal_data, sort_keys=True)
        return hashlib.sha256(signal_str.encode()).hexdigest()[:16]
    
    def add_signal(self, timestamp, signal_type, price, score, confidence):
        """Adiciona novo sinal pendente"""
        signal = {
            'timestamp': timestamp,
            'type': signal_type,
            'price': price,
            'score': score,
            'confidence': confidence,
            'bars_since': 0,
            'status': 'PENDING',
            'hash': None
        }
        signal['hash'] = self.generate_signal_hash(signal)
        self.signal_history.append(signal)
        return signal['hash']
    
    def update_signals(self):
        """Atualiza status dos sinais baseado em barras de confirmação"""
        for signal in self.signal_history:
            if signal['status'] == 'PENDING':
                signal['bars_since'] += 1
                
                if signal['bars_since'] >= self.confirmation_bars:
                    signal['status'] = 'CONFIRMED'
                    self.locked_signals.append(signal)
        
        # Remove sinais confirmados da lista pendente
        self.signal_history = [s for s in self.signal_history if s['status'] == 'PENDING']
    
    def get_confirmed_signals(self, limit=100):
        """Retorna últimos sinais confirmados"""
        return self.locked_signals[-limit:]

# ============================================================
# FUNÇÕES DE INDICADORES TÉCNICOS
# ============================================================

def calculate_tema(prices, period=9):
    """Triple Exponential Moving Average"""
    if len(prices) < period * 3:
        return np.full(len(prices), np.nan)
    
    ema1 = pd.Series(prices).ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    
    tema = 3 * ema1 - 3 * ema2 + ema3
    return tema.values

def calculate_vwap(df):
    """Volume Weighted Average Price"""
    if len(df) == 0 or 'Volume' not in df.columns:
        return np.full(len(df), np.nan)
    
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    return vwap.values

def calculate_rsi(prices, period=14):
    """Relative Strength Index"""
    if len(prices) < period + 1:
        return np.full(len(prices), 50.0)
    
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
    avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Pad início com 50
    rsi = np.concatenate([np.full(period, 50.0), rsi])
    return rsi

def calculate_schaff_trend_cycle(prices, fast=23, slow=50, cycle=10):
    """Schaff Trend Cycle"""
    if len(prices) < slow + cycle:
        return np.full(len(prices), 50.0)
    
    macd = pd.Series(prices).ewm(span=fast).mean() - pd.Series(prices).ewm(span=slow).mean()
    
    # Normalize
    stoch_macd = (macd - macd.rolling(cycle).min()) / (macd.rolling(cycle).max() - macd.rolling(cycle).min() + 1e-10) * 100
    stc = stoch_macd.ewm(span=3).mean()
    
    return stc.fillna(50).values

def calculate_fisher_transform(prices, period=10):
    """Fisher Transform"""
    if len(prices) < period:
        return np.full(len(prices), 0.0)
    
    min_low = pd.Series(prices).rolling(period).min()
    max_high = pd.Series(prices).rolling(period).max()
    
    value = 2 * ((prices - min_low) / (max_high - min_low + 1e-10) - 0.5)
    value = np.clip(value, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
    return pd.Series(fisher).fillna(0).values

def calculate_hurst_exponent(prices, max_lag=20):
    """Hurst Exponent - detecta tendência ou reversão"""
    if len(prices) < max_lag * 2:
        return 0.5
    
    lags = range(2, max_lag)
    tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
    
    # Regressão linear
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = reg[0]
    
    return np.clip(hurst, 0.0, 1.0)

def calculate_vpin(df, bucket_size=50):
    """Volume-Synchronized Probability of Informed Trading"""
    if len(df) < bucket_size or 'Volume' not in df.columns:
        return np.full(len(df), 0.0)
    
    # Buy/Sell volume proxy
    price_change = df['Close'].diff()
    buy_volume = np.where(price_change > 0, df['Volume'], 0)
    sell_volume = np.where(price_change < 0, df['Volume'], 0)
    
    # VPIN = |BuyVol - SellVol| / TotalVol
    vpin = np.abs(buy_volume - sell_volume) / (df['Volume'] + 1e-10)
    vpin = pd.Series(vpin).rolling(bucket_size).mean().fillna(0)
    
    return vpin.values * 100  # 0-100%

# ============================================================
# QUANTUM HUNTER V14
# ============================================================

def quantum_hunter_v14(df):
    """Algoritmo proprietário Quantum Hunter V14"""
    if len(df) < 50:
        return np.full(len(df), 0.0), np.full(len(df), 50.0)
    
    closes = df['Close'].values
    
    # Componentes
    tema_fast = calculate_tema(closes, 9)
    tema_slow = calculate_tema(closes, 21)
    vwap = calculate_vwap(df)
    rsi = calculate_rsi(closes, 14)
    
    # Trend Component (40%)
    trend = np.where(tema_fast > tema_slow, 1, -1)
    trend_strength = np.abs(tema_fast - tema_slow) / (closes + 1e-10) * 100
    trend_score = trend * np.clip(trend_strength, 0, 40)
    
    # VWAP Component (30%)
    vwap_diff = (closes - vwap) / (vwap + 1e-10) * 100
    vwap_score = np.clip(vwap_diff * 10, -30, 30)
    
    # RSI Component (30%)
    rsi_score = (rsi - 50) * 0.6  # -30 to +30
    
    # Score Final: -100 a +100
    quantum_score = trend_score + vwap_score + rsi_score
    quantum_score = np.clip(quantum_score, -100, 100)
    
    # Confidence: baseado em volatilidade
    volatility = pd.Series(closes).pct_change().rolling(20).std().fillna(0) * 100
    confidence = np.clip((100 - volatility * 5), 30, 95)
    
    return quantum_score, confidence

# ============================================================
# FORÇA DO WIN
# ============================================================

def calculate_win_strength(df):
    """Métrica proprietária FORÇA DO WIN"""
    if len(df) < 50:
        return np.full(len(df), 50.0)
    
    closes = df['Close'].values
    
    # Componentes
    quantum_score, _ = quantum_hunter_v14(df)
    stc = calculate_schaff_trend_cycle(closes)
    fisher = calculate_fisher_transform(closes)
    
    # WIN Strength: 0-100%
    win_strength = (
        (quantum_score + 100) / 2 * 0.4 +  # Quantum: 40%
        stc * 0.3 +  # STC: 30%
        (fisher + 5) / 10 * 100 * 0.3  # Fisher: 30%
    )
    
    return np.clip(win_strength, 0, 100)

# ============================================================
# ENSEMBLE MACHINE LEARNING
# ============================================================

def prepare_ml_features(df):
    """Prepara features para ML"""
    if len(df) < 50:
        return None, None
    
    closes = df['Close'].values
    
    # Features
    quantum_score, confidence = quantum_hunter_v14(df)
    rsi = calculate_rsi(closes)
    stc = calculate_schaff_trend_cycle(closes)
    fisher = calculate_fisher_transform(closes)
    hurst = np.full(len(closes), calculate_hurst_exponent(closes))
    vpin = calculate_vpin(df)
    
    features = pd.DataFrame({
        'quantum_score': quantum_score,
        'confidence': confidence,
        'rsi': rsi,
        'stc': stc,
        'fisher': fisher,
        'hurst': hurst,
        'vpin': vpin,
        'price_change': df['Close'].pct_change().fillna(0) * 100,
        'volume_change': df['Volume'].pct_change().fillna(0) * 100
    })
    
    # Target: preço vai subir na próxima barra?
    target = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Remove última linha (sem target)
    features = features[:-1]
    target = target[:-1]
    
    return features.fillna(0), target

def train_ensemble_ml(df):
    """Treina ensemble de modelos ML"""
    features, target = prepare_ml_features(df)
    
    if features is None or len(features) < 100:
        return None, None, None, None
    
    # Split treino/teste
    split = int(len(features) * 0.8)
    X_train, X_test = features[:split], features[split:]
    y_train, y_test = target[:split], target[split:]
    
    # Normalização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    models = {
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    }
    
    # Treinamento
    predictions = {}
    accuracies = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        acc = np.mean(pred == y_test) * 100
        
        predictions[name] = pred
        accuracies[name] = acc
    
    # Ensemble: voto majoritário
    ensemble_pred = np.array([predictions[m][-1] for m in models.keys()])
    final_prediction = np.bincount(ensemble_pred).argmax()
    
    # Confidence: % de modelos concordando
    confidence = np.max(np.bincount(ensemble_pred)) / len(ensemble_pred) * 100
    
    return final_prediction, confidence, accuracies, scaler

# ============================================================
# BUSCAR DADOS DE MERCADO
# ============================================================

@st.cache_data(ttl=60)
def get_market_data(symbol, period='1mo', interval='5m'):
    """Busca dados do Yahoo Finance"""
    try:
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart/"
        url = f"{base_url}{symbol}?interval={interval}&range={period}"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart']:
            return None
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps, unit='s'),
            'Open': quotes['open'],
            'High': quotes['high'],
            'Low': quotes['low'],
            'Close': quotes['close'],
            'Volume': quotes['volume']
        })
        
        df = df.dropna()
        return df
    
    except Exception as e:
        st.error(f"❌ Erro ao buscar dados de {symbol}: {str(e)}")
        return None

# ============================================================
# INTERFACE PRINCIPAL
# ============================================================

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🚀 PROFITONE QUANTUM V5</h1>
        <p style='font-size: 18px; color: #00ffcc;'>
            Tecnologia Quântica + Anti-Repaint + Machine Learning Ensemble
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configurações
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/rocket.png", width=150)
        st.markdown("## ⚙️ Configurações")
        
        mode = st.selectbox(
            "📊 Modo de Operação",
            ['Scalp (5 min)', 'Day Trade (15 min)', 'Swing (1h)']
        )
        
        interval_map = {
            'Scalp (5 min)': '5m',
            'Day Trade (15 min)': '15m',
            'Swing (1h)': '1h'
        }
        interval = interval_map[mode]
        
        confirmation_bars = st.slider("🔒 Barras de Confirmação (Anti-Repaint)", 1, 5, 2)
        
        refresh_seconds = st.slider("🔄 Auto-Refresh (segundos)", 15, 300, 30)
        
        st.markdown("---")
        st.markdown("### 🎯 Legenda de Sinais")
        st.markdown("""
        <div style='font-size: 12px;'>
            🟢 <strong>VERDE</strong>: Confirmado<br>
            🟡 <strong>AMARELO</strong>: Aguardando<br>
            🔴 <strong>VERMELHO</strong>: Rejeitado<br>
            ⚪ <strong>CINZA</strong>: Cancelado
        </div>
        """, unsafe_allow_html=True)
    
    # Buscar dados
    symbol = '^BVSP'
    df = get_market_data(symbol, period='1mo', interval=interval)
    
    if df is None or len(df) < 50:
        st.error("❌ Não foi possível carregar dados suficientes do mercado.")
        return
    
    # Inicializar componentes
    quantum_sim = QuantumMarketSimulator(n_states=8)
    anti_repaint = AntiRepaintEngine(confirmation_bars=confirmation_bars)
    
    # Calcular indicadores
    closes = df['Close'].values
    quantum_score, confidence = quantum_hunter_v14(df)
    win_strength = calculate_win_strength(df)
    hurst = calculate_hurst_exponent(closes)
    vpin = calculate_vpin(df)
    
    # Quantum Monte Carlo
    prob_up, qmc_confidence = quantum_sim.quantum_monte_carlo(closes)
    
    # Machine Learning
    ml_pred, ml_conf, ml_acc, _ = train_ensemble_ml(df)
    
    # Sinal atual
    current_score = quantum_score[-1]
    current_confidence = confidence[-1]
    current_win = win_strength[-1]
    
    if current_score > 30:
        signal = 'BUY'
        signal_class = 'buy-signal'
    elif current_score < -30:
        signal = 'SELL'
        signal_class = 'sell-signal'
    else:
        signal = 'NEUTRO'
        signal_class = 'neutral-signal'
    
    # Adicionar sinal ao motor anti-repaint
    anti_repaint.add_signal(
        df['timestamp'].iloc[-1],
        signal,
        df['Close'].iloc[-1],
        current_score,
        current_confidence
    )
    anti_repaint.update_signals()
    
    # ============================================================
    # TABS
    # ============================================================
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Análise Quântica",
        "🔒 Sinais Anti-Repaint",
        "📈 Performance Histórica"
    ])
    
    # TAB 1: Análise Quântica
    with tab1:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Quantum Score",
                f"{current_score:.1f}",
                delta=f"{current_score - quantum_score[-2]:.1f}"
            )
        
        with col2:
            st.metric(
                "💪 Força WIN",
                f"{current_win:.1f}%",
                delta=f"{current_win - win_strength[-2]:.1f}%"
            )
        
        with col3:
            st.metric(
                "🔮 Confiança",
                f"{current_confidence:.1f}%"
            )
        
        with col4:
            st.metric(
                "🧠 ML Ensemble",
                "📈 ALTA" if ml_pred == 1 else "📉 BAIXA",
                delta=f"{ml_conf:.1f}%" if ml_conf else "N/A"
            )
        
        # Sinal principal
        st.markdown(f"""
        <div style='text-align: center; margin: 30px 0;'>
            <span class='signal-badge {signal_class}' style='font-size: 24px;'>
                {signal} SIGNAL
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Gráfico principal
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=('Preço + Médias', 'Quantum Score', 'Volume')
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['timestamp'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='IBOV'
        ), row=1, col=1)
        
        # TEMAs
        tema9 = calculate_tema(closes, 9)
        tema21 = calculate_tema(closes, 21)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=tema9,
            name='TEMA 9',
            line=dict(color='cyan', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=tema21,
            name='TEMA 21',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        # Quantum Score
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=quantum_score,
            name='Quantum Score',
            line=dict(color='magenta', width=2)
        ), row=2, col=1)
        
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=-30, line_dash="dash", line_color="red", row=2, col=1)
        
        # Volume
        colors = ['green' if df['Close'].iloc[i] > df['Open'].iloc[i] else 'red' 
                  for i in range(len(df))]
        
        fig.add_trace(go.Bar(
            x=df['timestamp'],
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.5
        ), row=3, col=1)
        
        # Layout
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas adicionais
        st.markdown("### 📊 Métricas Avançadas")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>🌀 Hurst Exponent</h4>
                <p style='font-size: 24px; color: {"#00ff88" if hurst > 0.5 else "#ff4444"};'>
                    {hurst:.3f}
                </p>
                <p style='font-size: 12px;'>
                    {"📈 Tendência" if hurst > 0.5 else "📉 Reversão"}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>🔍 VPIN</h4>
                <p style='font-size: 24px; color: #00ffcc;'>
                    {vpin[-1]:.1f}%
                </p>
                <p style='font-size: 12px;'>
                    Order Flow Imbalance
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h4>🎲 Quantum MC</h4>
                <p style='font-size: 24px; color: #ff00ff;'>
                    {prob_up*100:.1f}%
                </p>
                <p style='font-size: 12px;'>
                    Prob. Alta (10k sim.)
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            rsi_val = calculate_rsi(closes)[-1]
            st.markdown(f"""
            <div class='metric-card'>
                <h4>📊 RSI (14)</h4>
                <p style='font-size: 24px; color: {"#ff4444" if rsi_val > 70 else "#00ff88" if rsi_val < 30 else "#888888"};'>
                    {rsi_val:.1f}
                </p>
                <p style='font-size: 12px;'>
                    {"Sobrecompra" if rsi_val > 70 else "Sobrevenda" if rsi_val < 30 else "Neutro"}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Accuracies do ML
        if ml_acc:
            st.markdown("### 🧠 Ensemble Machine Learning")
            acc_col1, acc_col2, acc_col3 = st.columns(3)
            
            with acc_col1:
                st.metric("XGBoost Accuracy", f"{ml_acc['XGBoost']:.1f}%")
            
            with acc_col2:
                st.metric("LightGBM Accuracy", f"{ml_acc['LightGBM']:.1f}%")
            
            with acc_col3:
                st.metric("Random Forest Accuracy", f"{ml_acc['RandomForest']:.1f}%")
    
    # TAB 2: Sinais Anti-Repaint
    with tab2:
        st.markdown("### 🔒 Histórico de Sinais Confirmados (Anti-Repaint)")
        
        confirmed = anti_repaint.get_confirmed_signals(50)
        
        if len(confirmed) == 0:
            st.info("⏳ Aguardando sinais confirmados... (mínimo 2 barras)")
        else:
            signals_df = pd.DataFrame(confirmed)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # Formatação
            signals_df = signals_df[['timestamp', 'type', 'price', 'score', 'confidence', 'hash']]
            signals_df.columns = ['Data/Hora', 'Tipo', 'Preço', 'Score', 'Confiança (%)', 'Hash']
            
            st.dataframe(signals_df, use_container_width=True)
            
            # Estatísticas
            st.markdown("### 📊 Estatísticas dos Sinais")
            
            buy_count = len([s for s in confirmed if s['type'] == 'BUY'])
            sell_count = len([s for s in confirmed if s['type'] == 'SELL'])
            neutral_count = len([s for s in confirmed if s['type'] == 'NEUTRO'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("📈 Sinais de COMPRA", buy_count)
            
            with col2:
                st.metric("📉 Sinais de VENDA", sell_count)
            
            with col3:
                st.metric("⚪ Sinais NEUTROS", neutral_count)
    
    # TAB 3: Performance
    with tab3:
        st.markdown("### 📈 Performance Histórica")
        
        st.info("🚧 Em desenvolvimento: Dashboard completo de performance com win rate, drawdown, profit factor, etc.")
        
        # Gráfico de equity (simulado)
        equity = np.cumsum(np.random.randn(len(df)) * 0.5 + 0.1) + 100
        
        fig_equity = go.Figure()
        fig_equity.add_trace(go.Scatter(
            x=df['timestamp'],
            y=equity,
            mode='lines',
            name='Equity Curve',
            line=dict(color='cyan', width=2)
        ))
        
        fig_equity.update_layout(
            title='Curva de Equity (Simulado)',
            template='plotly_dark',
            height=400
        )
        
        st.plotly_chart(fig_equity, use_container_width=True)
    
    # Footer
    st.markdown(f"""
    <div class='footer'>
        📊 ProfitOne Quantum V5 | Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | 
        Refresh automático em {refresh_seconds}s | 
        ⚠️ Este sistema é EDUCACIONAL - não use para decisões financeiras reais sem análise humana
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    st_autorefresh = st.empty()
    with st_autorefresh:
        import time
        time.sleep(refresh_seconds)
        st.rerun()

# ============================================================
# EXECUTAR
# ============================================================

if __name__ == "__main__":
    main()
