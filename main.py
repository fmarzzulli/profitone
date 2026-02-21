import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIG =========================
st.set_page_config(
    page_title="ProfitOne Quantum V8",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CSS =========================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        text-shadow: 0 0 10px rgba(0,255,204,0.5);
    }
    .signal-badge {
        padding: 15px 30px;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        font-size: 1.5rem;
        margin: 10px 0;
    }
    .buy { background: linear-gradient(135deg, #00ff88, #00cc66); color: #000; }
    .sell { background: linear-gradient(135deg, #ff4444, #cc0000); color: #fff; }
    .neutral { background: linear-gradient(135deg, #888, #666); color: #fff; }
</style>
""", unsafe_allow_html=True)

# ========================= FUNCTIONS =========================
def calculate_ema(prices, period):
    """Calcula EMA usando pandas"""
    return prices.ewm(span=period, adjust=False).mean()

def calculate_rsi(prices, period=14):
    """Calcula RSI"""
    if len(prices) < period + 1:
        return 50.0
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else 50.0

def quantum_score(df):
    """Calcula score quântico baseado em EMA e RSI"""
    if len(df) < 21:
        return 0.0
    
    close = df['close']
    ema9 = calculate_ema(close, 9).iloc[-1]
    ema21 = calculate_ema(close, 21).iloc[-1]
    rsi = calculate_rsi(close, 14)
    
    # Tendência
    trend = 50 if ema9 > ema21 else -50
    
    # Momentum
    momentum = (rsi - 50)
    
    # Score final
    score = trend + momentum
    return np.clip(score, -100, 100)

class AntiRepaint:
    """Motor anti-repaint com confirmação de barras"""
    def __init__(self, confirmation_bars=2):
        self.signals = []
        self.confirmation_bars = confirmation_bars
    
    def add_signal(self, timestamp, signal_type, price, score):
        """Adiciona novo sinal"""
        signal_hash = hashlib.sha256(
            f"{timestamp}{signal_type}{price}".encode()
        ).hexdigest()[:8]
        
        self.signals.append({
            'timestamp': timestamp,
            'type': signal_type,
            'price': price,
            'score': score,
            'hash': signal_hash,
            'bars_held': 0,
            'confirmed': False
        })
    
    def update_signals(self):
        """Atualiza e confirma sinais"""
        for sig in self.signals:
            if not sig['confirmed']:
                sig['bars_held'] += 1
                if sig['bars_held'] >= self.confirmation_bars:
                    sig['confirmed'] = True
    
    def get_confirmed(self):
        """Retorna apenas sinais confirmados"""
        return [s for s in self.signals if s['confirmed']]

@st.cache_data(ttl=60)
def get_data(symbol, interval='5m'):
    """Busca dados do Yahoo Finance"""
    try:
        interval_map = {
            '5m': '5m',
            '15m': '15m',
            '1h': '60m'
        }
        yf_interval = interval_map.get(interval, '5m')
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            'interval': yf_interval,
            'range': '5d'
        }
        
        response = requests.get(url, params=params, timeout=10)
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
        return df
    
    except Exception as e:
        st.error(f"Erro ao buscar dados: {str(e)}")
        return pd.DataFrame()

# ========================= MAIN APP =========================
def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/clouds/200/000000/rocket.png", width=150)
    st.sidebar.title("⚙️ Configurações")
    
    mode = st.sidebar.selectbox(
        "Modo de Operação",
        ["Scalp (5min)", "Day Trade (15min)", "Swing (1h)"]
    )
    
    interval_map = {
        "Scalp (5min)": "5m",
        "Day Trade (15min)": "15m",
        "Swing (1h)": "1h"
    }
    interval = interval_map[mode]
    
    confirmation_bars = st.sidebar.slider(
        "Barras de Confirmação",
        min_value=1,
        max_value=5,
        value=2
    )
    
    # Botão de refresh manual
    if st.sidebar.button("🔄 Atualizar Dados", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Header
    st.title("🚀 ProfitOne Quantum V8")
    st.markdown("**Motor Anti-Repaint | Quantum Score | Real-Time Analysis**")
    
    # Carrega dados
    symbol = "^BVSP"
    df = get_data(symbol, interval)
    
    if df.empty or len(df) < 50:
        st.error("❌ Dados insuficientes. Clique em 'Atualizar Dados' no sidebar.")
        st.stop()
    
    # Calcula indicadores
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    
    score = quantum_score(df)
    rsi = calculate_rsi(df['close'], 14)
    current_price = df['close'].iloc[-1]
    
    # Determina sinal
    if score > 30:
        signal = "COMPRA 🟢"
        signal_class = "buy"
    elif score < -30:
        signal = "VENDA 🔴"
        signal_class = "sell"
    else:
        signal = "NEUTRO ⚪"
        signal_class = "neutral"
    
    # Anti-repaint
    if 'anti_repaint' not in st.session_state:
        st.session_state.anti_repaint = AntiRepaint(confirmation_bars)
    
    ar = st.session_state.anti_repaint
    ar.confirmation_bars = confirmation_bars
    
    ar.add_signal(
        df['timestamp'].iloc[-1],
        signal,
        current_price,
        score
    )
    ar.update_signals()
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Quantum Score", f"{score:.1f}")
    
    with col2:
        st.metric("🎯 RSI (14)", f"{rsi:.1f}")
    
    with col3:
        st.metric("💰 Preço", f"R$ {current_price:,.2f}")
    
    with col4:
        st.markdown(f"""
        <div class="signal-badge {signal_class}">
            {signal}
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico
    st.markdown("---")
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("📈 Candlestick + EMAs", "📊 Quantum Score")
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Preço",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # EMAs
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ema9'],
            name="EMA 9",
            line=dict(color='cyan', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['ema21'],
            name="EMA 21",
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )
    
    # Quantum Score
    scores = [quantum_score(df.iloc[:i+1]) for i in range(21, len(df))]
    
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'].iloc[21:],
            y=scores,
            name="Quantum Score",
            line=dict(color='magenta', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,255,0.1)'
        ),
        row=2, col=1
    )
    
    # Linhas de referência
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=-30, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)
    
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(title_text="Tempo", row=2, col=1)
    fig.update_yaxes(title_text="Preço (R$)", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    
    st.plotly_chart(fig, width='stretch')
    
    # Sinais confirmados
    st.markdown("---")
    st.subheader("✅ Sinais Confirmados (Anti-Repaint)")
    
    confirmed = ar.get_confirmed()
    
    if confirmed:
        for sig in reversed(confirmed[-10:]):
            color = "#00ff88" if "COMPRA" in sig['type'] else "#ff4444" if "VENDA" in sig['type'] else "#888888"
            st.markdown(f"""
            <div style='background: linear-gradient(90deg, {color}22, transparent); 
                        padding: 10px; border-left: 4px solid {color}; margin: 5px 0; border-radius: 5px;'>
                <strong style='color: {color};'>{sig['type']}</strong> | 
                {sig['timestamp'].strftime('%d/%m/%Y %H:%M')} | 
                R$ {sig['price']:,.2f} | 
                Score: {sig['score']:.1f} | 
                Hash: <code>{sig['hash']}</code>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("⏳ Aguardando confirmação de sinais...")
    
    # Footer
    st.markdown("---")
    st.caption(
        f"⏰ Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
        f"📊 {len(df)} barras | "
        f"⚠️ Apenas para fins educacionais"
    )

if __name__ == "__main__":
    main()
