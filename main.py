import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.stats import entropy
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="ProfitOne IBOVESPA", layout="wide", page_icon="📊")

# CSS
st.markdown("""
<style>
    .main {background-color: #000000;}
    h1, h2, h3 {color: #00d9ff; text-shadow: 0 0 20px #00d9ff;}
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
        animation: pulse 2s infinite;
    }
    .signal-up {
        background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
        color: #000;
    }
    .signal-down {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        color: #fff;
    }
    @keyframes pulse {
        0%, 100% {transform: scale(1);}
        50% {transform: scale(1.02);}
    }
    .history-card {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# INDICADORES
# ==========================================

def tema(close, period=20):
    """TEMA + Velocity"""
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3
    velocity = tema.diff()
    return tema, velocity

def kalman_filter(data):
    """Kalman Filter"""
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
    """Entropia de Shannon"""
    def calc_ent(x):
        if len(x) < 2: return 0
        ret = np.diff(x)
        hist, _ = np.histogram(ret, bins=10, density=True)
        hist = hist[hist > 0]
        return entropy(hist, base=2)
    
    return data.rolling(window=window).apply(calc_ent, raw=False)

def fisher_transform(high, low, period=10):
    """Fisher Transform"""
    hl2 = (high + low) / 2
    max_h = hl2.rolling(period).max()
    min_l = hl2.rolling(period).min()
    
    value = 2 * ((hl2 - min_l) / (max_h - min_l + 1e-10) - 0.5)
    value = value.clip(-0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + value) / (1 - value + 1e-10))
    return fisher

def hurst_exponent(data, window=100):
    """Hurst Exponent"""
    def calc_h(ts):
        if len(ts) < 10: return 0.5
        lags = range(2, min(20, len(ts)//2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    return data.rolling(window=window).apply(calc_h, raw=False)

def z_score(data, window=20):
    """Z-Score"""
    mean = data.rolling(window).mean()
    std = data.rolling(window).std()
    return (data - mean) / (std + 1e-10)

# ==========================================
# BUSCAR DADOS
# ==========================================

@st.cache_data(ttl=60)
def get_ibovespa_data(period="5d", interval="5m"):
    """Buscar dados do IBOVESPA"""
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/^BVSP"
        params = {"interval": interval, "range": period}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return None
        
        data = response.json()
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
        
        return df
    
    except Exception as e:
        st.error(f"Erro: {str(e)}")
        return None

# ==========================================
# CALCULAR SCORE HISTÓRICO
# ==========================================

def calculate_historical_scores(df):
    """Calcular scores para todo o histórico"""
    
    # Calcular todos os indicadores
    tema_series, tema_vel = tema(df['close'])
    kalman_series = kalman_filter(df['close'])
    entropy_series = shannon_entropy(df['close'])
    fisher_series = fisher_transform(df['high'], df['low'])
    hurst_series = hurst_exponent(df['close'])
    z_score_series = z_score(df['close'])
    
    # Criar DataFrame com scores
    scores = pd.DataFrame(index=df.index)
    scores['price'] = df['close']
    scores['score'] = 0.0
    scores['signal'] = 'NEUTRO'
    
    for i in range(len(df)):
        score = 0
        count = 0
        
        # TEMA Velocity
        if not pd.isna(tema_vel.iloc[i]):
            score += 15 if tema_vel.iloc[i] > 0 else -15
            count += 1
        
        # Hurst
        if not pd.isna(hurst_series.iloc[i]):
            h = hurst_series.iloc[i]
            score += 20 if h > 0.5 else -10
            count += 1
        
        # Fisher
        if not pd.isna(fisher_series.iloc[i]):
            f = fisher_series.iloc[i]
            if f > 2:
                score -= 15
            elif f < -2:
                score += 15
            count += 1
        
        # Entropy
        if not pd.isna(entropy_series.iloc[i]):
            if entropy_series.iloc[i] < 1.5:
                score += 10
            count += 1
        
        if count > 0:
            scores.loc[scores.index[i], 'score'] = score / count
        
        # Classificar sinal
        final_score = scores.loc[scores.index[i], 'score']
        if final_score > 5:
            scores.loc[scores.index[i], 'signal'] = 'COMPRA'
        elif final_score < -5:
            scores.loc[scores.index[i], 'signal'] = 'VENDA'
        else:
            scores.loc[scores.index[i], 'signal'] = 'NEUTRO'
    
    return scores

# ==========================================
# CALCULAR PERFORMANCE
# ==========================================

def calculate_performance(scores, df):
    """Calcular acurácia dos sinais"""
    
    # Criar cópia do DataFrame
    performance = scores.copy()
    performance['next_price'] = df['close'].shift(-1)
    performance['price_change'] = ((performance['next_price'] - performance['price']) / performance['price'] * 100)
    
    # Verificar acertos
    performance['correct'] = False
    
    for i in range(len(performance) - 1):
        signal = performance['signal'].iloc[i]
        change = performance['price_change'].iloc[i]
        
        if pd.notna(change):
            if signal == 'COMPRA' and change > 0:
                performance.loc[performance.index[i], 'correct'] = True
            elif signal == 'VENDA' and change < 0:
                performance.loc[performance.index[i], 'correct'] = True
            elif signal == 'NEUTRO':
                performance.loc[performance.index[i], 'correct'] = True  # Neutro sempre certo
    
    return performance

# ==========================================
# INTERFACE
# ==========================================

st.markdown("<h1 style='text-align: center;'>📊 PROFITONE - IBOVESPA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Sistema Completo com Histórico de Performance</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configurações")

# Período histórico
period_map = {
    "1 dia": "1d",
    "5 dias": "5d",
    "1 mês": "1mo",
    "3 meses": "3mo",
    "6 meses": "6mo",
    "1 ano": "1y"
}

period_label = st.sidebar.selectbox("Período Histórico:", list(period_map.keys()), index=2)
period = period_map[period_label]

# Timeframe
timeframe_map = {
    "1 minuto": "1m",
    "5 minutos": "5m",
    "15 minutos": "15m",
    "30 minutos": "30m",
    "1 hora": "1h",
    "1 dia": "1d"
}

timeframe_label = st.sidebar.selectbox("Timeframe:", list(timeframe_map.keys()), index=1)
timeframe = timeframe_map[timeframe_label]

# Indicadores
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Indicadores")

show_tema = st.sidebar.checkbox("TEMA + Velocity", value=True)
show_kalman = st.sidebar.checkbox("Kalman Filter", value=True)
show_entropy = st.sidebar.checkbox("Entropia Shannon", value=True)
show_fisher = st.sidebar.checkbox("Fisher Transform", value=True)
show_hurst = st.sidebar.checkbox("Hurst Exponent", value=True)

# Abas
tab1, tab2, tab3 = st.tabs(["📊 Tempo Real", "📈 Histórico de Sinais", "📉 Performance"])

# ==========================================
# ABA 1: TEMPO REAL
# ==========================================

with tab1:
    
    placeholder_live = st.empty()
    
    with placeholder_live.container():
        
        df = get_ibovespa_data(period="1d", interval=timeframe)
        
        if df is not None and len(df) > 50:
            
            now = datetime.now().strftime("%H:%M:%S")
            st.markdown(f"<p style='text-align: center; color: #888;'>🕐 {now} | ✅ {len(df)} candles</p>", unsafe_allow_html=True)
            
            # Calcular indicadores
            indicators = {}
            
            if show_tema:
                indicators['tema'], indicators['tema_vel'] = tema(df['close'])
            
            if show_kalman:
                indicators['kalman'] = kalman_filter(df['close'])
            
            if show_entropy:
                indicators['entropy'] = shannon_entropy(df['close'])
            
            if show_fisher:
                indicators['fisher'] = fisher_transform(df['high'], df['low'])
            
            if show_hurst:
                indicators['hurst'] = hurst_exponent(df['close'])
            
            indicators['z_score'] = z_score(df['close'])
            
            # SCORE ATUAL
            score = 0
            count = 0
            
            if 'tema_vel' in indicators and not pd.isna(indicators['tema_vel'].iloc[-1]):
                score += 15 if indicators['tema_vel'].iloc[-1] > 0 else -15
                count += 1
            
            if 'hurst' in indicators and not pd.isna(indicators['hurst'].iloc[-1]):
                h = indicators['hurst'].iloc[-1]
                score += 20 if h > 0.5 else -10
                count += 1
            
            if 'fisher' in indicators and not pd.isna(indicators['fisher'].iloc[-1]):
                f = indicators['fisher'].iloc[-1]
                if f > 2:
                    score -= 15
                elif f < -2:
                    score += 15
                count += 1
            
            if 'entropy' in indicators and not pd.isna(indicators['entropy'].iloc[-1]):
                if indicators['entropy'].iloc[-1] < 1.5:
                    score += 10
                count += 1
            
            if count > 0:
                score = score / count
            
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
                if 'hurst' in indicators:
                    h = indicators['hurst'].iloc[-1]
                    regime = "Tendência" if h > 0.5 else "Lateral"
                    st.metric("📊 Regime", regime, f"H: {h:.2f}")
            
            with col3:
                if 'entropy' in indicators:
                    ent = indicators['entropy'].iloc[-1]
                    st.metric("⚛️ Entropia", f"{ent:.2f}")
            
            with col4:
                vol = df['volume'].iloc[-1]
                st.metric("📦 Volume", f"{vol/1e6:.0f}M")
            
            # GRÁFICO
            st.markdown("---")
            st.subheader("📊 Gráfico em Tempo Real")
            
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.25, 0.25],
                subplot_titles=('IBOVESPA', 'Oscillators', 'Regime')
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
            
            if 'tema' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['tema'],
                    name='TEMA', line=dict(color='#00d9ff', width=2)
                ), row=1, col=1)
            
            if 'kalman' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['kalman'],
                    name='Kalman', line=dict(color='#ff00ff', width=2)
                ), row=1, col=1)
            
            if 'fisher' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['fisher'],
                    name='Fisher', line=dict(color='#00d9ff')
                ), row=2, col=1)
                fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=-2, line_dash="dash", line_color="green", row=2, col=1)
            
            if 'hurst' in indicators:
                fig.add_trace(go.Scatter(
                    x=df.index, y=indicators['hurst'],
                    name='Hurst', line=dict(color='#00ff88')
                ), row=3, col=1)
                fig.add_hline(y=0.5, line_dash="dash", line_color="white", row=3, col=1)
            
            fig.update_layout(
                template='plotly_dark',
                height=800,
                xaxis_rangeslider_visible=False,
                plot_bgcolor='#000000',
                paper_bgcolor='#000000'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("❌ Erro ao carregar dados do IBOVESPA")

# ==========================================
# ABA 2: HISTÓRICO DE SINAIS
# ==========================================

with tab2:
    
    st.subheader("📈 Histórico de Sinais e Scores")
    
    df_hist = get_ibovespa_data(period=period, interval=timeframe)
    
    if df_hist is not None and len(df_hist) > 50:
        
        with st.spinner("Calculando histórico..."):
            scores_hist = calculate_historical_scores(df_hist)
        
        st.success(f"✅ {len(scores_hist)} pontos analisados no período de {period_label}")
        
        # Gráfico de Score histórico
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Scatter(
            x=scores_hist.index,
            y=scores_hist['score'],
            mode='lines',
            name='Score',
            line=dict(color='#00d9ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 217, 255, 0.2)'
        ))
        
        fig_hist.add_hline(y=5, line_dash="dash", line_color="green", annotation_text="Zona de Compra")
        fig_hist.add_hline(y=-5, line_dash="dash", line_color="red", annotation_text="Zona de Venda")
        fig_hist.add_hline(y=0, line_dash="dot", line_color="white")
        
        fig_hist.update_layout(
            template='plotly_dark',
            height=500,
            title="Evolução do Score Master ao Longo do Tempo",
            yaxis_title="Score",
            plot_bgcolor='#000000',
            paper_bgcolor='#000000'
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Tabela de sinais recentes
        st.markdown("---")
        st.subheader("📋 Últimos 50 Sinais")
        
        display_hist = scores_hist[['price', 'score', 'signal']].tail(50).copy()
        display_hist['price'] = display_hist['price'].apply(lambda x: f"{x:,.0f}")
        display_hist['score'] = display_hist['score'].apply(lambda x: f"{x:+.1f}")
        
        st.dataframe(display_hist, use_container_width=True)
    
    else:
        st.error("❌ Erro ao carregar histórico")

# ==========================================
# ABA 3: PERFORMANCE
# ==========================================

with tab3:
    
    st.subheader("📉 Análise de Performance")
    
    df_perf = get_ibovespa_data(period=period, interval=timeframe)
    
    if df_perf is not None and len(df_perf) > 50:
        
        with st.spinner("Analisando performance..."):
            scores_perf = calculate_historical_scores(df_perf)
            performance = calculate_performance(scores_perf, df_perf)
        
        # Estatísticas
        total_signals = len(performance[performance['signal'] != 'NEUTRO'])
        correct_signals = performance['correct'].sum()
        accuracy = (correct_signals / len(performance) * 100) if len(performance) > 0 else 0
        
        # Cards de estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total de Sinais", f"{total_signals}")
        
        with col2:
            st.metric("✅ Sinais Corretos", f"{correct_signals}")
        
        with col3:
            st.metric("🎯 Acurácia", f"{accuracy:.1f}%")
        
        with col4:
            compras = len(performance[performance['signal'] == 'COMPRA'])
            vendas = len(performance[performance['signal'] == 'VENDA'])
            st.metric("⚖️ Compras/Vendas", f"{compras}/{vendas}")
        
        st.markdown("---")
        
        # Distribuição de sinais
        st.subheader("📊 Distribuição de Sinais")
        
        signal_counts = performance['signal'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=signal_counts.index,
            values=signal_counts.values,
            marker=dict(colors=['#00ff88', '#ff4444', '#ffaa00']),
            textinfo='label+percent'
        )])
        
        fig_pie.update_layout(
            template='plotly_dark',
            height=400,
            paper_bgcolor='#000000'
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Tabela detalhada
        st.markdown("---")
        st.subheader("📋 Detalhamento (últimos 100)")
        
        display_perf = performance[['price', 'score', 'signal', 'price_change', 'correct']].tail(100).copy()
        display_perf['price'] = display_perf['price'].apply(lambda x: f"{x:,.0f}")
        display_perf['score'] = display_perf['score'].apply(lambda x: f"{x:+.1f}")
        display_perf['price_change'] = display_perf['price_change'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "-")
        display_perf['correct'] = display_perf['correct'].apply(lambda x: "✅" if x else "❌")
        
        st.dataframe(display_perf, use_container_width=True)
    
    else:
        st.error("❌ Erro ao carregar dados para análise")

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #444;'>📊 ProfitOne IBOVESPA | Sistema Profissional de Trading</div>", unsafe_allow_html=True)
