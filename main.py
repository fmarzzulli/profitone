"""
ProfitOne Quantum V8 - Sistema de Monitoramento Avançado
Sistema completo de análise técnica com indicadores de ponta
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Imports dos módulos
from modules.utils import (
    get_market_data, 
    AntiRepaintEngine, 
    format_signal_badge,
    calculate_ema,
    calculate_rsi
)
from modules.kinematics import (
    tema_velocity_signal,
    tema_entropy_signal,
    kalman_zscore_signal,
    jma_vortex_signal
)
from modules.thermodynamics import (
    entropy_bollinger_keltner_signal,
    vortex_adx_signal,
    reynolds_fvg_signal
)
from modules.statistics import (
    fisher_hurst_signal,
    zscore_vpin_signal,
    linear_regression_signal
)
from modules.microstructure import (
    vpin_wick_analysis,
    trapped_traders_signal,
    fvg_fibonacci_signal,
    synthetic_delta_signal
)
from modules.chaos import (
    hurst_bollinger_signal,
    laguerre_rsi_signal,
    cog_kalman_signal,
    vscore_vwap_signal
)
import config

# ============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="ProfitOne Quantum V8",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS CUSTOMIZADO
# ============================================================================

st.markdown("""
<style>
    /* Fundo escuro gradiente */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
    }
    
    /* Títulos */
    h1, h2, h3 {
        color: #00ff88 !important;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: bold !important;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Botões */
    .stButton > button {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 10px 20px;
        color: #00ff88;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%);
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

@st.cache_data(ttl=60)
def fetch_data(symbol: str, period: str, interval: str):
    """Busca dados com cache"""
    return get_market_data(symbol, period, interval)


def calculate_master_score(signals: dict) -> tuple:
    """
    Calcula score mestre a partir de todos os sinais
    Returns: (master_score, master_signal, signal_breakdown)
    """
    scores = []
    weights = {
        'kinematics': 1.2,      # Alta importância para momentum
        'thermodynamics': 1.0,  # Média importância para regime
        'statistics': 1.1,      # Alta importância para probabilidade
        'microstructure': 1.3,  # Muito alta para institutional flow
        'chaos': 0.9            # Menor peso para indicadores complexos
    }
    
    signal_breakdown = {}
    
    for category, category_signals in signals.items():
        category_weight = weights.get(category, 1.0)
        
        for signal_name, signal_data in category_signals.items():
            if 'score' in signal_data:
                weighted_score = signal_data['score'] * category_weight
                scores.append(weighted_score)
                signal_breakdown[signal_name] = {
                    'score': signal_data['score'],
                    'weighted_score': weighted_score,
                    'signal': signal_data.get('signal', 'NEUTRAL')
                }
    
    if scores:
        master_score = np.mean(scores)
    else:
        master_score = 0
    
    # Determinar sinal mestre
    if master_score > 30:
        master_signal = 'BUY'
    elif master_score < -30:
        master_signal = 'SELL'
    else:
        master_signal = 'NEUTRAL'
    
    return master_score, master_signal, signal_breakdown


def create_main_chart(df: pd.DataFrame, signals: dict, master_score: float):
    """Cria gráfico principal com candlesticks e indicadores"""
    
    # Criar subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.2, 0.15],
        subplot_titles=('Preço & Indicadores', 'Master Score', 'RSI & Fisher', 'VPIN & Volume')
    )
    
    # ========== ROW 1: CANDLESTICK ==========
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Preço',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # EMAs
    ema9 = calculate_ema(df['close'], 9)
    ema21 = calculate_ema(df['close'], 21)
    ema50 = calculate_ema(df['close'], 50)
    
    fig.add_trace(go.Scatter(x=df.index, y=ema9, name='EMA 9', line=dict(color='cyan', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema21, name='EMA 21', line=dict(color='orange', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=ema50, name='EMA 50', line=dict(color='purple', width=2)), row=1, col=1)
    
    # VWAP
    if 'vscore_vwap' in signals['chaos']:
        vwap_data = signals['chaos']['vscore_vwap']
        # Recalcular VWAP para plotar
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tpv = (typical_price * df['volume']).rolling(window=20).sum()
        cumulative_volume = df['volume'].rolling(window=20).sum()
        vwap = cumulative_tpv / (cumulative_volume + 1e-10)
        
        fig.add_trace(go.Scatter(x=df.index, y=vwap, name='VWAP', line=dict(color='yellow', width=2, dash='dot')), row=1, col=1)
    
    # ========== ROW 2: MASTER SCORE ==========
    master_score_series = pd.Series([master_score] * len(df), index=df.index)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=master_score_series,
            name='Master Score',
            line=dict(color='#00ff88' if master_score > 0 else '#ff4444', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.2)' if master_score > 0 else 'rgba(255, 68, 68, 0.2)'
        ),
        row=2, col=1
    )
    
    # Linhas de referência
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=-30, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
    
    # ========== ROW 3: RSI & FISHER ==========
    rsi = calculate_rsi(df['close'], 14)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple', width=2)), row=3, col=1)
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    
    # ========== ROW 4: VOLUME ==========
    colors = ['#00ff88' if df['close'].iloc[i] >= df['open'].iloc[i] else '#ff4444' for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=4, col=1
    )
    
    # ========== LAYOUT ==========
    fig.update_layout(
        height=config.CHART_HEIGHT,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.3)'
    )
    
    # Remover labels duplicados dos eixos x
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_xaxes(title_text="", row=2, col=1)
    fig.update_xaxes(title_text="", row=3, col=1)
    fig.update_xaxes(title_text="Data/Hora", row=4, col=1)
    
    fig.update_yaxes(title_text="Preço", row=1, col=1)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    return fig


# ============================================================================
# APLICAÇÃO PRINCIPAL
# ============================================================================

def main():
    # ========== HEADER ==========
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>🚀 PROFITONE QUANTUM V8 🚀</h1>
        <p style='font-size: 18px; color: #00ff88;'>Sistema Avançado de Monitoramento Multi-Indicador</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("## ⚙️ Configurações")
        
        # Símbolo
        symbol = st.text_input("📈 Símbolo", value=config.DEFAULT_SYMBOL)
        
        # Timeframe
        selected_timeframe = st.selectbox(
            "⏱️ Timeframe",
            list(config.TIMEFRAMES.keys()),
            index=1
        )
        interval = config.TIMEFRAMES[selected_timeframe]
        
        # Período
        period_options = {
            "1 dia": "1d",
            "5 dias": "5d",
            "1 mês": "1mo",
            "3 meses": "3mo",
            "6 meses": "6mo"
        }
        selected_period = st.selectbox("📅 Período", list(period_options.keys()), index=1)
        period = period_options[selected_period]
        
        # Anti-repaint
        st.markdown("### 🔒 Anti-Repaint")
        confirmation_bars = st.slider("Barras de Confirmação", 1, 10, config.MIN_CONFIRMATION_BARS)
        
        # Botões
        col1, col2 = st.columns(2)
        with col1:
            refresh_btn = st.button("🔄 Atualizar", use_container_width=True)
        with col2:
            if st.button("🗑️ Limpar Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache limpo!")
        
        # Legenda
        st.markdown("---")
        st.markdown("### 📊 Módulos Ativos")
        st.markdown("""
        - 🎯 **Cinemática**: TEMA, Kalman, JMA, Vortex
        - 🌡️ **Termodinâmica**: Entropy, ADX, Reynolds, FVG
        - 📈 **Estatística**: Fisher, Hurst, Z-Score, VPIN
        - 🔬 **Microestrutura**: Wicks, Trapped Traders, Delta
        - 🌀 **Caos**: Laguerre, COG, V-Score
        """)
    
    # ========== DADOS ==========
    with st.spinner("🔄 Carregando dados..."):
        df = fetch_data(symbol, period, interval)
    
    if df.empty:
        st.error("❌ Erro ao carregar dados. Verifique o símbolo e tente novamente.")
        return
    
    # Garantir que o índice seja datetime
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    elif 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    # ========== CÁLCULO DE SINAIS ==========
    with st.spinner("🧮 Calculando indicadores..."):
        signals = {
            'kinematics': {},
            'thermodynamics': {},
            'statistics': {},
            'microstructure': {},
            'chaos': {}
        }
        
        try:
            # Cinemática
            signals['kinematics']['tema_velocity'] = tema_velocity_signal(df)
            signals['kinematics']['tema_entropy'] = tema_entropy_signal(df)
            signals['kinematics']['kalman_zscore'] = kalman_zscore_signal(df)
            signals['kinematics']['jma_vortex'] = jma_vortex_signal(df)
            
            # Termodinâmica
            signals['thermodynamics']['entropy_bb_kc'] = entropy_bollinger_keltner_signal(df)
            signals['thermodynamics']['vortex_adx'] = vortex_adx_signal(df)
            signals['thermodynamics']['reynolds_fvg'] = reynolds_fvg_signal(df)
            
            # Estatística
            signals['statistics']['fisher_hurst'] = fisher_hurst_signal(df)
            signals['statistics']['zscore_vpin'] = zscore_vpin_signal(df)
            signals['statistics']['linear_regression'] = linear_regression_signal(df)
            
            # Microestrutura
            signals['microstructure']['vpin_wick'] = vpin_wick_analysis(df)
            signals['microstructure']['trapped_traders'] = trapped_traders_signal(df)
            signals['microstructure']['fvg_fibonacci'] = fvg_fibonacci_signal(df)
            signals['microstructure']['synthetic_delta'] = synthetic_delta_signal(df)
            
            # Caos
            signals['chaos']['hurst_bollinger'] = hurst_bollinger_signal(df)
            signals['chaos']['laguerre_rsi'] = laguerre_rsi_signal(df)
            signals['chaos']['cog_kalman'] = cog_kalman_signal(df)
            signals['chaos']['vscore_vwap'] = vscore_vwap_signal(df)
            
        except Exception as e:
            st.error(f"❌ Erro ao calcular indicadores: {e}")
            return
    
    # ========== MASTER SCORE ==========
    master_score, master_signal, signal_breakdown = calculate_master_score(signals)
    
    # ========== ANTI-REPAINT ==========
    if 'anti_repaint' not in st.session_state:
        st.session_state.anti_repaint = AntiRepaintEngine(confirmation_bars)
    
    # Adicionar sinal atual
    current_price = df['close'].iloc[-1]
    current_timestamp = df.index[-1]
    
    st.session_state.anti_repaint.add_signal(
        master_signal,
        current_price,
        current_timestamp,
        master_score
    )
    
    st.session_state.anti_repaint.update()
    confirmed_signals = st.session_state.anti_repaint.get_confirmed_signals(config.MAX_SIGNALS_DISPLAY)
    
    # ========== MÉTRICAS PRINCIPAIS ==========
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score_color = "normal" if -30 <= master_score <= 30 else ("off" if master_score < -30 else "normal")
        st.metric(
            "🎯 Master Score",
            f"{master_score:.1f}",
            delta=f"{master_score:.1f}",
            delta_color=score_color
        )
    
    with col2:
        rsi_current = calculate_rsi(df['close'], 14).iloc[-1]
        st.metric("📊 RSI (14)", f"{rsi_current:.1f}")
    
    with col3:
        st.metric("💰 Preço Atual", f"R$ {current_price:.2f}")
    
    with col4:
        st.markdown(f"### 🎪 Sinal: {format_signal_badge(master_signal)}", unsafe_allow_html=True)
    
    # ========== GRÁFICO PRINCIPAL ==========
    st.markdown("---")
    st.markdown("## 📈 Gráfico de Análise")
    
    fig = create_main_chart(df, signals, master_score)
    st.plotly_chart(fig, width='stretch')
    
    # ========== TABS DE ANÁLISE ==========
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Cinemática",
        "🌡️ Termodinâmica",
        "📈 Estatística",
        "🔬 Microestrutura",
        "🌀 Caos",
        "🔒 Sinais Confirmados"
    ])
    
    # TAB 1: Cinemática
    with tab1:
        st.markdown("### 🎯 Módulo de Cinemática & Velocidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### TEMA + Velocity")
            tv = signals['kinematics']['tema_velocity']
            st.write(f"**Sinal:** {tv['signal']}")
            st.write(f"**Score:** {tv['score']:.1f}")
            st.write(f"**TEMA:** {tv['tema']:.2f}")
            st.write(f"**Velocity:** {tv['velocity']:.4f}")
            
            st.markdown("#### Kalman + Z-Score")
            kz = signals['kinematics']['kalman_zscore']
            st.write(f"**Sinal:** {kz['signal']}")
            st.write(f"**Score:** {kz['score']:.1f}")
            st.write(f"**Fair Price:** {kz['fair_price']:.2f}")
            st.write(f"**Z-Score:** {kz['zscore']:.2f}")
        
        with col2:
            st.markdown("#### TEMA + Entropy")
            te = signals['kinematics']['tema_entropy']
            st.write(f"**Sinal:** {te['signal']}")
            st.write(f"**Score:** {te['score']:.1f}")
            st.write(f"**Regime:** {te['regime']}")
            st.write(f"**Entropy:** {te['entropy']:.3f}")
            
            st.markdown("#### JMA + Vortex")
            jv = signals['kinematics']['jma_vortex']
            st.write(f"**Sinal:** {jv['signal']}")
            st.write(f"**Score:** {jv['score']:.1f}")
            st.write(f"**VI+:** {jv['vi_plus']:.3f}")
            st.write(f"**VI-:** {jv['vi_minus']:.3f}")
    
    # TAB 2: Termodinâmica
    with tab2:
        st.markdown("### 🌡️ Módulo de Física & Termodinâmica")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Entropy + BB/KC")
            ebk = signals['thermodynamics']['entropy_bb_kc']
            st.write(f"**Sinal:** {ebk['signal']}")
            st.write(f"**Score:** {ebk['score']:.1f}")
            st.write(f"**Entropy:** {ebk['entropy']:.3f}")
            st.write(f"**Squeeze:** {'Sim' if ebk['squeeze'] else 'Não'}")
            st.write(f"**Breakout:** {ebk['breakout_direction']}")
            
            st.markdown("#### Reynolds + FVG")
            rf = signals['thermodynamics']['reynolds_fvg']
            st.write(f"**Sinal:** {rf['signal']}")
            st.write(f"**Score:** {rf['score']:.1f}")
            st.write(f"**Reynolds:** {rf['reynolds']:.1f}")
            st.write(f"**Fluxo:** {rf['flow_regime']}")
            st.write(f"**FVG:** {'Sim' if rf['fvg_detected'] else 'Não'}")
        
        with col2:
            st.markdown("#### Vortex + ADX")
            va = signals['thermodynamics']['vortex_adx']
            st.write(f"**Sinal:** {va['signal']}")
            st.write(f"**Score:** {va['score']:.1f}")
            st.write(f"**ADX:** {va['adx']:.1f}")
            st.write(f"**VI+:** {va['vi_plus']:.3f}")
            st.write(f"**VI-:** {va['vi_minus']:.3f}")
    
    # TAB 3: Estatística
    with tab3:
        st.markdown("### 📈 Módulo de Estatística & Probabilidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Fisher + Hurst")
            fh = signals['statistics']['fisher_hurst']
            st.write(f"**Sinal:** {fh['signal']}")
            st.write(f"**Score:** {fh['score']:.1f}")
            st.write(f"**Fisher:** {fh['fisher']:.3f}")
            st.write(f"**Hurst:** {fh['hurst']:.3f}")
            st.write(f"**Tipo de Mercado:** {fh['market_type']}")
            
            st.markdown("#### Regressão Linear")
            lr = signals['statistics']['linear_regression']
            st.write(f"**Sinal:** {lr['signal']}")
            st.write(f"**Score:** {lr['score']:.1f}")
            st.write(f"**R²:** {lr['r_squared']:.3f}")
            st.write(f"**Slope:** {lr['slope']:.4f}")
        
        with col2:
            st.markdown("#### Z-Score + VPIN")
            zv = signals['statistics']['zscore_vpin']
            st.write(f"**Sinal:** {zv['signal']}")
            st.write(f"**Score:** {zv['score']:.1f}")
            st.write(f"**Z-Score:** {zv['zscore']:.2f}")
            st.write(f"**VPIN:** {zv['vpin']:.3f}")
            st.write(f"**Anomalia:** {'Sim' if zv['anomaly_detected'] else 'Não'}")
    
    # TAB 4: Microestrutura
    with tab4:
        st.markdown("### 🔬 Módulo de Fluxo & Microestrutura")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### VPIN + Wicks")
            vw = signals['microstructure']['vpin_wick']
            st.write(f"**Sinal:** {vw['signal']}")
            st.write(f"**Score:** {vw['score']:.1f}")
            st.write(f"**VPIN:** {vw['vpin']:.3f}")
            st.write(f"**Upper Wick:** {vw['upper_wick_ratio']:.2%}")
            st.write(f"**Lower Wick:** {vw['lower_wick_ratio']:.2%}")
            
            st.markdown("#### FVG + Fibonacci")
            ff = signals['microstructure']['fvg_fibonacci']
            st.write(f"**Sinal:** {ff['signal']}")
            st.write(f"**Score:** {ff['score']:.1f}")
            st.write(f"**FVG:** {ff['fvg_type']}")
            if ff['fib_level']:
                st.write(f"**Fib 61.8%:** {ff['fib_level']:.2f}")
        
        with col2:
            st.markdown("#### Trapped Traders")
            tt = signals['microstructure']['trapped_traders']
            st.write(f"**Sinal:** {tt['signal']}")
            st.write(f"**Score:** {tt['score']:.1f}")
            st.write(f"**Longs Presos:** {'Sim' if tt['trapped_long'] else 'Não'}")
            st.write(f"**Shorts Presos:** {'Sim' if tt['trapped_short'] else 'Não'}")
            st.write(f"**Força:** {tt['trap_strength']:.2f}x")
            
            st.markdown("#### Delta Sintético")
            sd = signals['microstructure']['synthetic_delta']
            st.write(f"**Sinal:** {sd['signal']}")
            st.write(f"**Score:** {sd['score']:.1f}")
            st.write(f"**Delta:** {sd['delta']:.3f}")
            st.write(f"**Divergência:** {'Sim' if sd['divergence_detected'] else 'Não'}")
    
    # TAB 5: Caos
    with tab5:
        st.markdown("### 🌀 Módulo de Caos, Cibernética & Geometria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Hurst + Bollinger")
            hb = signals['chaos']['hurst_bollinger']
            st.write(f"**Sinal:** {hb['signal']}")
            st.write(f"**Score:** {hb['score']:.1f}")
            st.write(f"**Hurst:** {hb['hurst']:.3f}")
            st.write(f"**BB Position:** {hb['bb_position']:.2%}")
            st.write(f"**Mercado:** {hb['market_type']}")
            
            st.markdown("#### COG + Kalman")
            ck = signals['chaos']['cog_kalman']
            st.write(f"**Sinal:** {ck['signal']}")
            st.write(f"**Score:** {ck['score']:.1f}")
            st.write(f"**COG:** {ck['cog']:.3f}")
            st.write(f"**Kalman COG:** {ck['kalman_cog']:.3f}")
        
        with col2:
            st.markdown("#### Laguerre RSI")
            lr = signals['chaos']['laguerre_rsi']
            st.write(f"**Sinal:** {lr['signal']}")
            st.write(f"**Score:** {lr['score']:.1f}")
            st.write(f"**Laguerre RSI:** {lr['laguerre_rsi']:.3f}")
            
            st.markdown("#### V-Score (VWAP)")
            vs = signals['chaos']['vscore_vwap']
            st.write(f"**Sinal:** {vs['signal']}")
            st.write(f"**Score:** {vs['score']:.1f}")
            st.write(f"**VWAP:** {vs['vwap']:.2f}")
            st.write(f"**V-Score:** {vs['vscore']:.2f}")
            st.write(f"**Band Position:** {vs['band_position']:.2%}")
    
    # TAB 6: Sinais Confirmados
    with tab6:
        st.markdown("### 🔒 Sinais Confirmados (Anti-Repaint)")
        
        if confirmed_signals:
            for i, sig in enumerate(reversed(confirmed_signals), 1):
                col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**#{i}**")
                
                with col2:
                    badge_html = format_signal_badge(sig['type'])
                    st.markdown(badge_html, unsafe_allow_html=True)
                
                with col3:
                    st.write(f"💰 R$ {sig['price']:.2f}")
                
                with col4:
                    st.write(f"📊 Score: {sig['score']:.1f}")
                
                with col5:
                    st.write(f"🔐 {sig['hash']}")
                
                st.markdown(f"_📅 {sig['timestamp']}_")
                st.markdown("---")
        else:
            st.info("Aguardando confirmação de sinais...")
    
    # ========== FOOTER ==========
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(f"📅 Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    with col2:
        st.caption(f"📊 {len(df)} candles • {interval} • {symbol}")
    
    with col3:
        st.caption("🚀 ProfitOne Quantum V8 - All Rights Reserved")
    
    # Disclaimer
    st.markdown("""
    <div style='background: rgba(255, 68, 68, 0.1); padding: 15px; border-radius: 8px; border-left: 4px solid #ff4444; margin-top: 20px;'>
        <strong>⚠️ AVISO IMPORTANTE:</strong><br>
        Este sistema é apenas para fins educacionais e informativos. Não constitui recomendação de investimento.
        Operar no mercado financeiro envolve riscos. Consulte um profissional qualificado antes de tomar decisões.
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# EXECUTAR APP
# ============================================================================

if __name__ == "__main__":
    main()
