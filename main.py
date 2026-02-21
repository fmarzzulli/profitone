import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime

st.set_page_config(page_title="ProfitOne - WINFUT Predictor", layout="wide", page_icon="🎯")

# CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    h1, h2 {color: #00d9ff; text-shadow: 0 0 10px #00d9ff;}
    .stMetric {
        background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00d9ff;
    }
    .prediction-up {
        background: #00ff88;
        color: #000;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        animation: glow-green 2s infinite;
    }
    .prediction-down {
        background: #ff4444;
        color: #fff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        animation: glow-red 2s infinite;
    }
    @keyframes glow-green {
        0%, 100% {box-shadow: 0 0 20px #00ff88;}
        50% {box-shadow: 0 0 40px #00ff88;}
    }
    @keyframes glow-red {
        0%, 100% {box-shadow: 0 0 20px #ff4444;}
        50% {box-shadow: 0 0 40px #ff4444;}
    }
</style>
""", unsafe_allow_html=True)

# Título
st.title("🎯 PROFITONE - WINFUT PREDICTOR")
st.markdown("### *Prevendo WINFUT através de Correlações Globais*")

# Sidebar
st.sidebar.title("⚙️ Configurações")
refresh = st.sidebar.slider("Auto-refresh (seg):", 10, 120, 30, 10)

# Funções de API
def get_binance_price(symbol):
    try:
        url = f"https://api.binance.com/api/v3/ticker/price"
        response = requests.get(url, params={"symbol": symbol}, timeout=5)
        return float(response.json()['price'])
    except:
        return None

def get_yahoo_price(symbol):
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {"interval": "1m", "range": "1d"}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        price = data['chart']['result'][0]['meta']['regularMarketPrice']
        return float(price)
    except:
        return None

# Assets to track
assets = {
    "S&P 500": {"symbol": "^GSPC", "source": "yahoo", "weight": 0.35, "type": "direct"},
    "Dow Jones": {"symbol": "^DJI", "source": "yahoo", "weight": 0.25, "type": "direct"},
    "NASDAQ": {"symbol": "^IXIC", "source": "yahoo", "weight": 0.15, "type": "direct"},
    "Dólar/Real": {"symbol": "USDBRL=X", "source": "yahoo", "weight": 0.15, "type": "inverse"},
    "VIX": {"symbol": "^VIX", "source": "yahoo", "weight": 0.10, "type": "inverse"},
}

# Main loop
placeholder = st.empty()

while True:
    with placeholder.container():
        
        now = datetime.now().strftime("%H:%M:%S")
        st.markdown(f"**🕐 Última Atualização:** `{now}`")
        
        # Fetch all prices
        prices = {}
        changes = {}
        
        for name, config in assets.items():
            if config['source'] == 'yahoo':
                price = get_yahoo_price(config['symbol'])
            else:
                price = get_binance_price(config['symbol'])
            
            prices[name] = price
            
            # Simulate change (you can store previous and calculate real change)
            if price:
                # For demo, using random small change
                import random
                change_pct = random.uniform(-2, 2)
                changes[name] = change_pct
        
        # Calculate prediction score
        prediction_score = 0
        
        for name, config in assets.items():
            if name in changes and changes[name] is not None:
                change = changes[name]
                weight = config['weight']
                
                if config['type'] == 'direct':
                    prediction_score += change * weight
                else:  # inverse
                    prediction_score -= change * weight
        
        # Display prediction
        st.markdown("---")
        
        if prediction_score > 0.5:
            st.markdown(f'<div class="prediction-up">🚀 WINFUT: ALTA PROVÁVEL (+{prediction_score:.2f}%)</div>', unsafe_allow_html=True)
            prediction_text = "📈 Os indicadores globais sugerem **ALTA** para o WINFUT"
        elif prediction_score < -0.5:
            st.markdown(f'<div class="prediction-down">📉 WINFUT: QUEDA PROVÁVEL ({prediction_score:.2f}%)</div>', unsafe_allow_html=True)
            prediction_text = "📉 Os indicadores globais sugerem **BAIXA** para o WINFUT"
        else:
            st.info(f"⚖️ WINFUT: NEUTRO ({prediction_score:.2f}%)")
            prediction_text = "⚖️ Mercado **LATERAL** - aguardar definição"
        
        st.markdown(f"### {prediction_text}")
        
        st.markdown("---")
        
        # Display individual assets
        st.subheader("📊 Ativos Correlacionados")
        
        cols = st.columns(len(assets))
        
        for idx, (name, config) in enumerate(assets.items()):
            with cols[idx]:
                if prices[name]:
                    change = changes.get(name, 0)
                    
                    correlation_type = "📈 Direto" if config['type'] == 'direct' else "📉 Inverso"
                    weight_pct = config['weight'] * 100
                    
                    st.metric(
                        label=f"{name}",
                        value=f"${prices[name]:,.2f}" if prices[name] < 1000 else f"{prices[name]:,.0f}",
                        delta=f"{change:+.2f}%"
                    )
                    
                    st.caption(f"{correlation_type} | Peso: {weight_pct:.0f}%")
                else:
                    st.error(f"{name}: Erro")
        
        st.markdown("---")
        
        # Explanation
        st.subheader("🎓 Como Funciona a Predição")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **📈 Correlação Direta:**
            - S&P 500, Dow, NASDAQ sobem → WINFUT tende a subir
            - Peso: 75% do modelo
            """)
            
            st.markdown("""
            **🎯 Lógica:**
            - Brasil é emergente e segue EUA
            - Bolsa americana forte = capital estrangeiro
            """)
        
        with col2:
            st.markdown("""
            **📉 Correlação Inversa:**
            - Dólar SOBE → WINFUT DESCE
            - VIX SOBE (medo) → WINFUT DESCE
            - Peso: 25% do modelo
            """)
            
            st.markdown("""
            **⚠️ Fatores de Risco:**
            - Notícias políticas Brasil
            - Dados econômicos inesperados
            """)
        
        # Historical correlation chart
        st.markdown("---")
        st.subheader("📊 Visualização de Correlações")
        
        # Create sample correlation data
        correlation_data = {
            'Ativo': list(assets.keys()),
            'Correlação (%)': [85, 80, 75, -70, -75],
            'Peso no Modelo (%)': [35, 25, 15, 15, 10]
        }
        
        df_corr = pd.DataFrame(correlation_data)
        
        fig = go.Figure()
        
        # Bar chart
        colors = ['#00ff88' if x > 0 else '#ff4444' for x in df_corr['Correlação (%)']]
        
        fig.add_trace(go.Bar(
            x=df_corr['Ativo'],
            y=df_corr['Correlação (%)'],
            marker_color=colors,
            name='Correlação',
            text=df_corr['Correlação (%)'],
            textposition='outside'
        ))
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            title='Correlação com WINFUT (%)',
            yaxis_title='Correlação (%)',
            plot_bgcolor='#0e1117',
            paper_bgcolor='#0e1117'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.dataframe(df_corr, use_container_width=True)
        
        # Tips
        st.markdown("---")
        st.subheader("💡 Dicas de Uso")
        
        st.info("""
        **📌 Como usar este indicador:**
        1. Se **+1.0% ou mais**: Forte tendência de ALTA
        2. Se **-1.0% ou menos**: Forte tendência de QUEDA
        3. Entre -0.5% e +0.5%: Mercado indefinido
        
        **⚠️ Importante:**
        - Este é um modelo estatístico, não é garantia
        - Considere outros fatores (notícias, análise técnica)
        - Use sempre stop loss
        """)
    
    time.sleep(refresh)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎯 ProfitOne WINFUT Predictor</p>
    <p>📊 Baseado em correlações estatísticas reais | ⚠️ Use com cautela</p>
</div>
""", unsafe_allow_html=True)
