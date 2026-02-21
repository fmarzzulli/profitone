"""Configurações globais do ProfitOne Quantum V8"""

# Configurações de Trading
DEFAULT_SYMBOL = "^BVSP"
DEFAULT_INTERVAL = "15m"
DEFAULT_PERIOD = "5d"

# Timeframes disponíveis
TIMEFRAMES = {
    "Scalp 5m": "5m",
    "Day Trade 15m": "15m",
    "Swing 1h": "60m",
    "Swing 4h": "4h",
    "Position 1d": "1d"
}

# Configurações de indicadores
TEMA_PERIODS = [8, 21, 55]
EMA_PERIODS = [9, 21, 50, 200]
RSI_PERIOD = 14
ADX_PERIOD = 14
VORTEX_PERIOD = 14

# Configurações visuais
CHART_HEIGHT = 800
COLORS = {
    "buy": "#00ff88",
    "sell": "#ff4444",
    "neutral": "#ffaa00",
    "background": "#0e1117",
    "grid": "#1f2937"
}

# Anti-repaint
MIN_CONFIRMATION_BARS = 3
MAX_SIGNALS_DISPLAY = 10
