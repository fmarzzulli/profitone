"""
MÓDULO 4: FLUXO & MICROESTRUTURA
Tracking institucional e detecção de trapped traders
"""

import pandas as pd
import numpy as np


def vpin_wick_analysis(df: pd.DataFrame, vpin_window: int = 50) -> dict:
    """
    VPIN + Análise de Wicks (sombras de candles)
    Wicks grandes = rejeição de preço = possível reversão
    
    Returns:
        dict com 'signal', 'vpin', 'upper_wick_ratio', 'lower_wick_ratio', 'score'
    """
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # VPIN (simplificado)
    price_change = close.diff()
    buy_volume = volume.where(price_change > 0, 0)
    sell_volume = volume.where(price_change < 0, 0)
    
    volume_imbalance = np.abs(buy_volume - sell_volume)
    total_volume = volume
    
    vpin = (volume_imbalance.rolling(window=vpin_window).sum() / 
            total_volume.rolling(window=vpin_window).sum())
    
    current_vpin = vpin.iloc[-1]
    
    # Análise de Wicks
    body = np.abs(close - open_price)
    upper_wick = high - np.maximum(open_price, close)
    lower_wick = np.minimum(open_price, close) - low
    total_range = high - low
    
    # Ratios
    upper_wick_ratio = upper_wick / (total_range + 1e-10)
    lower_wick_ratio = lower_wick / (total_range + 1e-10)
    
    current_upper_wick_ratio = upper_wick_ratio.iloc[-1]
    current_lower_wick_ratio = lower_wick_ratio.iloc[-1]
    
    # Sinal combinado
    # Wick superior grande = rejeição de alta = bearish
    # Wick inferior grande = rejeição de baixa = bullish
    
    significant_upper_wick = current_upper_wick_ratio > 0.5
    significant_lower_wick = current_lower_wick_ratio > 0.5
    
    if significant_lower_wick and current_vpin < 0.5:
        signal = 'BUY'
        score = 65
    elif significant_upper_wick and current_vpin < 0.5:
        signal = 'SELL'
        score = -65
    elif current_vpin > 0.6:
        signal = 'NEUTRAL'
        score = 0
    else:
        signal = 'NEUTRAL'
        score = (current_lower_wick_ratio - current_upper_wick_ratio) * 50
    
    return {
        'signal': signal,
        'vpin': current_vpin,
        'upper_wick_ratio': current_upper_wick_ratio,
        'lower_wick_ratio': current_lower_wick_ratio,
        'score': np.clip(score, -100, 100)
    }


def trapped_traders_signal(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Detecta Trapped Traders (traders presos em posições perdedoras)
    
    Lógica:
    - Breakout falso seguido de reversão rápida = traders presos
    - Volume alto no breakout + reversão = ainda mais traders presos
    
    Returns:
        dict com 'signal', 'trapped_long', 'trapped_short', 'trap_strength', 'score'
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Identificar máximas e mínimas locais
    recent_high = high.rolling(window=lookback).max()
    recent_low = low.rolling(window=lookback).min()
    
    current_close = close.iloc[-1]
    current_high = high.iloc[-1]
    current_low = low.iloc[-1]
    current_volume = volume.iloc[-1]
    
    prev_recent_high = recent_high.iloc[-2]
    prev_recent_low = recent_low.iloc[-2]
    
    avg_volume = volume.rolling(window=lookback).mean().iloc[-1]
    
    # Detectar breakout
    breakout_up = current_high > prev_recent_high
    breakout_down = current_low < prev_recent_low
    
    # Detectar reversão (preço fecha do outro lado)
    # Breakout up mas fecha abaixo da máxima anterior = long traders trapped
    trapped_long = breakout_up and (current_close < prev_recent_high)
    
    # Breakout down mas fecha acima da mínima anterior = short traders trapped
    trapped_short = breakout_down and (current_close > prev_recent_low)
    
    # Força da armadilha baseada em volume
    volume_ratio = current_volume / (avg_volume + 1e-10)
    trap_strength = min(volume_ratio, 3.0)  # Cap at 3x
    
    # Sinal
    if trapped_short:
        # Shorts presos = squeeze up iminente
        signal = 'BUY'
        score = 50 * trap_strength
    elif trapped_long:
        # Longs presos = dump down iminente
        signal = 'SELL'
        score = -50 * trap_strength
    else:
        signal = 'NEUTRAL'
        score = 0
    
    return {
        'signal': signal,
        'trapped_long': trapped_long,
        'trapped_short': trapped_short,
        'trap_strength': trap_strength,
        'score': np.clip(score, -100, 100)
    }


def fvg_fibonacci_signal(df: pd.DataFrame, fvg_min_gap: float = 0.002) -> dict:
    """
    Fair Value Gaps + Fibonacci 61.8%
    
    FVG com retração de 61.8% = zona de alta probabilidade
    
    Returns:
        dict com 'signal', 'fvg_detected', 'fvg_type', 'fib_level', 'score'
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Detectar FVG
    # Bullish FVG: low[i] > high[i-2]
    # Bearish FVG: high[i] < low[i-2]
    
    bullish_fvg = low.iloc[-1] > high.iloc[-3]
    bearish_fvg = high.iloc[-1] < low.iloc[-3]
    
    fvg_detected = bullish_fvg or bearish_fvg
    
    if bullish_fvg:
        fvg_type = 'BULLISH'
        gap_top = low.iloc[-1]
        gap_bottom = high.iloc[-3]
        gap_size = (gap_top - gap_bottom) / close.iloc[-1]
        
        # Fibonacci 61.8% do gap
        fib_618 = gap_bottom + (gap_top - gap_bottom) * 0.618
        
        current_price = close.iloc[-1]
        
        # Se preço está próximo do fib 61.8% = zona de compra
        price_near_fib = np.abs(current_price - fib_618) / current_price < 0.01  # 1% de tolerância
        
        if gap_size > fvg_min_gap and price_near_fib:
            signal = 'BUY'
            score = 75
        elif gap_size > fvg_min_gap:
            signal = 'BUY'
            score = 50
        else:
            signal = 'NEUTRAL'
            score = 20
        
        fib_level = fib_618
        
    elif bearish_fvg:
        fvg_type = 'BEARISH'
        gap_top = low.iloc[-3]
        gap_bottom = high.iloc[-1]
        gap_size = (gap_top - gap_bottom) / close.iloc[-1]
        
        # Fibonacci 61.8% do gap
        fib_618 = gap_top - (gap_top - gap_bottom) * 0.618
        
        current_price = close.iloc[-1]
        
        # Se preço está próximo do fib 61.8% = zona de venda
        price_near_fib = np.abs(current_price - fib_618) / current_price < 0.01
        
        if gap_size > fvg_min_gap and price_near_fib:
            signal = 'SELL'
            score = -75
        elif gap_size > fvg_min_gap:
            signal = 'SELL'
            score = -50
        else:
            signal = 'NEUTRAL'
            score = -20
        
        fib_level = fib_618
        
    else:
        fvg_type = 'NONE'
        signal = 'NEUTRAL'
        score = 0
        fib_level = None
    
    return {
        'signal': signal,
        'fvg_detected': fvg_detected,
        'fvg_type': fvg_type,
        'fib_level': fib_level,
        'score': score
    }


def synthetic_delta_signal(df: pd.DataFrame, delta_period: int = 14) -> dict:
    """
    Delta Sintético + Divergência de Preço
    
    Delta = diferença entre pressão compradora e vendedora
    Aproximado por: close - open ponderado por volume
    
    Returns:
        dict com 'signal', 'delta', 'divergence_detected', 'score'
    """
    open_price = df['open']
    close = df['close']
    volume = df['volume']
    
    # Delta sintético
    # Delta positivo = mais compra, Delta negativo = mais venda
    delta = (close - open_price) * volume
    
    # Acumular delta
    cumulative_delta = delta.rolling(window=delta_period).sum()
    
    # Normalizar
    avg_volume = volume.rolling(window=delta_period).mean()
    normalized_delta = cumulative_delta / (avg_volume * delta_period + 1e-10)
    
    current_delta = normalized_delta.iloc[-1]
    
    # Detectar divergência
    # Preço fazendo máxima mais alta mas delta fazendo máxima mais baixa = bearish divergence
    # Preço fazendo mínima mais baixa mas delta fazendo mínima mais alta = bullish divergence
    
    price_slope = (close.iloc[-1] - close.iloc[-delta_period]) / close.iloc[-delta_period]
    delta_slope = (normalized_delta.iloc[-1] - normalized_delta.iloc[-delta_period]) / (np.abs(normalized_delta.iloc[-delta_period]) + 1e-10)
    
    # Divergência = sinais opostos
    bullish_divergence = (price_slope < 0) and (delta_slope > 0)
    bearish_divergence = (price_slope > 0) and (delta_slope < 0)
    
    divergence_detected = bullish_divergence or bearish_divergence
    
    # Sinal
    if bullish_divergence:
        signal = 'BUY'
        score = 70
    elif bearish_divergence:
        signal = 'SELL'
        score = -70
    elif current_delta > 0.5:
        signal = 'BUY'
        score = 40
    elif current_delta < -0.5:
        signal = 'SELL'
        score = -40
    else:
        signal = 'NEUTRAL'
        score = current_delta * 50
    
    return {
        'signal': signal,
        'delta': current_delta,
        'divergence_detected': divergence_detected,
        'score': np.clip(score, -100, 100)
    }
