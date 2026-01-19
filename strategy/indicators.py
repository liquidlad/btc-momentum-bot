"""
Technical indicators for the momentum scalping strategy.
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change (percentage)."""
    return ((series - series.shift(period)) / series.shift(period)) * 100


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Bollinger Bands.

    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower


def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate all strategy indicators.

    Args:
        df: DataFrame with OHLCV data
        params: Strategy parameters dict

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # EMAs
    df["ema_fast"] = ema(df["close"], params["ema_fast"])
    df["ema_slow"] = ema(df["close"], params["ema_slow"])

    # Rate of Change
    df["roc"] = roc(df["close"], params["roc_period"])

    # Volume SMA
    df["volume_sma"] = sma(df["volume"], params["volume_sma_period"])
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # RSI
    df["rsi"] = rsi(df["close"], params["rsi_period"])

    # Candle direction
    df["is_bullish"] = df["close"] > df["open"]
    df["is_bearish"] = df["close"] < df["open"]

    # Price position relative to EMAs
    df["above_ema_fast"] = df["close"] > df["ema_fast"]
    df["ema_fast_above_slow"] = df["ema_fast"] > df["ema_slow"]

    return df


def calculate_bb_rsi_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calculate RSI + Bollinger Band indicators for mean-reversion strategy.

    Args:
        df: DataFrame with OHLCV data
        params: Strategy parameters dict

    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()

    # Bollinger Bands
    bb_period = params.get("bb_period", 20)
    bb_std = params.get("bb_std", 2.0)
    df["bb_middle"], df["bb_upper"], df["bb_lower"] = bollinger_bands(
        df["close"], bb_period, bb_std
    )

    # RSI
    rsi_period = params.get("rsi_period", 7)
    df["rsi"] = rsi(df["close"], rsi_period)

    # Price position relative to bands
    df["above_upper_bb"] = df["close"] > df["bb_upper"]
    df["below_lower_bb"] = df["close"] < df["bb_lower"]

    return df


def generate_bb_rsi_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Generate SHORT-only signals for RSI+BB mean-reversion strategy.

    Entry: SHORT when price > upper BB AND RSI > threshold
    Exit: Handled separately (trailing stop, lower BB, stop loss)

    Args:
        df: DataFrame with BB+RSI indicators
        params: Strategy parameters with rsi_entry_threshold

    Returns:
        DataFrame with signal column (-1: short, 0: none)
    """
    df = df.copy()

    rsi_threshold = params.get("rsi_entry_threshold", 75)

    # SHORT entry: price above upper BB AND RSI overbought
    short_entry = df["above_upper_bb"] & (df["rsi"] > rsi_threshold)

    df["signal"] = 0
    df.loc[short_entry, "signal"] = -1

    return df


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Generate entry signals based on strategy rules.

    Args:
        df: DataFrame with indicators
        params: Strategy parameters

    Returns:
        DataFrame with signal column (-1: short, 0: none, 1: long)
    """
    df = df.copy()

    # Long conditions
    long_momentum = df["above_ema_fast"] & df["ema_fast_above_slow"]
    long_roc = df["roc"] > params["roc_threshold"]
    long_volume = df["volume_ratio"] > params["volume_multiplier"]
    long_rsi = (df["rsi"] > params["rsi_long_min"]) & (df["rsi"] < params["rsi_long_max"])
    long_candle = df["is_bullish"]

    # Short conditions
    short_momentum = ~df["above_ema_fast"] & ~df["ema_fast_above_slow"]
    short_roc = df["roc"] < -params["roc_threshold"]
    short_volume = df["volume_ratio"] > params["volume_multiplier"]
    short_rsi = (df["rsi"] > params["rsi_short_min"]) & (df["rsi"] < params["rsi_short_max"])
    short_candle = df["is_bearish"]

    # Score signals (count conditions met)
    df["long_score"] = (
        long_momentum.astype(int) +
        long_roc.astype(int) +
        long_volume.astype(int) +
        long_rsi.astype(int) +
        long_candle.astype(int)
    )

    df["short_score"] = (
        short_momentum.astype(int) +
        short_roc.astype(int) +
        short_volume.astype(int) +
        short_rsi.astype(int) +
        short_candle.astype(int)
    )

    # Use min_conditions parameter (default 5 = all conditions)
    min_conditions = params.get("min_conditions", 5)

    # Generate signal based on score threshold
    df["signal"] = 0
    df.loc[df["long_score"] >= min_conditions, "signal"] = 1
    df.loc[df["short_score"] >= min_conditions, "signal"] = -1

    # If both long and short score meet threshold, take the higher one
    both_meet = (df["long_score"] >= min_conditions) & (df["short_score"] >= min_conditions)
    df.loc[both_meet & (df["long_score"] > df["short_score"]), "signal"] = 1
    df.loc[both_meet & (df["short_score"] > df["long_score"]), "signal"] = -1
    df.loc[both_meet & (df["long_score"] == df["short_score"]), "signal"] = 0  # No signal if tied

    return df
