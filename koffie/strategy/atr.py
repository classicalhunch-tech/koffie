"""
koffie/strategy/atr.py

Causal ATR (Average True Range). known_atr[i] uses only bars up to i-1,
so it is fully known at the open of bar i (where market-on-close fills happen).

Imported from keithkulecho/spreadsheet_bot strategy/atr.py.
"""

import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """Calculate True Range for each candle.
    
    True Range = max(
        High - Low,
        |High - PreviousClose|,
        |Low - PreviousClose|
    )
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
    
    Returns:
        pd.Series of True Range values
    """
    prev_close = df["close"].shift(1)
    parts = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return parts.max(axis=1)


def atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR as a simple moving average of True Range.
    
    Args:
        df: DataFrame with OHLC columns
        period: Number of bars for moving average (default 14)
    
    Returns:
        pd.Series of ATR values
    
    Raises:
        ValueError: If period <= 0
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    return true_range(df).rolling(period, min_periods=period).mean()


def known_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate causal ATR (no lookahead bias).
    
    known_atr[i] = ATR as fully known at the OPEN of bar i.
    Uses only bars 0 to i-1, excluding bar i itself.
    
    This is essential for backtest fill logic that executes at bar open:
    the ATR used to widen stops must not include the bar being filled.
    
    Args:
        df: DataFrame with OHLC columns
        period: Number of bars for moving average (default 14)
    
    Returns:
        pd.Series of causal ATR values (shifted right by 1)
    """
    return atr_series(df, period).shift(1)
