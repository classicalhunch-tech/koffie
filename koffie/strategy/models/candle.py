"""Candle model for Koffie Strategy 1 (spec sections 2, 3 and 15).
 
This module defines only the data: the three timeframes, an immutable OHLCV
candle, and facts derived directly from one candle's own values. It contains
no strategy logic (no swings, no BRR/decisive test, no structure).
 
Timing
------
`open_time` is the START of the candle's period. A candle is CLOSED only once
its full timeframe has elapsed, i.e. `now >= close_time`.
"""
from __future__ import annotations
 
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
 
 
class Timeframe(Enum):
    M5 = "5M"
    M15 = "15M"
    H1 = "1H"
 
    @property
    def duration(self) -> timedelta:
        return _DURATIONS[self]
 
 
_DURATIONS = {
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(hours=1),
}
 
 
@dataclass(frozen=True)
class Candle:
    timeframe: Timeframe
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
 
    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError("timeframe must be a Timeframe")
        if not isinstance(self.open_time, datetime):
            raise ValueError("open_time must be a datetime")
        prices = (self.open, self.high, self.low, self.close)
        if not all(math.isfinite(p) for p in prices):
            raise ValueError("open, high, low and close must be finite numbers")
        if not (math.isfinite(self.volume) and self.volume >= 0):
            raise ValueError("volume must be a finite number >= 0")
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= open and close")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= open and close")
 
    # -- timing ---------------------------------------------------------
    @property
    def close_time(self) -> datetime:
        """Moment the candle's complete timeframe has elapsed."""
        return self.open_time + self.timeframe.duration
 
    def is_closed_at(self, now: datetime) -> bool:
        """True only once the complete timeframe has elapsed (now >= close_time)."""
        return now >= self.close_time
 
    # -- geometry (spec section 15) --------------------------------------
    @property
    def range(self) -> float:
        return self.high - self.low
 
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
 
    # -- direction (spec section 15) -------------------------------------
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
 
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
 
    @property
    def is_doji(self) -> bool:
        return self.close == self.open