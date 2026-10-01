"""Candle model for Koffie Strategy 1 (spec sections 2, 3 and 15).
 
This module defines the three timeframes, an immutable OHLCV candle, facts
derived directly from one candle's own values, and the locked Strategy 1
BRR candle classification. It contains no swing or structure logic.

BRR classification (locked)
---------------------------
    BRR = abs(close - open) / (high - low)

    high == low                 -> ZERO_RANGE_ANOMALY (never decisive)
    BRR >  0.70, close > open   -> BULLISH_DECISIVE
    BRR >  0.70, close < open   -> BEARISH_DECISIVE
    BRR <= 0.70 (incl. exactly 0.70 and close == open) -> NEUTRAL

The 0.70 default may be overridden via `classify(now, threshold=...)` for
backtesting; the comparison stays strict. BRR alone decides; no ATR, size, wick, displacement or volatility filter.
Only a fully CLOSED candle can be classified: `classify(now)` raises
ValueError while `now < close_time`, so a forming candle can never be decisive.
 
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
from typing import Optional


BRR_DECISIVE_THRESHOLD = 0.70


class CandleClass(Enum):
    BULLISH_DECISIVE = "BULLISH_DECISIVE"
    BEARISH_DECISIVE = "BEARISH_DECISIVE"
    NEUTRAL = "NEUTRAL"                        # DOJI / NEUTRAL: BRR <= 0.70
    ZERO_RANGE_ANOMALY = "ZERO_RANGE_ANOMALY"  # high == low

    @property
    def is_decisive(self) -> bool:
        return self in (CandleClass.BULLISH_DECISIVE, CandleClass.BEARISH_DECISIVE)


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
        """Literal close == open ONLY. This is NOT the Strategy 1 "DOJI / NEUTRAL"
        (BRR <= 0.70); use `classify(now)` for that."""
        return self.close == self.open

    # -- BRR classification (locked Strategy 1 rule) ----------------------
    @property
    def brr(self) -> Optional[float]:
        """abs(close - open) / (high - low), or None when high == low."""
        if self.high == self.low:
            return None
        return abs(self.close - self.open) / (self.high - self.low)

    def classify(
        self, now: datetime, threshold: float = BRR_DECISIVE_THRESHOLD
    ) -> CandleClass:
        """Classify this candle. Only a fully closed candle may be classified.

        Raises ValueError if the candle is not closed at `now` (TypeError if
        `now` is not a datetime), so an unclosed candle is never decisive.

        `threshold` defaults to the locked Strategy 1 value (0.70); it is
        configurable only for future backtesting. The comparison is always
        strict: BRR > threshold is decisive, BRR <= threshold is NEUTRAL.
        It must be a finite number in [0, 1].
        """
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a number")
        if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be a finite number between 0 and 1")
        if not self.is_closed_at(now):
            raise ValueError("candle is not closed yet; only closed candles may be classified")
        brr = self.brr
        if brr is None:
            return CandleClass.ZERO_RANGE_ANOMALY
        if brr > threshold:
            if self.close > self.open:
                return CandleClass.BULLISH_DECISIVE
            return CandleClass.BEARISH_DECISIVE   # close < open (close == open gives BRR 0)
        return CandleClass.NEUTRAL
