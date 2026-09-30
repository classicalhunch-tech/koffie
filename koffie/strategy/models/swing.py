"""Swing model for Koffie Strategy 1 (spec sections 4-7).
 
A `Swing` is a CONFIRMED swing: a candle N that satisfied the 3-candle rule
and whose right-hand neighbour N+1 has since CLOSED. Unconfirmed candidates
are not represented by this model.
 
Every confirmed local swing is automatically a structural swing (spec
section 6), so there is one model, not separate local/structural ones.
 
This module holds data only. It contains no swing detection, structure
classification, BOS/CHOCH, liquidity, zones, BRR or trading logic.
"""
from __future__ import annotations
 
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
 
from koffie.strategy.models.candle import Timeframe
 
 
class SwingType(Enum):
    HIGH = "HIGH"
    LOW = "LOW"
 
 
@dataclass(frozen=True)
class Swing:
    """An immutable, confirmed structural swing.
 
    swing_type    HIGH or LOW
    timeframe     timeframe the swing was found on
    price         the swing candle's high (HIGH) or low (LOW)
    candle_time   open_time of the swing candle N
    confirmed_at  close_time of the confirmation candle N+1, i.e. the first
                  moment the swing was knowable
    sequence      chronological confirmation order (0, 1, 2, ...)
 
    Causal invariant: candle N+1 begins no earlier than N's close and lasts a
    full timeframe, so confirmed_at >= candle_time + 2 * timeframe.duration.
    Gaps between candles (weekends, holidays) make it later, never earlier.
    """
 
    swing_type: SwingType
    timeframe: Timeframe
    price: float
    candle_time: datetime
    confirmed_at: datetime
    sequence: int
 
    def __post_init__(self) -> None:
        if not isinstance(self.swing_type, SwingType):
            raise ValueError("swing_type must be a SwingType")
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError("timeframe must be a Timeframe")
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float)):
            raise ValueError("price must be a number")
        if not math.isfinite(self.price):
            raise ValueError("price must be finite")
        if not isinstance(self.candle_time, datetime):
            raise ValueError("candle_time must be a datetime")
        if not isinstance(self.confirmed_at, datetime):
            raise ValueError("confirmed_at must be a datetime")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be an int")
        if self.sequence < 0:
            raise ValueError("sequence must be >= 0")
        if (self.candle_time.tzinfo is None) != (self.confirmed_at.tzinfo is None):
            raise ValueError(
                "candle_time and confirmed_at must both be naive or both timezone-aware"
            )
        if self.confirmed_at < self.candle_time + 2 * self.timeframe.duration:
            raise ValueError(
                "confirmed_at cannot precede the close of the confirmation candle "
                "(candle_time + 2 * timeframe duration)"
            )
 
    @property
    def is_high(self) -> bool:
        return self.swing_type is SwingType.HIGH
 
    @property
    def is_low(self) -> bool:
        return self.swing_type is SwingType.LOW