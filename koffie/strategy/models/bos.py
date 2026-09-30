"""BOS (Break of Structure) model for Koffie Strategy 1 (spec section 11).
 
A BOS is a CONTINUATION of the current structural direction:
 
    bullish BOS: a CLOSED candle closes above the active structural swing high
    bearish BOS: a CLOSED candle closes below the active structural swing low
 
Only the candle's CLOSE counts; a wick through the level is not a BOS. This
module is DATA ONLY: it records one BOS event and validates that the record
is internally consistent. It contains no detection logic, no state change, no
CHOCH, pivot, zone, liquidity or trading logic.
"""
from __future__ import annotations
 
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
 
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.swing import Swing, SwingType
 
 
class BOSDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
 
 
@dataclass(frozen=True)
class BOS:
    """Immutable record of one continuation break.
 
    timeframe      timeframe the break happened on
    direction      BULLISH (broke a swing high) or BEARISH (broke a swing low)
    broken_swing   the active swing whose level was broken: a HIGH swing for a
                   bullish BOS, a LOW swing for a bearish BOS
    candle_time    open_time of the breaking candle
    close_price    close of the breaking candle; strictly beyond the swing price
 
    `confirmed_at` is derived: the breaking candle's close time, the moment
    the BOS became knowable.
 
    The broken swing must already be confirmed by then. A swing confirmed by
    the very same candle is allowed (confirmed_at equal), because swings are
    processed before BOS/CHOCH within a candle.
    """
 
    timeframe: Timeframe
    direction: BOSDirection
    broken_swing: Swing
    candle_time: datetime
    close_price: float
 
    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError("timeframe must be a Timeframe")
        if not isinstance(self.direction, BOSDirection):
            raise ValueError("direction must be a BOSDirection")
        if not isinstance(self.broken_swing, Swing):
            raise ValueError("broken_swing must be a Swing")
        if not isinstance(self.candle_time, datetime):
            raise ValueError("candle_time must be a datetime")
        if isinstance(self.close_price, bool) or not isinstance(self.close_price, (int, float)):
            raise ValueError("close_price must be a number")
        if not math.isfinite(self.close_price):
            raise ValueError("close_price must be finite")
 
        if self.broken_swing.timeframe is not self.timeframe:
            raise ValueError("broken_swing belongs to a different timeframe")
 
        if self.direction is BOSDirection.BULLISH:
            if self.broken_swing.swing_type is not SwingType.HIGH:
                raise ValueError("a bullish BOS breaks a swing HIGH")
            if not self.close_price > self.broken_swing.price:
                raise ValueError("a bullish BOS must close strictly above the swing high")
        else:
            if self.broken_swing.swing_type is not SwingType.LOW:
                raise ValueError("a bearish BOS breaks a swing LOW")
            if not self.close_price < self.broken_swing.price:
                raise ValueError("a bearish BOS must close strictly below the swing low")
 
        if (self.candle_time.tzinfo is None) != (self.broken_swing.confirmed_at.tzinfo is None):
            raise ValueError(
                "candle_time and the broken swing's times must both be naive or both timezone-aware"
            )
        if self.broken_swing.confirmed_at > self.confirmed_at:
            raise ValueError("the broken swing must be confirmed by the time the BOS is confirmed")
 
    @property
    def confirmed_at(self) -> datetime:
        """Close time of the breaking candle."""
        return self.candle_time + self.timeframe.duration