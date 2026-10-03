"""Pivot models for Koffie Strategy 1.
 
A Pivot is the candle associated with a confirmed BOS:
 
    bullish BOS -> the MOST RECENT completed BEARISH_DECISIVE candle before the BOS candle
    bearish BOS -> the MOST RECENT completed BULLISH_DECISIVE candle before the BOS candle
 
The Pivot keeps the candle's ENTIRE range (full wick-to-wick high and low), not
its body. There is no size, ATR, distance or lookback rule: the only criteria
are recency and the existing BRR classification.
 
This module is DATA ONLY: frozen records that validate themselves. It contains
no search logic (see PivotEngine) and no BOS, CHOCH, swing, structure, zone,
liquidity, risk or trading logic.
 
`NO_PIVOT_FOUND` is an explicit outcome, not an error: the BOS stays valid.
"""
from __future__ import annotations
 
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
 
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import CandleClass, Timeframe
from koffie.strategy.models.swing import SwingType
 
 
# The opposite-colour decisive class each BOS direction searches for.
PIVOT_CANDLE_CLASS = {
    BOSDirection.BULLISH: CandleClass.BEARISH_DECISIVE,
    BOSDirection.BEARISH: CandleClass.BULLISH_DECISIVE,
}
 
 
class PivotStatus(Enum):
    FOUND = "FOUND"
    NO_PIVOT_FOUND = "NO_PIVOT_FOUND"
 
 
@dataclass(frozen=True)
class BOSIdentity:
    """Immutable identity of one BOS event, derived from the BOS's own fields.
 
    Two BOS records describe the same event when these fields are equal. It is
    never derived from Python object identity, so an equal BOS rebuilt from the
    same data has the same identity. The swing's price and the BOS close price
    are deliberately NOT part of the identity: a record with the same identity
    but different values is a conflicting record, not a new event.
    """
 
    timeframe: Timeframe
    direction: BOSDirection
    broken_swing_type: SwingType
    broken_swing_sequence: int
    broken_swing_candle_time: datetime
    candle_time: datetime
 
    @classmethod
    def from_bos(cls, bos: BOS) -> "BOSIdentity":
        if not isinstance(bos, BOS):
            raise TypeError("bos must be a BOS")
        swing = bos.broken_swing
        return cls(
            timeframe=bos.timeframe,
            direction=bos.direction,
            broken_swing_type=swing.swing_type,
            broken_swing_sequence=swing.sequence,
            broken_swing_candle_time=swing.candle_time,
            candle_time=bos.candle_time,
        )
 
 
@dataclass(frozen=True)
class Pivot:
    """Immutable record of the Pivot candle chosen for one BOS.
 
    bos           the BOS this Pivot belongs to
    candle_time   open_time of the Pivot candle (strictly before the BOS candle)
    high          the Pivot candle's HIGH (full wick-to-wick range)
    low           the Pivot candle's LOW
    candle_class  the Pivot candle's classification: BEARISH_DECISIVE for a
                  bullish BOS, BULLISH_DECISIVE for a bearish BOS
 
    `direction`, `timeframe` and `confirmed_at` are derived from the BOS;
    `confirmed_at` is always `bos.confirmed_at`, because the Pivot is known at
    the moment the BOS is confirmed.
    """
 
    bos: BOS
    candle_time: datetime
    high: float
    low: float
    candle_class: CandleClass
 
    def __post_init__(self) -> None:
        if not isinstance(self.bos, BOS):
            raise ValueError("bos must be a BOS")
        if not isinstance(self.candle_time, datetime):
            raise ValueError("candle_time must be a datetime")
        for name in ("high", "low"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(name + " must be a number")
            if not math.isfinite(value):
                raise ValueError(name + " must be finite")
        if self.high < self.low:
            raise ValueError("high must be >= low")
        if not isinstance(self.candle_class, CandleClass):
            raise ValueError("candle_class must be a CandleClass")
        if self.candle_class is not PIVOT_CANDLE_CLASS[self.bos.direction]:
            raise ValueError(
                "a " + self.bos.direction.value.lower() + " BOS requires a "
                + PIVOT_CANDLE_CLASS[self.bos.direction].value + " Pivot candle"
            )
        if (self.candle_time.tzinfo is None) != (self.bos.candle_time.tzinfo is None):
            raise ValueError("candle_time and the BOS times must both be naive or both timezone-aware")
        if not self.candle_time < self.bos.candle_time:
            raise ValueError("the Pivot candle must be strictly earlier than the BOS candle")
 
    @property
    def direction(self) -> BOSDirection:
        return self.bos.direction
 
    @property
    def timeframe(self) -> Timeframe:
        return self.bos.timeframe
 
    @property
    def candle_close_time(self) -> datetime:
        """When the Pivot candle itself finished closing."""
        return self.candle_time + self.bos.timeframe.duration
 
    @property
    def confirmed_at(self) -> datetime:
        return self.bos.confirmed_at
 
 
@dataclass(frozen=True)
class PivotOutcome:
    """The explicit result of Pivot selection for one BOS.
 
    FOUND          `pivot` is required and must belong to the same BOS
    NO_PIVOT_FOUND `pivot` must be None; the BOS remains valid
    """
 
    bos: BOS
    status: PivotStatus
    pivot: Optional[Pivot] = None
 
    def __post_init__(self) -> None:
        if not isinstance(self.bos, BOS):
            raise ValueError("bos must be a BOS")
        if not isinstance(self.status, PivotStatus):
            raise ValueError("status must be a PivotStatus")
        if self.status is PivotStatus.FOUND:
            if not isinstance(self.pivot, Pivot):
                raise ValueError("a FOUND outcome requires a Pivot")
            if self.pivot.bos != self.bos:
                raise ValueError("the Pivot belongs to a different BOS")
        else:
            if self.pivot is not None:
                raise ValueError("a NO_PIVOT_FOUND outcome must not carry a Pivot")
 
    @property
    def is_found(self) -> bool:
        return self.status is PivotStatus.FOUND
 
    @property
    def identity(self) -> BOSIdentity:
        return BOSIdentity.from_bos(self.bos)
 
    @property
    def confirmed_at(self) -> datetime:
        return self.bos.confirmed_at