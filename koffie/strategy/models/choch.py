"""CHOCH (Change of Character) model for Koffie Strategy 1 (spec sections 12-13).

A CHOCH records that the PROTECTED swing of an established trend was broken:

    BULLISH structure -> protected swing = active swing LOW
        broken by a candle trading BELOW it   -> CHOCH toward BEARISH
    BEARISH structure -> protected swing = active swing HIGH
        broken by a candle trading ABOVE it   -> CHOCH toward BULLISH

Current owner rule: the WHOLE CANDLE counts, so a wick strictly beyond the
protected swing is sufficient (touching the level exactly is not). NOTE: this
differs from spec sections 12, 38, 41 and 44, which still describe a
closing-price CHOCH; the document should be amended to match. The field is
called `break_price` (the candle price that went beyond the level) so the
record stays valid if the rule ever changes.

A CHOCH is not a trade entry and does not itself declare a new trend. The
structure moves to REVALUATING when a later component calls
StructureEngine.enter_revaluating(); the new direction is decided by swings.

This module is DATA ONLY: no detection, no state change, no BOS, pivot, zone,
liquidity or trading logic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.structure import StructureState
from koffie.strategy.models.swing import Swing, SwingType


class CHOCHDirection(Enum):
    """Direction of the POTENTIAL reversal the break points toward."""

    BEARISH = "BEARISH"   # a bullish structure's protected low was broken
    BULLISH = "BULLISH"   # a bearish structure's protected high was broken


@dataclass(frozen=True)
class CHOCH:
    """Immutable record of one protected-swing break.

    timeframe        timeframe the break happened on
    direction        BEARISH (protected LOW broken) or BULLISH (protected HIGH broken)
    protected_swing  the swing that was broken: LOW for BEARISH, HIGH for BULLISH
    candle_time      open_time of the breaking candle
    break_price      the breaking candle's price beyond the level: its low for a
                     BEARISH CHOCH, its high for a BULLISH one; strictly beyond
                     the protected swing's price

    `confirmed_at` is derived: the breaking candle's close time. Only closed
    candles are processed, so the break becomes knowable when the candle closes.

    The protected swing must already be confirmed by then; a swing confirmed by
    the very same candle is allowed, because swings are processed before
    BOS/CHOCH within a candle.
    """

    timeframe: Timeframe
    direction: CHOCHDirection
    protected_swing: Swing
    candle_time: datetime
    break_price: float

    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError("timeframe must be a Timeframe")
        if not isinstance(self.direction, CHOCHDirection):
            raise ValueError("direction must be a CHOCHDirection")
        if not isinstance(self.protected_swing, Swing):
            raise ValueError("protected_swing must be a Swing")
        if not isinstance(self.candle_time, datetime):
            raise ValueError("candle_time must be a datetime")
        if isinstance(self.break_price, bool) or not isinstance(self.break_price, (int, float)):
            raise ValueError("break_price must be a number")
        if not math.isfinite(self.break_price):
            raise ValueError("break_price must be finite")

        if self.protected_swing.timeframe is not self.timeframe:
            raise ValueError("protected_swing belongs to a different timeframe")

        if self.direction is CHOCHDirection.BEARISH:
            if self.protected_swing.swing_type is not SwingType.LOW:
                raise ValueError("a bearish CHOCH breaks a protected swing LOW")
            if not self.break_price < self.protected_swing.price:
                raise ValueError("a bearish CHOCH must trade strictly below the protected low")
        else:
            if self.protected_swing.swing_type is not SwingType.HIGH:
                raise ValueError("a bullish CHOCH breaks a protected swing HIGH")
            if not self.break_price > self.protected_swing.price:
                raise ValueError("a bullish CHOCH must trade strictly above the protected high")

        if (self.candle_time.tzinfo is None) != (self.protected_swing.confirmed_at.tzinfo is None):
            raise ValueError(
                "candle_time and the protected swing's times must both be naive or both timezone-aware"
            )
        if self.protected_swing.confirmed_at > self.confirmed_at:
            raise ValueError(
                "the protected swing must be confirmed by the time the CHOCH is confirmed"
            )

    @property
    def confirmed_at(self) -> datetime:
        """Close time of the breaking candle."""
        return self.candle_time + self.timeframe.duration

    @property
    def prior_state(self) -> StructureState:
        """The trend that existed before the break (BULLISH for a bearish CHOCH)."""
        if self.direction is CHOCHDirection.BEARISH:
            return StructureState.BULLISH
        return StructureState.BEARISH