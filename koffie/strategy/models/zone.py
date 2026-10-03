"""Zone model for Koffie Strategy 1.

A Zone is the supply/demand area created by a confirmed BOS that has a Pivot:

    bullish BOS + Pivot -> DEMAND zone
    bearish BOS + Pivot -> SUPPLY zone

The Zone keeps the Pivot candle's ENTIRE range (full wick-to-wick high and
low). There is NO buffer of any kind (no pip, ATR, percentage, body-only,
spread or volatility adjustment). The boundaries never move.

Single source of truth
----------------------
A Zone stores exactly ONE field: the `Pivot`. Everything else is derived from
it, so nothing can drift out of sync and no independent mutable copy exists:

    Zone -> Pivot -> BOS -> broken structural swing

Two different times (do not confuse them)
-----------------------------------------
origin_time   open_time of the historical PIVOT CANDLE the price range comes from.
created_at    when the BOS confirmed the zone (the BOS candle's close). This is
              the first moment the zone is known/usable. Downstream logic must
              NOT treat the zone as existing or usable before `created_at`,
              even though its price range comes from an earlier candle.
              `is_known_at(moment)` states this rule in one place.

A zone existing is NOT entry eligibility, setup validity or execution
permission; those belong to later components.

This module is DATA ONLY. It contains no search, touch, mitigation,
invalidation, merging, ranking, liquidity, setup, entry, SL/TP, risk, news or
execution logic, and no `consumed_for_entry` flag.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.pivot import BOSIdentity, Pivot
from koffie.strategy.models.swing import Swing


class ZoneType(Enum):
    DEMAND = "DEMAND"
    SUPPLY = "SUPPLY"


# Bullish BOS -> DEMAND, bearish BOS -> SUPPLY.
ZONE_TYPE_FOR_DIRECTION = {
    BOSDirection.BULLISH: ZoneType.DEMAND,
    BOSDirection.BEARISH: ZoneType.SUPPLY,
}


@dataclass(frozen=True)
class Zone:
    """Immutable supply/demand zone, derived entirely from its Pivot."""

    pivot: Pivot

    def __post_init__(self) -> None:
        if not isinstance(self.pivot, Pivot):
            raise ValueError("pivot must be a Pivot")

    # -- what kind of zone, where, on which timeframe ---------------------------
    @property
    def zone_type(self) -> ZoneType:
        return ZONE_TYPE_FOR_DIRECTION[self.pivot.bos.direction]

    @property
    def timeframe(self) -> Timeframe:
        return self.pivot.bos.timeframe

    @property
    def high(self) -> float:
        """Pivot candle HIGH (full wick), no buffer."""
        return self.pivot.high

    @property
    def low(self) -> float:
        """Pivot candle LOW (full wick), no buffer."""
        return self.pivot.low

    # -- the two times ----------------------------------------------------------
    @property
    def origin_time(self) -> datetime:
        """Open time of the Pivot candle the price range comes from."""
        return self.pivot.candle_time

    @property
    def created_at(self) -> datetime:
        """When the BOS confirmed the zone: the zone is not known before this."""
        return self.pivot.bos.confirmed_at

    def is_known_at(self, moment: datetime) -> bool:
        """True only from `created_at` onward; never at or before `origin_time`
        merely because the Pivot candle already existed.

        Raises TypeError if `moment` is not a datetime (and, as for any datetime
        comparison, if it mixes naive and timezone-aware values).
        """
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return moment >= self.created_at

    # -- provenance: Zone -> Pivot -> BOS -> broken structural swing ---------------
    @property
    def bos(self) -> BOS:
        return self.pivot.bos

    @property
    def broken_swing(self) -> Swing:
        return self.pivot.bos.broken_swing

    @property
    def identity(self) -> BOSIdentity:
        """Identity of the BOS that created this zone."""
        return BOSIdentity.from_bos(self.pivot.bos)
