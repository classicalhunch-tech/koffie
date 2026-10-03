"""Liquidity model for Koffie Strategy 1.
 
A LiquidityLevel is a confirmed swing that can act as a liquidity reference:
 
    swing LOW   -> liquidity BELOW price
    swing HIGH  -> liquidity ABOVE price
 
Single source of truth
----------------------
A LiquidityLevel stores exactly ONE field: the confirmed `Swing`. Everything
else is derived from it, so nothing can drift out of sync:
 
    LiquidityLevel -> Swing
 
Two different times (do not confuse them)
-----------------------------------------
origin_time   open_time of the swing candle the price comes from.
known_at      when the swing was confirmed (the close of its right-hand
              candle). This is the first moment the level is known. Downstream
              logic must NOT use the level before `known_at`.
              `is_known_at(moment)` states this rule in one place.
 
Relevance to a zone (`is_relevant_to`)
--------------------------------------
    DEMAND zone: a swing LOW strictly BELOW the zone's low
    SUPPLY zone: a swing HIGH strictly ABOVE the zone's high
 
Strict means a price exactly equal to the zone boundary is NOT relevant.
There is no distance limit, no tolerance (exact price equality only), no time
cutoff and no dependence on the zone's timeframe: a 5M level can serve a 1H,
15M or 5M zone. Two swings at the same price are simply two separate levels;
there is no clustering and no EQH/EQL object.
 
This module is DATA ONLY. It contains no detection, sweep, touch, expiry,
ranking, consumption, setup, entry, SL/TP, risk, news or execution logic, and
no flag of any kind. Whether a level has been swept belongs to a later
component.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
 
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone, ZoneType
 
 
class LiquiditySide(Enum):
    BELOW = "BELOW"   # a swing LOW: liquidity below price
    ABOVE = "ABOVE"   # a swing HIGH: liquidity above price
 
 
_SIDE_FOR_SWING_TYPE = {
    SwingType.LOW: LiquiditySide.BELOW,
    SwingType.HIGH: LiquiditySide.ABOVE,
}
 
 
@dataclass(frozen=True)
class LiquidityIdentity:
    """Immutable identity of one liquidity level, derived from its swing's fields.
 
    Never derived from Python object identity, so an equal swing rebuilt from
    the same data has the same identity. The swing's price and confirmed_at are
    deliberately NOT part of the identity: a record with the same identity but
    different values is a conflicting record, not a new level.
    """
 
    timeframe: Timeframe
    swing_type: SwingType
    swing_sequence: int
    swing_candle_time: datetime
 
    @classmethod
    def from_swing(cls, swing: Swing) -> "LiquidityIdentity":
        if not isinstance(swing, Swing):
            raise TypeError("swing must be a Swing")
        return cls(
            timeframe=swing.timeframe,
            swing_type=swing.swing_type,
            swing_sequence=swing.sequence,
            swing_candle_time=swing.candle_time,
        )
 
 
@dataclass(frozen=True)
class LiquidityLevel:
    """Immutable liquidity level, derived entirely from its confirmed Swing."""
 
    swing: Swing
 
    def __post_init__(self) -> None:
        if not isinstance(self.swing, Swing):
            raise ValueError("swing must be a Swing")
 
    # -- what and where -----------------------------------------------------------
    @property
    def side(self) -> LiquiditySide:
        return _SIDE_FOR_SWING_TYPE[self.swing.swing_type]
 
    @property
    def price(self) -> float:
        return self.swing.price
 
    @property
    def timeframe(self) -> Timeframe:
        return self.swing.timeframe
 
    @property
    def sequence(self) -> int:
        return self.swing.sequence
 
    @property
    def identity(self) -> LiquidityIdentity:
        return LiquidityIdentity.from_swing(self.swing)
 
    # -- the two times ------------------------------------------------------------
    @property
    def origin_time(self) -> datetime:
        """Open time of the swing candle the price comes from."""
        return self.swing.candle_time
 
    @property
    def known_at(self) -> datetime:
        """When the swing was confirmed: the level is not known before this."""
        return self.swing.confirmed_at
 
    def is_known_at(self, moment: datetime) -> bool:
        """True only from `known_at` onward.
 
        Raises TypeError if `moment` is not a datetime (and, as for any datetime
        comparison, if it mixes naive and timezone-aware values).
        """
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return moment >= self.known_at
 
    # -- relevance to a zone --------------------------------------------------------
    def is_relevant_to(self, zone: Zone) -> bool:
        """Pure check, stores nothing.
 
        DEMAND: a swing LOW strictly below the zone low.
        SUPPLY: a swing HIGH strictly above the zone high.
        """
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        if zone.zone_type is ZoneType.DEMAND:
            return self.swing.swing_type is SwingType.LOW and self.swing.price < zone.low
        return self.swing.swing_type is SwingType.HIGH and self.swing.price > zone.high