"""Confirmation model for Koffie Strategy 1.
 
After a Sweep, the 5M price action must CONFIRM it. Confirmation is evaluated on
completed 5M candles, whatever timeframe the sweeping zone came from, using the
existing Candle classifier (BRR > 0.70 = decisive). This module is DATA ONLY; the
sequence itself is run by ConfirmationEngine.
 
The locked sequence, LONG (demand zone); SHORT (supply zone) is the mirror:
 
    BULLISH_DECISIVE                          -> CONFIRMED
    BEARISH_DECISIVE, close >= zone.low       -> wait for the next decisive candle
    BEARISH_DECISIVE, close <  zone.low       -> INVALIDATED (boundary)
    next decisive: BULLISH_DECISIVE           -> CONFIRMED
    next decisive: BEARISH_DECISIVE           -> INVALIDATED
    NEUTRAL / ZERO_RANGE_ANOMALY              -> keep waiting (never consume the sequence)
 
    SHORT: BEARISH_DECISIVE confirms, BULLISH_DECISIVE is the wrong direction and the
    boundary is `close > zone.high`.
 
A wick beyond the zone boundary never invalidates; only a CLOSE beyond it does. A
confirming candle may close outside the zone. The sweep candle itself may confirm.
 
Records (all immutable)
-----------------------
ConfirmationStatus   the per-Sweep state held by the engine
Confirmation         the CONFIRMED event: exactly two stored fields, `sweep` and `candle`
Invalidation         the INVALIDATED event: `sweep`, `candle` and an explanatory `reason`
InvalidationReason   CLOSED_BEYOND_ZONE_BOUNDARY or SECOND_WRONG_DIRECTION_DECISIVE
 
Nothing is duplicated: the zone, its boundaries, the liquidity, the swept price and the
direction (`zone_type`: DEMAND = LONG, SUPPLY = SHORT) are all read from the Sweep; the
candle's class is derived from the Candle.
 
The invalidation `reason` is an audit label only; it never changes a transition. When a
candle satisfies both reasons, CLOSED_BEYOND_ZONE_BOUNDARY is the label.
 
This module contains no setup, entry, SL/TP, risk, news or execution logic, no BOS/CHOCH,
no tolerance, distance, ATR, displacement or session rule, no timeout and no flag.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
 
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.sweep import Sweep, SweepIdentity
from koffie.strategy.models.zone import Zone, ZoneType
 
# The decisive class that confirms, and the one that is the wrong direction, by zone type.
CONFIRMING_CLASS_FOR = {
    ZoneType.DEMAND: CandleClass.BULLISH_DECISIVE,
    ZoneType.SUPPLY: CandleClass.BEARISH_DECISIVE,
}
WRONG_CLASS_FOR = {
    ZoneType.DEMAND: CandleClass.BEARISH_DECISIVE,
    ZoneType.SUPPLY: CandleClass.BULLISH_DECISIVE,
}
 
 
class ConfirmationStatus(Enum):
    WAITING_FOR_DECISIVE = "WAITING_FOR_DECISIVE"
    WAITING_FOR_SECOND_DECISIVE = "WAITING_FOR_SECOND_DECISIVE"
    CONFIRMED = "CONFIRMED"            # terminal
    INVALIDATED = "INVALIDATED"        # terminal
 
 
class InvalidationReason(Enum):
    CLOSED_BEYOND_ZONE_BOUNDARY = "CLOSED_BEYOND_ZONE_BOUNDARY"
    SECOND_WRONG_DIRECTION_DECISIVE = "SECOND_WRONG_DIRECTION_DECISIVE"
 
 
def closes_beyond_boundary(zone_type: ZoneType, zone: Zone, candle: Candle) -> bool:
    """True when the candle CLOSES strictly beyond the zone boundary on the invalidating side.
 
    DEMAND: candle.close < zone.low.   SUPPLY: candle.close > zone.high.
    A wick beyond the boundary does not count, and a close exactly on it is not beyond it.
    Pure check: it does not look at the candle's class, times or timeframe.
    """
    if not isinstance(zone_type, ZoneType):
        raise TypeError("zone_type must be a ZoneType")
    if not isinstance(zone, Zone):
        raise TypeError("zone must be a Zone")
    if not isinstance(candle, Candle):
        raise TypeError("candle must be a Candle")
    if zone_type is ZoneType.DEMAND:
        return candle.close < zone.low
    return candle.close > zone.high
 
 
def _check_sweep_and_candle(sweep: object, candle: object) -> None:
    if not isinstance(sweep, Sweep):
        raise ValueError("sweep must be a Sweep")
    if not isinstance(candle, Candle):
        raise ValueError("candle must be a Candle")
    if candle.timeframe is not Timeframe.M5:
        raise ValueError("confirmation uses completed 5M candles")
    if candle.open_time < sweep.candle.open_time:
        raise ValueError("the candle is earlier than the sweep candle")
 
 
@dataclass(frozen=True)
class Confirmation:
    """Immutable record: this completed 5M candle CONFIRMED this Sweep."""
 
    sweep: Sweep
    candle: Candle
 
    def __post_init__(self) -> None:
        _check_sweep_and_candle(self.sweep, self.candle)
        wanted = CONFIRMING_CLASS_FOR[self.sweep.zone_type]
        if self.candle_class is not wanted:
            raise ValueError(f"a {self.sweep.zone_type.value} sweep is confirmed by a {wanted.value} candle")
 
    # -- derived values -------------------------------------------------------------
    @property
    def zone(self) -> Zone:
        return self.sweep.zone
 
    @property
    def zone_type(self) -> ZoneType:
        """The direction: DEMAND = LONG, SUPPLY = SHORT."""
        return self.sweep.zone_type
 
    @property
    def candle_class(self) -> CandleClass:
        """The confirming candle's class, from the existing classifier (it is closed at its own close time)."""
        return self.candle.classify(self.candle.close_time)
 
    @property
    def candle_time(self) -> datetime:
        """Open time of the confirming candle."""
        return self.candle.open_time
 
    @property
    def known_at(self) -> datetime:
        """Confirmation is known only once its candle has CLOSED."""
        return self.candle.close_time
 
    def is_known_at(self, moment: datetime) -> bool:
        """True only from `known_at` onward.
 
        Raises TypeError if `moment` is not a datetime (and, as for any datetime
        comparison, if it mixes naive and timezone-aware values).
        """
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return moment >= self.known_at
 
    @property
    def identity(self) -> SweepIdentity:
        """At most one Confirmation exists per Sweep, so its identity is the Sweep's."""
        return self.sweep.identity
 
 
@dataclass(frozen=True)
class Invalidation:
    """Immutable record: this completed 5M candle permanently INVALIDATED this Sweep's confirmation."""
 
    sweep: Sweep
    candle: Candle
    reason: InvalidationReason
 
    def __post_init__(self) -> None:
        _check_sweep_and_candle(self.sweep, self.candle)
        if not isinstance(self.reason, InvalidationReason):
            raise ValueError("reason must be an InvalidationReason")
        wrong = WRONG_CLASS_FOR[self.sweep.zone_type]
        if self.candle_class is not wrong:
            raise ValueError(f"a {self.sweep.zone_type.value} sweep is invalidated by a {wrong.value} candle")
        beyond = closes_beyond_boundary(self.sweep.zone_type, self.sweep.zone, self.candle)
        if beyond and self.reason is not InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY:
            raise ValueError("a candle that closes beyond the zone boundary is labelled CLOSED_BEYOND_ZONE_BOUNDARY")
        if not beyond and self.reason is InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY:
            raise ValueError("the candle does not close beyond the zone boundary")
 
    # -- derived values -------------------------------------------------------------
    @property
    def zone(self) -> Zone:
        return self.sweep.zone
 
    @property
    def zone_type(self) -> ZoneType:
        return self.sweep.zone_type
 
    @property
    def candle_class(self) -> CandleClass:
        return self.candle.classify(self.candle.close_time)
 
    @property
    def candle_time(self) -> datetime:
        return self.candle.open_time
 
    @property
    def known_at(self) -> datetime:
        """The invalidation is known only once its candle has CLOSED."""
        return self.candle.close_time
 
    def is_known_at(self, moment: datetime) -> bool:
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return moment >= self.known_at
 
    @property
    def identity(self) -> SweepIdentity:
        return self.sweep.identity