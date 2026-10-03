"""Setup model for Koffie Strategy 1.
 
A Setup is ONE independent trading opportunity on a Zone. It begins with the
first qualifying Zone Touch and then follows the existing Sweep and Confirmation
components:
 
    NO_SETUP -> ZONE_TOUCHED -> WAITING_FOR_SWEEP -> SWEPT
             -> WAITING_FOR_CONFIRMATION -> CONFIRMED -> TRADE_CREATED
    INVALIDATED is terminal.
 
Zone Touch (locked)
-------------------
A completed 5M candle touches a zone when its CLOSE is inside the zone, both
boundaries inclusive:
 
    zone.low <= candle.close <= zone.high
 
A wick into the zone with the close outside is NOT a touch. The candle's open is
irrelevant. The zone must already exist when the candle opens
(`candle.open_time >= zone.created_at`).
 
Single source of truth
----------------------
A Setup stores exactly TWO fields: the `Zone` and the qualifying touch `Candle`.
Everything else is derived. Sweep, Confirmation and Invalidation are NOT stored
here and no price is copied: they are owned by their own components and are read
through the SetupEngine. The mutable lifecycle status also lives in the engine
(as in ConfirmationEngine), so this record is immutable.
 
Identity
--------
A Setup is identified by its Zone and its touch event (`SetupIdentity`): the
zone's BOS identity plus the open time of the touch candle. The zone alone is NOT
an identity, because one persistent zone can have several independent Setups over
time.
 
This module is DATA ONLY (plus two pure helpers). It contains no sweep,
confirmation, entry, SL/TP, risk, news or execution logic, no BOS/CHOCH, no
timeout and no filter of any kind.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterable, Tuple
 
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.liquidity import LiquidityIdentity
from koffie.strategy.models.pivot import BOSIdentity
from koffie.strategy.models.zone import Zone, ZoneType
 
 
class SetupStatus(Enum):
    """The locked Setup lifecycle. Each member keeps its own distinct meaning."""
 
    NO_SETUP = "NO_SETUP"                                    # no Setup exists (never held by a Setup record)
    ZONE_TOUCHED = "ZONE_TOUCHED"
    WAITING_FOR_SWEEP = "WAITING_FOR_SWEEP"
    SWEPT = "SWEPT"
    WAITING_FOR_CONFIRMATION = "WAITING_FOR_CONFIRMATION"
    CONFIRMED = "CONFIRMED"
    TRADE_CREATED = "TRADE_CREATED"
    INVALIDATED = "INVALIDATED"                              # terminal, never revives
 
    @property
    def is_active(self) -> bool:
        """The Sweep/Confirmation sequence of this Setup is still running."""
        return self in (
            SetupStatus.ZONE_TOUCHED,
            SetupStatus.WAITING_FOR_SWEEP,
            SetupStatus.SWEPT,
            SetupStatus.WAITING_FOR_CONFIRMATION,
        )
 
    @property
    def is_resolved(self) -> bool:
        """The Sweep/Confirmation sequence is over (CONFIRMED, TRADE_CREATED or INVALIDATED)."""
        return self in (SetupStatus.CONFIRMED, SetupStatus.TRADE_CREATED, SetupStatus.INVALIDATED)
 
 
def is_touch(zone: Zone, candle: Candle) -> bool:
    """True when the candle CLOSES inside the zone, both boundaries inclusive.
 
    Pure check: it does not look at the open, the wicks, times, timeframes or
    whether the zone is known yet.
    """
    if not isinstance(zone, Zone):
        raise TypeError("zone must be a Zone")
    if not isinstance(candle, Candle):
        raise TypeError("candle must be a Candle")
    return zone.low <= candle.close <= zone.high
 
 
@dataclass(frozen=True)
class SetupIdentity:
    """Identity of one Setup: its Zone's BOS identity plus the open time of its touch candle."""
 
    zone_identity: BOSIdentity
    touch_time: datetime
 
    @classmethod
    def from_parts(cls, zone: Zone, touch_candle: Candle) -> "SetupIdentity":
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        if not isinstance(touch_candle, Candle):
            raise TypeError("touch_candle must be a Candle")
        return cls(zone_identity=zone.identity, touch_time=touch_candle.open_time)
 
 
@dataclass(frozen=True)
class Setup:
    """Immutable record of one independent Setup: a Zone and the candle that touched it."""
 
    zone: Zone
    touch_candle: Candle
 
    def __post_init__(self) -> None:
        if not isinstance(self.zone, Zone):
            raise ValueError("zone must be a Zone")
        if not isinstance(self.touch_candle, Candle):
            raise ValueError("touch_candle must be a Candle")
        if self.touch_candle.timeframe is not Timeframe.M5:
            raise ValueError("a zone touch is made by a completed 5M candle")
        if self.touch_candle.open_time < self.zone.created_at:
            raise ValueError("the zone was not known yet when the touch candle opened")
        if not is_touch(self.zone, self.touch_candle):
            raise ValueError("the touch candle does not close inside the zone")
 
    # -- derived values -------------------------------------------------------------
    @property
    def zone_type(self) -> ZoneType:
        """The direction: DEMAND = LONG, SUPPLY = SHORT."""
        return self.zone.zone_type
 
    @property
    def touch_time(self) -> datetime:
        """Open time of the touch candle."""
        return self.touch_candle.open_time
 
    @property
    def known_at(self) -> datetime:
        """The Setup exists only once its touch candle has CLOSED."""
        return self.touch_candle.close_time
 
    def is_known_at(self, moment: datetime) -> bool:
        """True only from `known_at` onward.
 
        Raises TypeError if `moment` is not a datetime (and, as for any datetime
        comparison, if it mixes naive and timezone-aware values).
        """
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return moment >= self.known_at
 
    @property
    def identity(self) -> SetupIdentity:
        return SetupIdentity.from_parts(self.zone, self.touch_candle)
 
 
@dataclass(frozen=True)
class SetupTransition:
    """One step of a Setup's lifecycle and the moment it became known."""
 
    status: SetupStatus
    at: datetime
 
    def __post_init__(self) -> None:
        if not isinstance(self.status, SetupStatus):
            raise ValueError("status must be a SetupStatus")
        if self.status is SetupStatus.NO_SETUP:
            raise ValueError("NO_SETUP is the absence of a Setup and cannot be a transition")
        if not isinstance(self.at, datetime):
            raise ValueError("at must be a datetime")
 
 
@dataclass(frozen=True)
class ConfirmationEventKey:
    """The causal Sweep + Confirmation event a confirmed Setup came from.
 
    Two confirmed Setups share an event when the same confirming candle confirmed a
    Sweep of the same liquidity level. One such event may produce at most ONE trade.
    """
 
    candle_time: datetime
    liquidity_identity: LiquidityIdentity
 
 
# Deterministic tie-break order for overlapping zones: 1H, then 15M, then 5M.
TIE_BREAK_RANK = {Timeframe.H1: 0, Timeframe.M15: 1, Timeframe.M5: 2}
 
 
def tie_break_order(setups: Iterable[Setup]) -> Tuple[Setup, ...]:
    """Order Setups 1H -> 15M -> 5M (stable for equal timeframes).
 
    This is ONLY a deterministic tie-breaker for Setups that genuinely share the
    same liquidity and confirmation event with no causal distinction. It never
    decides which zone is causal and never prefers a zone merely for being higher.
    """
    items = tuple(setups)
    for item in items:
        if not isinstance(item, Setup):
            raise TypeError("every item must be a Setup")
    return tuple(sorted(items, key=lambda s: TIE_BREAK_RANK[s.zone.timeframe]))