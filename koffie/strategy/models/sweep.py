"""Sweep model for Koffie Strategy 1.
 
A Sweep records that ONE completed 5M candle penetrated the required liquidity
level of ONE zone:
 
    DEMAND zone: the candle's LOW  is strictly BELOW the liquidity level
    SUPPLY zone: the candle's HIGH is strictly ABOVE the liquidity level
 
Strict means a wick that only reaches the level exactly is NOT a sweep. A wick
is sufficient: the candle does NOT have to close beyond the level, and where the
candle closes is irrelevant to the sweep itself.
 
Single source of truth
----------------------
A Sweep stores exactly three fields: the `Zone`, the `LiquidityLevel` and the
sweeping `Candle`. Everything else is derived from them:
 
    Sweep -> Zone -> Pivot -> BOS -> broken structural swing
    Sweep -> LiquidityLevel -> Swing
    Sweep -> Candle
 
The whole candle is kept (not just a price) so later components can read its
high and low; this module itself uses no SL/TP.
 
Causality (validated on construction)
-------------------------------------
- the candle and the level are both 5M;
- the level is relevant to the zone (LiquidityLevel.is_relevant_to);
- the level was already known when the candle OPENED
  (`level.known_at <= candle.open_time`);
- the zone was already known when the candle OPENED
  (`zone.created_at <= candle.open_time`);
- the candle's wick penetrates the level strictly.
 
A Sweep is known only when its candle has CLOSED (`known_at` = candle close time).
 
Identity
--------
One sweep exists per (zone, liquidity level) pair: `SweepIdentity`. Which candle
produced it is NOT part of the identity.
 
This module is DATA ONLY. It contains no zone-touch logic, no setup,
confirmation, entry, SL/TP, risk, news or execution logic, no BOS/CHOCH, no
tolerance, distance limit, ATR, displacement or other filter, and no flag of
any kind.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from datetime import datetime
 
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.liquidity import LiquidityIdentity, LiquidityLevel
from koffie.strategy.models.pivot import BOSIdentity
from koffie.strategy.models.zone import Zone, ZoneType
 
 
def penetrates(zone: Zone, level: LiquidityLevel, candle: Candle) -> bool:
    """True when the candle's wick goes strictly beyond the liquidity level.
 
    DEMAND: candle.low < level.price.   SUPPLY: candle.high > level.price.
    Pure check: it does not look at times, timeframes or relevance.
    """
    if not isinstance(zone, Zone):
        raise TypeError("zone must be a Zone")
    if not isinstance(level, LiquidityLevel):
        raise TypeError("level must be a LiquidityLevel")
    if not isinstance(candle, Candle):
        raise TypeError("candle must be a Candle")
    if zone.zone_type is ZoneType.DEMAND:
        return candle.low < level.price
    return candle.high > level.price
 
 
@dataclass(frozen=True)
class SweepIdentity:
    """Identity of one sweep: the zone's BOS identity plus the level's identity."""
 
    zone_identity: BOSIdentity
    liquidity_identity: LiquidityIdentity
 
    @classmethod
    def from_parts(cls, zone: Zone, level: LiquidityLevel) -> "SweepIdentity":
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        if not isinstance(level, LiquidityLevel):
            raise TypeError("level must be a LiquidityLevel")
        return cls(zone_identity=zone.identity, liquidity_identity=level.identity)
 
 
@dataclass(frozen=True)
class Sweep:
    """Immutable record of one completed 5M candle sweeping a zone's liquidity."""
 
    zone: Zone
    level: LiquidityLevel
    candle: Candle
 
    def __post_init__(self) -> None:
        if not isinstance(self.zone, Zone):
            raise ValueError("zone must be a Zone")
        if not isinstance(self.level, LiquidityLevel):
            raise ValueError("level must be a LiquidityLevel")
        if not isinstance(self.candle, Candle):
            raise ValueError("candle must be a Candle")
        if self.candle.timeframe is not Timeframe.M5:
            raise ValueError("a sweep is made by a 5M candle")
        if self.level.timeframe is not Timeframe.M5:
            raise ValueError("a sweep is of 5M liquidity")
        if not self.level.is_relevant_to(self.zone):
            raise ValueError("the liquidity level is not relevant to this zone")
        if self.candle.open_time < self.level.known_at:
            raise ValueError("the liquidity level was not known yet when the candle opened")
        if self.candle.open_time < self.zone.created_at:
            raise ValueError("the zone was not known yet when the candle opened")
        if not penetrates(self.zone, self.level, self.candle):
            raise ValueError("the candle's wick does not penetrate strictly beyond the liquidity level")
 
    # -- derived values -------------------------------------------------------------
    @property
    def zone_type(self) -> ZoneType:
        return self.zone.zone_type
 
    @property
    def swept_price(self) -> float:
        """The liquidity level's price."""
        return self.level.price
 
    @property
    def candle_time(self) -> datetime:
        """Open time of the sweeping candle."""
        return self.candle.open_time
 
    @property
    def known_at(self) -> datetime:
        """The sweep is known only once its candle has closed."""
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
        return SweepIdentity.from_parts(self.zone, self.level)
