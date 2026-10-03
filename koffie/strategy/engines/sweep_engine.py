"""SweepEngine for Koffie Strategy 1.
 
Detects when a completed 5M candle sweeps the required liquidity of a zone.
 
    (zone, completed 5M candle) -> a Sweep, or None
 
For each candle the engine asks the injected LiquidityEngine for the zone's
required liquidity AS OF THE CANDLE'S OPEN TIME:
 
    liquidity_engine.required_liquidity_for(zone, candle.open_time)
 
That is the existing "newest relevant liquidity" selection, used unchanged. Only
levels already known when the candle opened can be swept, and a level that
becomes known later can become the required level for LATER candles.
 
A sweep needs a STRICT wick penetration beyond the required level:
DEMAND candle.low < level.price, SUPPLY candle.high > level.price. A close
beyond the level is not required.
 
One sweep per (zone, liquidity level) pair: the first penetrating candle
creates it; later candles that penetrate the same level create nothing new. A
different, newer required level can produce its own sweep.
 
This engine only READS the LiquidityEngine; it never stores or changes
liquidity, and no sweep state exists inside Liquidity. It contains no zone-touch
logic, no setup, confirmation, entry, SL/TP, risk, news or execution logic, no
BOS/CHOCH, and no tolerance, distance, ATR, displacement or session filter.
 
Zones of every timeframe (1H, 15M, 5M) may be supplied; each zone is tracked
independently by its own identity.
 
Lifecycle: sweeps are append-only (acceptance order). Nothing is deleted,
expired, invalidated or consumed here.
 
Integrity checks (no strategy behaviour):
- zone / candle / now have the right types; the candle is 5M and CLOSED at `now`;
- the candle opens at or after `zone.created_at` (a zone cannot be swept before
  it exists);
- a zone identity is always the same zone (a different zone with the same
  identity is rejected);
- candles for one zone arrive in chronological order without overlap.
 
Replay: supplying the candle that produced a stored sweep again returns that
sweep; supplying the latest processed candle of a zone again returns the same
result; both change nothing. Any other candle that opens before the previous
candle of that zone closed raises ValueError. A rejected call changes nothing.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import Dict, List, Optional, Tuple
 
from koffie.strategy.engines.liquidity_engine import LiquidityEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import BOSIdentity
from koffie.strategy.models.sweep import Sweep, SweepIdentity, penetrates
from koffie.strategy.models.zone import Zone
 
 
class SweepEngine:
    def __init__(self, liquidity_engine: LiquidityEngine) -> None:
        if not isinstance(liquidity_engine, LiquidityEngine):
            raise TypeError("liquidity_engine must be a LiquidityEngine")
        self._liquidity = liquidity_engine
        self._sweeps: List[Sweep] = []                                   # append-only, acceptance order
        self._by_identity: Dict[SweepIdentity, Sweep] = {}
        self._by_zone: Dict[BOSIdentity, List[Sweep]] = {}
        self._zones: Dict[BOSIdentity, Zone] = {}                        # zone identity -> the zone seen
        self._last: Dict[BOSIdentity, Tuple[Candle, Optional[Sweep]]] = {}  # latest candle + result per zone
 
    # ------------------------------------------------------------- read-only state
    @property
    def timeframe(self) -> Timeframe:
        return Timeframe.M5
 
    @property
    def sweeps(self) -> Tuple[Sweep, ...]:
        """Every sweep recorded, in the order they were accepted. Never removed."""
        return tuple(self._sweeps)
 
    def sweep_for(self, zone: Zone, level: LiquidityLevel) -> Optional[Sweep]:
        """The sweep recorded for this zone and liquidity level, or None."""
        return self._by_identity.get(SweepIdentity.from_parts(zone, level))
 
    def sweeps_for_zone(self, zone: Zone) -> Tuple[Sweep, ...]:
        """Every sweep recorded for this zone, in acceptance order."""
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        return tuple(s for s in self._by_zone.get(zone.identity, ()) if s.zone == zone)
 
    def sweeps_known_at(self, moment: datetime) -> Tuple[Sweep, ...]:
        """Sweeps whose candle had closed by `moment`, in acceptance order."""
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(s for s in self._sweeps if s.is_known_at(moment))
 
    # --------------------------------------------------------------------- input
    def process_candle(self, zone: Zone, candle: Candle, now: datetime) -> Optional[Sweep]:
        """Evaluate one completed 5M candle against one zone.
 
        Returns the new Sweep, or None when there is nothing to record (no
        relevant liquidity known at the candle's open, the required level was
        already swept, or the wick did not penetrate strictly beyond it).
 
        Raises TypeError for wrong argument types (or mixed naive/aware
        datetimes) and ValueError for a non-5M candle, a candle not closed at
        `now`, a candle opening before the zone existed, a conflicting zone or
        an out-of-order candle. Nothing is changed when an exception is raised.
        """
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if candle.timeframe is not Timeframe.M5:
            raise ValueError(
                f"SweepEngine works on 5M candles only, but received a {candle.timeframe.value} candle"
            )
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only completed candles may sweep")
        if candle.open_time < zone.created_at:
            raise ValueError("the zone did not exist yet when this candle opened")
 
        zone_id = zone.identity
        seen_zone = self._zones.get(zone_id)
        if seen_zone is not None and seen_zone != zone:
            raise ValueError("a different zone with the same identity was already seen; conflicting zone rejected")
 
        last = self._last.get(zone_id)
        if last is not None:
            last_candle, last_result = last
            if candle == last_candle:
                return last_result                                   # replay of the latest candle: no state change
            if candle.open_time < last_candle.close_time:
                for earlier in self._by_zone.get(zone_id, ()):
                    if earlier.candle == candle:
                        return earlier                               # replay of a sweeping candle: no state change
                raise ValueError(
                    "candles for a zone must arrive in chronological order without overlap "
                    "(candle opens before the previous candle closed)"
                )
 
        result: Optional[Sweep] = None
        level = self._liquidity.required_liquidity_for(zone, candle.open_time)
        if level is not None:
            identity = SweepIdentity.from_parts(zone, level)
            if identity not in self._by_identity and penetrates(zone, level, candle):
                result = Sweep(zone, level, candle)
 
        # Commit only after validation and construction succeeded.
        self._zones[zone_id] = zone
        self._last[zone_id] = (candle, result)
        if result is not None:
            self._sweeps.append(result)
            self._by_identity[result.identity] = result
            self._by_zone.setdefault(zone_id, []).append(result)
        return result