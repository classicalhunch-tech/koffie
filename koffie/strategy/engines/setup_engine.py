"""SetupEngine for Koffie Strategy 1.
 
Coordinates the Setup lifecycle on completed 5M candles:
 
    (Zone, completed 5M candle) -> the Setup this candle started or advanced, or None
 
    NO_SETUP -> ZONE_TOUCHED -> WAITING_FOR_SWEEP -> SWEPT
             -> WAITING_FOR_CONFIRMATION -> CONFIRMED -> TRADE_CREATED
    INVALIDATED is terminal.
 
The engine only COORDINATES. Touch is the minimal locked test (close inside the
zone, both boundaries inclusive). Sweeps are produced by the existing SweepEngine
and confirmation by the existing ConfirmationEngine; this engine never repeats
their rules, never classifies candles and never creates or changes zones,
liquidity, sweeps or confirmations.
 
Independent Setups
------------------
Every Setup owns a PRIVATE SweepEngine and a PRIVATE ConfirmationEngine, created
when its touch starts it and fed only from that touch candle onward. This is how
a persistent zone can later start a completely new Setup with a genuinely new
Sweep: the existing engines record one Sweep per (zone, liquidity level) pair and
one confirmation track per Sweep identity, so a shared instance could never
produce a second Sweep of the same level. Per-Setup instances need no change to
those engines, and a Sweep that happened BEFORE a touch never reaches a Setup
because that Setup's SweepEngine has not seen those candles. The shared
LiquidityEngine is only read.
 
Per candle, in this fixed order and with no artificial delay between steps:
 
    1. active Setup of the zone: feed the candle to its sweep stage, then (if a
       Sweep exists) to its confirmation stage; a later touch is ignored
    2. no active Setup, zone not consumed, candle closes inside the zone:
       a new Setup starts and the SAME candle is fed to its sweep stage, then to
       its confirmation stage if it swept
 
So touch, Sweep and Confirmation can all happen on one candle. A candle advances
at most one Setup of a zone: the candle that resolves a Setup cannot also be the
touch that starts the next one.
 
A Setup is "active" until its Sweep/Confirmation sequence is over (CONFIRMED,
TRADE_CREATED or INVALIDATED). A resolved Setup stays in history; a new
qualifying touch may start a new independent Setup while the zone is unconsumed.
There is no timeout and no other invalidation rule while waiting.
 
Trade creation / zone consumption
---------------------------------
Confirmation and invalidation never consume a zone. Only `record_trade_created`,
called by the downstream Entry/Trade component once a trade is ACTUALLY created,
moves a CONFIRMED Setup to TRADE_CREATED and consumes its zone for entry. A zone
produces at most one entry, and one causal confirmation event (same confirming
candle, same liquidity) produces at most one trade, even when overlapping zones
were all confirmed by it. Other Setups, confirmed or not, stay in history. This
engine never builds an entry, SL, TP, risk check or order.
 
Integrity checks (no strategy behaviour): zone / candle / now have the right
types; the candle is 5M and CLOSED at `now`; it opens at or after the zone
existed; a zone identity is always the same zone; candles for one zone arrive in
chronological order without overlap. Replay: supplying the latest candle of a zone
again, or a candle that started or advanced a Setup, returns that result and
changes nothing. A rejected call changes nothing.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import Dict, List, Optional, Tuple
 
from koffie.strategy.engines.confirmation_engine import ConfirmationEngine
from koffie.strategy.engines.liquidity_engine import LiquidityEngine
from koffie.strategy.engines.sweep_engine import SweepEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.confirmation import Confirmation, Invalidation
from koffie.strategy.models.pivot import BOSIdentity
from koffie.strategy.models.setup import (
    ConfirmationEventKey,
    Setup,
    SetupIdentity,
    SetupStatus,
    SetupTransition,
    is_touch,
)
from koffie.strategy.models.sweep import Sweep
from koffie.strategy.models.zone import Zone
 
 
class _Track:
    """Mutable per-Setup state, private to the engine."""
 
    __slots__ = (
        "setup", "status", "sweep_engine", "confirmation_engine",
        "sweep", "confirmation", "invalidation", "transitions",
    )
 
    def __init__(self, setup: Setup, liquidity: LiquidityEngine) -> None:
        self.setup = setup
        self.status = SetupStatus.ZONE_TOUCHED
        self.sweep_engine = SweepEngine(liquidity)
        self.confirmation_engine = ConfirmationEngine()
        self.sweep: Optional[Sweep] = None
        self.confirmation: Optional[Confirmation] = None
        self.invalidation: Optional[Invalidation] = None
        self.transitions: List[SetupTransition] = [SetupTransition(SetupStatus.ZONE_TOUCHED, setup.known_at)]
 
 
class SetupEngine:
    def __init__(self, liquidity_engine: LiquidityEngine) -> None:
        if not isinstance(liquidity_engine, LiquidityEngine):
            raise TypeError("liquidity_engine must be a LiquidityEngine")
        self._liquidity = liquidity_engine                                  # only READ
        self._setups: List[Setup] = []                                      # append-only, acceptance order
        self._tracks: Dict[SetupIdentity, _Track] = {}
        self._by_zone: Dict[BOSIdentity, List[SetupIdentity]] = {}
        self._zones: Dict[BOSIdentity, Zone] = {}                           # zone identity -> the zone seen
        self._last: Dict[BOSIdentity, Tuple[Candle, Optional[Setup]]] = {}  # latest candle + result per zone
        self._changes: Dict[BOSIdentity, List[Tuple[Candle, Setup]]] = {}   # candles that started/advanced a Setup
        self._consumed: Dict[BOSIdentity, Setup] = {}                       # zone identity -> Setup that took the entry
        self._trade_events: Dict[ConfirmationEventKey, Setup] = {}          # causal event -> Setup that took the trade
 
    # ------------------------------------------------------------- read-only state
    @property
    def timeframe(self) -> Timeframe:
        return Timeframe.M5
 
    @property
    def setups(self) -> Tuple[Setup, ...]:
        """Every Setup ever started, in the order they started. Never removed."""
        return tuple(self._setups)
 
    def _tracked(self, setup: Setup) -> Optional[_Track]:
        if not isinstance(setup, Setup):
            raise TypeError("setup must be a Setup")
        track = self._tracks.get(setup.identity)
        return track if track is not None and track.setup == setup else None
 
    def setups_for_zone(self, zone: Zone) -> Tuple[Setup, ...]:
        """Every Setup of this zone, in the order they started."""
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        tracks = (self._tracks[i] for i in self._by_zone.get(zone.identity, ()))
        return tuple(t.setup for t in tracks if t.setup.zone == zone)
 
    def active_setup_for(self, zone: Zone) -> Optional[Setup]:
        """The zone's Setup whose Sweep/Confirmation sequence is still running, or None."""
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        track = self._active_track(zone.identity)
        return track.setup if track is not None and track.setup.zone == zone else None
 
    def status_for(self, setup: Setup) -> Optional[SetupStatus]:
        """The current lifecycle status of this Setup, or None if it is not tracked."""
        track = self._tracked(setup)
        return track.status if track is not None else None
 
    def status_for_zone(self, zone: Zone) -> SetupStatus:
        """Status of the zone's active Setup, or NO_SETUP when it has none."""
        active = self.active_setup_for(zone)
        return self._tracks[active.identity].status if active is not None else SetupStatus.NO_SETUP
 
    def sweep_for(self, setup: Setup) -> Optional[Sweep]:
        track = self._tracked(setup)
        return track.sweep if track is not None else None
 
    def confirmation_for(self, setup: Setup) -> Optional[Confirmation]:
        track = self._tracked(setup)
        return track.confirmation if track is not None else None
 
    def invalidation_for(self, setup: Setup) -> Optional[Invalidation]:
        track = self._tracked(setup)
        return track.invalidation if track is not None else None
 
    def transitions_for(self, setup: Setup) -> Tuple[SetupTransition, ...]:
        """Every lifecycle step of this Setup in order (empty if it is not tracked)."""
        track = self._tracked(setup)
        return tuple(track.transitions) if track is not None else ()
 
    def confirmed_setups_known_at(self, moment: datetime) -> Tuple[Setup, ...]:
        """CONFIRMED Setups (no trade yet) whose confirmation had closed by `moment`, oldest first.
 
        This is what the future Entry component reads. It never includes a Setup whose
        zone has been consumed.
        """
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(
            t.setup for t in (self._tracks[s.identity] for s in self._setups)
            if t.status is SetupStatus.CONFIRMED and t.confirmation.is_known_at(moment)
            and t.setup.zone.identity not in self._consumed
        )
 
    def confirmation_event_key_for(self, setup: Setup) -> Optional[ConfirmationEventKey]:
        """The causal event a CONFIRMED (or TRADE_CREATED) Setup came from, else None."""
        track = self._tracked(setup)
        return self._event_key(track) if track is not None else None
 
    @staticmethod
    def _event_key(track: _Track) -> Optional[ConfirmationEventKey]:
        if track.confirmation is None:
            return None
        return ConfirmationEventKey(track.confirmation.candle.open_time, track.sweep.level.identity)
 
    def is_zone_consumed(self, zone: Zone) -> bool:
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        return zone.identity in self._consumed
 
    def consumed_by(self, zone: Zone) -> Optional[Setup]:
        """The Setup whose actual trade consumed this zone for entry, or None."""
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        return self._consumed.get(zone.identity)
 
    def can_create_trade(self, setup: Setup) -> bool:
        """True when this Setup is CONFIRMED, its zone is unconsumed and its causal event has no trade yet."""
        track = self._tracked(setup)
        if track is None or track.status is not SetupStatus.CONFIRMED:
            return False
        return (setup.zone.identity not in self._consumed
                and self._event_key(track) not in self._trade_events)
 
    # --------------------------------------------------------------------- input
    def process_candle(self, zone: Zone, candle: Candle, now: datetime) -> Optional[Setup]:
        """Evaluate one completed 5M candle against one zone.
 
        Returns the Setup this candle STARTED or whose status it CHANGED, otherwise
        None (no touch, nothing happened, or the zone is consumed).
 
        Raises TypeError for wrong argument types (or mixed naive/aware datetimes) and
        ValueError for a non-5M candle, a candle not closed at `now`, a candle opening
        before the zone existed, a conflicting zone or an out-of-order candle. Nothing
        is changed when an exception is raised.
        """
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if candle.timeframe is not Timeframe.M5:
            raise ValueError(
                f"SetupEngine works on 5M candles only, but received a {candle.timeframe.value} candle"
            )
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only completed candles may advance a setup")
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
                for earlier_candle, earlier_result in self._changes.get(zone_id, ()):
                    if earlier_candle == candle:
                        return earlier_result                        # replay of an earlier result candle: no state change
                raise ValueError(
                    "candles for a zone must arrive in chronological order without overlap "
                    "(candle opens before the previous candle closed)"
                )
 
        result: Optional[Setup] = None
        active = self._active_track(zone_id)
        if active is not None:
            if self._advance(active, candle, now):
                result = active.setup
        elif zone_id not in self._consumed and is_touch(zone, candle):
            track = _Track(Setup(zone, candle), self._liquidity)    # not registered until it has been fed
            track.status = SetupStatus.WAITING_FOR_SWEEP
            track.transitions.append(SetupTransition(SetupStatus.WAITING_FOR_SWEEP, candle.close_time))
            self._advance(track, candle, now)                       # the touch candle itself may sweep (and confirm)
            self._register(track)
            result = track.setup
 
        # Commit only after validation and construction succeeded.
        self._zones[zone_id] = zone
        self._last[zone_id] = (candle, result)
        if result is not None:
            self._changes.setdefault(zone_id, []).append((candle, result))
        return result
 
    def record_trade_created(self, setup: Setup, now: datetime) -> None:
        """Called by the downstream Entry/Trade component when a trade was ACTUALLY created.
 
        Moves a CONFIRMED Setup to TRADE_CREATED and consumes its zone for entry. Calling it
        again for the same Setup is a no-op. Raises TypeError for wrong types and ValueError
        when the Setup is unknown or not CONFIRMED, `now` precedes the confirmation, the zone
        is already consumed, or the same causal confirmation event already produced a trade.
        Nothing is changed when an exception is raised.
        """
        if not isinstance(setup, Setup):
            raise TypeError("setup must be a Setup")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        track = self._tracked(setup)
        if track is None:
            raise ValueError("this setup is not tracked by the engine")
        if track.status is SetupStatus.TRADE_CREATED:
            return                                                   # replay: no state change
        if track.status is not SetupStatus.CONFIRMED:
            raise ValueError(f"a trade can only follow a CONFIRMED setup, not {track.status.value}")
        if not track.confirmation.is_known_at(now):
            raise ValueError("a trade cannot be created before the confirmation is known")
        zone_id = setup.zone.identity
        if zone_id in self._consumed:
            raise ValueError("this zone was already consumed for entry")
        key = self._event_key(track)
        if key in self._trade_events:
            raise ValueError("this causal confirmation event already produced a trade")
 
        # Commit only after validation succeeded.
        track.status = SetupStatus.TRADE_CREATED
        track.transitions.append(SetupTransition(SetupStatus.TRADE_CREATED, now))
        self._consumed[zone_id] = setup
        self._trade_events[key] = setup
 
    # ------------------------------------------------------------------ internals
    def _active_track(self, zone_id: BOSIdentity) -> Optional[_Track]:
        for identity in reversed(self._by_zone.get(zone_id, ())):
            track = self._tracks[identity]
            if track.status.is_active:
                return track
        return None
 
    def _register(self, track: _Track) -> None:
        identity = track.setup.identity
        self._setups.append(track.setup)
        self._tracks[identity] = track
        self._by_zone.setdefault(track.setup.zone.identity, []).append(identity)
 
    @staticmethod
    def _move(track: _Track, status: SetupStatus, at: datetime) -> None:
        track.status = status
        track.transitions.append(SetupTransition(status, at))
 
    def _advance(self, track: _Track, candle: Candle, now: datetime) -> bool:
        """Feed one candle to the Setup's own Sweep and Confirmation stages. True if its status changed."""
        before = track.status
        if track.status is SetupStatus.WAITING_FOR_SWEEP:
            sweep = track.sweep_engine.process_candle(track.setup.zone, candle, now)
            if sweep is not None:
                track.sweep = sweep
                self._move(track, SetupStatus.SWEPT, candle.close_time)
                self._move(track, SetupStatus.WAITING_FOR_CONFIRMATION, candle.close_time)
        if track.status is SetupStatus.WAITING_FOR_CONFIRMATION:
            event = track.confirmation_engine.process_candle(track.sweep, candle, now)
            if isinstance(event, Confirmation):
                track.confirmation = event
                self._move(track, SetupStatus.CONFIRMED, candle.close_time)
            elif isinstance(event, Invalidation):
                track.invalidation = event
                self._move(track, SetupStatus.INVALIDATED, candle.close_time)
        return track.status is not before