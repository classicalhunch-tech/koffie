"""ConfirmationEngine for Koffie Strategy 1.
 
Runs the locked confirmation sequence for each Sweep on completed 5M candles:
 
    (Sweep, completed 5M candle) -> a Confirmation, an Invalidation, or None
 
The caller supplies the Sweep (this engine does not own or read a SweepEngine) and then
the Sweep's candles in order. Each Sweep identity has its own independent state:
 
    WAITING_FOR_DECISIVE -> WAITING_FOR_SECOND_DECISIVE -> CONFIRMED | INVALIDATED
 
Transitions (LONG = DEMAND sweep; SHORT = SUPPLY sweep is the mirror image):
 
    BULLISH_DECISIVE (any close)                   -> CONFIRMED          (SHORT: BEARISH_DECISIVE)
    BEARISH_DECISIVE with close <  zone.low        -> INVALIDATED        (SHORT: BULLISH_DECISIVE, close > zone.high)
    BEARISH_DECISIVE otherwise, from the 1st state -> WAITING_FOR_SECOND_DECISIVE
    BEARISH_DECISIVE otherwise, from the 2nd state -> INVALIDATED
    NEUTRAL or ZERO_RANGE_ANOMALY                  -> no change (they never consume the sequence)
    anything once CONFIRMED or INVALIDATED         -> no change (terminal)
 
The boundary rule applies to every wrong-direction decisive candle, the first included. A
wick beyond the boundary never invalidates and a close exactly on it is not beyond it. A
confirming candle may close outside the zone. There is no timeout. Decisiveness comes from
Candle.classify; the BRR rule is not repeated here.
 
The first candle processed for a Sweep must be the Sweep's own candle, so evaluation
cannot silently start later; that candle may itself confirm, wait or invalidate. Later
candles may have gaps but must not overlap or go backwards.
 
`process_candle` returns the Confirmation or Invalidation THIS candle produced, otherwise
None (still waiting, or already terminal earlier); use `status_for(sweep)` for the state.
A Sweep produces at most one terminal event; later candles are still checked for validity
but never change a terminal result and return None.
 
Replay: supplying the latest candle of a Sweep again returns the same result; supplying
the candle that produced a terminal event returns that event; both change nothing. Any
other candle that opens before the previous one closed raises ValueError. A different
Sweep with an already-seen identity is rejected. A rejected call changes nothing.
 
This engine never creates or modifies swings, structure, BOS, CHOCH, pivots, zones,
liquidity, sweeps, setups, entries, SL/TP, risk, news or execution; it does not read any
other engine; and it contains no timeout and no filter beyond the locked sequence.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
 
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.confirmation import (
    CONFIRMING_CLASS_FOR,
    WRONG_CLASS_FOR,
    Confirmation,
    ConfirmationStatus,
    Invalidation,
    InvalidationReason,
    closes_beyond_boundary,
)
from koffie.strategy.models.sweep import Sweep, SweepIdentity
 
Event = Union[Confirmation, Invalidation]
 
_TERMINAL = (ConfirmationStatus.CONFIRMED, ConfirmationStatus.INVALIDATED)
 
 
class _Track:
    """Mutable per-Sweep state, private to the engine."""
 
    __slots__ = ("sweep", "status", "last_candle", "last_result", "terminal")
 
    def __init__(self, sweep: Sweep, status: ConfirmationStatus, candle: Candle, result: Optional[Event]) -> None:
        self.sweep = sweep
        self.status = status
        self.last_candle = candle
        self.last_result = result
        self.terminal: Optional[Event] = result
 
 
class ConfirmationEngine:
    def __init__(self) -> None:
        self._tracks: Dict[SweepIdentity, _Track] = {}
        self._confirmations: List[Confirmation] = []       # append-only, acceptance order
        self._invalidations: List[Invalidation] = []       # append-only, acceptance order
 
    # ------------------------------------------------------------- read-only state
    @property
    def timeframe(self) -> Timeframe:
        return Timeframe.M5
 
    @property
    def confirmations(self) -> Tuple[Confirmation, ...]:
        """Every Confirmation produced, in acceptance order. Never removed."""
        return tuple(self._confirmations)
 
    @property
    def invalidations(self) -> Tuple[Invalidation, ...]:
        """Every Invalidation produced, in acceptance order. Never removed."""
        return tuple(self._invalidations)
 
    def _tracked(self, sweep: Sweep) -> Optional[_Track]:
        if not isinstance(sweep, Sweep):
            raise TypeError("sweep must be a Sweep")
        track = self._tracks.get(sweep.identity)
        return track if track is not None and track.sweep == sweep else None
 
    def status_for(self, sweep: Sweep) -> Optional[ConfirmationStatus]:
        """The current state of this Sweep's confirmation sequence, or None if it was never processed."""
        track = self._tracked(sweep)
        return track.status if track is not None else None
 
    def confirmation_for(self, sweep: Sweep) -> Optional[Confirmation]:
        track = self._tracked(sweep)
        return track.terminal if track is not None and isinstance(track.terminal, Confirmation) else None
 
    def invalidation_for(self, sweep: Sweep) -> Optional[Invalidation]:
        track = self._tracked(sweep)
        return track.terminal if track is not None and isinstance(track.terminal, Invalidation) else None
 
    def confirmations_known_at(self, moment: datetime) -> Tuple[Confirmation, ...]:
        """Confirmations whose candle had closed by `moment`, in acceptance order."""
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(c for c in self._confirmations if c.is_known_at(moment))
 
    def invalidations_known_at(self, moment: datetime) -> Tuple[Invalidation, ...]:
        """Invalidations whose candle had closed by `moment`, in acceptance order."""
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(i for i in self._invalidations if i.is_known_at(moment))
 
    # --------------------------------------------------------------------- input
    def process_candle(self, sweep: Sweep, candle: Candle, now: datetime) -> Optional[Event]:
        """Evaluate one completed 5M candle for one Sweep.
 
        Returns the Confirmation or Invalidation this candle produced, otherwise None.
 
        Raises TypeError for wrong argument types (or mixed naive/aware datetimes) and
        ValueError for a non-5M candle, a candle not closed at `now`, a conflicting Sweep,
        a first candle that is not the Sweep's own candle, or an out-of-order candle.
        Nothing is changed when an exception is raised.
        """
        if not isinstance(sweep, Sweep):
            raise TypeError("sweep must be a Sweep")
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if candle.timeframe is not Timeframe.M5:
            raise ValueError(
                f"ConfirmationEngine works on 5M candles only, but received a {candle.timeframe.value} candle"
            )
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only completed candles may confirm")
 
        identity = sweep.identity
        track = self._tracks.get(identity)
        if track is not None and track.sweep != sweep:
            raise ValueError("a different sweep with the same identity was already seen; conflicting sweep rejected")
 
        if track is None:
            if candle != sweep.candle:
                raise ValueError("the first candle processed for a sweep must be the sweep's own candle")
        else:
            if candle == track.last_candle:
                return track.last_result                          # replay of the latest candle: no state change
            if candle.open_time < track.last_candle.close_time:
                if track.terminal is not None and track.terminal.candle == candle:
                    return track.terminal                         # replay of the terminal candle: no state change
                raise ValueError(
                    "candles for a sweep must arrive in chronological order without overlap "
                    "(candle opens before the previous candle closed)"
                )
 
        previous = ConfirmationStatus.WAITING_FOR_DECISIVE if track is None else track.status
        status, event = self._evaluate(sweep, candle, now, previous)
 
        # Commit only after validation and construction succeeded.
        if track is None:
            self._tracks[identity] = _Track(sweep, status, candle, event)
        else:
            track.status = status
            track.last_candle = candle
            track.last_result = event
            if event is not None:
                track.terminal = event
        if isinstance(event, Confirmation):
            self._confirmations.append(event)
        elif isinstance(event, Invalidation):
            self._invalidations.append(event)
        return event
 
    @staticmethod
    def _evaluate(
        sweep: Sweep, candle: Candle, now: datetime, previous: ConfirmationStatus
    ) -> Tuple[ConfirmationStatus, Optional[Event]]:
        """The locked sequence: (current state, completed candle) -> (new state, event)."""
        if previous in _TERMINAL:
            return previous, None                                  # a terminal result never changes
        zone_type = sweep.zone_type
        candle_class = candle.classify(now)                        # the existing classifier, closed candles only
        if candle_class is CONFIRMING_CLASS_FOR[zone_type]:
            return ConfirmationStatus.CONFIRMED, Confirmation(sweep, candle)
        if candle_class is WRONG_CLASS_FOR[zone_type]:
            if closes_beyond_boundary(zone_type, sweep.zone, candle):
                return (
                    ConfirmationStatus.INVALIDATED,
                    Invalidation(sweep, candle, InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY),
                )
            if previous is ConfirmationStatus.WAITING_FOR_SECOND_DECISIVE:
                return (
                    ConfirmationStatus.INVALIDATED,
                    Invalidation(sweep, candle, InvalidationReason.SECOND_WRONG_DIRECTION_DECISIVE),
                )
            return ConfirmationStatus.WAITING_FOR_SECOND_DECISIVE, None
        return previous, None                                      # NEUTRAL / ZERO_RANGE_ANOMALY: keep waiting