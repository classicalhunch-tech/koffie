"""ZoneEngine for Koffie Strategy 1.
 
Turns Pivot outcomes into supply/demand zones for ONE timeframe.
 
    PivotOutcome FOUND           -> one Zone (bullish BOS = DEMAND, bearish = SUPPLY)
    PivotOutcome NO_PIVOT_FOUND  -> no Zone; nothing is invented
 
The engine receives a PivotOutcome and `now`; it never sees candles, never
imports PivotEngine or any other engine, and adds no rule about zone size,
freshness, strength, touches or ranking.
 
Lifecycle: zones persist. Nothing here deletes, invalidates, mitigates, merges
or ranks a zone. Zones that overlap stay separate records, each with its own
causal chain (Zone -> Pivot -> BOS -> broken swing). Whether a zone may take
part in a setup is decided by a later component, not here.
 
Time: a zone is created when its BOS confirms (`Zone.created_at`). It must not
be treated as existing before that, which is why `now` may not be earlier than
the outcome's `confirmed_at`, and why `zones_known_at(moment)` only returns
zones whose `created_at` has been reached.
 
Idempotency: every outcome is keyed by its BOSIdentity. Supplying the exact same
outcome again returns the stored result and changes nothing. The same identity
with a different outcome raises ValueError (conflicting replay).
 
Integrity checks (no strategy behaviour): the outcome is a PivotOutcome on this
timeframe; `now` is a datetime not earlier than the outcome's confirmed_at; new
outcomes arrive with non-decreasing confirmed_at. A rejected call changes
nothing.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import Dict, List, Optional, Tuple
 
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.pivot import BOSIdentity, PivotOutcome
from koffie.strategy.models.zone import Zone
 
 
class ZoneEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
        self._zones: List[Zone] = []                       # append-only, creation order
        self._outcomes: List[PivotOutcome] = []            # append-only, every outcome seen
        self._by_identity: Dict[BOSIdentity, Tuple[PivotOutcome, Optional[Zone]]] = {}
 
    # ------------------------------------------------------------- read-only state
    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe
 
    @property
    def zones(self) -> Tuple[Zone, ...]:
        """Every zone created, oldest first. Zones are never removed."""
        return tuple(self._zones)
 
    @property
    def outcomes(self) -> Tuple[PivotOutcome, ...]:
        """Every accepted outcome (FOUND and NO_PIVOT_FOUND), oldest first."""
        return tuple(self._outcomes)
 
    def zone_for(self, identity: BOSIdentity) -> Optional[Zone]:
        """The zone created for this BOS identity, or None."""
        if not isinstance(identity, BOSIdentity):
            raise TypeError("identity must be a BOSIdentity")
        entry = self._by_identity.get(identity)
        return entry[1] if entry is not None else None
 
    def zones_known_at(self, moment: datetime) -> Tuple[Zone, ...]:
        """Zones whose `created_at` has been reached at `moment`, oldest first.
 
        A zone is not returned before its BOS confirmed it, even though its
        Pivot candle (origin_time) is earlier.
        """
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(z for z in self._zones if z.is_known_at(moment))
 
    # --------------------------------------------------------------------- input
    def process_outcome(self, outcome: PivotOutcome, now: datetime) -> Optional[Zone]:
        """Accept one PivotOutcome; return its Zone, or None for NO_PIVOT_FOUND.
 
        Raises TypeError for wrong argument types and ValueError for a wrong
        timeframe, a `now` before the BOS confirmed, a conflicting replay or an
        out-of-order outcome. Nothing is changed when an exception is raised.
        """
        if not isinstance(outcome, PivotOutcome):
            raise TypeError("outcome must be a PivotOutcome")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if outcome.bos.timeframe is not self._timeframe:
            raise ValueError(
                f"ZoneEngine is configured for {self._timeframe.value}, "
                f"but received a {outcome.bos.timeframe.value} outcome"
            )
        if now < outcome.confirmed_at:
            raise ValueError("the BOS is not confirmed yet at `now`; a zone cannot exist before created_at")
 
        identity = outcome.identity
        stored = self._by_identity.get(identity)
        if stored is not None:
            stored_outcome, stored_zone = stored
            if outcome != stored_outcome:
                raise ValueError(
                    "this BOS identity was already recorded with a different outcome; "
                    "conflicting replay rejected"
                )
            return stored_zone                                  # exact replay: no state change
 
        if self._outcomes and outcome.confirmed_at < self._outcomes[-1].confirmed_at:
            raise ValueError("outcomes must arrive in non-decreasing confirmed_at order")
 
        zone = Zone(outcome.pivot) if outcome.is_found else None
 
        # Commit only after validation and construction succeeded.
        self._outcomes.append(outcome)
        self._by_identity[identity] = (outcome, zone)
        if zone is not None:
            self._zones.append(zone)
        return zone
