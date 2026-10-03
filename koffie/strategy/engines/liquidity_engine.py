"""LiquidityEngine for Koffie Strategy 1.
 
Records 5M liquidity levels from confirmed swings. It is configured for the 5M
timeframe only.
 
    confirmed 5M Swing -> one LiquidityLevel (every accepted swing, no filter)
 
The engine receives swings that a SwingEngine has already confirmed, plus
`now`. It never sees candles and never imports any other engine.
 
Lifecycle: levels are append-only. Nothing here deletes, expires, merges,
clusters, ranks or consumes a level, and no sweep state exists. Whether a level
has been swept belongs to a later component.
 
Queries take a Zone as an argument. The engine never stores a zone and does not
depend on Zone state (it does not check whether the zone is known yet); that
gate belongs to a later component.
 
    levels_for(zone, moment)             known levels relevant to the zone
    required_liquidity_for(zone, moment) the newest of those, or None
 
Relevance (see LiquidityLevel.is_relevant_to): DEMAND -> swing LOW strictly
below the zone low; SUPPLY -> swing HIGH strictly above the zone high. No
distance limit, no tolerance, no time cutoff; a 5M level can serve a zone of
any timeframe. "Newest" = latest `known_at`, then the highest swing `sequence`.
 
Idempotency: every swing is keyed by its LiquidityIdentity. Supplying the exact
same swing again returns the stored level and changes nothing. The same
identity with different values raises ValueError (conflicting replay).
 
Integrity checks (no strategy behaviour): the swing is a Swing on this
timeframe; `now` is a datetime not earlier than the swing's confirmed_at; new
swings arrive with non-decreasing confirmed_at. A rejected call changes
nothing.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import Dict, List, Optional, Tuple
 
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.liquidity import LiquidityIdentity, LiquidityLevel
from koffie.strategy.models.swing import Swing
from koffie.strategy.models.zone import Zone
 
 
class LiquidityEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        if timeframe is not Timeframe.M5:
            raise ValueError("LiquidityEngine is configured for the 5M timeframe only")
        self._timeframe = timeframe
        self._levels: List[LiquidityLevel] = []                    # append-only, acceptance order
        self._by_identity: Dict[LiquidityIdentity, LiquidityLevel] = {}
 
    # ------------------------------------------------------------- read-only state
    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe
 
    @property
    def levels(self) -> Tuple[LiquidityLevel, ...]:
        """Every level recorded, oldest first. Levels are never removed."""
        return tuple(self._levels)
 
    def level_for(self, identity: LiquidityIdentity) -> Optional[LiquidityLevel]:
        """The level recorded for this identity, or None."""
        if not isinstance(identity, LiquidityIdentity):
            raise TypeError("identity must be a LiquidityIdentity")
        return self._by_identity.get(identity)
 
    def levels_known_at(self, moment: datetime) -> Tuple[LiquidityLevel, ...]:
        """Levels whose `known_at` has been reached at `moment`, oldest first."""
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(level for level in self._levels if level.is_known_at(moment))
 
    def levels_for(self, zone: Zone, moment: datetime) -> Tuple[LiquidityLevel, ...]:
        """Known levels that are relevant to `zone` at `moment`, oldest first."""
        if not isinstance(zone, Zone):
            raise TypeError("zone must be a Zone")
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return tuple(
            level
            for level in self._levels
            if level.is_known_at(moment) and level.is_relevant_to(zone)
        )
 
    def required_liquidity_for(self, zone: Zone, moment: datetime) -> Optional[LiquidityLevel]:
        """The newest relevant known level, or None.
 
        Newest = latest `known_at`, then the highest swing `sequence`.
        """
        candidates = self.levels_for(zone, moment)
        if not candidates:
            return None
        return max(candidates, key=lambda level: (level.known_at, level.sequence))
 
    # --------------------------------------------------------------------- input
    def process_swing(self, swing: Swing, now: datetime) -> LiquidityLevel:
        """Accept one confirmed swing and return its LiquidityLevel.
 
        Raises TypeError for wrong argument types and ValueError for a wrong
        timeframe, a `now` before the swing was confirmed, a conflicting replay
        or an out-of-order swing. Nothing is changed when an exception is raised.
        """
        if not isinstance(swing, Swing):
            raise TypeError("swing must be a Swing")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if swing.timeframe is not self._timeframe:
            raise ValueError(
                f"LiquidityEngine is configured for {self._timeframe.value}, "
                f"but received a {swing.timeframe.value} swing"
            )
        if now < swing.confirmed_at:
            raise ValueError("the swing is not confirmed yet at `now`; a level cannot exist before known_at")
 
        identity = LiquidityIdentity.from_swing(swing)
        stored = self._by_identity.get(identity)
        if stored is not None:
            if stored.swing != swing:
                raise ValueError(
                    "this swing identity was already recorded with different values; "
                    "conflicting replay rejected"
                )
            return stored                                       # exact replay: no state change
 
        if self._levels and swing.confirmed_at < self._levels[-1].known_at:
            raise ValueError("swings must arrive in non-decreasing confirmed_at order")
 
        level = LiquidityLevel(swing)
 
        # Commit only after validation and construction succeeded.
        self._levels.append(level)
        self._by_identity[identity] = level
        return level