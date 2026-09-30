"""BOSEngine for Koffie Strategy 1 (spec section 11).

Detects BOS (Break of Structure = continuation) on ONE timeframe from a CLOSED
candle and the current StructureSnapshot.

    BULLISH structure: candle.close strictly above the active swing HIGH -> bullish BOS
    BEARISH structure: candle.close strictly below the active swing LOW  -> bearish BOS
    UNDETERMINED / REVALUATING: no BOS is emitted (spec defines BOS only for
    bullish and bearish markets).

Only the CLOSE counts; a wick beyond the level is not a BOS. Every qualifying
close is its own BOS event (a repeated close beyond the same level emits again).

Ordering: within one candle, swings are confirmed and StructureEngine is
updated FIRST; the snapshot passed here must be that updated snapshot.

"CHOCH only" rule (Option A): if the same candle's range also goes strictly
beyond the PROTECTED swing (its low below the active low in BULLISH, its high
above the active high in BEARISH), no BOS is emitted, because a CHOCH is
recorded for that candle instead. This engine only compares prices for that
purpose; it creates no CHOCH and never changes any state.

NOT here: CHOCH detection, StructureEngine changes, pivots, zones, liquidity,
entries, risk, TP/SL or execution.

Integrity checks (no strategy behaviour): candle is a Candle on this timeframe
and closed at `now`; it must open at or after the previous candle's close (gaps
are fine, overlaps and repeats are not); the snapshot is a StructureSnapshot on
this timeframe and holds no swing confirmed after this candle closed (that
would be lookahead). A rejected call changes nothing.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.structure import StructureSnapshot, StructureState


class BOSEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
        self._history: List[BOS] = []
        self._last_candle: Optional[Candle] = None

    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def history(self) -> Tuple[BOS, ...]:
        """All BOS events in the order they occurred, oldest first."""
        return tuple(self._history)

    def process_candle(
        self, candle: Candle, now: datetime, snapshot: StructureSnapshot
    ) -> Optional[BOS]:
        """Evaluate one CLOSED candle against the (already updated) structure.

        Returns the BOS event, or None. Raises TypeError for wrong argument
        types and ValueError for a wrong timeframe, an unfinished, repeated or
        overlapping candle, or a snapshot containing future swings. Nothing is
        changed when an exception is raised.
        """
        self._validate(candle, now, snapshot)
        bos = self._detect(candle, snapshot)

        # Commit only after validation and construction succeeded.
        self._last_candle = candle
        if bos is not None:
            self._history.append(bos)
        return bos

    def _validate(self, candle: Candle, now: datetime, snapshot: StructureSnapshot) -> None:
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if not isinstance(snapshot, StructureSnapshot):
            raise TypeError("snapshot must be a StructureSnapshot")
        if candle.timeframe is not self._timeframe:
            raise ValueError(
                f"BOSEngine is configured for {self._timeframe.value}, "
                f"but received a {candle.timeframe.value} candle"
            )
        if snapshot.timeframe is not self._timeframe:
            raise ValueError(
                f"BOSEngine is configured for {self._timeframe.value}, "
                f"but received a {snapshot.timeframe.value} snapshot"
            )
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only closed candles may be processed")
        if self._last_candle is not None and candle.open_time < self._last_candle.close_time:
            raise ValueError(
                "candles must be supplied in chronological order without overlap "
                "(candle opens before the previous candle closed)"
            )
        for swing in (
            snapshot.active_swing_high, snapshot.prior_swing_high,
            snapshot.active_swing_low, snapshot.prior_swing_low,
        ):
            if swing is not None and swing.confirmed_at > candle.close_time:
                raise ValueError(
                    "snapshot contains a swing confirmed after this candle closed (lookahead)"
                )

    @staticmethod
    def _detect(candle: Candle, snapshot: StructureSnapshot) -> Optional[BOS]:
        state = snapshot.state
        if state is StructureState.BULLISH:
            level = snapshot.active_swing_high
            protected = snapshot.protected_swing              # active low
            if not candle.close > level.price:
                return None
            if candle.low < protected.price:                  # CHOCH candle: CHOCH only
                return None
            return BOS(candle.timeframe, BOSDirection.BULLISH, level,
                       candle.open_time, candle.close)
        if state is StructureState.BEARISH:
            level = snapshot.active_swing_low
            protected = snapshot.protected_swing              # active high
            if not candle.close < level.price:
                return None
            if candle.high > protected.price:                 # CHOCH candle: CHOCH only
                return None
            return BOS(candle.timeframe, BOSDirection.BEARISH, level,
                       candle.open_time, candle.close)
        return None                                           # UNDETERMINED / REVALUATING
