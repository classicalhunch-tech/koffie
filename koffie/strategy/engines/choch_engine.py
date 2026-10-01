"""CHOCHEngine for Koffie Strategy 1 (spec sections 12-13).

Detects CHOCH (Change of Character) on ONE timeframe from a CLOSED candle and
the current StructureSnapshot. It only DETECTS and RETURNS the event.

    BULLISH structure: candle.close strictly below the protected swing LOW  -> bearish CHOCH
    BEARISH structure: candle.close strictly above the protected swing HIGH -> bullish CHOCH
    UNDETERMINED / REVALUATING: no protected swing exists, so no CHOCH is emitted.

The protected swing is `snapshot.protected_swing` (the active swing low when
BULLISH, the active swing high when BEARISH). The rule is CLOSE-based (locked):
only a completed candle's CLOSE beyond the level is a CHOCH. A wick through the
level whose close stays on the original side is NOT a CHOCH, and a close exactly
on the level is not beyond it. The breaking candle is the CHOCH candle:
`candle_time` is its open time and `break_price` is its close. The CHOCH candle
does not become a swing; swings come only from SwingEngine.

The CHOCH condition is independent of BOS; each engine applies its own rule.

Caller responsibility (option (a))
----------------------------------
This engine never changes any state and holds no StructureEngine. After it
returns a CHOCH, the CALLER moves the structure with

    structure.enter_revaluating(candle.close_time)

A CHOCH does not itself establish a new trend; the new direction comes only
from later confirmed swings, decided by StructureEngine. The engine keeps no
memory of the structure between calls: it judges each candle only against the
snapshot it is given. If the caller does not move the structure to
REVALUATING, a later candle that again breaks the same protected swing is
again a CHOCH.

Ordering: within one candle, swings are confirmed and StructureEngine is
updated FIRST; the snapshot passed here must be that updated snapshot.

NOT here: BOS detection, StructureEngine changes, pivots, zones, liquidity,
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

from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.choch import CHOCH, CHOCHDirection
from koffie.strategy.models.structure import StructureSnapshot, StructureState


class CHOCHEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
        self._history: List[CHOCH] = []
        self._last_candle: Optional[Candle] = None

    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def history(self) -> Tuple[CHOCH, ...]:
        """All CHOCH events in the order they occurred, oldest first."""
        return tuple(self._history)

    def process_candle(
        self, candle: Candle, now: datetime, snapshot: StructureSnapshot
    ) -> Optional[CHOCH]:
        """Evaluate one CLOSED candle against the (already updated) structure.

        Returns the CHOCH event, or None. Raises TypeError for wrong argument
        types and ValueError for a wrong timeframe, an unfinished, repeated or
        overlapping candle, or a snapshot containing future swings. Nothing is
        changed when an exception is raised.
        """
        self._validate(candle, now, snapshot)
        choch = self._detect(candle, snapshot)

        # Commit only after validation and construction succeeded.
        self._last_candle = candle
        if choch is not None:
            self._history.append(choch)
        return choch

    def _validate(self, candle: Candle, now: datetime, snapshot: StructureSnapshot) -> None:
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if not isinstance(snapshot, StructureSnapshot):
            raise TypeError("snapshot must be a StructureSnapshot")
        if candle.timeframe is not self._timeframe:
            raise ValueError(
                f"CHOCHEngine is configured for {self._timeframe.value}, "
                f"but received a {candle.timeframe.value} candle"
            )
        if snapshot.timeframe is not self._timeframe:
            raise ValueError(
                f"CHOCHEngine is configured for {self._timeframe.value}, "
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
    def _detect(candle: Candle, snapshot: StructureSnapshot) -> Optional[CHOCH]:
        state = snapshot.state
        if state is StructureState.BULLISH:
            protected = snapshot.protected_swing              # active low
            if not candle.close < protected.price:
                return None
            return CHOCH(candle.timeframe, CHOCHDirection.BEARISH, protected,
                         candle.open_time, candle.close)
        if state is StructureState.BEARISH:
            protected = snapshot.protected_swing              # active high
            if not candle.close > protected.price:
                return None
            return CHOCH(candle.timeframe, CHOCHDirection.BULLISH, protected,
                         candle.open_time, candle.close)
        return None                                           # UNDETERMINED / REVALUATING