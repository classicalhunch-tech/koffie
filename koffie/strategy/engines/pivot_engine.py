"""PivotEngine for Koffie Strategy 1.
 
Selects the Pivot candle for a confirmed BOS on ONE timeframe.
 
    bullish BOS -> most recent BEARISH_DECISIVE candle before the BOS candle
    bearish BOS -> most recent BULLISH_DECISIVE candle before the BOS candle
 
The search starts at the candle immediately before the BOS candle and walks
backward through ALL completed history the engine has accepted. Candles that do
not match (neutral, zero-range anomaly, same-colour decisive) are skipped and do
not stop the search. There is no lookback limit and no size, price, ATR or
distance rule. The BOS candle can never be its own Pivot.
 
If no qualifying candle exists the outcome is NO_PIVOT_FOUND, recorded
explicitly; the BOS stays valid and no fallback candle is invented.
 
Causality: the Pivot is decided in the same call that supplies the BOS, from
candles that had already closed. Outcomes are append-only and are never
modified, replaced or deleted by later candles.
 
Idempotency (owned by this engine, not by BOSEngine): every outcome is keyed by
an immutable BOSIdentity built from the BOS's own fields. Supplying the exact
same BOS event again together with the exact same, most recently accepted
candle returns the stored outcome and changes nothing. If that identity is
supplied with different BOS values, ValueError is raised. Any other repeated,
overlapping or out-of-order candle is rejected by the normal validation.
 
Integrity checks (no strategy behaviour): candle is a Candle on this timeframe
and closed at `now`; it opens at or after the previous candle's close (gaps are
fine, overlaps and repeats are not); a supplied BOS is a BOS on this timeframe
that belongs to this candle (same candle_time and close_price). A rejected call
changes nothing.
 
NOT here: BOS/CHOCH detection, swings, structure, zones, liquidity, entries,
risk, TP/SL or execution. This engine imports none of those engines.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import Dict, List, Optional, Tuple
 
from koffie.strategy.models.bos import BOS
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.pivot import (
    PIVOT_CANDLE_CLASS,
    BOSIdentity,
    Pivot,
    PivotOutcome,
    PivotStatus,
)
 
 
class PivotEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
        self._candles: List[Tuple[Candle, CandleClass]] = []   # every accepted candle, oldest first
        self._outcomes: List[PivotOutcome] = []                # append-only
        self._by_identity: Dict[BOSIdentity, PivotOutcome] = {}
 
    # ------------------------------------------------------------- read-only state
    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe
 
    @property
    def history(self) -> Tuple[PivotOutcome, ...]:
        """Every outcome (FOUND and NO_PIVOT_FOUND) in order, oldest first."""
        return tuple(self._outcomes)
 
    @property
    def pivots(self) -> Tuple[Pivot, ...]:
        """Only the Pivots that were found, in order, oldest first."""
        return tuple(o.pivot for o in self._outcomes if o.pivot is not None)
 
    @property
    def candle_count(self) -> int:
        return len(self._candles)
 
    @property
    def last_candle(self) -> Optional[Candle]:
        return self._candles[-1][0] if self._candles else None
 
    # --------------------------------------------------------------------- input
    def process_candle(
        self, candle: Candle, now: datetime, bos: Optional[BOS] = None
    ) -> Optional[PivotOutcome]:
        """Accept one CLOSED candle and, if `bos` is given, decide its Pivot.
 
        Returns the PivotOutcome when a BOS is supplied, otherwise None.
        Raises TypeError for wrong argument types and ValueError for a wrong
        timeframe, an unfinished, repeated, overlapping or out-of-order candle,
        or a BOS that does not belong to this candle. Nothing is changed when an
        exception is raised.
        """
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if bos is not None and not isinstance(bos, BOS):
            raise TypeError("bos must be a BOS or None")
        if candle.timeframe is not self._timeframe:
            raise ValueError(
                f"PivotEngine is configured for {self._timeframe.value}, "
                f"but received a {candle.timeframe.value} candle"
            )
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only closed candles may be processed")
 
        if bos is not None:
            replay = self._recognize_replay(candle, bos)
            if replay is not None:
                return replay                                   # exact duplicate: no state change
 
        if self._candles and candle.open_time < self._candles[-1][0].close_time:
            raise ValueError(
                "candles must be supplied in chronological order without overlap "
                "(candle opens before the previous candle closed)"
            )
        if bos is not None:
            if bos.timeframe is not self._timeframe:
                raise ValueError(
                    f"PivotEngine is configured for {self._timeframe.value}, "
                    f"but received a {bos.timeframe.value} BOS"
                )
            if bos.candle_time != candle.open_time:
                raise ValueError("the BOS does not belong to the current candle (candle_time differs)")
            if bos.close_price != candle.close:
                raise ValueError("the BOS does not belong to the current candle (close_price differs)")
 
        candle_class = candle.classify(now)
        outcome = self._select(bos) if bos is not None else None
 
        # Commit only after validation and construction succeeded.
        self._candles.append((candle, candle_class))
        if outcome is not None:
            self._outcomes.append(outcome)
            self._by_identity[outcome.identity] = outcome
        return outcome
 
    # ------------------------------------------------------------------ internals
    def _recognize_replay(self, candle: Candle, bos: BOS) -> Optional[PivotOutcome]:
        """Return the stored outcome for an exact duplicate BOS replay, else None.
 
        A replay is recognised only when the BOS identity is already recorded AND
        the supplied candle is exactly the most recently accepted candle. If the
        identity matches but the BOS values differ, the record conflicts with the
        stored one and ValueError is raised. Anything else falls through to the
        normal validation (which rejects repeated/overlapping candles).
        """
        stored = self._by_identity.get(BOSIdentity.from_bos(bos))
        if stored is None or not self._candles or candle != self._candles[-1][0]:
            return None
        if bos != stored.bos:
            raise ValueError(
                "this BOS identity was already recorded with different values; "
                "conflicting BOS record rejected"
            )
        return stored
 
    def _select(self, bos: BOS) -> PivotOutcome:
        """Walk backward over completed candles; first opposite-colour decisive wins.
 
        The BOS candle is not in the history yet, so it cannot select itself.
        """
        wanted = PIVOT_CANDLE_CLASS[bos.direction]
        for candle, candle_class in reversed(self._candles):
            if candle_class is wanted:
                pivot = Pivot(bos, candle.open_time, candle.high, candle.low, candle_class)
                return PivotOutcome(bos, PivotStatus.FOUND, pivot)
        return PivotOutcome(bos, PivotStatus.NO_PIVOT_FOUND)