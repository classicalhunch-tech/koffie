"""SwingEngine for Koffie Strategy 1 (spec sections 4-7).
 
Causal, incremental 3-candle swing detection on ONE timeframe.
 
    swing high at N:  high[N] > high[N-1]  AND  high[N] > high[N+1]
    swing low  at N:  low[N]  < low[N-1]   AND  low[N]  < low[N+1]
 
Equality never qualifies. Candle N becomes a confirmed swing only when candle
N+1 has CLOSED; the candle handed to `process_candle` is that N+1 candle.
Every confirmed local swing is automatically a structural swing (no filter).
 
The engine only detects and records swings. It does NOT determine HH/HL/LH/LL,
BOS, CHOCH, trend, zones, liquidity, EQH/EQL, BRR, entries or SL/TP.
 
Behaviour
---------
- Outside bar: a candle satisfying both rules emits BOTH swings, HIGH first,
  then LOW (consecutive sequence numbers, same `confirmed_at`).
- Input is validated before any state changes; a rejected candle leaves the
  engine exactly as it was.
- Candles must not overlap: a candle must open at or after the previous
  candle's close_time. Gaps (weekends, holidays) are accepted.
- `now` is required so the closed-candle rule cannot be skipped. A backtest
  replaying stored candles passes `now=candle.close_time`.
- Datetimes must be consistently naive or consistently timezone-aware; mixing
  them raises TypeError from Python's datetime comparison (before any state
  change).
"""
from __future__ import annotations
 
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple
 
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.swing import Swing, SwingType
 
 
class SwingEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
 
        # The last two accepted candles: (previous, candidate). The incoming
        # candle is always the right-hand confirmation candle.
        self._window: Deque[Candle] = deque(maxlen=2)
 
        # Append-only history. A swing's `sequence` equals its index here.
        self._history: List[Swing] = []
 
        # Latest two CONFIRMED swings per type. "Active" means most recently
        # confirmed, NOT the highest high or lowest low.
        self._active_high: Optional[Swing] = None
        self._prior_high: Optional[Swing] = None
        self._active_low: Optional[Swing] = None
        self._prior_low: Optional[Swing] = None
 
    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe
 
    @property
    def history(self) -> Tuple[Swing, ...]:
        """All confirmed swings in confirmation order, oldest first."""
        return tuple(self._history)
 
    @property
    def active_swing_high(self) -> Optional[Swing]:
        return self._active_high
 
    @property
    def prior_swing_high(self) -> Optional[Swing]:
        return self._prior_high
 
    @property
    def active_swing_low(self) -> Optional[Swing]:
        return self._active_low
 
    @property
    def prior_swing_low(self) -> Optional[Swing]:
        return self._prior_low
 
    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_candle(self, candle: Candle, now: datetime) -> List[Swing]:
        """Process one CLOSED candle as the confirmation candle (N+1).
 
        Returns the swings newly confirmed by this candle: zero, one or two.
 
        Raises:
            TypeError:  `candle` is not a Candle or `now` is not a datetime.
            ValueError: wrong timeframe, candle not closed at `now`, or candle
                        not chronologically after the previous candle.
        Nothing is changed when an exception is raised.
        """
        self._validate(candle, now)
 
        new_swings: List[Swing] = []
        if len(self._window) == 2:
            previous, candidate = self._window
            sequence = len(self._history)
 
            if candidate.high > previous.high and candidate.high > candle.high:
                new_swings.append(
                    Swing(
                        swing_type=SwingType.HIGH,
                        timeframe=self._timeframe,
                        price=candidate.high,
                        candle_time=candidate.open_time,
                        confirmed_at=candle.close_time,
                        sequence=sequence,
                    )
                )
                sequence += 1
 
            if candidate.low < previous.low and candidate.low < candle.low:
                new_swings.append(
                    Swing(
                        swing_type=SwingType.LOW,
                        timeframe=self._timeframe,
                        price=candidate.low,
                        candle_time=candidate.open_time,
                        confirmed_at=candle.close_time,
                        sequence=sequence,
                    )
                )
 
        # Commit only after validation and swing construction succeeded.
        self._window.append(candle)
        for swing in new_swings:
            self._record(swing)
        return list(new_swings)
 
    def _validate(self, candle: Candle, now: datetime) -> None:
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if candle.timeframe is not self._timeframe:
            raise ValueError(
                f"SwingEngine is configured for {self._timeframe.value}, "
                f"but received a {candle.timeframe.value} candle"
            )
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only closed candles may be processed")
        if self._window and candle.open_time < self._window[-1].close_time:
            raise ValueError(
                "candles must be supplied in chronological order without overlap "
                "(candle opens before the previous candle closed)"
            )
 
    def _record(self, swing: Swing) -> None:
        self._history.append(swing)  # every confirmed swing is structural
        if swing.swing_type is SwingType.HIGH:
            self._prior_high = self._active_high
            self._active_high = swing
        else:
            self._prior_low = self._active_low
            self._active_low = swing