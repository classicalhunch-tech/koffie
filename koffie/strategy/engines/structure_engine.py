"""StructureEngine for Koffie Strategy 1 (spec sections 8-10, 13, 35).
 
Holds the structure state of ONE timeframe. It is handed the swings a
SwingEngine confirms (it never reads candles or SwingEngine) and applies the
locked literal rules:
 
* HH/HL/LL/LH compare CONSECUTIVE confirmed structural swings of the same
  type ("active" = most recently confirmed), never historical extremes.
  Equal prices qualify as none of them.
* After EACH supplied swing the pair is evaluated:
    HH + HL            -> BULLISH   (if not already BULLISH)
    LL + LH            -> BEARISH   (if not already BEARISH)
    mixed/tied/partial -> state unchanged
  This applies from every state: UNDETERMINED, a direct BULLISH<->BEARISH
  flip, and restoration out of REVALUATING. The newest swing may combine with
  the existing opposite swing; nothing requires swings to post-date a CHOCH.
* enter_revaluating(at) is the interface a later BOS/CHOCH component calls.
  It only moves BULLISH/BEARISH -> REVALUATING and never evaluates a pair, so
  an old HH+HL / LL+LH pair cannot restore the trend at the CHOCH instant;
  evaluation happens only when a newly confirmed swing is supplied.
* An outside bar's swings arrive as HIGH then LOW and are evaluated one by
  one, in the order supplied, with no special case.
 
NOT in this engine: BOS or CHOCH detection, zones, liquidity, entries, risk,
news, TP/SL or execution.
 
Integrity checks (no strategy behaviour): supplied swings must be Swing
objects on this timeframe, with confirmed_at not earlier than the previous
supplied swing's and strictly increasing candle_time per swing type. `sequence`
is metadata only: it is never used to order, skip or reject swings (gaps and
any starting value are fine). One consequence comes from the existing
StructureSnapshot model, not from this engine: within a single swing type the
prior swing's sequence must be lower than the active swing's, so a call that
would violate that is rejected as a whole. A rejected call changes nothing.
"""
from __future__ import annotations
 
from datetime import datetime
from typing import List, Optional, Sequence, Tuple
 
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.structure import (
    DirectionalPermission,
    StructureSnapshot,
    StructureState,
    StructureTransition,
    TransitionReason,
    classify_pair,
    compare_highs,
    compare_lows,
)
from koffie.strategy.models.swing import Swing, SwingType
 
 
class StructureEngine:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
        self._snapshot = StructureSnapshot(timeframe)
        self._transitions: List[StructureTransition] = []
        self._last_confirmed_at: Optional[datetime] = None
 
    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe
 
    @property
    def snapshot(self) -> StructureSnapshot:
        return self._snapshot
 
    @property
    def state(self) -> StructureState:
        return self._snapshot.state
 
    @property
    def permission(self) -> DirectionalPermission:
        return self._snapshot.permission
 
    @property
    def protected_swing(self) -> Optional[Swing]:
        return self._snapshot.protected_swing
 
    @property
    def transitions(self) -> Tuple[StructureTransition, ...]:
        """Append-only history of state changes, oldest first."""
        return tuple(self._transitions)
 
    # ------------------------------------------------------------------
    # Swings in
    # ------------------------------------------------------------------
    def process_swings(self, swings: Sequence[Swing]) -> List[StructureTransition]:
        """Apply newly confirmed swings in order; return the transitions made.
 
        Raises TypeError for a non-list/tuple argument or a non-Swing item,
        and ValueError for a wrong timeframe or an ordering violation
        (confirmed_at or same-type candle_time). Nothing is changed when an
        exception is raised.
        """
        if not isinstance(swings, (list, tuple)):
            raise TypeError("swings must be a list or tuple of Swing objects")
 
        snap = self._snapshot
        state = snap.state
        active_high, prior_high = snap.active_swing_high, snap.prior_swing_high
        active_low, prior_low = snap.active_swing_low, snap.prior_swing_low
        last_confirmed = self._last_confirmed_at
        made: List[StructureTransition] = []
 
        for swing in swings:
            self._check_swing(swing, last_confirmed, active_high, active_low)
            last_confirmed = swing.confirmed_at
 
            if swing.swing_type is SwingType.HIGH:
                prior_high, active_high = active_high, swing
            else:
                prior_low, active_low = active_low, swing
 
            pair = self._directional_pair(active_high, prior_high, active_low, prior_low)
            if pair is not None and pair is not state:
                made.append(
                    StructureTransition(
                        timeframe=self._timeframe,
                        from_state=state,
                        to_state=pair,
                        reason=TransitionReason.DIRECTIONAL_PAIR,
                        at=swing.confirmed_at,
                        trigger_swing_sequence=swing.sequence,
                    )
                )
                state = pair
 
        new_snapshot = StructureSnapshot(
            self._timeframe, state, active_high, prior_high, active_low, prior_low
        )
 
        # Commit only after every swing was accepted.
        self._snapshot = new_snapshot
        self._last_confirmed_at = last_confirmed
        self._transitions.extend(made)
        return list(made)
 
    def _check_swing(
        self,
        swing: Swing,
        last_confirmed_at: Optional[datetime],
        active_high: Optional[Swing],
        active_low: Optional[Swing],
    ) -> None:
        if not isinstance(swing, Swing):
            raise TypeError("swings must contain Swing objects")
        if swing.timeframe is not self._timeframe:
            raise ValueError(
                f"StructureEngine is configured for {self._timeframe.value}, "
                f"but received a {swing.timeframe.value} swing"
            )
        if last_confirmed_at is not None and swing.confirmed_at < last_confirmed_at:
            raise ValueError("swing confirmed_at goes backwards")
        same_type_active = active_high if swing.swing_type is SwingType.HIGH else active_low
        if same_type_active is not None and swing.candle_time <= same_type_active.candle_time:
            raise ValueError("swing candle_time must increase within each swing type")
 
    @staticmethod
    def _directional_pair(
        active_high: Optional[Swing],
        prior_high: Optional[Swing],
        active_low: Optional[Swing],
        prior_low: Optional[Swing],
    ) -> Optional[StructureState]:
        high_relation = (
            compare_highs(active_high, prior_high)
            if active_high is not None and prior_high is not None else None
        )
        low_relation = (
            compare_lows(active_low, prior_low)
            if active_low is not None and prior_low is not None else None
        )
        return classify_pair(high_relation, low_relation)
 
    # ------------------------------------------------------------------
    # CHOCH interface (detection lives in a later component)
    # ------------------------------------------------------------------
    def enter_revaluating(self, at: datetime) -> StructureTransition:
        """Move BULLISH/BEARISH -> REVALUATING because the protected swing broke.
 
        `at` is when the breaking candle closed. Raises ValueError from any
        other state and TypeError if `at` is not a datetime; nothing changes
        on error. No pair is evaluated here.
        """
        if not isinstance(at, datetime):
            raise TypeError("at must be a datetime")
        snap = self._snapshot
        if snap.state not in (StructureState.BULLISH, StructureState.BEARISH):
            raise ValueError(
                f"can only enter REVALUATING from BULLISH or BEARISH, not {snap.state.value}"
            )
        transition = StructureTransition(
            timeframe=self._timeframe,
            from_state=snap.state,
            to_state=StructureState.REVALUATING,
            reason=TransitionReason.CHOCH,
            at=at,
        )
        new_snapshot = StructureSnapshot(
            self._timeframe,
            StructureState.REVALUATING,
            snap.active_swing_high, snap.prior_swing_high,
            snap.active_swing_low, snap.prior_swing_low,
        )
        self._snapshot = new_snapshot
        self._transitions.append(transition)
        return transition