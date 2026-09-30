"""Structure model for Koffie Strategy 1 (spec sections 8-10, 13, 35).

DATA AND PURE DEFINITIONS ONLY. This module contains no engine, no BOS
detection, no CHOCH detection, and no liquidity, zone, entry, risk, news,
TP/SL or execution logic. A later StructureEngine will maintain a
`StructureSnapshot`; a later BOS/CHOCH component will decide when the
protected swing is broken and ask the engine to enter REVALUATING.

Definitions used (all from consecutive CONFIRMED structural swings, never
raw historical extremes; "active" = most recently confirmed of that type):

    HH = active high > prior high      LH = active high < prior high
    HL = active low  > prior low       LL = active low  < prior low

Equal prices qualify as none of HH/LH/HL/LL (they are EQUAL).

    BULLISH pair  = HH + HL
    BEARISH pair  = LL + LH
    any other combination (mixed or tied) is NOT a directional pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.swing import Swing, SwingType


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DirectionalPermission(Enum):
    LONG_ONLY = "LONG_ONLY"
    SHORT_ONLY = "SHORT_ONLY"
    NO_TRADE_PERMITTED = "NO_TRADE_PERMITTED"


class StructureState(Enum):
    UNDETERMINED = "UNDETERMINED"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    REVALUATING = "REVALUATING"

    @property
    def permission(self) -> DirectionalPermission:
        """Directional permission this state grants (spec sections 2 and 35)."""
        return _PERMISSIONS[self]


_PERMISSIONS = {
    StructureState.BULLISH: DirectionalPermission.LONG_ONLY,
    StructureState.BEARISH: DirectionalPermission.SHORT_ONLY,
    StructureState.REVALUATING: DirectionalPermission.NO_TRADE_PERMITTED,
    StructureState.UNDETERMINED: DirectionalPermission.NO_TRADE_PERMITTED,
}


class HighRelation(Enum):
    HH = "HH"          # active high > prior high
    LH = "LH"          # active high < prior high
    EQUAL = "EQUAL"    # ties qualify as neither


class LowRelation(Enum):
    HL = "HL"          # active low > prior low
    LL = "LL"          # active low < prior low
    EQUAL = "EQUAL"    # ties qualify as neither


class TransitionReason(Enum):
    DIRECTIONAL_PAIR = "DIRECTIONAL_PAIR"   # swing-pair evaluation set BULLISH/BEARISH
    CHOCH = "CHOCH"                         # protected swing broken -> REVALUATING


# ---------------------------------------------------------------------------
# Pure definitions
# ---------------------------------------------------------------------------
def _require_swing(swing: Swing, swing_type: SwingType, name: str) -> None:
    if not isinstance(swing, Swing):
        raise ValueError(f"{name} must be a Swing")
    if swing.swing_type is not swing_type:
        raise ValueError(f"{name} must be a {swing_type.value} swing")


def compare_highs(active: Swing, prior: Swing) -> HighRelation:
    _require_swing(active, SwingType.HIGH, "active")
    _require_swing(prior, SwingType.HIGH, "prior")
    if active.price > prior.price:
        return HighRelation.HH
    if active.price < prior.price:
        return HighRelation.LH
    return HighRelation.EQUAL


def compare_lows(active: Swing, prior: Swing) -> LowRelation:
    _require_swing(active, SwingType.LOW, "active")
    _require_swing(prior, SwingType.LOW, "prior")
    if active.price > prior.price:
        return LowRelation.HL
    if active.price < prior.price:
        return LowRelation.LL
    return LowRelation.EQUAL


def classify_pair(
    high_relation: Optional[HighRelation],
    low_relation: Optional[LowRelation],
) -> Optional[StructureState]:
    """BULLISH for HH+HL, BEARISH for LL+LH, otherwise None (no direction).

    None means "not a directional pair": mixed (HH+LL, LH+HL), tied, or
    incomplete. It never means UNDETERMINED; the caller leaves its current
    state unchanged.
    """
    if high_relation is HighRelation.HH and low_relation is LowRelation.HL:
        return StructureState.BULLISH
    if low_relation is LowRelation.LL and high_relation is HighRelation.LH:
        return StructureState.BEARISH
    return None


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructureSnapshot:
    """Immutable picture of one timeframe's structure at one moment.

    BULLISH, BEARISH and REVALUATING require two confirmed highs AND two
    confirmed lows (spec section 8; REVALUATING only follows an established
    trend). UNDETERMINED may hold any swings, because a mixed or tied pair
    leaves the state unchanged.
    """

    timeframe: Timeframe
    state: StructureState = StructureState.UNDETERMINED
    active_swing_high: Optional[Swing] = None
    prior_swing_high: Optional[Swing] = None
    active_swing_low: Optional[Swing] = None
    prior_swing_low: Optional[Swing] = None

    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError("timeframe must be a Timeframe")
        if not isinstance(self.state, StructureState):
            raise ValueError("state must be a StructureState")

        slots = (
            ("active_swing_high", self.active_swing_high, SwingType.HIGH),
            ("prior_swing_high", self.prior_swing_high, SwingType.HIGH),
            ("active_swing_low", self.active_swing_low, SwingType.LOW),
            ("prior_swing_low", self.prior_swing_low, SwingType.LOW),
        )
        for name, swing, swing_type in slots:
            if swing is None:
                continue
            _require_swing(swing, swing_type, name)
            if swing.timeframe is not self.timeframe:
                raise ValueError(f"{name} belongs to a different timeframe")

        for kind, active, prior in (
            ("high", self.active_swing_high, self.prior_swing_high),
            ("low", self.active_swing_low, self.prior_swing_low),
        ):
            if prior is None:
                continue
            if active is None:
                raise ValueError(f"prior swing {kind} requires an active swing {kind}")
            if not (prior.sequence < active.sequence and prior.candle_time < active.candle_time):
                raise ValueError(
                    f"prior swing {kind} must be confirmed chronologically before the active one"
                )

        if self.state is not StructureState.UNDETERMINED and not self.has_two_highs_and_two_lows:
            raise ValueError(
                f"{self.state.value} requires two confirmed swing highs and two confirmed swing lows"
            )

    # -- availability -------------------------------------------------------
    @property
    def has_two_highs_and_two_lows(self) -> bool:
        return self.prior_swing_high is not None and self.prior_swing_low is not None

    # -- relationships between consecutive confirmed swings ---------------------
    @property
    def high_relation(self) -> Optional[HighRelation]:
        if self.active_swing_high is None or self.prior_swing_high is None:
            return None
        return compare_highs(self.active_swing_high, self.prior_swing_high)

    @property
    def low_relation(self) -> Optional[LowRelation]:
        if self.active_swing_low is None or self.prior_swing_low is None:
            return None
        return compare_lows(self.active_swing_low, self.prior_swing_low)

    @property
    def directional_pair(self) -> Optional[StructureState]:
        """BULLISH (HH+HL), BEARISH (LL+LH) or None, from the held swings."""
        return classify_pair(self.high_relation, self.low_relation)

    # -- protected swing (spec section 10) -----------------------------------
    @property
    def protected_swing(self) -> Optional[Swing]:
        """Active low if BULLISH, active high if BEARISH, else undefined (None)."""
        if self.state is StructureState.BULLISH:
            return self.active_swing_low
        if self.state is StructureState.BEARISH:
            return self.active_swing_high
        return None

    # -- permission (spec sections 2 and 35) ------------------------------------
    @property
    def permission(self) -> DirectionalPermission:
        return self.state.permission


# ---------------------------------------------------------------------------
# Transition record (auditability, spec section 39)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StructureTransition:
    """Immutable record that the structure state changed.

    DIRECTIONAL_PAIR: swing-pair evaluation set BULLISH or BEARISH;
        `trigger_swing_sequence` is the newest swing that was evaluated and
        `at` is normally that swing's confirmed_at.
    CHOCH: the protected swing was broken while BULLISH/BEARISH, entering
        REVALUATING; it is caused by a candle, not a swing, so
        `trigger_swing_sequence` must be None.
    """

    timeframe: Timeframe
    from_state: StructureState
    to_state: StructureState
    reason: TransitionReason
    at: datetime
    trigger_swing_sequence: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.timeframe, Timeframe):
            raise ValueError("timeframe must be a Timeframe")
        if not isinstance(self.from_state, StructureState):
            raise ValueError("from_state must be a StructureState")
        if not isinstance(self.to_state, StructureState):
            raise ValueError("to_state must be a StructureState")
        if not isinstance(self.reason, TransitionReason):
            raise ValueError("reason must be a TransitionReason")
        if not isinstance(self.at, datetime):
            raise ValueError("at must be a datetime")
        if self.from_state is self.to_state:
            raise ValueError("a transition must change the state")

        directional = (StructureState.BULLISH, StructureState.BEARISH)
        seq = self.trigger_swing_sequence
        if self.reason is TransitionReason.DIRECTIONAL_PAIR:
            if self.to_state not in directional:
                raise ValueError("a directional-pair transition must end BULLISH or BEARISH")
            if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
                raise ValueError("a directional-pair transition needs a swing sequence >= 0")
        else:  # CHOCH
            if self.from_state not in directional:
                raise ValueError("CHOCH can only occur from BULLISH or BEARISH")
            if self.to_state is not StructureState.REVALUATING:
                raise ValueError("CHOCH must enter REVALUATING")
            if seq is not None:
                raise ValueError("a CHOCH transition has no trigger swing")
