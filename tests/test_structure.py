import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

import koffie.strategy.models.structure as structure_module
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.structure import (
    DirectionalPermission,
    HighRelation,
    LowRelation,
    StructureSnapshot,
    StructureState,
    StructureTransition,
    TransitionReason,
    classify_pair,
    compare_highs,
    compare_lows,
)
from koffie.strategy.models.swing import Swing, SwingType

T0 = datetime(2026, 1, 1, 10, 0)
M5, M15 = Timeframe.M5, Timeframe.M15
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = StructureState


def sw(kind, price, seq, tf=M5):
    ct = T0 + seq * tf.duration
    return Swing(kind, tf, float(price), ct, ct + 2 * tf.duration, seq)


def snap(state=S.UNDETERMINED, ah=12, ph=10, al=6, pl=4, tf=M5):
    """Defaults describe HH + HL. Pass None to leave a swing out."""
    def maybe(kind, price, seq):
        return None if price is None else sw(kind, price, seq, tf)
    return StructureSnapshot(
        timeframe=tf, state=state,
        active_swing_high=maybe(HIGH, ah, 2), prior_swing_high=maybe(HIGH, ph, 0),
        active_swing_low=maybe(LOW, al, 3), prior_swing_low=maybe(LOW, pl, 1),
    )


# ------------------------------------------------------------- enumerations
def test_structure_states_are_exactly_the_four_specified():
    assert {s.name for s in S} == {"UNDETERMINED", "BULLISH", "BEARISH", "REVALUATING"}


def test_permissions_are_exactly_the_three_specified():
    assert {p.name for p in DirectionalPermission} == {"LONG_ONLY", "SHORT_ONLY", "NO_TRADE_PERMITTED"}


def test_state_to_permission_mapping():
    assert S.BULLISH.permission is DirectionalPermission.LONG_ONLY
    assert S.BEARISH.permission is DirectionalPermission.SHORT_ONLY
    assert S.REVALUATING.permission is DirectionalPermission.NO_TRADE_PERMITTED
    assert S.UNDETERMINED.permission is DirectionalPermission.NO_TRADE_PERMITTED


def test_relation_enums():
    assert {r.name for r in HighRelation} == {"HH", "LH", "EQUAL"}
    assert {r.name for r in LowRelation} == {"HL", "LL", "EQUAL"}
    assert {r.name for r in TransitionReason} == {"DIRECTIONAL_PAIR", "CHOCH"}


# ------------------------------------------------------------- relationships
def test_compare_highs():
    assert compare_highs(sw(HIGH, 12, 1), sw(HIGH, 10, 0)) is HighRelation.HH
    assert compare_highs(sw(HIGH, 9, 1), sw(HIGH, 10, 0)) is HighRelation.LH
    assert compare_highs(sw(HIGH, 10, 1), sw(HIGH, 10, 0)) is HighRelation.EQUAL


def test_compare_lows():
    assert compare_lows(sw(LOW, 6, 1), sw(LOW, 4, 0)) is LowRelation.HL
    assert compare_lows(sw(LOW, 3, 1), sw(LOW, 4, 0)) is LowRelation.LL
    assert compare_lows(sw(LOW, 4, 1), sw(LOW, 4, 0)) is LowRelation.EQUAL


def test_relations_use_consecutive_swings_not_historical_extremes():
    # Prior high 15 (the historical extreme), latest high 14 -> LH, not "still a high".
    assert compare_highs(sw(HIGH, 14, 1), sw(HIGH, 15, 0)) is HighRelation.LH
    # Prior low 5 (the historical extreme), latest low 7 -> HL.
    assert compare_lows(sw(LOW, 7, 1), sw(LOW, 5, 0)) is LowRelation.HL


def test_compare_rejects_wrong_swing_types_and_non_swings():
    with pytest.raises(ValueError):
        compare_highs(sw(LOW, 12, 1), sw(HIGH, 10, 0))
    with pytest.raises(ValueError):
        compare_highs(sw(HIGH, 12, 1), sw(LOW, 10, 0))
    with pytest.raises(ValueError):
        compare_lows(sw(HIGH, 6, 1), sw(LOW, 4, 0))
    with pytest.raises(ValueError):
        compare_lows(sw(LOW, 6, 1), sw(HIGH, 4, 0))
    with pytest.raises(ValueError):
        compare_highs(12, sw(HIGH, 10, 0))


# ------------------------------------------------------------- pair classification
def test_bullish_pair_is_hh_plus_hl():
    assert classify_pair(HighRelation.HH, LowRelation.HL) is S.BULLISH


def test_bearish_pair_is_ll_plus_lh():
    assert classify_pair(HighRelation.LH, LowRelation.LL) is S.BEARISH


def test_mixed_pairs_give_no_direction():
    assert classify_pair(HighRelation.HH, LowRelation.LL) is None
    assert classify_pair(HighRelation.LH, LowRelation.HL) is None


def test_equal_comparisons_never_qualify():
    for high in HighRelation:
        for low in LowRelation:
            result = classify_pair(high, low)
            if HighRelation.EQUAL in (high,) or LowRelation.EQUAL in (low,):
                assert result is None
    assert classify_pair(HighRelation.EQUAL, LowRelation.HL) is None
    assert classify_pair(HighRelation.HH, LowRelation.EQUAL) is None
    assert classify_pair(HighRelation.EQUAL, LowRelation.LL) is None
    assert classify_pair(HighRelation.LH, LowRelation.EQUAL) is None
    assert classify_pair(HighRelation.EQUAL, LowRelation.EQUAL) is None


def test_missing_relations_give_no_direction():
    assert classify_pair(None, None) is None
    assert classify_pair(HighRelation.HH, None) is None
    assert classify_pair(None, LowRelation.HL) is None


def test_classify_pair_never_returns_undetermined_or_revaluating():
    for high in list(HighRelation) + [None]:
        for low in list(LowRelation) + [None]:
            assert classify_pair(high, low) in (None, S.BULLISH, S.BEARISH)


def test_all_nine_combinations_exactly_two_are_directional():
    directional = [(h, l) for h in HighRelation for l in LowRelation
                   if classify_pair(h, l) is not None]
    assert set(directional) == {(HighRelation.HH, LowRelation.HL),
                                (HighRelation.LH, LowRelation.LL)}


# ------------------------------------------------------------- snapshot basics
def test_default_snapshot_is_undetermined_and_empty():
    sn = StructureSnapshot(M5)
    assert sn.state is S.UNDETERMINED
    assert sn.timeframe is M5
    for name in ("active_swing_high", "prior_swing_high", "active_swing_low", "prior_swing_low"):
        assert getattr(sn, name) is None
    assert sn.protected_swing is None
    assert sn.high_relation is None and sn.low_relation is None
    assert sn.directional_pair is None
    assert sn.has_two_highs_and_two_lows is False
    assert sn.permission is DirectionalPermission.NO_TRADE_PERMITTED


def test_snapshot_relations_and_directional_pair_from_swings():
    sn = snap()
    assert sn.high_relation is HighRelation.HH and sn.low_relation is LowRelation.HL
    assert sn.directional_pair is S.BULLISH
    bear = snap(ah=8, ph=10, al=3, pl=4)
    assert bear.high_relation is HighRelation.LH and bear.low_relation is LowRelation.LL
    assert bear.directional_pair is S.BEARISH


def test_snapshot_mixed_and_tied_pairs_are_not_directional():
    assert snap(ah=12, ph=10, al=3, pl=4).directional_pair is None      # HH + LL
    assert snap(ah=8, ph=10, al=6, pl=4).directional_pair is None       # LH + HL
    assert snap(ah=10, ph=10, al=6, pl=4).directional_pair is None      # tied high
    assert snap(ah=12, ph=10, al=4, pl=4).directional_pair is None      # tied low


def test_snapshot_with_single_swing_of_a_type_has_no_relation():
    sn = snap(ph=None)
    assert sn.high_relation is None and sn.low_relation is LowRelation.HL
    assert sn.directional_pair is None
    assert sn.has_two_highs_and_two_lows is False


def test_has_two_highs_and_two_lows():
    assert snap().has_two_highs_and_two_lows is True
    assert snap(pl=None).has_two_highs_and_two_lows is False


# ------------------------------------------------------------- protected swing
def test_protected_swing_bullish_is_active_low():
    sn = snap(state=S.BULLISH)
    assert sn.protected_swing is sn.active_swing_low


def test_protected_swing_bearish_is_active_high():
    sn = snap(state=S.BEARISH, ah=8, ph=10, al=3, pl=4)
    assert sn.protected_swing is sn.active_swing_high


def test_protected_swing_undefined_when_undetermined_or_revaluating():
    assert snap(state=S.UNDETERMINED).protected_swing is None
    assert snap(state=S.REVALUATING).protected_swing is None


def test_protected_swing_is_the_latest_confirmed_low_not_the_lowest():
    # active low 6 is higher than prior low 4 but is the protected one.
    sn = snap(state=S.BULLISH, al=6, pl=4)
    assert sn.protected_swing.price == 6


# ------------------------------------------------------------- permission
def test_snapshot_permission_follows_state():
    assert snap(state=S.BULLISH).permission is DirectionalPermission.LONG_ONLY
    assert snap(state=S.BEARISH).permission is DirectionalPermission.SHORT_ONLY
    assert snap(state=S.REVALUATING).permission is DirectionalPermission.NO_TRADE_PERMITTED
    assert snap(state=S.UNDETERMINED).permission is DirectionalPermission.NO_TRADE_PERMITTED


# ------------------------------------------------------------- invariants
def test_directional_and_revaluating_states_need_two_highs_and_two_lows():
    for state in (S.BULLISH, S.BEARISH, S.REVALUATING):
        snap(state=state)                                  # complete set is fine
        for missing in ({"ph": None}, {"pl": None}, {"ah": None, "ph": None},
                        {"al": None, "pl": None}, {"ah": None, "ph": None, "al": None, "pl": None}):
            with pytest.raises(ValueError):
                snap(state=state, **missing)


def test_undetermined_may_hold_any_swings():
    snap(state=S.UNDETERMINED, ph=None)
    snap(state=S.UNDETERMINED, ah=None, ph=None)
    snap(state=S.UNDETERMINED, ah=12, ph=10, al=3, pl=4)   # complete but mixed


def test_state_does_not_have_to_equal_the_current_pair():
    # A mixed pair leaves BULLISH unchanged, so BULLISH with HH+LL is a valid snapshot.
    sn = snap(state=S.BULLISH, ah=12, ph=10, al=3, pl=4)
    assert sn.state is S.BULLISH and sn.directional_pair is None


def test_swing_slots_reject_wrong_type_of_swing():
    good = dict(ah=12, ph=10, al=6, pl=4)
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, active_swing_high=sw(LOW, 12, 2))
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, prior_swing_high=sw(LOW, 10, 0))
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, active_swing_low=sw(HIGH, 6, 3))
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, prior_swing_low=sw(HIGH, 4, 1))
    assert snap(**good).state is S.UNDETERMINED


def test_swing_slots_reject_non_swings_including_lookalikes():
    class Lookalike:
        swing_type = HIGH
        timeframe = M5
        price = 12.0
        candle_time = T0
        confirmed_at = T0 + timedelta(minutes=10)
        sequence = 2

    for bad in (Lookalike(), 12.0, "swing", object()):
        with pytest.raises(ValueError):
            StructureSnapshot(M5, S.UNDETERMINED, active_swing_high=bad)


def test_swings_must_share_the_snapshot_timeframe():
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, active_swing_high=sw(HIGH, 12, 2, tf=M15))
    with pytest.raises(ValueError):
        StructureSnapshot(M15, S.UNDETERMINED, active_swing_low=sw(LOW, 6, 3, tf=M5))
    assert snap(tf=M15).timeframe is M15


def test_prior_requires_active():
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, prior_swing_high=sw(HIGH, 10, 0))
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, prior_swing_low=sw(LOW, 4, 1))


def test_prior_must_be_chronologically_before_active():
    for a_seq, p_seq in ((2, 2), (1, 2)):   # equal and reversed sequence
        with pytest.raises(ValueError):
            StructureSnapshot(M5, S.UNDETERMINED,
                              active_swing_high=sw(HIGH, 12, a_seq),
                              prior_swing_high=sw(HIGH, 10, p_seq))
        with pytest.raises(ValueError):
            StructureSnapshot(M5, S.UNDETERMINED,
                              active_swing_low=sw(LOW, 6, a_seq),
                              prior_swing_low=sw(LOW, 4, p_seq))
    # sequence in order but candle_time reversed
    late_prior = Swing(HIGH, M5, 10.0, T0 + timedelta(hours=5), T0 + timedelta(hours=5, minutes=10), 0)
    with pytest.raises(ValueError):
        StructureSnapshot(M5, S.UNDETERMINED, active_swing_high=sw(HIGH, 12, 2), prior_swing_high=late_prior)


def test_snapshot_type_validation():
    with pytest.raises(ValueError):
        StructureSnapshot("5M")
    with pytest.raises(ValueError):
        StructureSnapshot(M5, "BULLISH")
    with pytest.raises(ValueError):
        StructureSnapshot(M5, None)


def test_snapshot_is_immutable_hashable_and_value_equal():
    sn = snap(state=S.BULLISH)
    for name, value in (("state", S.BEARISH), ("timeframe", M15), ("active_swing_low", None)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(sn, name, value)
    assert snap(state=S.BULLISH) == snap(state=S.BULLISH)
    assert hash(snap(state=S.BULLISH)) == hash(snap(state=S.BULLISH))
    assert snap(state=S.BULLISH) != snap(state=S.BEARISH, ah=8, ph=10, al=3, pl=4)


def test_snapshot_fields_are_exactly_the_specified_ones():
    assert [f.name for f in dataclasses.fields(StructureSnapshot)] == [
        "timeframe", "state", "active_swing_high", "prior_swing_high",
        "active_swing_low", "prior_swing_low"]


# ------------------------------------------------------------- compatibility
def test_snapshot_can_be_built_from_real_swing_engine_state():
    """Compatibility with existing Candle/Swing/SwingEngine (engine is only read)."""
    pairs = [(10, 4), (14, 5), (11, 6), (12, 3), (13, 7), (12, 5), (9, 4), (8, 6)]
    eng = SwingEngine(M5)
    for i, (h, l) in enumerate(pairs):
        mid = (h + l) / 2
        cd = Candle(M5, T0 + i * M5.duration, mid, float(h), float(l), mid)
        eng.process_candle(cd, now=cd.close_time)
    sn = StructureSnapshot(
        timeframe=eng.timeframe,
        active_swing_high=eng.active_swing_high, prior_swing_high=eng.prior_swing_high,
        active_swing_low=eng.active_swing_low, prior_swing_low=eng.prior_swing_low,
    )
    assert sn.active_swing_high is eng.active_swing_high
    assert sn.high_relation is not None or eng.prior_swing_high is None
    assert isinstance(sn.active_swing_high, Swing)


def test_snapshot_works_for_every_timeframe():
    for tf in Timeframe:
        sn = snap(state=S.BULLISH, tf=tf)
        assert sn.timeframe is tf and sn.protected_swing.timeframe is tf


# ------------------------------------------------------------- transitions
def tr(frm, to, reason=TransitionReason.DIRECTIONAL_PAIR, seq=5, at=T0, tf=M5):
    return StructureTransition(tf, frm, to, reason, at, seq)


def test_valid_directional_pair_transitions():
    for frm in (S.UNDETERMINED, S.REVALUATING):
        for to in (S.BULLISH, S.BEARISH):
            t = tr(frm, to)
            assert t.from_state is frm and t.to_state is to and t.trigger_swing_sequence == 5


def test_valid_choch_transitions():
    for frm in (S.BULLISH, S.BEARISH):
        t = StructureTransition(M5, frm, S.REVALUATING, TransitionReason.CHOCH, T0)
        assert t.to_state is S.REVALUATING and t.trigger_swing_sequence is None


def test_transition_must_change_state():
    for s in S:
        with pytest.raises(ValueError):
            tr(s, s)


def test_directional_pair_transition_must_end_bullish_or_bearish():
    with pytest.raises(ValueError):
        tr(S.BULLISH, S.REVALUATING)
    with pytest.raises(ValueError):
        tr(S.BULLISH, S.UNDETERMINED)


def test_directional_pair_transition_needs_a_valid_swing_sequence():
    for bad in (None, -1, True, 1.0, "5"):
        with pytest.raises(ValueError):
            tr(S.UNDETERMINED, S.BULLISH, seq=bad)
    assert tr(S.UNDETERMINED, S.BULLISH, seq=0).trigger_swing_sequence == 0


def test_choch_only_from_a_trend_and_only_into_revaluating():
    for frm in (S.UNDETERMINED, S.REVALUATING):
        with pytest.raises(ValueError):
            StructureTransition(M5, frm, S.BULLISH, TransitionReason.CHOCH, T0)
        with pytest.raises(ValueError):
            StructureTransition(M5, frm, S.REVALUATING, TransitionReason.CHOCH, T0)
    with pytest.raises(ValueError):
        StructureTransition(M5, S.BULLISH, S.BEARISH, TransitionReason.CHOCH, T0)
    with pytest.raises(ValueError):
        StructureTransition(M5, S.BULLISH, S.UNDETERMINED, TransitionReason.CHOCH, T0)


def test_choch_transition_has_no_trigger_swing():
    with pytest.raises(ValueError):
        StructureTransition(M5, S.BULLISH, S.REVALUATING, TransitionReason.CHOCH, T0, 3)


def test_transition_type_validation():
    with pytest.raises(ValueError):
        StructureTransition("5M", S.UNDETERMINED, S.BULLISH, TransitionReason.DIRECTIONAL_PAIR, T0, 1)
    with pytest.raises(ValueError):
        StructureTransition(M5, "UNDETERMINED", S.BULLISH, TransitionReason.DIRECTIONAL_PAIR, T0, 1)
    with pytest.raises(ValueError):
        StructureTransition(M5, S.UNDETERMINED, "BULLISH", TransitionReason.DIRECTIONAL_PAIR, T0, 1)
    with pytest.raises(ValueError):
        StructureTransition(M5, S.UNDETERMINED, S.BULLISH, "CHOCH", T0, 1)
    with pytest.raises(ValueError):
        StructureTransition(M5, S.UNDETERMINED, S.BULLISH, TransitionReason.DIRECTIONAL_PAIR, "now", 1)


def test_transition_accepts_timezone_aware_time_and_is_immutable():
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t = tr(S.UNDETERMINED, S.BULLISH, at=at)
    assert t.at == at
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.to_state = S.BEARISH
    assert tr(S.UNDETERMINED, S.BULLISH) == tr(S.UNDETERMINED, S.BULLISH)


# ------------------------------------------------------------- scope
def test_model_module_contains_no_engine_or_other_strategy_logic():
    for name in ("StructureEngine", "detect_bos", "detect_choch", "BOS", "CHOCH", "Zone",
                 "Liquidity", "SwingEngine", "enter_revaluating", "process_swing"):
        assert not hasattr(structure_module, name)
    sn = snap(state=S.BULLISH)
    for attr in ("bos", "choch", "zone", "zones", "liquidity", "entry", "sl", "tp",
                 "enter_revaluating", "update"):
        assert not hasattr(sn, attr)
