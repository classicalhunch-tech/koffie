import dataclasses
import random
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.structure_engine as engine_module
from koffie.strategy.engines.structure_engine import StructureEngine
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.structure import (
    DirectionalPermission,
    StructureSnapshot,
    StructureState,
    StructureTransition,
    TransitionReason,
)
from koffie.strategy.models.swing import Swing, SwingType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = StructureState
PAIR, CHOCH = TransitionReason.DIRECTIONAL_PAIR, TransitionReason.CHOCH
AT = datetime(2026, 2, 1, 12, 0)
 
 
class Feed:
    """Builds real Swing objects with increasing candle times."""
 
    def __init__(self, tf=M5):
        self.tf, self.seq, self.i = tf, 0, 0
 
    def _mk(self, kind, price, i, seq=None):
        ct = T0 + i * self.tf.duration
        s = Swing(kind, self.tf, float(price), ct, ct + 2 * self.tf.duration,
                  self.seq if seq is None else seq)
        self.seq = s.sequence + 1
        return s
 
    def high(self, price, seq=None):
        s = self._mk(HIGH, price, self.i, seq); self.i += 1; return s
 
    def low(self, price, seq=None):
        s = self._mk(LOW, price, self.i, seq); self.i += 1; return s
 
    def outside(self, high, low):
        i = self.i
        a, b = self._mk(HIGH, high, i), self._mk(LOW, low, i)
        self.i += 1
        return [a, b]
 
 
def bullish(tf=M5):
    """Highs 10,12 and lows 4,6 -> HH+HL -> BULLISH."""
    f, e = Feed(tf), StructureEngine(tf)
    for s in (f.high(10), f.low(4), f.high(12), f.low(6)):
        e.process_swings([s])
    assert e.state is S.BULLISH
    return e, f
 
 
def bearish(tf=M5):
    """Highs 12,10 and lows 6,4 -> LH+LL -> BEARISH."""
    f, e = Feed(tf), StructureEngine(tf)
    for s in (f.high(12), f.low(6), f.high(10), f.low(4)):
        e.process_swings([s])
    assert e.state is S.BEARISH
    return e, f
 
 
def feed_all(e, swings):
    for s in swings:
        e.process_swings([s])
 
 
def full_state(e):
    return (e.snapshot, e.transitions)
 
 
# ------------------------------------------------------------ construction
def test_initial_state():
    e = StructureEngine(M5)
    assert e.timeframe is M5
    assert e.state is S.UNDETERMINED
    assert e.snapshot == StructureSnapshot(M5)
    assert e.transitions == ()
    assert e.protected_swing is None
    assert e.permission is DirectionalPermission.NO_TRADE_PERMITTED
 
 
def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            StructureEngine(bad)
 
 
def test_uses_existing_models_not_replacements():
    assert engine_module.Swing is Swing
    assert engine_module.StructureSnapshot is StructureSnapshot
    assert engine_module.StructureTransition is StructureTransition
    assert engine_module.StructureState is StructureState
 
 
def test_state_is_read_only():
    e = StructureEngine(M5)
    for name in ("state", "snapshot", "transitions", "timeframe", "permission", "protected_swing"):
        with pytest.raises(AttributeError):
            setattr(e, name, None)
 
 
# ------------------------------------------------------------ classification
def test_becomes_bullish_on_hh_plus_hl_with_full_transition_record():
    f, e = Feed(), StructureEngine(M5)
    swings = [f.high(10), f.low(4), f.high(12)]
    for s in swings:
        assert e.process_swings([s]) == []
        assert e.state is S.UNDETERMINED
    last = f.low(6)
    made = e.process_swings([last])
    assert len(made) == 1
    t = made[0]
    assert (t.timeframe, t.from_state, t.to_state, t.reason) == (M5, S.UNDETERMINED, S.BULLISH, PAIR)
    assert t.at == last.confirmed_at and t.trigger_swing_sequence == last.sequence
    assert e.state is S.BULLISH and e.transitions == (t,)
    assert e.permission is DirectionalPermission.LONG_ONLY
    assert e.snapshot == StructureSnapshot(M5, S.BULLISH, swings[2], swings[0], last, swings[1])
 
 
def test_becomes_bearish_on_ll_plus_lh():
    e, _ = bearish()
    t = e.transitions[0]
    assert (t.from_state, t.to_state, t.reason) == (S.UNDETERMINED, S.BEARISH, PAIR)
    assert e.permission is DirectionalPermission.SHORT_ONLY
 
 
def test_protected_swing_follows_state():
    e, _ = bullish()
    assert e.protected_swing is e.snapshot.active_swing_low
    e2, _ = bearish()
    assert e2.protected_swing is e2.snapshot.active_swing_high
 
 
def test_fewer_than_two_highs_or_two_lows_stays_undetermined():
    f, e = Feed(), StructureEngine(M5)
    feed_all(e, [f.high(10), f.high(12), f.high(14)])           # no lows at all
    assert e.state is S.UNDETERMINED
    f, e = Feed(), StructureEngine(M5)
    feed_all(e, [f.high(10), f.low(4), f.high(12)])             # only one low
    assert e.state is S.UNDETERMINED and e.transitions == ()
 
 
def test_mixed_pairs_do_not_set_a_direction():
    f, e = Feed(), StructureEngine(M5)
    feed_all(e, [f.high(10), f.low(6), f.high(12), f.low(4)])   # HH + LL
    assert e.state is S.UNDETERMINED
    f, e = Feed(), StructureEngine(M5)
    feed_all(e, [f.high(12), f.low(4), f.high(10), f.low(6)])   # LH + HL
    assert e.state is S.UNDETERMINED
    assert e.transitions == ()
 
 
def test_equal_prices_never_qualify():
    for seq in (
        lambda f: [f.high(10), f.low(4), f.high(10), f.low(6)],   # equal high + HL
        lambda f: [f.high(10), f.low(4), f.high(12), f.low(4)],   # HH + equal low
        lambda f: [f.high(10), f.low(4), f.high(10), f.low(4)],   # both equal
        lambda f: [f.high(12), f.low(6), f.high(12), f.low(4)],   # equal high + LL
        lambda f: [f.high(12), f.low(6), f.high(10), f.low(6)],   # LH + equal low
    ):
        f, e = Feed(), StructureEngine(M5)
        feed_all(e, seq(f))
        assert e.state is S.UNDETERMINED and e.transitions == ()
 
 
def test_relations_use_consecutive_swings_not_historical_extremes():
    f, e = Feed(), StructureEngine(M5)
    feed_all(e, [f.high(15), f.low(4), f.high(14), f.low(6)])   # 14 < 15 -> LH, HL
    assert e.snapshot.high_relation.name == "LH"
    assert e.state is S.UNDETERMINED
    f, e = Feed(), StructureEngine(M5)
    feed_all(e, [f.high(15), f.low(4), f.high(14), f.high(13), f.low(3)])
    assert e.snapshot.active_swing_high.price == 13
    assert e.snapshot.prior_swing_high.price == 14          # latest two, not the 15 extreme
    assert e.state is S.BEARISH                              # LH + LL
 
 
def test_mixed_pair_leaves_bullish_unchanged_but_updates_swings():
    e, f = bullish()
    before = e.transitions
    assert e.process_swings([f.low(3)]) == []                # HH + LL is mixed
    assert e.state is S.BULLISH and e.transitions == before
    assert e.snapshot.active_swing_low.price == 3 and e.snapshot.prior_swing_low.price == 6
    assert e.protected_swing.price == 3
 
 
def test_same_direction_pair_again_is_not_a_new_transition():
    e, f = bullish()
    assert e.process_swings([f.high(14)]) == []
    assert e.process_swings([f.low(7)]) == []
    assert e.state is S.BULLISH and len(e.transitions) == 1
 
 
# ------------------------------------------------------------ direct flips
def test_direct_bullish_to_bearish_without_revaluating():
    e, f = bullish()
    assert e.process_swings([f.high(11)]) == []              # LH + HL: mixed
    made = e.process_swings([f.low(5)])                      # LL + LH
    assert [(t.from_state, t.to_state, t.reason) for t in made] == [(S.BULLISH, S.BEARISH, PAIR)]
    assert e.state is S.BEARISH
    assert S.REVALUATING not in [t.to_state for t in e.transitions]
    assert e.protected_swing is e.snapshot.active_swing_high
 
 
def test_direct_bearish_to_bullish_without_revaluating():
    e, f = bearish()
    assert e.process_swings([f.low(5)]) == []                # LH + HL: mixed
    made = e.process_swings([f.high(11)])                    # HH + HL
    assert [(t.from_state, t.to_state) for t in made] == [(S.BEARISH, S.BULLISH)]
    assert S.REVALUATING not in [t.to_state for t in e.transitions]
    assert e.protected_swing is e.snapshot.active_swing_low
 
 
# ------------------------------------------------------------ CHOCH interface
def test_enter_revaluating_from_bullish():
    e, _ = bullish()
    snap_before = e.snapshot
    t = e.enter_revaluating(AT)
    assert (t.from_state, t.to_state, t.reason, t.at, t.trigger_swing_sequence, t.timeframe) == \
           (S.BULLISH, S.REVALUATING, CHOCH, AT, None, M5)
    assert e.state is S.REVALUATING and e.transitions[-1] == t
    assert e.protected_swing is None
    assert e.permission is DirectionalPermission.NO_TRADE_PERMITTED
    for name in ("active_swing_high", "prior_swing_high", "active_swing_low", "prior_swing_low"):
        assert getattr(e.snapshot, name) is getattr(snap_before, name)
 
 
def test_enter_revaluating_from_bearish():
    e, _ = bearish()
    t = e.enter_revaluating(AT)
    assert (t.from_state, t.to_state) == (S.BEARISH, S.REVALUATING)
    assert e.state is S.REVALUATING
 
 
def test_enter_revaluating_rejected_from_undetermined_and_revaluating():
    e = StructureEngine(M5)
    with pytest.raises(ValueError):
        e.enter_revaluating(AT)
    assert full_state(e) == (StructureSnapshot(M5), ())
    e, _ = bullish()
    e.enter_revaluating(AT)
    before = full_state(e)
    with pytest.raises(ValueError):
        e.enter_revaluating(AT + timedelta(minutes=5))
    assert full_state(e) == before
 
 
def test_enter_revaluating_rejects_non_datetime_and_changes_nothing():
    e, _ = bullish()
    before = full_state(e)
    for bad in (None, "2026-01-01", 123, AT.date()):
        with pytest.raises(TypeError):
            e.enter_revaluating(bad)
    assert full_state(e) == before and e.state is S.BULLISH
 
 
def test_choch_does_not_restore_the_old_trend_by_itself():
    e, _ = bullish()
    e.enter_revaluating(AT)
    assert e.snapshot.directional_pair is S.BULLISH          # old pair is still HH+HL
    assert e.state is S.REVALUATING
    assert e.process_swings([]) == []
    assert e.state is S.REVALUATING
 
 
def test_engine_has_no_choch_detection_or_other_strategy_logic():
    e = StructureEngine(M5)
    for attr in ("detect_choch", "detect_bos", "process_candle", "bos", "choch", "zone",
                 "zones", "liquidity", "entry", "sl", "tp", "swing_engine"):
        assert not hasattr(e, attr)
 
 
# ------------------------------------------------------------ restoration
def test_restore_bullish_from_newest_swing_plus_existing_opposite_swing():
    e, f = bullish()
    e.enter_revaluating(AT)
    new_high = f.high(13)                                    # HH; existing low pair is still HL
    made = e.process_swings([new_high])
    assert [(t.from_state, t.to_state, t.reason, t.trigger_swing_sequence) for t in made] == \
           [(S.REVALUATING, S.BULLISH, PAIR, new_high.sequence)]
    assert e.state is S.BULLISH and e.protected_swing is e.snapshot.active_swing_low
 
 
def test_restore_bullish_from_a_new_higher_low():
    e, f = bullish()
    e.enter_revaluating(AT)
    e.process_swings([f.low(7)])                             # HL + existing HH
    assert e.state is S.BULLISH
 
 
def test_restore_bearish_from_newest_swing_plus_existing_opposite_swing():
    e, f = bearish()
    e.enter_revaluating(AT)
    e.process_swings([f.low(3)])                             # LL + existing LH
    assert e.state is S.BEARISH
 
 
def test_bullish_choch_then_bearish_pair_gives_bearish():
    e, f = bullish()
    e.enter_revaluating(AT)
    assert e.process_swings([f.high(11)]) == []              # LH + HL: mixed, stays
    assert e.state is S.REVALUATING
    made = e.process_swings([f.low(5)])                      # LL + LH
    assert [(t.from_state, t.to_state) for t in made] == [(S.REVALUATING, S.BEARISH)]
 
 
def test_bearish_choch_then_bullish_pair_gives_bullish():
    e, f = bearish()
    e.enter_revaluating(AT)
    assert e.process_swings([f.high(11)]) == []              # HH + LL: mixed
    assert e.state is S.REVALUATING
    e.process_swings([f.low(5)])                             # HH + HL
    assert e.state is S.BULLISH
 
 
def test_mixed_or_tied_pair_keeps_revaluating():
    e, f = bullish()
    e.enter_revaluating(AT)
    e.process_swings([f.low(3)])                             # HH + LL mixed
    assert e.state is S.REVALUATING
    e, f = bullish()
    e.enter_revaluating(AT)
    e.process_swings([f.high(12)])                           # equal high with HL
    assert e.state is S.REVALUATING
 
 
def test_can_enter_revaluating_again_after_restoration():
    e, f = bullish()
    e.enter_revaluating(AT)
    e.process_swings([f.high(13)])
    e.enter_revaluating(AT + timedelta(hours=1))
    assert [t.to_state for t in e.transitions] == [S.BULLISH, S.REVALUATING, S.BULLISH, S.REVALUATING]
 
 
# ------------------------------------------------------------ outside bars
def test_outside_bar_is_evaluated_swing_by_swing_high_then_low():
    e, f = bullish()
    e.enter_revaluating(AT)
    high, low = f.outside(13, 3)
    made = e.process_swings([high, low])
    # HIGH: HH + HL -> BULLISH restored. LOW: HH + LL mixed -> stays BULLISH.
    assert [(t.from_state, t.to_state, t.trigger_swing_sequence) for t in made] == \
           [(S.REVALUATING, S.BULLISH, high.sequence)]
    assert e.state is S.BULLISH
    assert e.snapshot.active_swing_high is high and e.snapshot.active_swing_low is low
 
 
def test_outside_bar_can_flip_on_its_low_after_a_neutral_high():
    e, f = bullish()
    high, low = f.outside(11, 5)          # HIGH: LH+HL mixed; LOW: LL+LH bearish
    made = e.process_swings([high, low])
    assert [(t.from_state, t.to_state, t.trigger_swing_sequence) for t in made] == \
           [(S.BULLISH, S.BEARISH, low.sequence)]
 
 
def test_outside_bar_gives_same_result_as_supplying_the_swings_separately():
    for build in (bullish, bearish):
        a, fa = build()
        b, fb = build()
        a.enter_revaluating(AT)
        b.enter_revaluating(AT)
        pair_a = fa.outside(13, 3)
        pair_b = fb.outside(13, 3)
        a.process_swings(pair_a)
        b.process_swings([pair_b[0]])
        b.process_swings([pair_b[1]])
        assert full_state(a) == full_state(b)
 
 
def test_swings_are_processed_in_the_order_supplied():
    e1, f1 = bullish()
    e2, f2 = bullish()
    e1.enter_revaluating(AT)
    e2.enter_revaluating(AT)
    h, l = f1.outside(13, 3)
    h2, l2 = f2.outside(13, 3)
    e1.process_swings([h, l])             # HIGH first: HH+HL restores BULLISH, LOW is then mixed
    e2.process_swings([l2, h2])           # LOW first: HH+LL mixed, then HIGH still HH+LL mixed
    assert e1.state is S.BULLISH
    assert e2.state is S.REVALUATING      # supplied order decides; no reordering by the engine
    assert [t.trigger_swing_sequence for t in e1.transitions] == [3, None, h.sequence]   # pair, CHOCH, restore
    assert [t.to_state for t in e2.transitions] == [S.BULLISH, S.REVALUATING]
 
 
# ------------------------------------------------------------ validation
def test_rejects_non_swings_including_lookalikes_and_non_sequences():
    class Lookalike:
        swing_type, timeframe, price = HIGH, M5, 12.0
        candle_time, confirmed_at, sequence = T0, T0 + timedelta(minutes=10), 0
 
    e = StructureEngine(M5)
    for bad in (Lookalike(), None, 12.0, "swing", {"price": 1}):
        with pytest.raises(TypeError):
            e.process_swings([bad])
    for bad in (Feed().high(10), None, "abc", iter([Feed().high(10)]), {1}):
        with pytest.raises(TypeError):
            e.process_swings(bad)
    assert full_state(e) == (StructureSnapshot(M5), ())
 
 
def test_rejects_wrong_timeframe_for_every_pairing():
    for etf in (M5, M15, H1):
        for stf in (M5, M15, H1):
            e = StructureEngine(etf)
            swing = Feed(stf).high(10)
            if etf is stf:
                e.process_swings([swing])
            else:
                with pytest.raises(ValueError):
                    e.process_swings([swing])
 
 
def test_empty_batch_is_a_no_op():
    e, _ = bullish()
    before = full_state(e)
    assert e.process_swings([]) == [] and e.process_swings(()) == []
    assert full_state(e) == before
 
 
def test_rejects_confirmed_at_going_backwards():
    e = StructureEngine(M5)
    late = Swing(HIGH, M5, 10.0, T0 + 10 * M5.duration, T0 + 12 * M5.duration, 0)
    early = Swing(LOW, M5, 5.0, T0, T0 + 2 * M5.duration, 1)
    e.process_swings([late])
    before = full_state(e)
    with pytest.raises(ValueError):
        e.process_swings([early])
    assert full_state(e) == before
 
 
def test_accepts_equal_confirmed_at():
    f, e = Feed(), StructureEngine(M5)
    e.process_swings(f.outside(10, 5))       # both share one confirmed_at
 
 
def test_rejects_same_type_candle_time_not_increasing():
    e = StructureEngine(M5)
    a = Swing(HIGH, M5, 10.0, T0 + 5 * M5.duration, T0 + 7 * M5.duration, 0)
    same = Swing(HIGH, M5, 12.0, T0 + 5 * M5.duration, T0 + 8 * M5.duration, 1)
    earlier = Swing(HIGH, M5, 12.0, T0 + 3 * M5.duration, T0 + 9 * M5.duration, 2)
    e.process_swings([a])
    before = full_state(e)
    for bad in (same, earlier):
        with pytest.raises(ValueError):
            e.process_swings([bad])
        assert full_state(e) == before
 
 
def test_candle_time_order_is_only_checked_within_a_swing_type():
    e = StructureEngine(M5)
    high = Swing(HIGH, M5, 10.0, T0 + 5 * M5.duration, T0 + 7 * M5.duration, 0)
    low = Swing(LOW, M5, 5.0, T0 + 4 * M5.duration, T0 + 8 * M5.duration, 1)
    e.process_swings([high, low])            # low's candle is earlier, different type: fine
    assert e.snapshot.active_swing_low is low
 
 
def test_mixing_naive_and_aware_datetimes_raises_and_changes_nothing():
    e = StructureEngine(M5)
    naive = Feed().high(10)
    e.process_swings([naive])
    aware_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    aware = Swing(LOW, M5, 4.0, aware_t, aware_t + 2 * M5.duration, 1)
    before = full_state(e)
    with pytest.raises(TypeError):
        e.process_swings([aware])
    assert full_state(e) == before
 
 
def test_timezone_aware_swings_work():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    e = StructureEngine(M5)
    seq = [(HIGH, 10), (LOW, 4), (HIGH, 12), (LOW, 6)]
    for i, (kind, price) in enumerate(seq):
        ct = t0 + i * M5.duration
        e.process_swings([Swing(kind, M5, float(price), ct, ct + 2 * M5.duration, i)])
    assert e.state is S.BULLISH
 
 
def test_a_rejected_batch_is_atomic():
    e, f = bullish()
    before = full_state(e)
    good = f.high(13)
    bad = Swing(LOW, M15, 3.0, T0 + 99 * M5.duration, T0 + 99 * M5.duration + 2 * M15.duration, 99)
    with pytest.raises(ValueError):
        e.process_swings([good, bad])
    assert full_state(e) == before
    e.process_swings([good])                 # engine is healthy afterwards
    assert e.snapshot.active_swing_high is good
 
 
def test_transition_history_is_append_only_and_read_only():
    e, f = bullish()
    snap = e.transitions
    assert isinstance(snap, tuple)
    e.process_swings([f.high(11)])
    made = e.process_swings([f.low(5)])          # direct flip -> one new transition
    assert len(made) == 1
    assert e.transitions[: len(snap)] == snap and len(e.transitions) == 2
    made.clear()                                  # returned list is a copy
    assert len(e.transitions) == 2
 
 
def test_snapshot_and_transitions_are_immutable():
    e, _ = bullish()
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.snapshot.state = S.BEARISH
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.transitions[0].to_state = S.BEARISH
 
 
# ------------------------------------------------------------ sequence is metadata
def test_sequence_gaps_are_accepted():
    f, e = Feed(), StructureEngine(M5)
    swings = [f.high(10, seq=0), f.low(4, seq=1), f.high(12, seq=5), f.low(6, seq=9)]
    for s in swings:
        e.process_swings([s])
    assert e.state is S.BULLISH
    assert e.transitions[0].trigger_swing_sequence == 9
 
 
def test_first_swing_may_have_any_non_negative_sequence():
    f, e = Feed(), StructureEngine(M5)
    e.process_swings([f.high(10, seq=5)])
    e.process_swings([f.low(4, seq=9)])
    assert e.snapshot.active_swing_high.sequence == 5
 
 
def test_sequence_5_after_3_is_accepted_and_batches_need_not_contain_earlier_ones():
    f, e = Feed(), StructureEngine(M5)
    e.process_swings([f.high(10, seq=3)])
    e.process_swings([f.high(12, seq=5)])
    assert e.snapshot.prior_swing_high.sequence == 3
    assert e.snapshot.active_swing_high.sequence == 5
 
 
def test_sequence_is_not_compared_across_swing_types():
    f, e = Feed(), StructureEngine(M5)
    e.process_swings([f.high(10, seq=5)])
    e.process_swings([f.low(4, seq=2)])      # lower sequence, other type: accepted
    assert e.snapshot.active_swing_low.sequence == 2
 
 
def test_same_type_non_increasing_sequence_is_rejected_by_the_structure_model():
    # This comes from StructureSnapshot's own invariant (prior.sequence < active.sequence
    # for one swing type), not from an engine rule. The whole call is rejected.
    for second_seq in (5, 3):
        f, e = Feed(), StructureEngine(M5)
        e.process_swings([f.high(10, seq=5)])
        before = full_state(e)
        with pytest.raises(ValueError):
            e.process_swings([f.high(12, seq=second_seq)])
        assert full_state(e) == before
 
 
# ------------------------------------------------------------ timeframes
def test_every_timeframe_works_and_is_recorded():
    for tf in (M5, M15, H1):
        e, _ = bullish(tf)
        assert e.timeframe is tf and e.snapshot.timeframe is tf
        assert e.transitions[0].timeframe is tf
        assert e.enter_revaluating(AT).timeframe is tf
 
 
# ------------------------------------------------------------ integration + oracle
class Reference:
    """Independent plain-price implementation of the locked literal rules."""
 
    def __init__(self):
        self.state = "UNDETERMINED"
        self.h = []
        self.l = []
 
    def swing(self, kind, price):
        (self.h if kind is HIGH else self.l).append(price)
        if len(self.h) < 2 or len(self.l) < 2:
            return
        hh, lh = self.h[-1] > self.h[-2], self.h[-1] < self.h[-2]
        hl, ll = self.l[-1] > self.l[-2], self.l[-1] < self.l[-2]
        if hh and hl:
            self.state = "BULLISH"
        elif ll and lh:
            self.state = "BEARISH"
 
    def choch(self):
        assert self.state in ("BULLISH", "BEARISH")
        self.state = "REVALUATING"
 
 
def test_engine_matches_independent_reference_with_real_swing_engine():
    for seed in range(40):
        rng = random.Random(seed)
        swing_engine, structure = SwingEngine(M5), StructureEngine(M5)
        ref = Reference()
        for i in range(80):
            high = rng.randint(5, 15)
            low = high - rng.randint(1, 5)
            mid = (high + low) / 2
            candle = Candle(M5, T0 + i * M5.duration, mid, float(high), float(low), mid)
            swings = swing_engine.process_candle(candle, now=candle.close_time)
            structure.process_swings(swings)
            for s in swings:
                ref.swing(s.swing_type, s.price)
            if structure.state.value in ("BULLISH", "BEARISH") and rng.random() < 0.25:
                structure.enter_revaluating(candle.close_time)
                ref.choch()
            assert structure.state.value == ref.state, (seed, i)
            assert structure.permission is structure.state.permission
 
 
def test_deterministic_replay():
    def run():
        e, f = bullish()
        e.enter_revaluating(AT)
        e.process_swings(f.outside(13, 3))
        e.process_swings([f.high(11)])
        return full_state(e)
    assert run() == run()