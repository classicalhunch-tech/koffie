import random
from datetime import datetime, timedelta, timezone

import pytest

import koffie.strategy.engines.choch_engine as engine_module
from koffie.strategy.engines.bos_engine import BOSEngine
from koffie.strategy.engines.choch_engine import CHOCHEngine
from koffie.strategy.engines.structure_engine import StructureEngine
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.choch import CHOCH, CHOCHDirection
from koffie.strategy.models.structure import StructureSnapshot, StructureState
from koffie.strategy.models.swing import Swing, SwingType

T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = StructureState
BEAR, BULL = CHOCHDirection.BEARISH, CHOCHDirection.BULLISH


def sw(kind, price, seq, tf=M5, idx=0):
    ct = T0 + idx * tf.duration
    return Swing(kind, tf, float(price), ct, ct + 2 * tf.duration, seq)


def bull_snap(tf=M5):
    """Highs 10,12 / lows 4,6: BULLISH. Protected low = 6 (active, not the lowest 4)."""
    return StructureSnapshot(tf, S.BULLISH,
                             sw(HIGH, 12, 2, tf, 2), sw(HIGH, 10, 0, tf, 0),
                             sw(LOW, 6, 3, tf, 3), sw(LOW, 4, 1, tf, 1))


def bear_snap(tf=M5):
    """Highs 12,10 / lows 6,4: BEARISH. Protected high = 10 (active, not the highest 12)."""
    return StructureSnapshot(tf, S.BEARISH,
                             sw(HIGH, 10, 2, tf, 2), sw(HIGH, 12, 0, tf, 0),
                             sw(LOW, 4, 3, tf, 3), sw(LOW, 6, 1, tf, 1))


def undetermined_snap(tf=M5):
    return StructureSnapshot(tf, S.UNDETERMINED,
                             sw(HIGH, 12, 2, tf, 2), sw(HIGH, 10, 0, tf, 0),
                             sw(LOW, 6, 3, tf, 3), sw(LOW, 4, 1, tf, 1))


def revaluating_snap(tf=M5):
    b = bull_snap(tf)
    return StructureSnapshot(tf, S.REVALUATING, b.active_swing_high, b.prior_swing_high,
                             b.active_swing_low, b.prior_swing_low)


def cd(i, o, h, l, c, tf=M5):
    return Candle(tf, T0 + (20 + i) * tf.duration, o, h, l, c)


def run(engine, candle, snapshot):
    return engine.process_candle(candle, candle.close_time, snapshot)


# ---------------------------------------------------------------- construction
def test_initial_state():
    e = CHOCHEngine(M5)
    assert e.timeframe is M5 and e.history == ()


def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            CHOCHEngine(bad)


def test_uses_existing_models():
    assert engine_module.CHOCH is CHOCH and engine_module.CHOCHDirection is CHOCHDirection
    assert engine_module.StructureSnapshot is StructureSnapshot
    assert engine_module.Candle is Candle


def test_state_is_read_only():
    e = CHOCHEngine(M5)
    for name in ("timeframe", "history"):
        with pytest.raises(AttributeError):
            setattr(e, name, None)


# ---------------------------------------------------------------- CHOCH detection
def test_bearish_choch_when_low_goes_below_the_protected_low():
    e, snap = CHOCHEngine(M5), bull_snap()
    candle = cd(0, 8, 9, 5, 7)                                   # low 5 < protected low 6
    choch = run(e, candle, snap)
    assert isinstance(choch, CHOCH)
    assert choch.timeframe is M5 and choch.direction is BEAR
    assert choch.protected_swing is snap.protected_swing is snap.active_swing_low
    assert choch.candle_time == candle.open_time and choch.break_price == 5
    assert choch.confirmed_at == candle.close_time and choch.prior_state is S.BULLISH
    assert e.history == (choch,)


def test_bullish_choch_when_high_goes_above_the_protected_high():
    e, snap = CHOCHEngine(M5), bear_snap()
    candle = cd(0, 8, 11, 7, 9)                                  # high 11 > protected high 10
    choch = run(e, candle, snap)
    assert isinstance(choch, CHOCH)
    assert choch.direction is BULL
    assert choch.protected_swing is snap.protected_swing is snap.active_swing_high
    assert choch.candle_time == candle.open_time and choch.break_price == 11
    assert choch.confirmed_at == candle.close_time and choch.prior_state is S.BEARISH
    assert e.history == (choch,)


def test_break_price_is_the_candle_low_for_bearish_and_the_high_for_bullish():
    assert run(CHOCHEngine(M5), cd(0, 8, 20, 2, 7), bull_snap()).break_price == 2
    assert run(CHOCHEngine(M5), cd(0, 8, 20, 2, 7), bear_snap()).break_price == 20


def test_the_wick_is_enough_even_if_the_close_is_back_inside():
    # BULLISH: low 5 < 6, close 9 is well above the protected low.
    choch = run(CHOCHEngine(M5), cd(0, 8, 10, 5, 9), bull_snap())
    assert choch is not None and choch.break_price == 5
    # BEARISH: high 11 > 10, close 8 is well below the protected high.
    choch = run(CHOCHEngine(M5), cd(0, 9, 11, 7, 8), bear_snap())
    assert choch is not None and choch.break_price == 11


def test_a_close_beyond_the_level_also_qualifies_because_the_wick_is_beyond_it():
    assert run(CHOCHEngine(M5), cd(0, 8, 8.5, 3, 4), bull_snap()) is not None
    assert run(CHOCHEngine(M5), cd(0, 8, 14, 7.5, 13), bear_snap()) is not None


def test_touching_the_protected_swing_exactly_is_not_a_choch():
    e = CHOCHEngine(M5)
    assert run(e, cd(0, 8, 9, 6, 7), bull_snap()) is None        # low == 6
    assert run(e, cd(1, 8, 10, 7, 9), bear_snap()) is None       # high == 10
    assert e.history == ()


def test_strictness_holds_just_either_side_of_the_level():
    assert run(CHOCHEngine(M5), cd(0, 8, 9, 6.0001, 7), bull_snap()) is None
    assert run(CHOCHEngine(M5), cd(0, 8, 9, 5.9999, 7), bull_snap()) is not None
    assert run(CHOCHEngine(M5), cd(0, 8, 9.9999, 7, 9), bear_snap()) is None
    assert run(CHOCHEngine(M5), cd(0, 8, 10.0001, 7, 9), bear_snap()) is not None


def test_no_choch_when_the_candle_stays_inside_the_protected_level():
    assert run(CHOCHEngine(M5), cd(0, 8, 13, 7, 12.5), bull_snap()) is None
    assert run(CHOCHEngine(M5), cd(0, 8, 9, 3, 3.5), bear_snap()) is None


def test_the_protected_swing_is_the_active_swing_not_an_older_or_extreme_one():
    # Low 5 is below the active low 6 but above nothing older that matters:
    # the prior low is 4, the active (protected) low is 6, so 5 breaks it.
    snap = bull_snap()
    assert snap.prior_swing_low.price == 4
    choch = run(CHOCHEngine(M5), cd(0, 8, 9, 5, 7), snap)
    assert choch.protected_swing is snap.active_swing_low and choch.protected_swing.price == 6
    # Same for BEARISH: high 11 breaks the active high 10 although the prior high is 12.
    snap = bear_snap()
    assert snap.prior_swing_high.price == 12
    choch = run(CHOCHEngine(M5), cd(0, 8, 11, 7, 9), snap)
    assert choch.protected_swing is snap.active_swing_high and choch.protected_swing.price == 10


def test_the_close_never_decides_the_choch():
    # Same low, different closes: same CHOCH condition result.
    for close in (5.5, 8, 9.9):
        assert run(CHOCHEngine(M5), cd(0, 8, 10, 5, close), bull_snap()) is not None
    for close in (6.1, 8, 9.9):
        assert run(CHOCHEngine(M5), cd(0, 8, 10, 6.1, close), bull_snap()) is None


def test_only_the_structures_own_protected_side_can_break():
    # BULLISH structure, high far above the active high: that is BOS territory, not CHOCH.
    assert run(CHOCHEngine(M5), cd(0, 8, 30, 7, 25), bull_snap()) is None
    # BEARISH structure, low far below the active low.
    assert run(CHOCHEngine(M5), cd(0, 8, 9, -5, -3), bear_snap()) is None


def test_no_choch_in_undetermined_or_revaluating():
    for snap in (undetermined_snap(), revaluating_snap(), StructureSnapshot(M5)):
        e = CHOCHEngine(M5)
        assert snap.protected_swing is None
        assert run(e, cd(0, 8, 30, -5, 9), snap) is None          # breaks everything
        assert run(e, cd(1, 8, 13, 3, 9), snap) is None
        assert e.history == ()


def test_engine_judges_each_candle_against_the_snapshot_it_is_given():
    # Same candle: breaks the old protected low 6 but not a newer active low 4.
    older = bull_snap()
    newer = StructureSnapshot(M5, S.BULLISH, older.active_swing_high, older.prior_swing_high,
                              sw(LOW, 4, 4, M5, 4), older.active_swing_low)
    candle = cd(0, 8, 9, 5, 7)
    assert run(CHOCHEngine(M5), candle, older) is not None
    assert run(CHOCHEngine(M5), candle, newer) is None


def test_engine_has_no_memory_of_the_structure_between_calls():
    # If the caller keeps giving a BULLISH snapshot, each breaking candle is a CHOCH again.
    e, snap = CHOCHEngine(M5), bull_snap()
    events = [run(e, cd(i, 8, 9, 5 - i, 7), snap) for i in range(3)]
    assert all(isinstance(c, CHOCH) for c in events)
    assert len({c.candle_time for c in events}) == 3
    assert all(c.protected_swing is snap.active_swing_low for c in events)
    assert e.history == tuple(events)
    # Once the caller passes a REVALUATING snapshot, nothing more is emitted.
    assert run(e, cd(3, 8, 9, 0, 7), revaluating_snap()) is None
    assert len(e.history) == 3


def test_a_non_breaking_candle_does_not_affect_later_candles():
    e, snap = CHOCHEngine(M5), bull_snap()
    assert run(e, cd(0, 8, 9, 7, 8), snap) is None
    assert run(e, cd(1, 8, 9, 5, 7), snap) is not None


# ---------------------------------------------------------------- independence from BOS (Option A)
def test_candle_meeting_both_conditions_is_a_choch_and_no_bos():
    # BULLISH: close 13 > active high 12 (BOS condition) and low 5 < protected low 6.
    candle, snap = cd(0, 12, 14, 5, 13), bull_snap()
    assert run(CHOCHEngine(M5), candle, snap) is not None
    assert run(BOSEngine(M5), candle, snap) is None
    # BEARISH: close 3 < active low 4 (BOS condition) and high 11 > protected high 10.
    candle, snap = cd(0, 5, 11, 2, 3), bear_snap()
    assert run(CHOCHEngine(M5), candle, snap) is not None
    assert run(BOSEngine(M5), candle, snap) is None


def test_choch_emission_does_not_depend_on_the_bos_condition():
    # The CHOCH engine fires the same whether or not the close would also be a BOS.
    with_bos_close = run(CHOCHEngine(M5), cd(0, 12, 14, 5, 13), bull_snap())
    without_bos_close = run(CHOCHEngine(M5), cd(0, 8, 9, 5, 7), bull_snap())
    assert with_bos_close is not None and without_bos_close is not None


def test_a_bos_candle_that_does_not_break_the_protected_swing_is_not_a_choch():
    candle, snap = cd(0, 11, 13, 10, 12.5), bull_snap()
    assert run(BOSEngine(M5), candle, snap) is not None
    assert run(CHOCHEngine(M5), candle, snap) is None


def test_choch_and_bos_are_never_both_emitted_for_one_candle():
    rng = random.Random(7)
    both = only_choch = only_bos = 0
    for _ in range(3000):
        snap = rng.choice((bull_snap(), bear_snap()))
        high = float(rng.randint(0, 20))
        low = high - rng.randint(0, 8)
        o, c = rng.uniform(low, high), rng.uniform(low, high)
        candle = cd(0, o, high, low, c)
        got_choch = run(CHOCHEngine(M5), candle, snap)
        got_bos = run(BOSEngine(M5), candle, snap)
        assert not (got_choch is not None and got_bos is not None)
        only_choch += got_choch is not None
        only_bos += got_bos is not None
    assert only_choch > 0 and only_bos > 0 and both == 0


# ---------------------------------------------------------------- validation
def test_rejects_wrong_argument_types():
    class Lookalike:
        timeframe, high, low, open, close = M5, 14.0, 10.0, 11.0, 13.0
        open_time = T0
        close_time = T0 + timedelta(minutes=5)

        def is_closed_at(self, now):
            return True

    e = CHOCHEngine(M5)
    good = cd(0, 8, 9, 5, 7)
    for bad in (Lookalike(), None, "candle", (1, 2)):
        with pytest.raises(TypeError):
            e.process_candle(bad, good.close_time, bull_snap())
    for bad in (None, "now", 5, good.open_time.date()):
        with pytest.raises(TypeError):
            e.process_candle(good, bad, bull_snap())
    for bad in (None, "snap", object()):
        with pytest.raises(TypeError):
            e.process_candle(good, good.close_time, bad)
    assert e.history == ()


def test_rejects_wrong_timeframe_candle_or_snapshot():
    for etf in (M5, M15, H1):
        for other in (M5, M15, H1):
            e = CHOCHEngine(etf)
            candle, snap = cd(0, 8, 9, 5, 7, other), bull_snap(other)
            if etf is other:
                run(e, candle, snap)
            else:
                with pytest.raises(ValueError):
                    run(e, candle, snap)
    e = CHOCHEngine(M5)
    with pytest.raises(ValueError):
        run(e, cd(0, 8, 9, 5, 7), bull_snap(M15))
    with pytest.raises(ValueError):
        run(e, cd(0, 8, 9, 5, 7, M15), bull_snap(M5))


def test_rejects_unfinished_candle():
    e = CHOCHEngine(M5)
    candle = cd(0, 8, 9, 5, 7)
    for offset in (timedelta(0), timedelta(minutes=4, seconds=59)):
        with pytest.raises(ValueError):
            e.process_candle(candle, candle.open_time + offset, bull_snap())
    assert e.history == ()
    assert e.process_candle(candle, candle.close_time, bull_snap()) is not None


def test_rejects_repeated_earlier_and_overlapping_candles():
    e, snap = CHOCHEngine(M5), bull_snap()
    first = cd(3, 8, 9, 5, 7)
    run(e, first, snap)
    for bad in (first, cd(2, 8, 9, 5, 7),
                Candle(M5, first.open_time + timedelta(minutes=1), 8, 9, 5, 7)):
        with pytest.raises(ValueError):
            e.process_candle(bad, bad.close_time + timedelta(hours=1), snap)
    assert len(e.history) == 1


def test_accepts_gaps_and_back_to_back_candles():
    e, snap = CHOCHEngine(M5), bull_snap()
    run(e, cd(0, 8, 9, 5, 7), snap)
    run(e, cd(1, 8, 9, 5, 7), snap)                              # opens exactly at previous close
    run(e, cd(500, 8, 9, 5, 7), snap)                            # weekend-sized gap
    assert len(e.history) == 3


def test_rejects_a_snapshot_containing_a_future_swing():
    e = CHOCHEngine(M5)
    candle = cd(0, 8, 9, 5, 7)
    future = Swing(LOW, M5, 6.0, candle.close_time, candle.close_time + 2 * M5.duration, 4)
    snap = StructureSnapshot(M5, S.BULLISH, sw(HIGH, 12, 2, M5, 2), sw(HIGH, 10, 0, M5, 0),
                             future, sw(LOW, 4, 1, M5, 1))
    with pytest.raises(ValueError):
        run(e, candle, snap)
    assert e.history == ()


def test_swing_confirmed_exactly_at_the_candle_close_is_allowed():
    e = CHOCHEngine(M5)
    candle = cd(0, 8, 9, 5, 7)
    same_close = Swing(LOW, M5, 6.0, candle.open_time - M5.duration, candle.close_time, 4)
    assert same_close.confirmed_at == candle.close_time
    snap = StructureSnapshot(M5, S.BULLISH, sw(HIGH, 12, 2, M5, 2), sw(HIGH, 10, 0, M5, 0),
                             same_close, sw(LOW, 4, 1, M5, 1))
    choch = run(e, candle, snap)
    assert choch is not None and choch.protected_swing is same_close


def test_rejected_calls_change_nothing():
    e, snap = CHOCHEngine(M5), bull_snap()
    good = cd(0, 8, 9, 5, 7)
    with pytest.raises(ValueError):
        e.process_candle(good, good.open_time, snap)                 # unfinished
    with pytest.raises(ValueError):
        run(e, good, bull_snap(M15))                                 # wrong snapshot timeframe
    assert e.history == ()
    assert run(e, good, snap) is not None                            # same candle now accepted


def test_mixing_naive_and_aware_datetimes_raises_and_changes_nothing():
    e = CHOCHEngine(M5)
    aware_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    aware_candle = Candle(M5, aware_t, 8, 9, 5, 7)
    with pytest.raises(TypeError):
        run(e, aware_candle, bull_snap())                            # naive swings, aware candle
    assert e.history == ()
    run(e, cd(0, 8, 9, 5, 7), bull_snap())                           # still healthy


def test_history_is_append_only_and_a_tuple():
    e, snap = CHOCHEngine(M5), bull_snap()
    run(e, cd(0, 8, 9, 5, 7), snap)
    first = e.history
    assert isinstance(first, tuple)
    run(e, cd(1, 8, 9, 5, 7), snap)
    assert e.history[:1] == first and len(e.history) == 2


def test_every_timeframe_works():
    for tf in (M5, M15, H1):
        e = CHOCHEngine(tf)
        candle = cd(0, 8, 9, 5, 7, tf)
        choch = run(e, candle, bull_snap(tf))
        assert choch.timeframe is tf and choch.confirmed_at == candle.close_time


def test_every_timeframe_works_for_bullish_choch():
    for tf in (M5, M15, H1):
        candle = cd(0, 8, 11, 7, 9, tf)
        choch = run(CHOCHEngine(tf), candle, bear_snap(tf))
        assert choch.timeframe is tf and choch.direction is BULL


# ---------------------------------------------------------------- scope
def test_engine_never_changes_the_snapshot_and_has_no_other_logic():
    e, snap = CHOCHEngine(M5), bull_snap()
    before = snap
    run(e, cd(0, 8, 9, 5, 7), snap)
    assert snap == before and snap.state is S.BULLISH                # the caller moves the state
    for attr in ("enter_revaluating", "structure", "structure_engine", "process_swings",
                 "detect_bos", "bos", "zone", "liquidity", "entry", "sl", "tp", "pivot"):
        assert not hasattr(e, attr)
    for name in ("StructureEngine", "SwingEngine", "BOS", "BOSEngine"):
        assert not hasattr(engine_module, name)


def test_engine_holds_no_structure_engine_and_never_calls_enter_revaluating():
    structure = StructureEngine(M5)
    calls = []
    structure.enter_revaluating = lambda at: calls.append(at)        # spy on the caller's engine
    e = CHOCHEngine(M5)
    run(e, cd(0, 8, 9, 5, 7), bull_snap())
    assert calls == []
    assert not any(isinstance(v, StructureEngine) for v in vars(e).values())


# ---------------------------------------------------------------- pipeline with real engines
class Pipeline:
    """SwingEngine -> StructureEngine.process_swings -> CHOCHEngine + BOSEngine, per candle.

    Like a real caller, this harness (not CHOCHEngine) calls
    structure.enter_revaluating(candle.close_time) after a CHOCH.
    """

    def __init__(self, tf=M5):
        self.swing, self.structure = SwingEngine(tf), StructureEngine(tf)
        self.choch, self.bos = CHOCHEngine(tf), BOSEngine(tf)
        self.tf, self.i = tf, 0

    def feed(self, high, low, enter_revaluating=True):
        tf = self.tf
        mid = (high + low) / 2
        c = Candle(tf, T0 + self.i * tf.duration, mid, float(high), float(low), mid)
        self.i += 1
        self.structure.process_swings(self.swing.process_candle(c, c.close_time))
        snap = self.structure.snapshot
        choch = self.choch.process_candle(c, c.close_time, snap)
        bos = self.bos.process_candle(c, c.close_time, snap)
        if choch is not None and enter_revaluating:
            self.structure.enter_revaluating(c.close_time)
        return choch, bos


BULLISH_SETUP = [(8, 6), (10, 7), (9, 5), (8, 4), (11, 5), (12, 8), (11, 7), (10, 6), (11, 7)]
BEARISH_SETUP = [(20 - l, 20 - h) for h, l in BULLISH_SETUP]     # mirror image around 10


def bullish_pipeline():
    p = Pipeline()
    for h, l in BULLISH_SETUP:
        assert p.feed(h, l) == (None, None)
    assert p.structure.state is S.BULLISH
    assert p.structure.snapshot.active_swing_high.price == 12
    assert p.structure.snapshot.active_swing_low.price == 6
    assert p.choch.history == ()
    return p


def bearish_pipeline():
    p = Pipeline()
    for h, l in BEARISH_SETUP:
        assert p.feed(h, l) == (None, None)
    assert p.structure.state is S.BEARISH
    assert p.structure.snapshot.active_swing_low.price == 8
    assert p.structure.snapshot.active_swing_high.price == 14
    return p


def test_pipeline_bearish_choch_then_caller_enters_revaluating():
    p = bullish_pipeline()
    protected = p.structure.protected_swing
    choch, bos = p.feed(9, 5, enter_revaluating=False)           # low 5 < protected low 6
    assert choch is not None and choch.direction is BEAR and choch.protected_swing is protected
    assert bos is None
    # The engine did not touch the structure; the caller has not acted yet.
    assert p.structure.state is S.BULLISH
    p.structure.enter_revaluating(choch.confirmed_at)
    assert p.structure.state is S.REVALUATING                    # not BEARISH: no new trend yet
    assert p.structure.transitions[-1].at == choch.confirmed_at


def test_pipeline_bullish_choch_from_a_bearish_structure():
    p = bearish_pipeline()
    protected = p.structure.protected_swing                      # active high 14
    choch, bos = p.feed(15, 8)                                   # high 15 > 14
    assert choch is not None and choch.direction is BULL and choch.protected_swing is protected
    assert bos is None and p.structure.state is S.REVALUATING


def test_pipeline_wick_only_break_is_a_choch_even_with_the_close_inside():
    p = bullish_pipeline()
    choch, bos = p.feed(11, 5)                                   # close 8, well above 6
    assert choch is not None and choch.break_price == 5 and bos is None


def test_pipeline_choch_candle_is_choch_only_never_bos():
    p = bullish_pipeline()
    choch, bos = p.feed(20, 5)                                   # close 12.5 > 12 (BOS condition)
    assert choch is not None and bos is None
    assert p.bos.history == () and len(p.choch.history) == 1


def test_pipeline_touching_the_protected_low_is_not_a_choch():
    p = bullish_pipeline()
    choch, bos = p.feed(14, 6)                                   # low == 6; close 10 <= 12 too
    assert choch is None and p.structure.state is S.BULLISH


def test_pipeline_no_further_choch_once_revaluating():
    p = bullish_pipeline()
    assert p.feed(9, 5)[0] is not None
    for high, low in ((9, 1), (30, -5), (12, 2)):
        assert p.feed(high, low) == (None, None)
    assert len(p.choch.history) == 1


def test_pipeline_choch_does_not_pick_the_new_direction():
    p = bullish_pipeline()
    p.feed(9, 5)
    assert p.structure.state is S.REVALUATING
    assert p.structure.permission.value == "NO_TRADE_PERMITTED"
    assert p.structure.protected_swing is None


# ---------------------------------------------------------------- randomised comparison
class Reference:
    """Plain restatement of the locked rules, evaluated from the snapshot."""

    @staticmethod
    def expected(snapshot, candle):
        if snapshot.state is S.BULLISH:
            low = snapshot.active_swing_low
            if candle.low < low.price:
                return (BEAR, low.sequence, candle.open_time, candle.low)
        if snapshot.state is S.BEARISH:
            high = snapshot.active_swing_high
            if candle.high > high.price:
                return (BULL, high.sequence, candle.open_time, candle.high)
        return None


def test_random_pipeline_matches_reference():
    total_choch = total_bos = 0
    for seed in range(60):
        rng = random.Random(seed)
        swing, structure = SwingEngine(M5), StructureEngine(M5)
        choch_engine, bos_engine = CHOCHEngine(M5), BOSEngine(M5)
        for i in range(100):
            high = rng.randint(5, 20)
            low = high - rng.randint(1, 6)
            mid = (high + low) / 2
            c = Candle(M5, T0 + i * M5.duration, mid, float(high), float(low), mid)
            structure.process_swings(swing.process_candle(c, c.close_time))
            snap = structure.snapshot
            got = choch_engine.process_candle(c, c.close_time, snap)
            bos = bos_engine.process_candle(c, c.close_time, snap)
            want = Reference.expected(snap, c)
            if want is None:
                assert got is None, (seed, i)
            else:
                assert got is not None, (seed, i)
                assert (got.direction, got.protected_swing.sequence,
                        got.candle_time, got.break_price) == want, (seed, i)
                assert bos is None, (seed, i)                    # Option A: CHOCH wins
                total_choch += 1
            total_bos += bos is not None
            if got is not None:
                structure.enter_revaluating(c.close_time)        # the caller's job
                assert structure.state is S.REVALUATING
    assert total_choch > 0 and total_bos > 0