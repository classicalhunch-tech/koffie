import random
from datetime import datetime, timedelta, timezone

import pytest

import koffie.strategy.engines.bos_engine as engine_module
from koffie.strategy.engines.bos_engine import BOSEngine
from koffie.strategy.engines.structure_engine import StructureEngine
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.structure import StructureSnapshot, StructureState
from koffie.strategy.models.swing import Swing, SwingType

T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = StructureState
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH


def sw(kind, price, seq, tf=M5, idx=0):
    ct = T0 + idx * tf.duration
    return Swing(kind, tf, float(price), ct, ct + 2 * tf.duration, seq)


def bull_snap(tf=M5):
    """Highs 10,12 / lows 4,6: BULLISH. Level = 12, protected low = 6."""
    return StructureSnapshot(tf, S.BULLISH,
                             sw(HIGH, 12, 2, tf, 2), sw(HIGH, 10, 0, tf, 0),
                             sw(LOW, 6, 3, tf, 3), sw(LOW, 4, 1, tf, 1))


def bear_snap(tf=M5):
    """Highs 12,10 / lows 6,4: BEARISH. Level = 4, protected high = 10."""
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
    e = BOSEngine(M5)
    assert e.timeframe is M5 and e.history == ()


def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            BOSEngine(bad)


def test_uses_existing_models():
    assert engine_module.BOS is BOS and engine_module.StructureSnapshot is StructureSnapshot
    assert engine_module.Candle is Candle


def test_state_is_read_only():
    e = BOSEngine(M5)
    for name in ("timeframe", "history"):
        with pytest.raises(AttributeError):
            setattr(e, name, None)


# ---------------------------------------------------------------- BOS detection
def test_bullish_bos_on_close_above_active_high():
    e, snap = BOSEngine(M5), bull_snap()
    candle = cd(0, 11, 13, 10, 12.5)
    bos = run(e, candle, snap)
    assert isinstance(bos, BOS)
    assert bos.timeframe is M5 and bos.direction is BULL
    assert bos.broken_swing is snap.active_swing_high
    assert bos.candle_time == candle.open_time and bos.close_price == 12.5
    assert bos.confirmed_at == candle.close_time
    assert e.history == (bos,)


def test_bearish_bos_on_close_below_active_low():
    e, snap = BOSEngine(M5), bear_snap()
    candle = cd(0, 5, 6, 3, 3.5)
    bos = run(e, candle, snap)
    assert bos.direction is BEAR and bos.broken_swing is snap.active_swing_low
    assert bos.close_price == 3.5 and e.history == (bos,)


def test_close_equal_to_the_level_is_not_a_bos():
    assert run(BOSEngine(M5), cd(0, 11, 13, 10, 12.0), bull_snap()) is None
    assert run(BOSEngine(M5), cd(0, 5, 6, 3, 4.0), bear_snap()) is None


def test_wick_beyond_the_level_with_close_inside_is_not_a_bos():
    assert run(BOSEngine(M5), cd(0, 11, 15, 10, 11.5), bull_snap()) is None
    assert run(BOSEngine(M5), cd(0, 5, 6, 1, 4.5), bear_snap()) is None


def test_no_bos_when_close_stays_between_the_levels():
    assert run(BOSEngine(M5), cd(0, 8, 9, 7, 8.5), bull_snap()) is None
    assert run(BOSEngine(M5), cd(0, 8, 9, 7, 8.5), bear_snap()) is None


def test_only_the_structures_own_direction_can_break():
    # BULLISH structure, close below the active low: that is CHOCH territory, not a BOS.
    assert run(BOSEngine(M5), cd(0, 7, 7.5, 4, 5), bull_snap()) is None
    # BEARISH structure, close above the active high.
    assert run(BOSEngine(M5), cd(0, 9, 13, 8.5, 12), bear_snap()) is None


def test_no_bos_in_undetermined_or_revaluating():
    for snap in (undetermined_snap(), revaluating_snap(), StructureSnapshot(M5)):
        e = BOSEngine(M5)
        assert run(e, cd(0, 11, 13, 10, 12.5), snap) is None
        assert run(e, cd(1, 5, 6, 3, 3.5), snap) is None
        assert e.history == ()


def test_every_qualifying_close_is_its_own_bos():
    e, snap = BOSEngine(M5), bull_snap()
    events = [run(e, cd(i, 12.5, 20, 12.2, 13 + i), snap) for i in range(3)]
    assert all(isinstance(b, BOS) for b in events)
    assert len({b.candle_time for b in events}) == 3
    assert all(b.broken_swing is snap.active_swing_high for b in events)
    assert e.history == tuple(events)


def test_engine_evaluates_against_the_snapshot_it_is_given():
    # Same candle: breaks the old level 12 but not a newer active high 13.
    older = bull_snap()
    newer = StructureSnapshot(M5, S.BULLISH, sw(HIGH, 13, 4, M5, 4), older.active_swing_high,
                              older.active_swing_low, older.prior_swing_low)
    candle = cd(0, 12, 12.9, 11.5, 12.5)
    assert run(BOSEngine(M5), candle, older) is not None
    assert run(BOSEngine(M5), candle, newer) is None


# ---------------------------------------------------------------- CHOCH-only rule (Option A)
def test_no_bos_when_the_candle_also_breaks_the_protected_low():
    # close 13 > 12 (BOS condition) but low 5 < protected low 6 (CHOCH condition).
    e = BOSEngine(M5)
    assert run(e, cd(0, 12, 14, 5, 13), bull_snap()) is None
    assert e.history == ()


def test_no_bos_when_the_candle_also_breaks_the_protected_high():
    # close 3 < 4 (BOS condition) but high 11 > protected high 10 (CHOCH condition).
    e = BOSEngine(M5)
    assert run(e, cd(0, 5, 11, 2, 3), bear_snap()) is None
    assert e.history == ()


def test_touching_the_protected_swing_exactly_does_not_suppress_the_bos():
    assert run(BOSEngine(M5), cd(0, 12, 14, 6, 13), bull_snap()) is not None      # low == 6
    assert run(BOSEngine(M5), cd(0, 5, 10, 2, 3), bear_snap()) is not None        # high == 10


def test_suppression_does_not_affect_later_candles():
    e, snap = BOSEngine(M5), bull_snap()
    assert run(e, cd(0, 12, 14, 5, 13), snap) is None
    assert run(e, cd(1, 12.5, 14, 12.2, 13), snap) is not None


# ---------------------------------------------------------------- validation
def test_rejects_wrong_argument_types():
    class Lookalike:
        timeframe, high, low, open, close = M5, 14.0, 10.0, 11.0, 13.0
        open_time = T0
        close_time = T0 + timedelta(minutes=5)

        def is_closed_at(self, now):
            return True

    e = BOSEngine(M5)
    good = cd(0, 11, 13, 10, 12.5)
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
            e = BOSEngine(etf)
            candle, snap = cd(0, 11, 13, 10, 12.5, other), bull_snap(other)
            if etf is other:
                run(e, candle, snap)
            else:
                with pytest.raises(ValueError):
                    run(e, candle, snap)
    e = BOSEngine(M5)
    with pytest.raises(ValueError):
        run(e, cd(0, 11, 13, 10, 12.5), bull_snap(M15))
    with pytest.raises(ValueError):
        run(e, cd(0, 11, 13, 10, 12.5, M15), bull_snap(M5))


def test_rejects_unfinished_candle():
    e = BOSEngine(M5)
    candle = cd(0, 11, 13, 10, 12.5)
    for offset in (timedelta(0), timedelta(minutes=4, seconds=59)):
        with pytest.raises(ValueError):
            e.process_candle(candle, candle.open_time + offset, bull_snap())
    assert e.history == ()
    assert e.process_candle(candle, candle.close_time, bull_snap()) is not None


def test_rejects_repeated_earlier_and_overlapping_candles():
    e, snap = BOSEngine(M5), bull_snap()
    first = cd(3, 11, 13, 10, 12.5)
    run(e, first, snap)
    for bad in (first, cd(2, 11, 13, 10, 12.5),
                Candle(M5, first.open_time + timedelta(minutes=1), 11, 13, 10, 12.5)):
        with pytest.raises(ValueError):
            e.process_candle(bad, bad.close_time + timedelta(hours=1), snap)
    assert len(e.history) == 1


def test_accepts_gaps_and_back_to_back_candles():
    e, snap = BOSEngine(M5), bull_snap()
    run(e, cd(0, 11, 13, 10, 12.5), snap)
    run(e, cd(1, 11, 13, 10, 12.5), snap)                  # opens exactly at previous close
    run(e, cd(500, 11, 13, 10, 12.5), snap)                # weekend-sized gap
    assert len(e.history) == 3


def test_rejects_a_snapshot_containing_a_future_swing():
    e = BOSEngine(M5)
    candle = cd(0, 11, 13, 10, 12.5)
    future = Swing(HIGH, M5, 12.0, candle.close_time, candle.close_time + 2 * M5.duration, 4)
    snap = StructureSnapshot(M5, S.BULLISH, future, sw(HIGH, 10, 0, M5, 0),
                             sw(LOW, 6, 3, M5, 3), sw(LOW, 4, 1, M5, 1))
    with pytest.raises(ValueError):
        run(e, candle, snap)
    assert e.history == ()


def test_swing_confirmed_exactly_at_the_candle_close_is_allowed():
    e = BOSEngine(M5)
    candle = cd(0, 11, 13, 10, 12.5)
    same_close = Swing(HIGH, M5, 12.0, candle.open_time - M5.duration,
                       candle.close_time, 4)
    assert same_close.confirmed_at == candle.close_time
    snap = StructureSnapshot(M5, S.BULLISH, same_close, sw(HIGH, 10, 0, M5, 0),
                             sw(LOW, 6, 3, M5, 3), sw(LOW, 4, 1, M5, 1))
    assert run(e, candle, snap) is not None


def test_rejected_calls_change_nothing():
    e, snap = BOSEngine(M5), bull_snap()
    good = cd(0, 11, 13, 10, 12.5)
    with pytest.raises(ValueError):
        e.process_candle(good, good.open_time, snap)                # unfinished
    with pytest.raises(ValueError):
        run(e, good, bull_snap(M15))                                # wrong snapshot timeframe
    assert e.history == ()
    assert run(e, good, snap) is not None                           # same candle now accepted


def test_mixing_naive_and_aware_datetimes_raises_and_changes_nothing():
    e = BOSEngine(M5)
    aware_t = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    aware_candle = Candle(M5, aware_t, 11, 13, 10, 12.5)
    with pytest.raises(TypeError):
        run(e, aware_candle, bull_snap())                           # naive swings, aware candle
    assert e.history == ()
    run(e, cd(0, 11, 13, 10, 12.5), bull_snap())                    # still healthy


def test_history_is_append_only_and_a_tuple():
    e, snap = BOSEngine(M5), bull_snap()
    run(e, cd(0, 11, 13, 10, 12.5), snap)
    first = e.history
    assert isinstance(first, tuple)
    run(e, cd(1, 11, 13, 10, 12.5), snap)
    assert e.history[:1] == first and len(e.history) == 2


def test_every_timeframe_works():
    for tf in (M5, M15, H1):
        e = BOSEngine(tf)
        bos = run(e, cd(0, 11, 13, 10, 12.5, tf), bull_snap(tf))
        assert bos.timeframe is tf and bos.confirmed_at == cd(0, 11, 13, 10, 12.5, tf).close_time


def test_engine_never_changes_the_snapshot_and_has_no_other_logic():
    e, snap = BOSEngine(M5), bull_snap()
    before = snap
    run(e, cd(0, 11, 13, 10, 12.5), snap)
    assert snap == before and snap.state is S.BULLISH
    for attr in ("detect_choch", "enter_revaluating", "structure", "zone", "liquidity",
                 "entry", "sl", "tp", "pivot"):
        assert not hasattr(e, attr)
    for name in ("StructureEngine", "CHOCH", "SwingEngine"):
        assert not hasattr(engine_module, name)


# ---------------------------------------------------------------- pipeline with real engines
class Pipeline:
    """SwingEngine -> StructureEngine.process_swings -> BOSEngine, per candle."""

    def __init__(self, tf=M5):
        self.swing, self.structure, self.bos = SwingEngine(tf), StructureEngine(tf), BOSEngine(tf)
        self.tf, self.i = tf, 0
        self.events = []

    def candle(self, high, low, tf=None):
        tf = self.tf
        mid = (high + low) / 2
        c = Candle(tf, T0 + self.i * tf.duration, mid, float(high), float(low), mid)
        self.i += 1
        return c

    def feed(self, high, low):
        c = self.candle(high, low)
        swings = self.swing.process_candle(c, c.close_time)
        self.structure.process_swings(swings)
        event = self.bos.process_candle(c, c.close_time, self.structure.snapshot)
        self.events.append(event)
        return event


BULLISH_SETUP = [(8, 6), (10, 7), (9, 5), (8, 4), (11, 5), (12, 8), (11, 7), (10, 6), (11, 7)]


def bullish_pipeline():
    p = Pipeline()
    for h, l in BULLISH_SETUP:
        p.feed(h, l)
    assert p.structure.state is S.BULLISH
    assert p.structure.snapshot.active_swing_high.price == 12
    assert p.structure.snapshot.active_swing_low.price == 6
    assert p.bos.history == ()
    return p


def test_pipeline_bullish_bos_and_literal_repeat():
    p = bullish_pipeline()
    first = p.feed(14, 11.5)                     # close 12.75 > 12, low above protected low
    second = p.feed(15, 12.5)                    # close 13.75 > 12, same active high
    assert first.direction is BULL and second.direction is BULL
    assert first.broken_swing is second.broken_swing is p.structure.snapshot.active_swing_high
    assert len(p.bos.history) == 2


def test_pipeline_wick_only_break_is_not_a_bos():
    p = bullish_pipeline()
    assert p.feed(14, 8) is None                 # close 11 < 12 despite the high of 14
    assert p.bos.history == ()


def test_pipeline_choch_candle_emits_no_bos():
    p = bullish_pipeline()
    assert p.feed(20, 5) is None                 # close 12.5 > 12 but low 5 < protected low 6
    assert p.bos.history == ()


def test_pipeline_no_bos_after_the_structure_enters_revaluating():
    p = bullish_pipeline()
    p.structure.enter_revaluating(T0 + timedelta(days=1))
    assert p.feed(14, 11.5) is None
    assert p.bos.history == ()


class Reference:
    """Plain restatement of the locked rules, evaluated from the snapshot."""

    @staticmethod
    def expected(snapshot, candle):
        if snapshot.state is S.BULLISH:
            high, low = snapshot.active_swing_high, snapshot.active_swing_low
            if candle.close > high.price and not candle.low < low.price:
                return (BULL, high.sequence, candle.open_time)
        if snapshot.state is S.BEARISH:
            high, low = snapshot.active_swing_high, snapshot.active_swing_low
            if candle.close < low.price and not candle.high > high.price:
                return (BEAR, low.sequence, candle.open_time)
        return None


def test_random_pipeline_matches_reference():
    total_bos = 0
    for seed in range(60):
        rng = random.Random(seed)
        swing, structure, bos_engine = SwingEngine(M5), StructureEngine(M5), BOSEngine(M5)
        for i in range(100):
            high = rng.randint(5, 20)
            low = high - rng.randint(1, 6)
            mid = (high + low) / 2
            c = Candle(M5, T0 + i * M5.duration, mid, float(high), float(low), mid)
            structure.process_swings(swing.process_candle(c, c.close_time))
            snap = structure.snapshot
            got = bos_engine.process_candle(c, c.close_time, snap)
            want = Reference.expected(snap, c)
            if want is None:
                assert got is None, (seed, i)
            else:
                assert got is not None, (seed, i)
                assert (got.direction, got.broken_swing.sequence, got.candle_time) == want, (seed, i)
                total_bos += 1
            if snap.state in (S.BULLISH, S.BEARISH) and rng.random() < 0.1:
                structure.enter_revaluating(c.close_time)
    assert total_bos > 0
