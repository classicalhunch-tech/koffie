import dataclasses
import random
from datetime import datetime, timedelta, timezone

import pytest

import koffie.strategy.engines.structure_processor as module
from koffie.strategy.engines.bos_engine import BOSEngine
from koffie.strategy.engines.choch_engine import CHOCHEngine
from koffie.strategy.engines.structure_engine import StructureEngine
from koffie.strategy.engines.structure_processor import CandleResult, StructureProcessor
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.choch import CHOCH, CHOCHDirection
from koffie.strategy.models.structure import StructureState, TransitionReason

T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
S = StructureState
BEAR, BULL = CHOCHDirection.BEARISH, CHOCHDirection.BULLISH

# Highs 10,12 / lows 4,6 -> BULLISH after the last candle. Active high 12, protected low 6.
BULLISH_SETUP = [(8, 6), (10, 7), (9, 5), (8, 4), (11, 5), (12, 8), (11, 7), (10, 6), (11, 7)]
# Mirror image around 10 -> BEARISH. Active low 8, protected high 14.
BEARISH_SETUP = [(20 - l, 20 - h) for h, l in BULLISH_SETUP]


class Feeder:
    """Builds closed candles one after another and feeds them to a StructureProcessor."""

    def __init__(self, tf=M5):
        self.p = StructureProcessor(tf)
        self.tf, self.i = tf, 0

    def candle(self, high, low):
        mid = (high + low) / 2
        c = Candle(self.tf, T0 + self.i * self.tf.duration, mid, float(high), float(low), mid)
        self.i += 1
        return c

    def feed(self, high, low):
        c = self.candle(high, low)
        return c, self.p.process_candle(c, c.close_time)


def bullish():
    f = Feeder()
    for h, l in BULLISH_SETUP:
        _, r = f.feed(h, l)
        assert r.choch is None and r.bos is None and r.revaluating is None
    assert f.p.structure.state is S.BULLISH
    assert f.p.structure.protected_swing.price == 6
    return f


def bearish():
    f = Feeder()
    for h, l in BEARISH_SETUP:
        _, r = f.feed(h, l)
        assert r.choch is None and r.bos is None and r.revaluating is None
    assert f.p.structure.state is S.BEARISH
    assert f.p.structure.protected_swing.price == 14
    return f


def engine_state(p):
    """Everything observable about all four engines, for before/after comparisons."""
    return (p.swing_engine.history, p.structure.snapshot, p.structure.transitions,
            p.choch_engine.history, p.bos_engine.history)


# ---------------------------------------------------------------- construction and shape
def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            StructureProcessor(bad)


def test_initial_state_and_engines_use_the_existing_classes():
    p = StructureProcessor(M15)
    assert p.timeframe is M15
    assert type(p.swing_engine) is SwingEngine and type(p.structure) is StructureEngine
    assert type(p.choch_engine) is CHOCHEngine and type(p.bos_engine) is BOSEngine
    for e in (p.swing_engine, p.structure, p.choch_engine, p.bos_engine):
        assert e.timeframe is M15
    assert p.structure.state is S.UNDETERMINED


def test_properties_are_read_only():
    p = StructureProcessor(M5)
    for name in ("timeframe", "swing_engine", "structure", "choch_engine", "bos_engine"):
        with pytest.raises(AttributeError):
            setattr(p, name, None)


def test_candle_result_keeps_events_separate_and_is_immutable():
    assert [f.name for f in dataclasses.fields(CandleResult)] == [
        "swings", "structure_transitions", "choch", "bos", "revaluating", "snapshot"]
    f = bullish()
    _, r = f.feed(14, 11.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.choch = None
    for name in ("signal", "verdict", "event", "trade", "direction"):
        assert not hasattr(r, name)                       # no merged BOS/CHOCH signal


# ---------------------------------------------------------------- 1. bearish CHOCH from BULLISH
def test_bearish_choch_from_bullish_enters_revaluating_at_the_candle_close_time():
    f = bullish()
    protected = f.p.structure.protected_swing
    c, r = f.feed(9, 5)                                    # low 5 < protected low 6
    assert isinstance(r.choch, CHOCH) and r.choch.direction is BEAR
    assert r.choch.protected_swing is protected and r.choch.candle_time == c.open_time
    assert r.choch.break_price == 5 and r.choch.confirmed_at == c.close_time
    t = r.revaluating
    assert (t.from_state, t.to_state, t.reason) == (S.BULLISH, S.REVALUATING, TransitionReason.CHOCH)
    assert t.at == c.close_time == r.choch.confirmed_at and t.trigger_swing_sequence is None
    assert f.p.structure.state is S.REVALUATING and f.p.structure.transitions[-1] == t
    assert r.snapshot.state is S.REVALUATING and r.bos is None
    assert f.p.choch_engine.history == (r.choch,)


# ---------------------------------------------------------------- 2. bullish CHOCH from BEARISH
def test_bullish_choch_from_bearish_enters_revaluating_at_the_candle_close_time():
    f = bearish()
    protected = f.p.structure.protected_swing
    c, r = f.feed(15, 9)                                   # high 15 > protected high 14
    assert isinstance(r.choch, CHOCH) and r.choch.direction is BULL
    assert r.choch.protected_swing is protected and r.choch.candle_time == c.open_time
    assert r.choch.break_price == 15 and r.choch.confirmed_at == c.close_time
    t = r.revaluating
    assert (t.from_state, t.to_state, t.reason) == (S.BEARISH, S.REVALUATING, TransitionReason.CHOCH)
    assert t.at == c.close_time and f.p.structure.state is S.REVALUATING
    assert r.bos is None


def test_wick_break_with_the_close_back_inside_still_enters_revaluating():
    f = bullish()
    c, r = f.feed(11, 5)                                   # close 8 is above the protected low
    assert r.choch is not None and r.revaluating.at == c.close_time
    f = bearish()
    c, r = f.feed(15, 9)                                   # close 12 is below the protected high
    assert r.choch is not None and r.revaluating.at == c.close_time


def test_the_transition_time_is_the_candle_close_not_the_open_and_not_now():
    f = bullish()
    c = f.candle(9, 5)
    later = c.close_time + timedelta(hours=3)
    r = f.p.process_candle(c, later)
    assert r.revaluating.at == c.close_time
    assert r.revaluating.at != c.open_time and r.revaluating.at != later


def test_every_timeframe_enters_revaluating_at_its_own_close_time():
    for tf in (M5, M15, H1):
        f = Feeder(tf)
        for h, l in BULLISH_SETUP:
            f.feed(h, l)
        c, r = f.feed(9, 5)
        assert r.choch.timeframe is tf and r.revaluating.timeframe is tf
        assert r.revaluating.at == c.close_time == c.open_time + tf.duration


# ---------------------------------------------------------------- 3. no CHOCH, no REVALUATING
def test_candle_without_a_choch_does_not_enter_revaluating():
    f = bullish()
    before = f.p.structure.transitions
    for high, low in ((13, 8), (11, 7), (12, 6.5)):
        _, r = f.feed(high, low)
        assert r.choch is None and r.revaluating is None
        assert f.p.structure.state is S.BULLISH
    assert f.p.structure.transitions[:len(before)] == before
    assert all(t.reason is not TransitionReason.CHOCH for t in f.p.structure.transitions)


def test_touching_the_protected_swing_does_not_enter_revaluating():
    f = bullish()
    _, r = f.feed(9, 6)                                    # low == protected low 6
    assert r.choch is None and r.revaluating is None and f.p.structure.state is S.BULLISH
    f = bearish()
    _, r = f.feed(14, 9)                                   # high == protected high 14
    assert r.choch is None and r.revaluating is None and f.p.structure.state is S.BEARISH


def test_a_bos_candle_is_a_bos_only_and_does_not_enter_revaluating():
    f = bullish()
    _, r = f.feed(14, 11.5)                                # close 12.75 > 12, low above 6
    assert isinstance(r.bos, BOS) and r.bos.direction is BOSDirection.BULLISH
    assert r.choch is None and r.revaluating is None
    assert f.p.structure.state is S.BULLISH


def test_the_other_side_breaking_does_not_enter_revaluating():
    f = bullish()
    _, r = f.feed(40, 25)                                  # far above, nowhere near the protected low
    assert r.choch is None and r.revaluating is None
    f = bearish()
    _, r = f.feed(-5, -20)
    assert r.choch is None and r.revaluating is None


# ---------------------------------------------------------------- 4. UNDETERMINED
def test_undetermined_produces_no_choch_and_no_revaluating():
    f = Feeder()
    for h, l in BULLISH_SETUP[:8]:                         # two highs, one low: still UNDETERMINED
        _, r = f.feed(h, l)
        assert r.choch is None and r.revaluating is None
        assert r.snapshot.state is S.UNDETERMINED
    for high, low in ((30, -5), (31, -6)):                 # breaks every level in sight
        _, r = f.feed(high, low)
        assert r.snapshot.state is S.UNDETERMINED
        assert r.choch is None and r.revaluating is None and r.bos is None
    assert f.p.structure.transitions == ()
    assert f.p.choch_engine.history == () and f.p.bos_engine.history == ()


def test_a_brand_new_processor_produces_no_choch():
    f = Feeder()
    _, r = f.feed(10, 5)
    assert r.choch is None and r.revaluating is None and r.swings == ()


# ---------------------------------------------------------------- 5. REVALUATING
def test_revaluating_does_not_produce_another_choch():
    f = bullish()
    _, first = f.feed(9, 5)
    assert first.choch is not None and f.p.structure.state is S.REVALUATING
    count = len(f.p.structure.transitions)
    for high, low in ((9, 4), (9, 3), (9, 2)):             # lower lows, no new swings form
        _, r = f.feed(high, low)
        assert f.p.structure.state is S.REVALUATING
        assert r.choch is None and r.revaluating is None and r.bos is None
    assert len(f.p.structure.transitions) == count
    assert f.p.choch_engine.history == (first.choch,)


def test_choch_does_not_establish_the_new_trend():
    f = bullish()
    _, r = f.feed(9, 5)
    assert r.snapshot.state is S.REVALUATING               # not BEARISH
    assert f.p.structure.permission.value == "NO_TRADE_PERMITTED"
    assert f.p.structure.protected_swing is None
    f = bearish()
    _, r = f.feed(15, 9)
    assert r.snapshot.state is S.REVALUATING               # not BULLISH


def test_the_new_direction_comes_only_from_later_swings():
    f = bullish()
    f.feed(9, 5)
    directional = []
    for high, low in ((9, 4), (12, 3), (8, 6), (13, 7), (6, 3), (5, 1)):
        _, r = f.feed(high, low)
        directional += [t for t in r.structure_transitions
                        if t.reason is TransitionReason.DIRECTIONAL_PAIR]
    for t in f.p.structure.transitions:
        if t.reason is TransitionReason.DIRECTIONAL_PAIR and t.from_state is S.REVALUATING:
            assert t.trigger_swing_sequence is not None    # a swing decided it, never the caller
    assert all(t.trigger_swing_sequence is not None for t in directional)


# ---------------------------------------------------------------- 6. only after the candle has closed
def test_an_unfinished_candle_is_rejected_and_nothing_changes():
    f = bullish()
    c = f.candle(9, 5)
    before = engine_state(f.p)
    for at in (c.open_time, c.open_time + timedelta(minutes=4, seconds=59)):
        with pytest.raises(ValueError):
            f.p.process_candle(c, at)
        assert engine_state(f.p) == before
        assert f.p.structure.state is S.BULLISH
    r = f.p.process_candle(c, c.close_time)                # now it is closed
    assert r.revaluating.at == c.close_time and f.p.structure.state is S.REVALUATING


def test_the_transition_happens_at_the_first_moment_the_candle_is_closed():
    f = bullish()
    c = f.candle(9, 5)
    r = f.p.process_candle(c, c.close_time)                # exactly at close_time
    assert r.revaluating is not None and r.revaluating.at == c.close_time


def test_rejected_input_changes_nothing_in_any_engine():
    f = bullish()
    good = f.candle(9, 5)
    before = engine_state(f.p)
    wrong_tf = Candle(M15, good.open_time, 7, 9, 5, 7)
    aware = Candle(M5, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), 7, 9, 5, 7)
    for args, exc in (((None, good.close_time), TypeError), (("c", good.close_time), TypeError),
                      ((good, None), TypeError), ((good, "now"), TypeError),
                      ((wrong_tf, wrong_tf.close_time), ValueError),
                      ((aware, aware.close_time), TypeError)):
        with pytest.raises(exc):
            f.p.process_candle(*args)
        assert engine_state(f.p) == before
    old = Candle(M5, T0, 8, 9, 7, 8)                       # overlaps candles already processed
    with pytest.raises(ValueError):
        f.p.process_candle(old, old.close_time)
    assert engine_state(f.p) == before
    assert f.p.process_candle(good, good.close_time).choch is not None   # still healthy


# ---------------------------------------------------------------- 7. CHOCHEngine independence
def test_enter_revaluating_is_called_once_by_the_caller_with_the_candle_close_time():
    f = bullish()
    calls = []
    real = f.p.structure.enter_revaluating
    f.p.structure.enter_revaluating = lambda at: (calls.append(at), real(at))[1]
    c1, _ = f.feed(14, 11.5)                               # BOS candle, no CHOCH
    c2, _ = f.feed(12, 10)                                 # nothing
    assert calls == []
    c3, r = f.feed(9, 5)                                   # CHOCH
    assert calls == [c3.close_time] and r.revaluating.at == c3.close_time


def test_if_the_caller_does_not_call_enter_revaluating_the_structure_never_moves():
    f = bullish()
    calls = []
    f.p.structure.enter_revaluating = lambda at: calls.append(at)    # caller step disabled
    c, r = f.feed(9, 5)
    assert r.choch is not None and calls == [c.close_time]
    assert f.p.structure.state is S.BULLISH                # CHOCHEngine itself moved nothing
    assert all(t.reason is not TransitionReason.CHOCH for t in f.p.structure.transitions)


def test_choch_engine_holds_no_structure_engine_and_the_processor_owns_the_call():
    f = bullish()
    assert not any(isinstance(v, StructureEngine) for v in vars(f.p.choch_engine).values())
    assert not hasattr(f.p.choch_engine, "enter_revaluating")
    # A stand-alone CHOCHEngine given a snapshot returns the event and changes nothing.
    standalone = CHOCHEngine(M5)
    snap = f.p.structure.snapshot
    c = f.candle(9, 5)
    event = standalone.process_candle(c, c.close_time, snap)
    assert event is not None and f.p.structure.snapshot == snap
    assert f.p.structure.state is S.BULLISH and f.p.structure.transitions[-1].reason \
        is TransitionReason.DIRECTIONAL_PAIR


# ---------------------------------------------------------------- 8. BOS behaviour unchanged
def test_bullish_structure_candle_meeting_both_conditions_gives_choch_only():
    f = bullish()
    _, r = f.feed(20, 5)                                   # close 12.5 > 12 (BOS) and low 5 < 6 (CHOCH)
    assert r.choch is not None and r.bos is None and r.revaluating is not None
    assert f.p.bos_engine.history == ()                    # Option A is applied inside BOSEngine


def test_bearish_structure_candle_meeting_both_conditions_gives_choch_only():
    f = bearish()
    c0 = f.candle(15, 1)
    c = Candle(M5, c0.open_time, 10, 15, 1, 5)             # close 5 < active low 8 (BOS), high 15 > 14 (CHOCH)
    r = f.p.process_candle(c, c.close_time)
    assert r.choch is not None and r.choch.direction is BULL
    assert r.bos is None and r.revaluating is not None
    assert f.p.bos_engine.history == ()


def test_bos_is_unchanged_for_repeated_closes_beyond_the_level():
    f = bullish()
    first = f.feed(14, 11.5)[1].bos
    second = f.feed(15, 12.5)[1].bos
    assert first.direction is second.direction is BOSDirection.BULLISH
    assert first.broken_swing is second.broken_swing is f.p.structure.snapshot.active_swing_high
    assert len(f.p.bos_engine.history) == 2
    assert f.p.structure.state is S.BULLISH


def test_no_bos_after_the_structure_has_entered_revaluating():
    f = bullish()
    f.feed(9, 5)
    _, r = f.feed(9, 4)
    assert r.bos is None and f.p.bos_engine.history == ()


class Reference:
    """Hand-wired restatement of the locked rules, driving separate engines."""

    def __init__(self, tf=M5):
        self.swing, self.structure = SwingEngine(tf), StructureEngine(tf)

    def step(self, c):
        self.structure.process_swings(self.swing.process_candle(c, c.close_time))
        snap = self.structure.snapshot
        choch = bos = None
        if snap.state is S.BULLISH:
            hi, lo = snap.active_swing_high, snap.active_swing_low
            if c.low < lo.price:
                choch = (BEAR, lo.sequence, c.open_time, c.low)
            elif c.close > hi.price:
                bos = (BOSDirection.BULLISH, hi.sequence, c.open_time)
        elif snap.state is S.BEARISH:
            hi, lo = snap.active_swing_high, snap.active_swing_low
            if c.high > hi.price:
                choch = (BULL, hi.sequence, c.open_time, c.high)
            elif c.close < lo.price:
                bos = (BOSDirection.BEARISH, lo.sequence, c.open_time)
        if choch is not None:
            self.structure.enter_revaluating(c.close_time)
        return choch, bos


def test_random_flow_matches_the_reference_and_existing_bos_behaviour():
    total_choch = total_bos = 0
    seeds_with_two_choch = 0
    for seed in range(80):
        rng = random.Random(seed)
        p, ref = StructureProcessor(M5), Reference()
        chochs = 0
        for i in range(120):
            high = rng.randint(5, 20)
            low = high - rng.randint(1, 6)
            mid = (high + low) / 2
            c = Candle(M5, T0 + i * M5.duration, mid, float(high), float(low), mid)
            r = p.process_candle(c, c.close_time)
            want_choch, want_bos = ref.step(c)
            got_choch = None if r.choch is None else (
                r.choch.direction, r.choch.protected_swing.sequence,
                r.choch.candle_time, r.choch.break_price)
            got_bos = None if r.bos is None else (
                r.bos.direction, r.bos.broken_swing.sequence, r.bos.candle_time)
            assert got_choch == want_choch, (seed, i)
            assert got_bos == want_bos, (seed, i)
            assert not (r.choch is not None and r.bos is not None), (seed, i)
            assert (r.revaluating is not None) == (r.choch is not None), (seed, i)
            if r.revaluating is not None:
                assert r.revaluating.at == c.close_time and r.snapshot.state is S.REVALUATING
            assert p.structure.snapshot == ref.structure.snapshot, (seed, i)
            chochs += r.choch is not None
            total_bos += r.bos is not None
        assert p.structure.transitions == ref.structure.transitions, seed
        total_choch += chochs
        seeds_with_two_choch += chochs >= 2
    assert total_choch > 0 and total_bos > 0
    assert seeds_with_two_choch > 0          # after restoration a later CHOCH is possible again


# ---------------------------------------------------------------- scope
def test_module_wires_the_existing_engines_and_adds_no_strategy_logic():
    for attr in ("zone", "liquidity", "entry", "sl", "tp", "pivot", "signal", "trade"):
        assert not hasattr(StructureProcessor(M5), attr)
    for name in ("Zone", "Liquidity", "H4", "MT5"):
        assert not hasattr(module, name)
    assert "H4" not in [t.name for t in Timeframe]         # H4 support is not part of this step