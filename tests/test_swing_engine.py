import random
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.swing_engine as engine_module
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.swing import Swing, SwingType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
 
def c(i, high, low, tf=M5, t0=T0):
    mid = (high + low) / 2
    return Candle(tf, t0 + i * tf.duration, mid, high, low, mid)
 
 
def feed(eng, candle):
    return eng.process_candle(candle, now=candle.close_time)
 
 
def run(pairs, tf=M5):
    eng = SwingEngine(tf)
    outputs = [feed(eng, c(i, h, l, tf)) for i, (h, l) in enumerate(pairs)]
    return eng, outputs
 
 
def snapshot(eng):
    return (eng.history, eng.active_swing_high, eng.prior_swing_high,
            eng.active_swing_low, eng.prior_swing_low)
 
 
def reference(candles):
    """Non-incremental definition of the rules, used as an oracle."""
    out = []
    for n in range(1, len(candles) - 1):
        p, cd, nx = candles[n - 1], candles[n], candles[n + 1]
        if cd.high > p.high and cd.high > nx.high:
            out.append((HIGH, cd.high, cd.open_time, nx.close_time))
        if cd.low < p.low and cd.low < nx.low:
            out.append((LOW, cd.low, cd.open_time, nx.close_time))
    return out
 
 
def keys(history):
    return [(s.swing_type, s.price, s.candle_time, s.confirmed_at) for s in history]
 
 
# ---------------------------------------------------------------- construction
def test_initial_state_is_empty():
    eng = SwingEngine(M5)
    assert eng.timeframe is M5
    assert eng.history == ()
    assert eng.active_swing_high is None and eng.prior_swing_high is None
    assert eng.active_swing_low is None and eng.prior_swing_low is None
 
 
def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            SwingEngine(bad)
 
 
def test_uses_existing_models_not_replacements():
    assert engine_module.Swing is Swing
    assert engine_module.Candle is Candle
    assert Swing.__module__ == "koffie.strategy.models.swing"
    assert Candle.__module__ == "koffie.strategy.models.candle"
    _, out = run([(10, 5), (12, 6), (11, 7)])
    assert type(out[2][0]) is Swing
 
 
# ---------------------------------------------------------------- swing highs
def test_swing_high_detected_with_correct_fields():
    eng = SwingEngine(M5)
    a, b, d = c(0, 10, 5), c(1, 12, 6), c(2, 11, 7)
    assert feed(eng, a) == []
    assert feed(eng, b) == []
    new = feed(eng, d)
    assert len(new) == 1
    s = new[0]
    assert s.swing_type is HIGH
    assert s.price == 12
    assert s.timeframe is M5
    assert s.candle_time == b.open_time
    assert s.confirmed_at == d.close_time
    assert s.sequence == 0
 
 
def test_swing_high_not_known_until_next_candle_closes():
    eng = SwingEngine(M5)
    feed(eng, c(0, 10, 5))
    feed(eng, c(1, 12, 6))          # candidate present, N+1 not yet processed
    assert eng.history == ()
    assert eng.active_swing_high is None
 
 
def test_swing_high_uses_candle_high_not_close():
    eng, out = run([(10, 5), (12, 6), (11, 7)])
    assert out[2][0].price == 12
 
 
# ---------------------------------------------------------------- swing lows
def test_swing_low_detected_with_correct_fields():
    eng = SwingEngine(M5)
    a, b, d = c(0, 10, 8), c(1, 9, 5), c(2, 11, 7)
    feed(eng, a)
    feed(eng, b)
    new = feed(eng, d)
    assert len(new) == 1
    s = new[0]
    assert s.swing_type is LOW and s.price == 5
    assert s.candle_time == b.open_time and s.confirmed_at == d.close_time
 
 
# ---------------------------------------------------------------- equality
def test_equal_highs_never_qualify():
    for series in (
        [(10, 1), (12, 2), (12, 3), (10, 4)],    # 10,12,12,10 from the spec
        [(12, 1), (12, 2), (10, 3)],             # equals previous
        [(10, 1), (12, 2), (12, 3)],             # equals next
        [(12, 1), (12, 2), (12, 3)],             # plateau
    ):
        eng, _ = run(series)
        assert [s for s in eng.history if s.swing_type is HIGH] == []
 
 
def test_equal_lows_never_qualify():
    for series in (
        [(20, 10), (20, 8), (20, 8), (20, 10)],
        [(20, 8), (20, 8), (20, 10)],
        [(20, 10), (20, 8), (20, 8)],
        [(20, 8), (20, 8), (20, 8)],
    ):
        eng, _ = run(series)
        assert [s for s in eng.history if s.swing_type is LOW] == []
 
 
def test_monotonic_series_has_no_swings():
    eng, _ = run([(10 + i, 1 + i) for i in range(10)])
    assert eng.history == ()
    eng, _ = run([(20 - i, 10 - i) for i in range(10)])
    assert eng.history == ()
 
 
def test_first_two_candles_never_confirm_anything():
    eng = SwingEngine(M5)
    assert feed(eng, c(0, 10, 5)) == []
    assert feed(eng, c(1, 20, 1)) == []
 
 
# ---------------------------------------------------------------- outside bar
def test_outside_bar_emits_both_swings_high_first():
    eng, out = run([(10, 5), (15, 2), (11, 6)])
    new = out[2]
    assert [s.swing_type for s in new] == [HIGH, LOW]
    assert [s.price for s in new] == [15, 2]
    assert [s.sequence for s in new] == [0, 1]
    assert new[0].candle_time == new[1].candle_time
    assert new[0].confirmed_at == new[1].confirmed_at
    assert eng.history == tuple(new)
    assert eng.active_swing_high is new[0] and eng.active_swing_low is new[1]
 
 
def test_outside_bar_after_existing_swings_updates_both_sides():
    eng, _ = run([(10, 4), (14, 5), (11, 6), (12, 3), (13, 7),   # highs 14, 13; low 3
                  (12, 5), (20, 1), (13, 6)])                    # outside bar
    assert [s.price for s in eng.history if s.swing_type is HIGH] == [14, 13, 20]
    assert [s.price for s in eng.history if s.swing_type is LOW] == [3, 1]
    assert eng.active_swing_high.price == 20 and eng.prior_swing_high.price == 13
    assert eng.active_swing_low.price == 1 and eng.prior_swing_low.price == 3
    assert eng.history[-2].swing_type is HIGH and eng.history[-1].swing_type is LOW
 
 
# ---------------------------------------------------------------- active / prior
def test_latest_high_is_latest_confirmed_not_historical_extreme():
    eng, _ = run([(10, 1), (15, 2), (12, 3), (14, 4), (11, 5)])
    assert eng.active_swing_high.price == 14      # later and LOWER than 15
    assert eng.prior_swing_high.price == 15
 
 
def test_latest_low_is_latest_confirmed_not_historical_extreme():
    eng, _ = run([(20, 10), (15, 5), (18, 9), (16, 7), (19, 8)])
    assert eng.active_swing_low.price == 7        # later and HIGHER than 5
    assert eng.prior_swing_low.price == 5
 
 
def test_prior_is_none_until_second_swing_of_that_type():
    eng, _ = run([(10, 1), (15, 2), (12, 3)])
    assert eng.active_swing_high.price == 15 and eng.prior_swing_high is None
    assert eng.active_swing_low is None and eng.prior_swing_low is None
 
 
def test_only_the_latest_two_are_tracked():
    eng, _ = run([(10, 1), (15, 2), (12, 3), (14, 4), (11, 5), (13, 4), (9, 6)])
    highs = [s for s in eng.history if s.swing_type is HIGH]
    assert [s.price for s in highs] == [15, 14, 13]
    assert eng.active_swing_high is highs[-1]
    assert eng.prior_swing_high is highs[-2]
 
 
def test_high_and_low_tracking_are_independent():
    eng, _ = run([(10, 6), (15, 7), (12, 3), (14, 6), (11, 5)])
    assert eng.active_swing_high.price == 14 and eng.prior_swing_high.price == 15
    assert eng.active_swing_low.price == 3 and eng.prior_swing_low is None
 
 
def test_active_and_prior_are_read_only():
    eng, _ = run([(10, 1), (15, 2), (12, 3)])
    for name in ("active_swing_high", "prior_swing_high", "active_swing_low",
                 "prior_swing_low", "history", "timeframe"):
        with pytest.raises(AttributeError):
            setattr(eng, name, None)
 
 
# ---------------------------------------------------------------- history
def test_history_is_append_only_ordered_and_sequenced():
    eng = SwingEngine(M5)
    data = [(10, 1), (15, 2), (12, 3), (14, 4), (11, 5), (13, 2), (12, 6), (16, 3), (9, 4)]
    previous = ()
    for i, (h, l) in enumerate(data):
        feed(eng, c(i, h, l))
        now_hist = eng.history
        assert now_hist[: len(previous)] == previous       # nothing rewritten
        assert len(now_hist) >= len(previous)
        previous = now_hist
    assert [s.sequence for s in eng.history] == list(range(len(eng.history)))
    confirmations = [s.confirmed_at for s in eng.history]
    assert confirmations == sorted(confirmations)
 
 
def test_history_is_an_immutable_snapshot_and_returned_list_is_a_copy():
    eng, out = run([(10, 1), (15, 2), (12, 3)])
    assert isinstance(eng.history, tuple)
    out[2].clear()
    assert len(eng.history) == 1
    snap = eng.history
    feed(eng, c(3, 14, 4))
    feed(eng, c(4, 11, 5))
    assert len(snap) == 1 and len(eng.history) == 2
 
 
def test_swings_are_immutable_and_hashable():
    eng, _ = run([(10, 1), (15, 2), (12, 3)])
    with pytest.raises(Exception):
        eng.history[0].price = 1
    assert len({eng.history[0], eng.history[0]}) == 1
 
 
# ---------------------------------------------------------------- closed candles
def test_rejects_unfinished_candle():
    eng = SwingEngine(M5)
    cd = c(0, 10, 5)
    for offset in (timedelta(0), timedelta(minutes=4, seconds=59)):
        with pytest.raises(ValueError):
            eng.process_candle(cd, now=cd.open_time + offset)
 
 
def test_accepts_candle_exactly_at_close_and_later():
    eng = SwingEngine(M5)
    cd = c(0, 10, 5)
    eng.process_candle(cd, now=cd.close_time)
    eng.process_candle(c(1, 10, 5), now=c(1, 10, 5).close_time + timedelta(hours=1))
 
 
def test_unfinished_confirmation_candle_does_not_confirm_a_swing():
    eng = SwingEngine(M5)
    feed(eng, c(0, 10, 5))
    feed(eng, c(1, 12, 6))
    n1 = c(2, 11, 7)
    with pytest.raises(ValueError):
        eng.process_candle(n1, now=n1.open_time + timedelta(minutes=1))
    assert eng.history == ()
    assert len(feed(eng, n1)) == 1               # once closed, it confirms
 
 
def test_unfinished_check_applies_to_every_timeframe():
    for tf in (M5, M15, H1):
        eng = SwingEngine(tf)
        cd = c(0, 10, 5, tf)
        with pytest.raises(ValueError):
            eng.process_candle(cd, now=cd.close_time - timedelta(seconds=1))
 
 
def test_now_is_required():
    eng = SwingEngine(M5)
    with pytest.raises(TypeError):
        eng.process_candle(c(0, 10, 5))
 
 
# ---------------------------------------------------------------- timeframe
def test_rejects_wrong_timeframe_for_every_pairing():
    for engine_tf in (M5, M15, H1):
        for candle_tf in (M5, M15, H1):
            eng = SwingEngine(engine_tf)
            cd = c(0, 10, 5, candle_tf)
            if engine_tf is candle_tf:
                feed(eng, cd)
            else:
                with pytest.raises(ValueError):
                    feed(eng, cd)
 
 
def test_swings_carry_the_engine_timeframe():
    for tf in (M5, M15, H1):
        _, out = run([(10, 5), (12, 6), (11, 7)], tf)
        assert out[2][0].timeframe is tf
        assert out[2][0].confirmed_at == c(2, 11, 7, tf).close_time
 
 
# ---------------------------------------------------------------- chronology
def test_rejects_duplicate_and_earlier_candles():
    eng = SwingEngine(M5)
    feed(eng, c(1, 10, 5))
    for i in (1, 0):
        with pytest.raises(ValueError):
            feed(eng, c(i, 10, 5))
 
 
def test_rejects_overlapping_candle():
    eng = SwingEngine(M5)
    feed(eng, c(0, 10, 5))
    overlapping = Candle(M5, T0 + timedelta(minutes=1), 7, 8, 6, 7)   # opens mid-candle
    with pytest.raises(ValueError):
        feed(eng, overlapping)
 
 
def test_accepts_candle_opening_exactly_at_previous_close():
    eng = SwingEngine(M5)
    feed(eng, c(0, 10, 5))
    feed(eng, c(1, 10, 5))
 
 
def test_accepts_gaps_between_candles():
    eng = SwingEngine(M5)
    feed(eng, c(0, 10, 5))
    feed(eng, c(1, 12, 6))
    weekend = c(1, 11, 7, t0=T0 + timedelta(days=2))
    new = feed(eng, weekend)
    assert len(new) == 1
    assert new[0].confirmed_at == weekend.close_time
 
 
def test_rejects_non_candle_objects_including_lookalikes():
    class Lookalike:
        timeframe = M5
        open_time = T0
        high, low, open, close = 10.0, 5.0, 6.0, 7.0
        close_time = T0 + timedelta(minutes=5)
 
        def is_closed_at(self, now):
            return True
 
    eng = SwingEngine(M5)
    for bad in (Lookalike(), None, (10, 5), {"high": 10, "low": 5}, "candle", 5):
        with pytest.raises(TypeError):
            eng.process_candle(bad, now=T0 + timedelta(hours=1))
 
 
def test_rejects_non_datetime_now():
    eng = SwingEngine(M5)
    for bad in (None, "2026-01-01", 123, T0.date()):
        with pytest.raises(TypeError):
            eng.process_candle(c(0, 10, 5), now=bad)
 
 
def test_mixing_naive_and_aware_datetimes_raises_and_changes_nothing():
    eng = SwingEngine(M5)
    feed(eng, c(0, 10, 5))
    aware = datetime(2026, 1, 1, 10, 5, tzinfo=timezone.utc)
    aware_candle = Candle(M5, aware, 6, 7, 5, 6)
    with pytest.raises(TypeError):
        feed(eng, aware_candle)                       # aware vs naive history
    naive_next = c(1, 10, 5)
    with pytest.raises(TypeError):
        eng.process_candle(naive_next, now=aware + timedelta(hours=1))   # aware now
    feed(eng, c(1, 12, 6))                            # engine still healthy
 
 
def test_timezone_aware_candles_work():
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    eng = SwingEngine(M5)
    out = [feed(eng, c(i, h, l, t0=t0)) for i, (h, l) in enumerate([(10, 5), (12, 6), (11, 7)])]
    assert out[2][0].confirmed_at == t0 + timedelta(minutes=15)
 
 
# ---------------------------------------------------------------- atomicity
def test_rejected_input_leaves_state_untouched_and_engine_behaves_as_if_never_called():
    data = [(10, 1), (15, 2), (12, 3), (14, 4), (11, 5), (13, 2), (12, 6)]
    control, _ = run(data)
 
    eng = SwingEngine(M5)
    for i, (h, l) in enumerate(data):
        feed(eng, c(i, h, l))
        before = snapshot(eng)
        bad_calls = (
            lambda: feed(eng, c(i, 99, 0)),                                   # duplicate
            lambda: feed(eng, c(i + 1, 99, 0, M15)),                          # timeframe
            lambda: eng.process_candle(c(i + 1, 99, 0), now=T0),              # unfinished
            lambda: eng.process_candle("x", now=T0),                          # type
        )
        for call in bad_calls:
            with pytest.raises((ValueError, TypeError)):
                call()
            assert snapshot(eng) == before
    assert snapshot(eng) == snapshot(control)
 
 
# ---------------------------------------------------------------- causality
def test_engine_matches_non_incremental_reference_after_every_candle():
    for seed in range(20):
        rng = random.Random(seed)
        candles = []
        for i in range(60):
            high = rng.randint(5, 15)                # small range -> many ties
            low = high - rng.randint(1, 5)
            candles.append(c(i, float(high), float(low)))
        eng = SwingEngine(M5)
        for k, candle in enumerate(candles, start=1):
            feed(eng, candle)
            assert keys(eng.history) == reference(candles[:k]), (seed, k)
 
 
def test_future_candles_never_change_swings_already_confirmed():
    base = [(10, 1), (15, 2), (12, 3), (14, 4), (11, 5)]
    tails = [[(30, 20), (5, 1)], [(1, 0), (40, 30)], [(14, 4), (14, 4)]]
    reference_hist = None
    for tail in tails:
        eng = SwingEngine(M5)
        for i, (h, l) in enumerate(base):
            feed(eng, c(i, h, l))
        hist = keys(eng.history)
        if reference_hist is None:
            reference_hist = hist
        assert hist == reference_hist
        for j, (h, l) in enumerate(tail, start=len(base)):
            feed(eng, c(j, h, l))
        assert keys(eng.history)[: len(reference_hist)] == reference_hist
 
 
def test_confirmation_never_precedes_the_confirmation_candle_close():
    eng = SwingEngine(M5)
    for i, (h, l) in enumerate([(10, 1), (15, 2), (12, 3), (14, 4), (11, 5), (16, 1), (9, 8)]):
        candle = c(i, h, l)
        for s in feed(eng, candle):
            assert s.confirmed_at == candle.close_time
            assert s.candle_time == c(i - 1, 0, 0).open_time
 
 
def test_deterministic_replay():
    data = [(10, 1), (15, 2), (12, 3), (14, 4), (11, 5), (13, 2), (12, 6), (16, 3), (9, 4)]
    a, _ = run(data)
    b, _ = run(data)
    assert a.history == b.history and snapshot(a) == snapshot(b)
 
 
# ---------------------------------------------------------------- scope
def test_engine_contains_no_other_strategy_logic():
    eng = SwingEngine(M5)
    for attr in ("trend", "structure", "bos", "choch", "hh", "hl", "lh", "ll", "zone",
                 "zones", "liquidity", "eqh", "eql", "brr", "entry", "sl", "tp",
                 "is_superseded", "superseded_by"):
        assert not hasattr(eng, attr)