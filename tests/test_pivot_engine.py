import random
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.pivot_engine as engine_module
from koffie.strategy.engines.bos_engine import BOSEngine
from koffie.strategy.engines.pivot_engine import PivotEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.choch import CHOCH, CHOCHDirection
from koffie.strategy.models.pivot import BOSIdentity, Pivot, PivotOutcome, PivotStatus
from koffie.strategy.models.structure import StructureSnapshot, StructureState
from koffie.strategy.models.swing import Swing, SwingType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
FOUND, NOT_FOUND = PivotStatus.FOUND, PivotStatus.NO_PIVOT_FOUND
S = StructureState
 
# (open, high, low, close) for each candle kind. BRR = |close-open| / (high-low).
SPECS = {
    "BULL": (100.0, 110.0, 99.0, 109.0),      # BRR 0.818 -> BULLISH_DECISIVE
    "BEAR": (109.0, 110.0, 99.0, 100.0),      # BRR 0.818 -> BEARISH_DECISIVE
    "DOJI": (104.0, 110.0, 99.0, 105.0),      # BRR 0.091 -> NEUTRAL
    "BULL50": (100.0, 110.0, 100.0, 105.0),   # BRR exactly 0.50 -> NEUTRAL
    "BEAR50": (105.0, 110.0, 100.0, 100.0),   # BRR exactly 0.50 -> NEUTRAL
    "ZERO": (105.0, 105.0, 105.0, 105.0),     # high == low -> ZERO_RANGE_ANOMALY
    "BIGBEAR": (150.0, 160.0, 100.0, 105.0),  # huge BEARISH_DECISIVE (range 60)
    "BIGBULL": (100.0, 160.0, 90.0, 155.0),   # huge BULLISH_DECISIVE (range 70)
}
EXPECTED_CLASS = {
    "BULL": CandleClass.BULLISH_DECISIVE, "BEAR": CandleClass.BEARISH_DECISIVE,
    "DOJI": CandleClass.NEUTRAL, "BULL50": CandleClass.NEUTRAL, "BEAR50": CandleClass.NEUTRAL,
    "ZERO": CandleClass.ZERO_RANGE_ANOMALY,
    "BIGBEAR": CandleClass.BEARISH_DECISIVE, "BIGBULL": CandleClass.BULLISH_DECISIVE,
}
 
 
def mk(i, kind, tf=M5):
    o, h, l, c = SPECS[kind]
    return Candle(tf, T0 + i * tf.duration, o, h, l, c)
 
 
def custom(i, o, h, l, c, tf=M5):
    return Candle(tf, T0 + i * tf.duration, o, h, l, c)
 
 
def bos_for(candle, direction, seq=0):
    """A valid BOS whose breaking candle is `candle`."""
    tf = candle.timeframe
    price = candle.close - 1.0 if direction is BULL else candle.close + 1.0
    swing = Swing(HIGH if direction is BULL else LOW, tf, price,
                  candle.open_time - 4 * tf.duration, candle.open_time - 2 * tf.duration, seq)
    return BOS(tf, direction, swing, candle.open_time, candle.close)
 
 
def feed(engine, candle, bos=None):
    return engine.process_candle(candle, candle.close_time, bos)
 
 
def build(kinds, direction=BULL, tf=M5, bos_kind="DOJI"):
    """Feed `kinds` (oldest first), then a BOS candle. Returns (engine, outcome, candles, bos)."""
    e = PivotEngine(tf)
    candles = [mk(i, k, tf) for i, k in enumerate(kinds)]
    for c in candles:
        feed(e, c)
    bos_candle = mk(len(kinds), bos_kind, tf)
    bos = bos_for(bos_candle, direction)
    return e, feed(e, bos_candle, bos), candles, bos
 
 
def state(e):
    return (e.history, e.pivots, e.candle_count, e.last_candle)
 
 
# ---------------------------------------------------------------- sanity of the fixtures
def test_fixture_candles_have_the_intended_classification():
    for kind, expected in EXPECTED_CLASS.items():
        c = mk(0, kind)
        assert c.classify(c.close_time) is expected, kind
    assert mk(0, "BULL50").brr == 0.5 and mk(0, "BEAR50").brr == 0.5
 
 
# ---------------------------------------------------------------- construction
def test_initial_state():
    e = PivotEngine(M5)
    assert e.timeframe is M5 and e.history == () and e.pivots == ()
    assert e.candle_count == 0 and e.last_candle is None
 
 
def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            PivotEngine(bad)
 
 
def test_state_is_read_only():
    e = PivotEngine(M5)
    for name in ("timeframe", "history", "pivots", "candle_count", "last_candle"):
        with pytest.raises(AttributeError):
            setattr(e, name, None)
 
 
def test_candle_without_bos_returns_none_and_creates_no_outcome():
    e = PivotEngine(M5)
    assert feed(e, mk(0, "BEAR")) is None
    assert e.history == () and e.pivots == () and e.candle_count == 1
 
 
# ---------------------------------------------------------------- 1, 2: selection by colour
def test_bullish_bos_selects_most_recent_bearish_decisive_candle():
    e, out, candles, bos = build(["BEAR", "BULL", "BEAR", "BULL"], BULL)
    assert out.status is FOUND and out.pivot.candle_time == candles[2].open_time
    assert out.pivot.candle_class is CandleClass.BEARISH_DECISIVE
    assert out.bos is bos and out.pivot.bos is bos and out.pivot.direction is BULL
 
 
def test_bearish_bos_selects_most_recent_bullish_decisive_candle():
    e, out, candles, bos = build(["BULL", "BEAR", "BULL", "BEAR"], BEAR)
    assert out.status is FOUND and out.pivot.candle_time == candles[2].open_time
    assert out.pivot.candle_class is CandleClass.BULLISH_DECISIVE
    assert out.pivot.direction is BEAR
 
 
def test_same_colour_decisive_candles_are_not_pivots():
    _, out, _, _ = build(["BULL", "BULL", "BULL"], BULL)       # bullish BOS needs a BEARISH one
    assert out.status is NOT_FOUND
    _, out, _, _ = build(["BEAR", "BEAR"], BEAR)               # bearish BOS needs a BULLISH one
    assert out.status is NOT_FOUND
 
 
# ---------------------------------------------------------------- 3: BOS candle is never its own pivot
def test_bos_candle_cannot_be_its_own_pivot():
    # The BOS candle is itself a BEARISH_DECISIVE candle that still closes above the swing high.
    e = PivotEngine(M5)
    c = custom(0, 120.0, 121.0, 110.0, 111.0)
    assert c.classify(c.close_time) is CandleClass.BEARISH_DECISIVE
    swing = Swing(HIGH, M5, 110.0, c.open_time - 4 * M5.duration, c.open_time - 2 * M5.duration, 0)
    bos = BOS(M5, BULL, swing, c.open_time, c.close)
    out = feed(e, c, bos)
    assert out.status is NOT_FOUND and out.pivot is None
    # Mirror: bearish BOS candle that is itself BULLISH_DECISIVE.
    e = PivotEngine(M5)
    c = custom(0, 100.0, 110.0, 99.0, 109.0)
    swing = Swing(LOW, M5, 110.0, c.open_time - 4 * M5.duration, c.open_time - 2 * M5.duration, 0)
    out = feed(e, c, BOS(M5, BEAR, swing, c.open_time, c.close))
    assert out.status is NOT_FOUND
 
 
def test_search_starts_at_the_candle_immediately_before_the_bos_candle():
    e, out, candles, _ = build(["BEAR", "BEAR"], BULL)
    assert out.pivot.candle_time == candles[1].open_time
 
 
def test_pivot_candle_is_strictly_before_the_bos_candle():
    _, out, candles, bos = build(["BEAR", "DOJI"], BULL)
    assert out.pivot.candle_time < bos.candle_time
    assert out.pivot.candle_close_time <= bos.candle_time
 
 
# ---------------------------------------------------------------- 4-7: skipped candles
def test_dojis_are_skipped():
    _, out, candles, _ = build(["BEAR", "DOJI", "DOJI"], BULL)
    assert out.status is FOUND and out.pivot.candle_time == candles[0].open_time
 
 
def test_brr_exactly_050_is_skipped():
    _, out, candles, _ = build(["BEAR", "BEAR50"], BULL)
    assert out.pivot.candle_time == candles[0].open_time
    _, out, candles, _ = build(["BULL", "BULL50"], BEAR)
    assert out.pivot.candle_time == candles[0].open_time
    _, out, _, _ = build(["BEAR50", "BEAR50"], BULL)
    assert out.status is NOT_FOUND
 
 
def test_zero_range_candle_is_skipped():
    _, out, candles, _ = build(["BEAR", "ZERO"], BULL)
    assert out.pivot.candle_time == candles[0].open_time
    _, out, _, _ = build(["ZERO", "ZERO"], BULL)
    assert out.status is NOT_FOUND
 
 
def test_multiple_nonmatching_candles_are_skipped():
    # Spec example: BOS <- doji <- bullish decisive <- doji <- bearish decisive.
    _, out, candles, _ = build(["BEAR", "DOJI", "BULL", "DOJI"], BULL)
    assert out.pivot.candle_time == candles[0].open_time
    _, out, candles, _ = build(["BULL", "BEAR50", "ZERO", "BULL50", "DOJI", "BULL"], BEAR)
    assert out.pivot.candle_time == candles[5].open_time
    _, out, candles, _ = build(["BEAR", "BULL", "DOJI", "ZERO", "BULL", "BEAR50", "DOJI"], BULL)
    assert out.pivot.candle_time == candles[0].open_time
 
 
# ---------------------------------------------------------------- 8-11: recency only
def test_most_recent_qualifying_candle_wins():
    _, out, candles, _ = build(["BEAR", "BEAR", "BEAR"], BULL)
    assert out.pivot.candle_time == candles[2].open_time
    _, out, candles, _ = build(["BULL", "BULL"], BEAR)
    assert out.pivot.candle_time == candles[1].open_time
 
 
def test_selection_is_not_based_on_candle_size():
    # An older, much larger bearish decisive candle must lose to a newer, smaller one.
    e = PivotEngine(M5)
    big, small = mk(0, "BIGBEAR"), mk(2, "BEAR")
    assert big.range > small.range
    for c in (big, mk(1, "DOJI"), small):
        feed(e, c)
    bos_candle = mk(3, "DOJI")
    out = feed(e, bos_candle, bos_for(bos_candle, BULL))
    assert out.pivot.candle_time == small.open_time and out.pivot.high == small.high
    # And the reverse: a newer, larger one wins over an older, smaller one.
    e = PivotEngine(M5)
    for c in (mk(0, "BEAR"), mk(1, "DOJI"), mk(2, "BIGBEAR")):
        feed(e, c)
    bos_candle = mk(3, "DOJI")
    out = feed(e, bos_candle, bos_for(bos_candle, BULL))
    assert out.pivot.candle_time == mk(2, "BIGBEAR").open_time
 
 
def test_selection_is_not_based_on_price_extremes():
    # The oldest candle has the highest high AND the lowest low; the newest must still win.
    e = PivotEngine(M5)
    older = custom(0, 190.0, 200.0, 50.0, 60.0)             # extreme range, BEARISH_DECISIVE
    middle = custom(1, 109.0, 110.0, 99.0, 100.0)
    newest = custom(2, 104.5, 105.0, 100.0, 100.4)           # BRR 0.82 -> BEARISH_DECISIVE, mid-range prices
    assert newest.classify(newest.close_time) is CandleClass.BEARISH_DECISIVE
    for c in (older, middle, newest):
        feed(e, c)
    bos_candle = mk(3, "DOJI")
    out = feed(e, bos_candle, bos_for(bos_candle, BULL))
    assert out.pivot.candle_time == newest.open_time
    assert out.pivot.high == 105.0 and out.pivot.low == 100.0
 
 
def test_no_arbitrary_search_limit():
    for n in (6, 25, 100, 600):
        _, out, candles, _ = build(["BEAR"] + ["DOJI"] * n, BULL)
        assert out.status is FOUND and out.pivot.candle_time == candles[0].open_time, n
    _, out, candles, _ = build(["BULL"] + ["BEAR50", "ZERO", "BULL50"] * 200, BEAR)
    assert out.status is FOUND and out.pivot.candle_time == candles[0].open_time
 
 
# ---------------------------------------------------------------- 12: NO_PIVOT_FOUND
def test_no_pivot_found_is_explicit_and_recorded():
    e, out, _, bos = build(["DOJI", "BULL", "ZERO"], BULL)
    assert out.status is NOT_FOUND and out.pivot is None and out.bos is bos
    assert e.history == (out,) and e.pivots == ()
    e, out, _, bos = build(["DOJI", "BEAR"], BEAR)
    assert out.status is NOT_FOUND and e.history == (out,)
 
 
def test_no_pivot_found_with_no_prior_candles():
    e, out, _, _ = build([], BULL)
    assert out.status is NOT_FOUND and out.pivot is None and e.candle_count == 1
 
 
def test_no_pivot_found_does_not_invalidate_the_bos_or_use_a_fallback():
    e, out, candles, bos = build(["BULL", "DOJI"], BULL)
    assert out.status is NOT_FOUND and out.pivot is None
    assert out.bos == bos and e.pivots == ()          # no synthetic or fallback Pivot
 
 
# ---------------------------------------------------------------- 13-15: record contents
def test_pivot_preserves_full_wick_to_wick_range():
    e = PivotEngine(M5)
    wicky = custom(0, 118.0, 120.0, 90.0, 92.0)             # BRR 26/30 = 0.867
    assert wicky.classify(wicky.close_time) is CandleClass.BEARISH_DECISIVE
    feed(e, wicky)
    bos_candle = mk(1, "DOJI")
    out = feed(e, bos_candle, bos_for(bos_candle, BULL))
    assert (out.pivot.high, out.pivot.low) == (120.0, 90.0)
    assert out.pivot.high != max(wicky.open, wicky.close) and out.pivot.low != min(wicky.open, wicky.close)
 
 
def test_pivot_confirmed_at_equals_bos_confirmed_at():
    for direction, kinds in ((BULL, ["BEAR", "DOJI"]), (BEAR, ["BULL", "DOJI"])):
        _, out, _, bos = build(kinds, direction)
        assert out.pivot.confirmed_at == bos.confirmed_at == out.confirmed_at
 
 
def test_pivot_is_determined_at_bos_confirmation():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    bos_candle = mk(1, "DOJI")
    bos = bos_for(bos_candle, BULL)
    out = e.process_candle(bos_candle, bos_candle.close_time, bos)      # known in the SAME call
    assert out is not None and e.history == (out,)
    assert out.pivot.confirmed_at == bos_candle.close_time
    assert out.pivot.candle_close_time <= bos_candle.open_time          # pivot closed before the BOS candle began
 
 
def test_outcome_pivot_matches_the_selected_candle_exactly():
    e, out, candles, _ = build(["DOJI", "BEAR", "DOJI"], BULL)
    c = candles[1]
    assert (out.pivot.candle_time, out.pivot.high, out.pivot.low) == (c.open_time, c.high, c.low)
    assert out.pivot.candle_class is c.classify(c.close_time)
 
 
# ---------------------------------------------------------------- 16, 20: causality, append-only
def test_future_candles_cannot_change_an_existing_pivot():
    e, out, candles, bos = build(["BEAR", "DOJI"], BULL)
    pivot, history = out.pivot, e.history
    start = len(candles) + 1
    for i, kind in enumerate(["BEAR", "BIGBEAR", "BEAR", "DOJI", "BULL"]):
        feed(e, mk(start + i, kind))
    assert e.history == history and e.history[0] is out and e.history[0].pivot is pivot
    assert e.pivots == (pivot,)
 
 
def test_a_later_bos_does_not_alter_the_earlier_pivot():
    e, first, candles, _ = build(["BEAR", "DOJI"], BULL)
    feed(e, mk(3, "BEAR"))
    later_candle = mk(4, "DOJI")
    later = feed(e, later_candle, bos_for(later_candle, BULL, seq=1))
    assert later.pivot.candle_time == mk(3, "BEAR").open_time
    assert e.history == (first, later) and e.history[0].pivot.candle_time == candles[0].open_time
 
 
def test_history_is_append_only_and_a_tuple():
    e = PivotEngine(M5)
    snapshots = []
    seq = 0
    kinds = ["BEAR", "DOJI", "BULL", "BEAR", "DOJI", "DOJI", "BULL", "DOJI"]
    for i, k in enumerate(kinds):
        c = mk(i, k)
        bos = None
        if i in (2, 5, 7):
            bos = bos_for(c, BULL if i != 7 else BEAR, seq=seq)
            seq += 1
        feed(e, c, bos)
        snapshots.append(e.history)
    assert all(isinstance(s, tuple) for s in snapshots) and isinstance(e.pivots, tuple)
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later[:len(earlier)] == earlier
    assert len(e.history) == 3 and all(o.status is FOUND for o in e.history)
 
 
def test_returned_collections_cannot_be_used_to_alter_history():
    e, out, _, _ = build(["BEAR"], BULL)
    h = e.history
    assert not hasattr(h, "append")
    assert e.history == h == (out,)
 
 
# ---------------------------------------------------------------- 17: rejected input changes nothing
def test_rejects_unfinished_candle_without_changing_state():
    e, _, _, _ = build(["BEAR", "DOJI"], BULL)
    before = state(e)
    c = mk(5, "DOJI")
    for offset in (timedelta(0), timedelta(minutes=4, seconds=59)):
        with pytest.raises(ValueError):
            e.process_candle(c, c.open_time + offset, None)
        with pytest.raises(ValueError):
            e.process_candle(c, c.open_time + offset, bos_for(c, BULL, seq=9))
        assert state(e) == before
    assert feed(e, c) is None                              # the same candle is fine once closed
    assert e.candle_count == before[2] + 1
 
 
def test_rejects_repeated_earlier_overlapping_and_out_of_order_candles():
    e = PivotEngine(M5)
    first = mk(3, "BEAR")
    feed(e, first)
    before = state(e)
    bad = (first, mk(2, "BEAR"), mk(1, "BULL"),
           Candle(M5, first.open_time + timedelta(minutes=1), 109.0, 110.0, 99.0, 100.0))
    for c in bad:
        with pytest.raises(ValueError):
            e.process_candle(c, c.close_time + timedelta(hours=1), None)
        assert state(e) == before
    assert feed(e, mk(4, "DOJI")) is None
 
 
def test_rejected_candle_with_bos_does_not_record_an_outcome():
    e = PivotEngine(M5)
    feed(e, mk(3, "BEAR"))
    before = state(e)
    stale = mk(2, "DOJI")                                  # out of order, carrying a valid BOS
    with pytest.raises(ValueError):
        feed(e, stale, bos_for(stale, BULL))
    assert state(e) == before and e.history == ()
 
 
def test_rejects_wrong_argument_types_without_changing_state():
    class Lookalike:
        timeframe, high, low, open, close = M5, 14.0, 10.0, 11.0, 13.0
        open_time = T0
        close_time = T0 + timedelta(minutes=5)
 
        def is_closed_at(self, now):
            return True
 
    e, _, _, _ = build(["BEAR"], BULL)
    before = state(e)
    good = mk(5, "DOJI")
    for bad in (Lookalike(), None, "candle", (1, 2)):
        with pytest.raises(TypeError):
            e.process_candle(bad, good.close_time, None)
    for bad in (None, "now", 5, good.open_time.date()):
        with pytest.raises(TypeError):
            e.process_candle(good, bad, None)
    for bad in ("bos", 5, object(), Lookalike()):
        with pytest.raises(TypeError):
            e.process_candle(good, good.close_time, bad)
    assert state(e) == before
 
 
def test_rejects_wrong_candle_timeframe():
    for etf in (M5, M15, H1):
        for other in (M5, M15, H1):
            e = PivotEngine(etf)
            c = mk(0, "BEAR", other)
            if etf is other:
                assert feed(e, c) is None
            else:
                with pytest.raises(ValueError):
                    feed(e, c)
                assert e.candle_count == 0 and e.history == ()
 
 
def test_mixing_naive_and_aware_datetimes_changes_nothing():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    before = state(e)
    aware = Candle(M5, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), 109.0, 110.0, 99.0, 100.0)
    with pytest.raises(TypeError):
        e.process_candle(aware, aware.close_time, None)
    assert state(e) == before
    assert feed(e, mk(1, "DOJI")) is None
 
 
def test_aware_datetimes_work_consistently():
    e = PivotEngine(M5)
    base = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    c0 = Candle(M5, base, 109.0, 110.0, 99.0, 100.0)
    c1 = Candle(M5, base + M5.duration, 104.0, 110.0, 99.0, 105.0)
    feed(e, c0)
    out = feed(e, c1, bos_for(c1, BULL))
    assert out.status is FOUND and out.pivot.candle_time == base
 
 
# ---------------------------------------------------------------- 23-26: BOS validation
def test_rejects_bos_with_wrong_timeframe():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    before = state(e)
    c = mk(1, "DOJI")
    m15_candle = Candle(M15, c.open_time, c.open, c.high, c.low, c.close)
    wrong = bos_for(m15_candle, BULL)                      # a 15M BOS with the same open time and close
    assert wrong.timeframe is M15 and wrong.candle_time == c.open_time and wrong.close_price == c.close
    with pytest.raises(ValueError):
        feed(e, c, wrong)
    assert state(e) == before
 
 
def test_rejects_bos_with_candle_time_mismatch():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    before = state(e)
    c = mk(1, "DOJI")
    swing = Swing(HIGH, M5, c.close - 1.0, c.open_time - 4 * M5.duration, c.open_time - 2 * M5.duration, 0)
    for shift in (M5.duration, -M5.duration, timedelta(seconds=1)):
        bos = BOS(M5, BULL, swing, c.open_time + shift, c.close)
        with pytest.raises(ValueError):
            feed(e, c, bos)
        assert state(e) == before
 
 
def test_rejects_bos_with_close_price_mismatch():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    before = state(e)
    c = mk(1, "DOJI")
    for delta in (0.5, 5.0, -0.0001):
        bos = BOS(M5, BULL, bos_for(c, BULL).broken_swing, c.open_time, c.close + delta) \
            if delta > 0 else BOS(M5, BULL, Swing(HIGH, M5, c.close - 2.0, c.open_time - 4 * M5.duration,
                                                  c.open_time - 2 * M5.duration, 0), c.open_time, c.close + delta)
        with pytest.raises(ValueError):
            feed(e, c, bos)
        assert state(e) == before
    assert feed(e, c, bos_for(c, BULL)).status is FOUND  # the correct BOS is accepted afterwards
 
 
def test_rejects_bos_that_belongs_to_another_candle():
    e = PivotEngine(M5)
    first = mk(0, "BEAR")
    feed(e, first)
    second = mk(1, "DOJI")
    third = mk(2, "DOJI")
    other = bos_for(second, BULL)                          # a BOS of the second candle...
    before = state(e)
    with pytest.raises(ValueError):
        feed(e, third, other)                              # ...handed in with the third candle
    assert state(e) == before and e.candle_count == 1
    assert feed(e, second, other).status is FOUND          # and fine with its own candle
 
 
# ---------------------------------------------------------------- 18, 19: idempotency
def test_duplicate_same_bos_returns_the_stored_outcome_and_changes_nothing():
    e, out, candles, bos = build(["BEAR", "DOJI"], BULL)
    bos_candle = mk(len(candles), "DOJI")                  # a freshly built, equal candle...
    same_bos = bos_for(bos_candle, BULL)                   # ...and an equal, separately built BOS
    assert bos_candle is not e.last_candle and same_bos is not bos
    before = state(e)
    again = feed(e, bos_candle, same_bos)
    assert again is out
    assert state(e) == before
    assert len(e.history) == 1 and len(e.pivots) == 1 and e.candle_count == before[2]
 
 
def test_duplicate_bos_does_not_create_a_second_pivot_or_outcome():
    e, out, candles, bos = build(["BEAR"], BULL)
    bos_candle = e.last_candle
    for _ in range(5):
        assert feed(e, bos_candle, bos) is out
    assert e.history == (out,) and e.pivots == (out.pivot,)
 
 
def test_duplicate_no_pivot_found_bos_is_idempotent_too():
    e, out, _, bos = build(["DOJI", "BULL"], BULL)
    assert out.status is NOT_FOUND
    bos_candle = e.last_candle
    before = state(e)
    assert feed(e, bos_candle, bos) is out
    assert state(e) == before and len(e.history) == 1
 
 
def test_duplicate_is_detected_immediately_without_a_later_candle():
    e, out, _, bos = build(["BEAR"], BULL)
    assert feed(e, e.last_candle, bos) is out              # no further candle was needed
    assert e.candle_count == 2
 
 
def test_duplicate_detection_does_not_rely_on_object_identity():
    e, out, candles, bos = build(["BEAR", "BEAR"], BULL)
    rebuilt_candle = Candle(M5, e.last_candle.open_time, *SPECS["DOJI"])
    rebuilt_bos = BOS(M5, BULL, Swing(HIGH, M5, bos.broken_swing.price, bos.broken_swing.candle_time,
                                      bos.broken_swing.confirmed_at, bos.broken_swing.sequence),
                      bos.candle_time, bos.close_price)
    assert rebuilt_bos is not bos and rebuilt_candle is not e.last_candle
    assert feed(e, rebuilt_candle, rebuilt_bos) is out and len(e.history) == 1
 
 
def test_duplicate_works_after_the_engine_is_fed_other_dedupe_history():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    c1 = mk(1, "DOJI")
    first = feed(e, c1, bos_for(c1, BULL, seq=0))
    c2 = mk(2, "DOJI")
    second = feed(e, c2, bos_for(c2, BULL, seq=1))
    assert feed(e, c2, bos_for(c2, BULL, seq=1)) is second
    assert e.history == (first, second)
 
 
def test_same_identity_with_different_values_raises_and_changes_nothing():
    e, out, candles, bos = build(["BEAR"], BULL)
    c = e.last_candle
    before = state(e)
    other_close = BOS(M5, BULL, bos.broken_swing, bos.candle_time, bos.close_price + 0.5)
    other_price = BOS(M5, BULL, Swing(HIGH, M5, bos.broken_swing.price - 0.25, bos.broken_swing.candle_time,
                                      bos.broken_swing.confirmed_at, bos.broken_swing.sequence),
                      bos.candle_time, bos.close_price)
    other_confirmed = BOS(M5, BULL, Swing(HIGH, M5, bos.broken_swing.price, bos.broken_swing.candle_time,
                                          bos.broken_swing.confirmed_at + M5.duration,
                                          bos.broken_swing.sequence), bos.candle_time, bos.close_price)
    for conflicting in (other_close, other_price, other_confirmed):
        assert BOSIdentity.from_bos(conflicting) == BOSIdentity.from_bos(bos) and conflicting != bos
        with pytest.raises(ValueError):
            feed(e, c, conflicting)
        assert state(e) == before
 
 
def test_duplicate_bos_with_a_non_latest_candle_is_rejected_by_normal_validation():
    e, out, candles, bos = build(["BEAR"], BULL)
    bos_candle = e.last_candle
    feed(e, mk(5, "DOJI"))                                  # the BOS candle is no longer the latest
    before = state(e)
    with pytest.raises(ValueError):
        feed(e, bos_candle, bos)
    assert state(e) == before and len(e.history) == 1
 
 
def test_repeated_candle_without_the_bos_is_still_rejected():
    e, out, _, _ = build(["BEAR"], BULL)
    before = state(e)
    with pytest.raises(ValueError):
        feed(e, e.last_candle, None)
    assert state(e) == before
 
 
def test_repeated_latest_candle_with_a_different_bos_event_is_rejected():
    e, out, _, bos = build(["BEAR"], BULL)
    c = e.last_candle
    before = state(e)
    different = bos_for(c, BULL, seq=99)                    # different swing -> different identity
    assert BOSIdentity.from_bos(different) != BOSIdentity.from_bos(bos)
    with pytest.raises(ValueError):
        feed(e, c, different)
    assert state(e) == before
 
 
def test_each_new_bos_event_gets_exactly_one_outcome():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    outs = []
    for i in range(1, 5):
        c = mk(i, "DOJI")
        outs.append(feed(e, c, bos_for(c, BULL, seq=i)))
        assert feed(e, c, bos_for(c, BULL, seq=i)) is outs[-1]          # replay is a no-op
    assert e.history == tuple(outs) and len(e.pivots) == 4
 
 
# ---------------------------------------------------------------- 21, 22: independence
def test_pivot_engine_has_no_dependency_on_other_engines():
    for name in ("StructureEngine", "SwingEngine", "BOSEngine", "CHOCHEngine", "StructureProcessor",
                 "StructureSnapshot", "StructureState", "CHOCH", "Zone", "Liquidity"):
        assert not hasattr(engine_module, name), name
    e = PivotEngine(M5)
    for attr in ("snapshot", "structure", "detect_bos", "detect_choch", "enter_revaluating",
                 "zone", "liquidity", "entry", "sl", "tp", "risk"):
        assert not hasattr(e, attr)
 
 
def _bull_snap():
    def sw(kind, price, seq, idx):
        ct = T0 + idx * M5.duration
        return Swing(kind, M5, float(price), ct, ct + 2 * M5.duration, seq)
    return StructureSnapshot(M5, S.BULLISH, sw(HIGH, 12, 2, 2), sw(HIGH, 10, 0, 0),
                             sw(LOW, 6, 3, 3), sw(LOW, 4, 1, 1))
 
 
def _cd(i, o, h, l, c):
    return Candle(M5, T0 + (20 + i) * M5.duration, o, h, l, c)
 
 
def test_pivot_works_downstream_of_the_real_bos_engine_without_touching_the_snapshot():
    snap = _bull_snap()
    snap_before = snap
    bos_engine, pivot_engine = BOSEngine(M5), PivotEngine(M5)
    candles = [
        _cd(0, 11.0, 11.2, 9.0, 9.2),       # BEARISH_DECISIVE, closes below the level
        _cd(1, 9.0, 11.4, 8.9, 11.2),       # BULLISH_DECISIVE, closes below the level
        _cd(2, 11.0, 13.0, 10.0, 12.5),     # closes above 12 -> bullish BOS
        _cd(3, 12.5, 20.0, 12.2, 13.0),     # same unchanged level -> BOSEngine emits nothing
    ]
    outcomes = []
    for c in candles:
        bos = bos_engine.process_candle(c, c.close_time, snap)
        outcomes.append(pivot_engine.process_candle(c, c.close_time, bos))
    assert outcomes[0] is None and outcomes[1] is None and outcomes[3] is None
    out = outcomes[2]
    assert out.status is FOUND and out.bos is bos_engine.history[0]
    assert out.pivot.candle_time == candles[0].open_time
    assert (out.pivot.high, out.pivot.low) == (11.2, 9.0)
    assert pivot_engine.history == (out,) and len(bos_engine.history) == 1
    assert snap == snap_before and snap.state is S.BULLISH          # structure input untouched
 
 
def test_pivot_never_relies_on_bosengine_dedupe_when_a_bos_is_replayed():
    snap = _bull_snap()
    bos_engine, pivot_engine = BOSEngine(M5), PivotEngine(M5)
    c0, c1 = _cd(0, 11.0, 11.2, 9.0, 9.2), _cd(1, 11.0, 13.0, 10.0, 12.5)
    pivot_engine.process_candle(c0, c0.close_time, bos_engine.process_candle(c0, c0.close_time, snap))
    bos = bos_engine.process_candle(c1, c1.close_time, snap)
    first = pivot_engine.process_candle(c1, c1.close_time, bos)
    # BOSEngine would never emit this twice, but the SAME BOS handed to Pivot again is a no-op.
    assert pivot_engine.process_candle(c1, c1.close_time, bos) is first
    assert len(pivot_engine.history) == 1 and len(pivot_engine.pivots) == 1
 
 
def test_real_structure_engine_is_not_modified_by_pivot_processing():
    swing_mod = pytest.importorskip("koffie.strategy.engines.swing_engine")
    struct_mod = pytest.importorskip("koffie.strategy.engines.structure_engine")
    setup = [(8, 6), (10, 7), (9, 5), (8, 4), (11, 5), (12, 8), (11, 7), (10, 6), (11, 7)]
    bearish_decisive_rows = {2: (8.8, 5.2), 7: (9.8, 6.2)}          # same high/low, bearish decisive body
 
    def run(with_pivot):
        swing, structure = swing_mod.SwingEngine(M5), struct_mod.StructureEngine(M5)
        bos_engine, pivot_engine = BOSEngine(M5), PivotEngine(M5)
        rows = [(h, l) for h, l in setup] + [(14, 11.5)]
        pivots = []
        for i, (h, l) in enumerate(rows):
            o, c = bearish_decisive_rows.get(i, ((h + l) / 2, (h + l) / 2))
            candle = Candle(M5, T0 + i * M5.duration, o, float(h), float(l), c)
            structure.process_swings(swing.process_candle(candle, candle.close_time))
            bos = bos_engine.process_candle(candle, candle.close_time, structure.snapshot)
            if with_pivot:
                pivots.append(pivot_engine.process_candle(candle, candle.close_time, bos))
        return structure, bos_engine, pivot_engine, pivots
 
    plain_structure, plain_bos, _, _ = run(False)
    structure, bos_engine, pivot_engine, pivots = run(True)
    assert structure.state is S.BULLISH and structure.snapshot == plain_structure.snapshot
    assert bos_engine.history == plain_bos.history and len(bos_engine.history) == 1
    out = pivots[-1]
    assert out.status is FOUND and out.pivot.candle_time == T0 + 7 * M5.duration
    assert pivot_engine.history == (out,)
 
 
def _choch_for(candle):
    swing = Swing(LOW, M5, candle.close + 1.0, candle.open_time - 4 * M5.duration,
                  candle.open_time - 2 * M5.duration, 0)
    return CHOCH(M5, CHOCHDirection.BEARISH, swing, candle.open_time, candle.close)
 
 
def test_choch_does_not_create_a_pivot():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    c = mk(1, "BEAR")
    choch = _choch_for(c)
    before = state(e)
    with pytest.raises(TypeError):
        e.process_candle(c, c.close_time, choch)            # a CHOCH is not a BOS
    assert state(e) == before and e.history == () and e.pivots == ()
    # A candle that is a CHOCH on the structure side, fed without a BOS, produces nothing.
    assert feed(e, c) is None
    assert e.history == () and e.pivots == ()
    assert not hasattr(engine_module, "CHOCH")
 
 
def test_choch_style_candles_never_appear_as_outcomes_even_with_later_bos():
    e = PivotEngine(M5)
    for i, k in enumerate(["BEAR", "BEAR", "BULL"]):
        feed(e, mk(i, k))
    assert e.history == ()
    c = mk(3, "DOJI")
    out = feed(e, c, bos_for(c, BULL))
    assert len(e.history) == 1 and out.bos.direction is BULL
 
 
# ---------------------------------------------------------------- misc
def test_every_timeframe_works():
    for tf in (M5, M15, H1):
        e, out, candles, bos = build(["BEAR", "DOJI"], BULL, tf)
        assert out.status is FOUND and out.pivot.timeframe is tf
        assert out.pivot.confirmed_at == bos.confirmed_at == mk(2, "DOJI", tf).close_time
 
 
def test_gaps_between_candles_are_allowed():
    e = PivotEngine(M5)
    feed(e, mk(0, "BEAR"))
    c = mk(500, "DOJI")                                     # weekend-sized gap
    out = feed(e, c, bos_for(c, BULL))
    assert out.status is FOUND and out.pivot.candle_time == mk(0, "BEAR").open_time
 
 
def test_uses_existing_models():
    assert engine_module.Candle is Candle and engine_module.BOS is BOS
    assert engine_module.Pivot is Pivot and engine_module.PivotOutcome is PivotOutcome
 
 
def test_engine_never_changes_the_bos_or_candles_it_is_given():
    e = PivotEngine(M5)
    c0, c1 = mk(0, "BEAR"), mk(1, "DOJI")
    bos = bos_for(c1, BULL)
    c1_before, bos_before = c1, bos
    feed(e, c0)
    out = feed(e, c1, bos)
    assert c1 == c1_before and bos == bos_before and out.bos is bos
 
 
# ---------------------------------------------------------------- randomized check against a plain reference
def reference(kinds_seq, i, direction):
    wanted = "BEARISH_DECISIVE" if direction is BULL else "BULLISH_DECISIVE"
    for j in range(i - 1, -1, -1):
        if EXPECTED_CLASS[kinds_seq[j]].value == wanted:
            return j
    return None
 
 
def test_random_sequences_match_a_plain_reference_search():
    kinds_pool = list(SPECS)
    total = found = missing = 0
    for seed in range(40):
        rng = random.Random(seed)
        n = rng.randint(5, 80)
        seq = [rng.choice(kinds_pool) for _ in range(n)]
        e, bos_seq = PivotEngine(M5), 0
        expected_outcomes = 0
        for i, kind in enumerate(seq):
            c = mk(i, kind)
            if rng.random() < 0.3:
                direction = rng.choice((BULL, BEAR))
                out = feed(e, c, bos_for(c, direction, seq=bos_seq))
                bos_seq += 1
                expected_outcomes += 1
                j = reference(seq, i, direction)
                total += 1
                if j is None:
                    assert out.status is NOT_FOUND and out.pivot is None, (seed, i)
                    missing += 1
                else:
                    assert out.status is FOUND, (seed, i)
                    assert out.pivot.candle_time == mk(j, seq[j]).open_time, (seed, i)
                    assert (out.pivot.high, out.pivot.low) == SPECS[seq[j]][1:3], (seed, i)
                    assert out.pivot.confirmed_at == c.close_time
                    found += 1
            else:
                assert feed(e, c) is None
        assert len(e.history) == expected_outcomes and e.candle_count == n
    assert found > 0 and missing > 0 and total == found + missing