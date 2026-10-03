import random
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.liquidity_engine as engine_module
from koffie.strategy.engines.liquidity_engine import LiquidityEngine
from koffie.strategy.engines.pivot_engine import PivotEngine
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.engines.zone_engine import ZoneEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.liquidity import LiquidityIdentity, LiquidityLevel, LiquiditySide
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
 
# ------------------------------------------------------------------ helpers
def mk_swing(swing_type, price, i=0, seq=0, tf=M5, t0=T0):
    candle_time = t0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5):
    """A Zone built directly from a BOS and a Pivot (no engines involved)."""
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=2, seq=0, tf=tf)
    close_price = 101.0 if direction is BULL else 99.0
    bos = BOS(tf, direction, broken, T0 + 10 * tf.duration, close_price)
    pivot = Pivot(bos, T0 + 5 * tf.duration, high, low, pivot_class)
    return Zone(pivot)
 
 
def accept(engine, swing, now=None):
    return engine.process_swing(swing, swing.confirmed_at if now is None else now)
 
 
def state(engine):
    return engine.levels
 
 
def feed(swing_engine, specs, tf=M5, start=0):
    """Feed (low, high) candles to a real SwingEngine; return the swings it emits."""
    swings = []
    for k, (low, high) in enumerate(specs):
        c = Candle(tf, T0 + (start + k) * tf.duration, low, high, low, high)
        swings.extend(swing_engine.process_candle(c, c.close_time))
    return swings
 
 
# Same real-PivotEngine pattern used by test_zone_engine.py.
SPECS = {
    "BULL": (100.0, 110.0, 99.0, 109.0),
    "BEAR": (109.0, 110.0, 99.0, 100.0),
    "DOJI": (104.0, 110.0, 99.0, 105.0),
}
 
 
def mk_candle(i, kind, tf=M5, t0=T0):
    o, h, l, c = SPECS[kind]
    return Candle(tf, t0 + i * tf.duration, o, h, l, c)
 
 
def bos_for(candle, direction, seq=0):
    tf = candle.timeframe
    price = candle.close - 1.0 if direction is BULL else candle.close + 1.0
    swing = Swing(HIGH if direction is BULL else LOW, tf, price,
                  candle.open_time - 4 * tf.duration, candle.open_time - 2 * tf.duration, seq)
    return BOS(tf, direction, swing, candle.open_time, candle.close)
 
 
def real_zone(direction=BULL):
    """A Zone produced by a real PivotEngine and a real ZoneEngine (range 99..110)."""
    pe, ze = PivotEngine(M5), ZoneEngine(M5)
    kinds = ["BEAR" if direction is BULL else "BULL", "DOJI"]
    for i, kind in enumerate(kinds):
        c = mk_candle(i, kind)
        pe.process_candle(c, c.close_time)
    c = mk_candle(2, "DOJI")
    outcome = pe.process_candle(c, c.close_time, bos_for(c, direction))
    assert outcome.is_found
    return ze.process_outcome(outcome, c.close_time)
 
 
# ------------------------------------------------------------------ construction
def test_initial_state():
    e = LiquidityEngine(M5)
    assert e.timeframe is M5 and e.levels == ()
 
 
def test_constructor_accepts_only_the_5m_timeframe():
    for tf in (M15, H1):
        with pytest.raises(ValueError):
            LiquidityEngine(tf)
 
 
def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            LiquidityEngine(bad)
 
 
def test_state_is_read_only():
    e = LiquidityEngine(M5)
    for name in ("timeframe", "levels"):
        with pytest.raises(AttributeError):
            setattr(e, name, None)
 
 
# ------------------------------------------------------------------ creation
def test_a_swing_creates_exactly_one_level():
    e = LiquidityEngine(M5)
    s = mk_swing(LOW, 98.0)
    level = accept(e, s)
    assert isinstance(level, LiquidityLevel) and level.swing is s
    assert e.levels == (level,)
    assert e.level_for(level.identity) is level
 
 
def test_every_swing_becomes_a_level_without_filtering():
    e = LiquidityEngine(M5)
    swings = [mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(HIGH, 120.0, i=1, seq=1),
              mk_swing(LOW, 98.0, i=2, seq=2), mk_swing(HIGH, 120.0, i=3, seq=3)]
    for s in swings:
        accept(e, s)
    assert [l.swing for l in e.levels] == swings            # equal prices are separate levels
 
 
def test_level_for_validates_and_returns_none_for_unknown():
    e = LiquidityEngine(M5)
    for bad in (None, "id", 5, mk_swing(LOW, 98.0)):
        with pytest.raises(TypeError):
            e.level_for(bad)
    assert e.level_for(LiquidityIdentity.from_swing(mk_swing(LOW, 98.0))) is None
 
 
# ------------------------------------------------------------------ idempotent replay
def test_exact_replay_returns_the_stored_level_and_changes_nothing():
    e = LiquidityEngine(M5)
    s = mk_swing(LOW, 98.0)
    first = accept(e, s)
    before = state(e)
    again = accept(e, s)
    assert again is first
    assert state(e) == before and len(e.levels) == 1
 
 
def test_equal_but_rebuilt_swing_is_recognised_as_a_replay():
    a, b = mk_swing(LOW, 98.0), mk_swing(LOW, 98.0)
    assert a is not b and a == b
    e = LiquidityEngine(M5)
    first = accept(e, a)
    assert accept(e, b) is first and len(e.levels) == 1
 
 
def test_replay_of_an_older_swing_after_newer_ones_is_still_idempotent():
    e = LiquidityEngine(M5)
    s0, s1, s2 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 97.0, i=3, seq=1), mk_swing(HIGH, 120.0, i=6, seq=2)
    l0 = accept(e, s0)
    accept(e, s1)
    accept(e, s2)
    before = state(e)
    assert accept(e, s0) is l0
    assert state(e) == before
 
 
# ------------------------------------------------------------------ conflicting replay
def test_same_identity_with_a_different_price_is_rejected():
    e = LiquidityEngine(M5)
    s = mk_swing(LOW, 98.0, i=2, seq=4)
    accept(e, s)
    before = state(e)
    conflicting = mk_swing(LOW, 97.0, i=2, seq=4)
    assert LiquidityIdentity.from_swing(conflicting) == LiquidityIdentity.from_swing(s)
    with pytest.raises(ValueError):
        accept(e, conflicting)
    assert state(e) == before
 
 
def test_same_identity_with_a_different_confirmed_at_is_rejected():
    e = LiquidityEngine(M5)
    s = mk_swing(LOW, 98.0, i=2, seq=4)
    accept(e, s)
    before = state(e)
    later = Swing(LOW, M5, 98.0, s.candle_time, s.confirmed_at + M5.duration, 4)
    assert LiquidityIdentity.from_swing(later) == LiquidityIdentity.from_swing(s)
    with pytest.raises(ValueError):
        accept(e, later)
    assert state(e) == before
 
 
# ------------------------------------------------------------------ validation
def test_rejects_wrong_argument_types():
    s = mk_swing(LOW, 98.0)
 
    class Lookalike:
        swing_type, timeframe, price = s.swing_type, s.timeframe, s.price
        candle_time, confirmed_at, sequence = s.candle_time, s.confirmed_at, s.sequence
 
    e = LiquidityEngine(M5)
    for bad in (Lookalike(), None, "swing", 5, (s,), LiquidityLevel(s)):
        with pytest.raises(TypeError):
            e.process_swing(bad, s.confirmed_at)
    for bad in (None, "now", 5, s.confirmed_at.date()):
        with pytest.raises(TypeError):
            e.process_swing(s, bad)
    assert state(e) == ()
 
 
def test_rejects_a_swing_from_another_timeframe():
    for tf in (M15, H1):
        e = LiquidityEngine(M5)
        with pytest.raises(ValueError):
            accept(e, mk_swing(LOW, 98.0, tf=tf))
        assert state(e) == ()
    assert accept(LiquidityEngine(M5), mk_swing(LOW, 98.0)) is not None
 
 
def test_now_may_not_precede_swing_confirmation():
    s = mk_swing(LOW, 98.0)
    e = LiquidityEngine(M5)
    for delta in (timedelta(seconds=1), timedelta(minutes=5), timedelta(days=1)):
        with pytest.raises(ValueError):
            e.process_swing(s, s.confirmed_at - delta)
    assert state(e) == ()
    assert e.process_swing(s, s.confirmed_at) is not None            # exactly at known_at is fine
 
 
def test_now_after_confirmation_is_fine():
    s = mk_swing(LOW, 98.0)
    assert LiquidityEngine(M5).process_swing(s, s.confirmed_at + timedelta(days=3)) is not None
 
 
def test_swings_must_arrive_in_non_decreasing_confirmed_at_order():
    e = LiquidityEngine(M5)
    early, late = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 97.0, i=5, seq=1)
    accept(e, late)
    before = state(e)
    with pytest.raises(ValueError):
        accept(e, early)
    assert state(e) == before
    assert early.confirmed_at < late.confirmed_at
 
 
def test_equal_confirmed_at_is_accepted_for_distinct_swings():
    e = LiquidityEngine(M5)
    high = mk_swing(HIGH, 120.0, i=0, seq=0)
    low = mk_swing(LOW, 95.0, i=0, seq=1)
    assert high.confirmed_at == low.confirmed_at
    accept(e, high)
    accept(e, low)
    assert len(e.levels) == 2
 
 
def test_rejected_calls_change_nothing_and_the_engine_stays_healthy():
    s = mk_swing(LOW, 98.0)
    e = LiquidityEngine(M5)
    with pytest.raises(ValueError):
        e.process_swing(s, s.confirmed_at - timedelta(seconds=1))
    with pytest.raises(TypeError):
        e.process_swing("x", s.confirmed_at)
    assert state(e) == ()
    assert accept(e, s) is not None and len(e.levels) == 1
 
 
def test_mixing_naive_and_aware_datetimes_raises_and_changes_nothing():
    e = LiquidityEngine(M5)
    s = mk_swing(LOW, 98.0)
    with pytest.raises(TypeError):
        e.process_swing(s, datetime(2027, 1, 1, tzinfo=timezone.utc))
    assert state(e) == ()
    aware = mk_swing(LOW, 98.0, t0=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc))
    level = accept(e, aware)
    assert level.known_at.tzinfo is timezone.utc
    with pytest.raises(TypeError):
        accept(e, mk_swing(LOW, 97.0, i=9, seq=1))                    # naive swing after aware history
    assert e.levels == (level,)
 
 
# ------------------------------------------------------------------ append-only
def test_levels_are_append_only_and_nothing_is_removed():
    e = LiquidityEngine(M5)
    previous = ()
    for k in range(6):
        accept(e, mk_swing(LOW if k % 2 else HIGH, 100.0 + k, i=k * 3, seq=k))
        assert e.levels[: len(previous)] == previous
        previous = e.levels
    assert len(e.levels) == 6 and isinstance(e.levels, tuple)
 
 
# ------------------------------------------------------------------ levels_known_at
def test_levels_known_at_respects_known_at_not_origin_time():
    e = LiquidityEngine(M5)
    level = accept(e, mk_swing(LOW, 98.0))
    assert e.levels_known_at(level.origin_time) == ()
    assert e.levels_known_at(level.origin_time + M5.duration) == ()
    assert e.levels_known_at(level.known_at - timedelta(seconds=1)) == ()
    assert e.levels_known_at(level.known_at) == (level,)
    assert e.levels_known_at(level.known_at + timedelta(days=9)) == (level,)
 
 
def test_levels_known_at_returns_only_levels_already_known_in_order():
    e = LiquidityEngine(M5)
    l0 = accept(e, mk_swing(LOW, 98.0, i=0, seq=0))
    l1 = accept(e, mk_swing(HIGH, 120.0, i=5, seq=1))
    assert e.levels_known_at(l0.known_at) == (l0,)
    assert e.levels_known_at(l1.known_at - timedelta(seconds=1)) == (l0,)
    assert e.levels_known_at(l1.known_at) == (l0, l1)
 
 
def test_levels_known_at_validates_its_argument():
    e = LiquidityEngine(M5)
    for bad in (None, "now", 5):
        with pytest.raises(TypeError):
            e.levels_known_at(bad)
 
 
# ------------------------------------------------------------------ levels_for: DEMAND
def test_levels_for_demand_returns_only_swing_lows_strictly_below_the_zone_low():
    e = LiquidityEngine(M5)
    below = accept(e, mk_swing(LOW, 98.0, i=0, seq=0))
    equal = accept(e, mk_swing(LOW, 99.0, i=2, seq=1))
    above = accept(e, mk_swing(LOW, 101.0, i=4, seq=2))
    a_high = accept(e, mk_swing(HIGH, 90.0, i=6, seq=3))
    zone = mk_zone(BULL, low=99.0, high=110.0)
    moment = a_high.known_at
    assert e.levels_for(zone, moment) == (below,)
    assert equal not in e.levels_for(zone, moment) and above not in e.levels_for(zone, moment)
 
 
def test_levels_for_demand_keeps_old_levels_there_is_no_time_cutoff():
    e = LiquidityEngine(M5)
    old = accept(e, mk_swing(LOW, 50.0, i=0, seq=0))
    new = accept(e, mk_swing(LOW, 98.0, i=500, seq=1))
    zone = mk_zone(BULL)
    assert old.known_at < zone.created_at
    assert e.levels_for(zone, new.known_at) == (old, new)
 
 
# ------------------------------------------------------------------ levels_for: SUPPLY
def test_levels_for_supply_returns_only_swing_highs_strictly_above_the_zone_high():
    e = LiquidityEngine(M5)
    above = accept(e, mk_swing(HIGH, 111.0, i=0, seq=0))
    equal = accept(e, mk_swing(HIGH, 110.0, i=2, seq=1))
    below = accept(e, mk_swing(HIGH, 105.0, i=4, seq=2))
    a_low = accept(e, mk_swing(LOW, 130.0, i=6, seq=3))
    zone = mk_zone(BEAR, low=99.0, high=110.0)
    moment = a_low.known_at
    assert e.levels_for(zone, moment) == (above,)
    assert equal not in e.levels_for(zone, moment) and below not in e.levels_for(zone, moment)
 
 
# ------------------------------------------------------------------ levels_for: timing and types
def test_levels_for_ignores_levels_not_yet_known():
    e = LiquidityEngine(M5)
    level = accept(e, mk_swing(LOW, 98.0, i=5))
    zone = mk_zone(BULL)
    assert e.levels_for(zone, level.known_at - timedelta(seconds=1)) == ()
    assert e.levels_for(zone, level.known_at) == (level,)
 
 
def test_levels_for_validates_its_arguments():
    e = LiquidityEngine(M5)
    zone = mk_zone(BULL)
    for bad in (None, "zone", 5, zone.pivot):
        with pytest.raises(TypeError):
            e.levels_for(bad, T0)
        with pytest.raises(TypeError):
            e.required_liquidity_for(bad, T0)
    for bad in (None, "now", 5):
        with pytest.raises(TypeError):
            e.levels_for(zone, bad)
        with pytest.raises(TypeError):
            e.required_liquidity_for(zone, bad)
 
 
# ------------------------------------------------------------------ required_liquidity_for
def test_required_liquidity_is_the_newest_relevant_known_level():
    e = LiquidityEngine(M5)
    l0 = accept(e, mk_swing(LOW, 98.0, i=0, seq=0))
    l1 = accept(e, mk_swing(LOW, 97.0, i=3, seq=1))
    l2 = accept(e, mk_swing(LOW, 96.0, i=6, seq=2))
    zone = mk_zone(BULL)
    assert e.required_liquidity_for(zone, l0.known_at) is l0
    assert e.required_liquidity_for(zone, l1.known_at) is l1
    assert e.required_liquidity_for(zone, l2.known_at) is l2
    assert e.required_liquidity_for(zone, l2.known_at + timedelta(days=3)) is l2
 
 
def test_required_liquidity_newest_means_newest_not_lowest_or_highest_price():
    e = LiquidityEngine(M5)
    accept(e, mk_swing(LOW, 90.0, i=0, seq=0))                         # lowest, but older
    newest = accept(e, mk_swing(LOW, 98.0, i=3, seq=1))
    assert e.required_liquidity_for(mk_zone(BULL), newest.known_at) is newest
    e2 = LiquidityEngine(M5)
    accept(e2, mk_swing(HIGH, 150.0, i=0, seq=0))                      # highest, but older
    newest2 = accept(e2, mk_swing(HIGH, 111.0, i=3, seq=1))
    assert e2.required_liquidity_for(mk_zone(BEAR), newest2.known_at) is newest2
 
 
def test_required_liquidity_breaks_a_known_at_tie_with_the_swing_sequence():
    e = LiquidityEngine(M5)
    a = accept(e, mk_swing(LOW, 97.0, i=0, seq=3))
    b = accept(e, mk_swing(LOW, 98.0, i=0, seq=4))                     # same confirmed_at, higher sequence
    assert a.known_at == b.known_at
    zone = mk_zone(BULL)
    assert e.required_liquidity_for(zone, b.known_at) is b
    e2 = LiquidityEngine(M5)
    accept(e2, mk_swing(LOW, 97.0, i=0, seq=3))
    accept(e2, mk_swing(LOW, 98.0, i=0, seq=2))
    assert e2.required_liquidity_for(zone, b.known_at).sequence == 3
 
 
def test_required_liquidity_is_none_when_nothing_is_relevant():
    e = LiquidityEngine(M5)
    zone = mk_zone(BULL)
    assert e.required_liquidity_for(zone, T0 + timedelta(days=1)) is None
    level = accept(e, mk_swing(LOW, 99.0))                             # equal to the boundary
    accept(e, mk_swing(HIGH, 50.0, i=4, seq=1))                        # wrong type
    assert e.required_liquidity_for(zone, level.known_at + timedelta(days=1)) is None
    assert e.required_liquidity_for(mk_zone(BULL), level.known_at - timedelta(seconds=1)) is None
 
 
def test_a_newer_irrelevant_swing_does_not_displace_an_older_relevant_one():
    e = LiquidityEngine(M5)
    relevant = accept(e, mk_swing(LOW, 98.0, i=0, seq=0))
    accept(e, mk_swing(LOW, 105.0, i=3, seq=1))                        # newer but above the zone low
    later = accept(e, mk_swing(HIGH, 90.0, i=6, seq=2))                # newer but wrong type
    assert e.required_liquidity_for(mk_zone(BULL), later.known_at) is relevant
 
 
def test_required_liquidity_for_supply_uses_swing_highs_only():
    e = LiquidityEngine(M5)
    accept(e, mk_swing(HIGH, 112.0, i=0, seq=0))
    newest = accept(e, mk_swing(HIGH, 111.0, i=3, seq=1))
    later_low = accept(e, mk_swing(LOW, 95.0, i=6, seq=2))
    assert e.required_liquidity_for(mk_zone(BEAR), later_low.known_at) is newest
 
 
def test_a_5m_engine_serves_zones_of_every_timeframe():
    e = LiquidityEngine(M5)
    low = accept(e, mk_swing(LOW, 95.0, i=0, seq=0))
    high = accept(e, mk_swing(HIGH, 120.0, i=3, seq=1))
    for tf in (M5, M15, H1):
        demand = mk_zone(BULL, tf=tf)
        supply = mk_zone(BEAR, tf=tf)
        assert e.required_liquidity_for(demand, high.known_at) is low
        assert e.required_liquidity_for(supply, high.known_at) is high
 
 
# ------------------------------------------------------------------ independence from Zone state
def test_the_engine_does_not_depend_on_whether_the_zone_is_known_yet():
    e = LiquidityEngine(M5)
    level = accept(e, mk_swing(LOW, 95.0, i=0))
    zone = mk_zone(BULL)
    moment = level.known_at
    assert zone.is_known_at(moment) is False                           # the zone does not exist yet at `moment`
    assert e.required_liquidity_for(zone, moment) is level             # liquidity is unaffected
 
 
def test_queries_change_no_state():
    e = LiquidityEngine(M5)
    accept(e, mk_swing(LOW, 98.0, i=0, seq=0))
    accept(e, mk_swing(HIGH, 120.0, i=3, seq=1))
    zones = [mk_zone(BULL), mk_zone(BEAR)]
    before = (e.levels, list(zones))
    for zone in zones:
        e.levels_for(zone, T0 + timedelta(days=1))
        e.required_liquidity_for(zone, T0 + timedelta(days=1))
        e.levels_known_at(T0 + timedelta(days=1))
    assert (e.levels, zones) == before
 
 
def test_the_engine_stores_no_zone_and_no_sweep_state():
    e = LiquidityEngine(M5)
    accept(e, mk_swing(LOW, 98.0))
    e.required_liquidity_for(mk_zone(BULL), T0 + timedelta(days=1))
    assert not any(isinstance(v, Zone) for v in vars(e).values())
    assert not any(isinstance(v, (list, tuple)) and any(isinstance(x, Zone) for x in v) for v in vars(e).values())
 
 
# ------------------------------------------------------------------ integration with the real engines
def test_integration_with_a_real_swing_engine_for_demand():
    se, le = SwingEngine(M5), LiquidityEngine(M5)
    swings = feed(se, [(100.0, 110.0), (98.0, 110.0), (100.0, 110.0), (97.0, 110.0), (99.0, 110.0)])
    assert [(s.swing_type, s.price) for s in swings] == [(LOW, 98.0), (LOW, 97.0)]
    levels = [accept(le, s) for s in swings]
    assert [l.side for l in levels] == [LiquiditySide.BELOW, LiquiditySide.BELOW]
    zone = mk_zone(BULL, low=99.0, high=110.0)
    assert le.levels_for(zone, swings[-1].confirmed_at) == tuple(levels)
    assert le.required_liquidity_for(zone, swings[-1].confirmed_at) is levels[1]
    assert le.required_liquidity_for(zone, swings[0].confirmed_at) is levels[0]
    assert le.required_liquidity_for(mk_zone(BULL, low=97.0, high=110.0), swings[-1].confirmed_at) is None
    assert le.required_liquidity_for(mk_zone(BULL, low=98.0, high=110.0), swings[-1].confirmed_at) is levels[1]
 
 
def test_integration_with_a_real_swing_engine_for_supply():
    se, le = SwingEngine(M5), LiquidityEngine(M5)
    swings = feed(se, [(90.0, 105.0), (90.0, 108.0), (90.0, 105.0), (90.0, 109.0), (90.0, 106.0)])
    assert [(s.swing_type, s.price) for s in swings] == [(HIGH, 108.0), (HIGH, 109.0)]
    levels = [accept(le, s) for s in swings]
    moment = swings[-1].confirmed_at
    assert [l.side for l in levels] == [LiquiditySide.ABOVE, LiquiditySide.ABOVE]
    assert le.required_liquidity_for(mk_zone(BEAR, low=99.0, high=107.0), moment) is levels[1]
    assert le.levels_for(mk_zone(BEAR, low=99.0, high=108.0), moment) == (levels[1],)   # 108 is the boundary
    assert le.required_liquidity_for(mk_zone(BEAR, low=99.0, high=109.0), moment) is None
 
 
def test_an_outside_bar_gives_a_high_and_a_low_with_the_same_confirmed_at():
    se, le = SwingEngine(M5), LiquidityEngine(M5)
    swings = feed(se, [(100.0, 105.0), (95.0, 110.0), (99.0, 106.0)])
    assert [(s.swing_type, s.price) for s in swings] == [(HIGH, 110.0), (LOW, 95.0)]
    assert swings[0].confirmed_at == swings[1].confirmed_at
    high_level, low_level = [accept(le, s) for s in swings]
    moment = swings[0].confirmed_at
    assert le.required_liquidity_for(mk_zone(BEAR, low=99.0, high=107.0), moment) is high_level
    assert le.required_liquidity_for(mk_zone(BULL, low=99.0, high=107.0), moment) is low_level
 
 
def test_integration_with_a_zone_from_a_real_pivot_and_zone_engine():
    demand, supply = real_zone(BULL), real_zone(BEAR)
    assert demand.zone_type is ZoneType.DEMAND and supply.zone_type is ZoneType.SUPPLY
    assert (demand.low, demand.high) == (99.0, 110.0) and (supply.low, supply.high) == (99.0, 110.0)
    le = LiquidityEngine(M5)
    low = accept(le, mk_swing(LOW, 98.0, i=0, seq=0))
    high = accept(le, mk_swing(HIGH, 111.0, i=3, seq=1))
    moment = high.known_at
    assert le.required_liquidity_for(demand, moment) is low
    assert le.required_liquidity_for(supply, moment) is high
 
 
# ------------------------------------------------------------------ randomized integration vs brute force
def test_random_integration_matches_an_independent_brute_force():
    relevant_total = 0
    for seed in range(40):
        rng = random.Random(seed)
        se, le = SwingEngine(M5), LiquidityEngine(M5)
        specs = []
        for _ in range(50):
            low = float(rng.randint(90, 100))
            specs.append((low, low + float(rng.randint(1, 10))))
        swings = feed(se, specs)
        for s in swings:
            accept(le, s)
        assert le.levels == tuple(LiquidityLevel(s) for s in swings)
        assert [l.swing for l in le.levels] == list(se.history)
 
        for direction in (BULL, BEAR):
            low = float(rng.randint(90, 100))
            zone = mk_zone(direction, low=low, high=low + float(rng.randint(1, 10)))
            for k in range(0, 54, 4):
                moment = T0 + k * M5.duration
                if direction is BULL:
                    expected = [s for s in swings
                                if s.confirmed_at <= moment and s.swing_type is LOW and s.price < zone.low]
                else:
                    expected = [s for s in swings
                                if s.confirmed_at <= moment and s.swing_type is HIGH and s.price > zone.high]
                assert [l.swing for l in le.levels_for(zone, moment)] == expected
                required = le.required_liquidity_for(zone, moment)
                if expected:
                    newest = max(expected, key=lambda s: (s.confirmed_at, s.sequence))
                    assert required is not None and required.swing == newest
                    relevant_total += 1
                else:
                    assert required is None
    assert relevant_total > 0
 
 
def test_deterministic_replay_of_the_same_inputs():
    def run_once():
        se, le = SwingEngine(M5), LiquidityEngine(M5)
        for s in feed(se, [(100.0, 110.0), (98.0, 110.0), (100.0, 110.0), (97.0, 110.0), (99.0, 110.0)]):
            accept(le, s)
        zone = mk_zone(BULL)
        return le.levels, le.required_liquidity_for(zone, T0 + timedelta(days=1))
    assert run_once() == run_once()
 
 
# ------------------------------------------------------------------ scope
def test_engine_has_no_deletion_sweep_setup_or_other_logic():
    e = LiquidityEngine(M5)
    for attr in ("delete", "remove", "invalidate", "mitigate", "merge", "rank", "score", "cluster",
                 "touch", "sweep", "swept", "is_swept", "mark_swept", "process_candle", "process_price",
                 "consume", "mark_consumed", "expire", "clear", "setup", "entry", "sl", "tp",
                 "tolerance", "distance", "equal_levels"):
        assert not hasattr(e, attr)
    for name in ("ZoneEngine", "SwingEngine", "PivotEngine", "BOSEngine", "StructureEngine", "Candle"):
        assert not hasattr(engine_module, name)