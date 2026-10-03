import dataclasses
import random
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.zone_engine as engine_module
from koffie.strategy.engines.pivot_engine import PivotEngine
from koffie.strategy.engines.zone_engine import ZoneEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.pivot import BOSIdentity, Pivot, PivotOutcome, PivotStatus
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
SPECS = {
    "BULL": (100.0, 110.0, 99.0, 109.0),
    "BEAR": (109.0, 110.0, 99.0, 100.0),
    "DOJI": (104.0, 110.0, 99.0, 105.0),
    "BIGBEAR": (150.0, 160.0, 100.0, 105.0),
    "BIGBULL": (100.0, 160.0, 90.0, 155.0),
}
 
 
def mk(i, kind, tf=M5, t0=T0):
    o, h, l, c = SPECS[kind]
    return Candle(tf, t0 + i * tf.duration, o, h, l, c)
 
 
def bos_for(candle, direction, seq=0):
    tf = candle.timeframe
    price = candle.close - 1.0 if direction is BULL else candle.close + 1.0
    swing = Swing(HIGH if direction is BULL else LOW, tf, price,
                  candle.open_time - 4 * tf.duration, candle.open_time - 2 * tf.duration, seq)
    return BOS(tf, direction, swing, candle.open_time, candle.close)
 
 
class Scenario:
    """Feeds a real PivotEngine and hands back its real PivotOutcomes."""
 
    def __init__(self, tf=M5, t0=T0):
        self.tf, self.t0, self.i = tf, t0, 0
        self.engine = PivotEngine(tf)
        self.candles = []
 
    def add(self, kind):
        c = mk(self.i, kind, self.tf, self.t0)
        self.i += 1
        self.engine.process_candle(c, c.close_time)
        self.candles.append(c)
        return c
 
    def bos(self, direction, kind="DOJI", seq=0):
        c = mk(self.i, kind, self.tf, self.t0)
        self.i += 1
        outcome = self.engine.process_candle(c, c.close_time, bos_for(c, direction, seq))
        self.candles.append(c)
        return outcome
 
 
def found_outcome(direction=BULL, tf=M5, pivot_kind=None, t0=T0):
    s = Scenario(tf, t0)
    s.add(pivot_kind or ("BEAR" if direction is BULL else "BULL"))
    s.add("DOJI")
    o = s.bos(direction)
    assert o.is_found
    return o, s
 
 
def empty_outcome(direction=BULL, tf=M5):
    s = Scenario(tf)
    s.add("DOJI")
    o = s.bos(direction)
    assert o.status is PivotStatus.NO_PIVOT_FOUND
    return o
 
 
def run(engine, outcome, now=None):
    return engine.process_outcome(outcome, outcome.confirmed_at if now is None else now)
 
 
def state(e):
    return (e.zones, e.outcomes)
 
 
# ------------------------------------------------------------------ construction
def test_initial_state():
    e = ZoneEngine(M5)
    assert e.timeframe is M5 and e.zones == () and e.outcomes == ()
 
 
def test_constructor_requires_a_timeframe():
    for bad in ("5M", 5, None, object()):
        with pytest.raises(TypeError):
            ZoneEngine(bad)
 
 
def test_state_is_read_only():
    e = ZoneEngine(M5)
    for name in ("timeframe", "zones", "outcomes"):
        with pytest.raises(AttributeError):
            setattr(e, name, None)
 
 
# ------------------------------------------------------------------ creation
def test_found_outcome_creates_exactly_one_zone():
    o, _ = found_outcome(BULL)
    e = ZoneEngine(M5)
    z = run(e, o)
    assert isinstance(z, Zone) and z.pivot is o.pivot
    assert e.zones == (z,) and e.outcomes == (o,)
    assert e.zone_for(o.identity) is z
 
 
def test_bullish_creates_demand_and_bearish_creates_supply():
    e = ZoneEngine(M5)
    assert run(e, found_outcome(BULL)[0]).zone_type is ZoneType.DEMAND
    assert run(ZoneEngine(M5), found_outcome(BEAR)[0]).zone_type is ZoneType.SUPPLY
 
 
def test_zone_range_is_the_real_pivot_candle_range_no_buffer():
    for direction, kind, expected in ((BULL, "BIGBEAR", (160.0, 100.0)), (BEAR, "BIGBULL", (160.0, 90.0)),
                                      (BULL, "BEAR", (110.0, 99.0))):
        o, s = found_outcome(direction, pivot_kind=kind)
        z = run(ZoneEngine(M5), o)
        pivot_candle = next(c for c in s.candles if c.open_time == o.pivot.candle_time)
        assert (z.high, z.low) == (pivot_candle.high, pivot_candle.low) == expected
 
 
def test_created_at_and_origin_time_are_the_two_distinct_times():
    o, s = found_outcome(BULL)
    z = run(ZoneEngine(M5), o)
    assert z.origin_time == s.candles[0].open_time
    assert z.created_at == s.candles[-1].close_time == o.bos.confirmed_at
    assert z.origin_time < z.created_at
 
 
def test_no_pivot_found_creates_no_zone_and_no_fallback():
    o = empty_outcome()
    e = ZoneEngine(M5)
    assert run(e, o) is None
    assert e.zones == () and e.outcomes == (o,)
    assert e.zone_for(o.identity) is None
 
 
def test_no_pivot_found_is_remembered_so_a_replay_is_a_no_op():
    o = empty_outcome()
    e = ZoneEngine(M5)
    run(e, o)
    before = state(e)
    assert run(e, o) is None
    assert state(e) == before
 
 
def test_zone_for_validates_its_argument():
    e = ZoneEngine(M5)
    for bad in (None, "id", 5):
        with pytest.raises(TypeError):
            e.zone_for(bad)
    assert e.zone_for(found_outcome()[0].identity) is None
 
 
# ------------------------------------------------------------------ idempotent replay
def test_exact_replay_returns_the_stored_zone_and_changes_nothing():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    first = run(e, o)
    before = state(e)
    again = run(e, o)
    assert again is first
    assert state(e) == before and len(e.zones) == 1
 
 
def test_equal_but_rebuilt_outcome_is_recognised_as_a_replay():
    a, _ = found_outcome()
    b, _ = found_outcome()
    assert a is not b and a == b
    e = ZoneEngine(M5)
    first = run(e, a)
    assert run(e, b) is first and len(e.zones) == 1
 
 
def test_replay_of_an_older_outcome_after_newer_ones_is_still_idempotent():
    s = Scenario()
    s.add("BEAR")
    o1 = s.bos(BULL)
    s.add("BEAR")
    o2 = s.bos(BULL)
    e = ZoneEngine(M5)
    z1 = run(e, o1)
    run(e, o2)
    before = state(e)
    assert run(e, o1) is z1
    assert state(e) == before
 
 
# ------------------------------------------------------------------ conflicting replay
def test_same_identity_with_different_bos_values_is_rejected():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    run(e, o)
    before = state(e)
    other_bos = BOS(M5, BULL, o.bos.broken_swing, o.bos.candle_time, o.bos.close_price + 5.0)
    conflicting = PivotOutcome(other_bos, PivotStatus.NO_PIVOT_FOUND)
    assert conflicting.identity == o.identity and conflicting != o
    with pytest.raises(ValueError):
        run(e, conflicting)
    assert state(e) == before
 
 
def test_same_identity_with_a_different_pivot_is_rejected():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    run(e, o)
    before = state(e)
    other_pivot = Pivot(o.bos, o.pivot.candle_time - M5.duration, 120.0, 95.0, CandleClass.BEARISH_DECISIVE)
    conflicting = PivotOutcome(o.bos, PivotStatus.FOUND, other_pivot)
    assert conflicting.identity == o.identity
    with pytest.raises(ValueError):
        run(e, conflicting)
    assert state(e) == before
 
 
def test_found_versus_no_pivot_for_the_same_bos_is_a_conflict_both_ways():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    run(e, o)
    with pytest.raises(ValueError):
        run(e, PivotOutcome(o.bos, PivotStatus.NO_PIVOT_FOUND))
    empty = empty_outcome()
    e2 = ZoneEngine(M5)
    run(e2, empty)
    pivot = Pivot(empty.bos, empty.bos.candle_time - 3 * M5.duration, 110.0, 99.0, CandleClass.BEARISH_DECISIVE)
    with pytest.raises(ValueError):
        run(e2, PivotOutcome(empty.bos, PivotStatus.FOUND, pivot))
    assert e2.zones == ()
 
 
# ------------------------------------------------------------------ validation
def test_rejects_wrong_argument_types():
    o, _ = found_outcome()
 
    class Lookalike:
        bos, status, pivot = o.bos, o.status, o.pivot
        identity, confirmed_at, is_found = o.identity, o.confirmed_at, True
 
    e = ZoneEngine(M5)
    for bad in (Lookalike(), o.pivot, o.bos, None, "outcome", (o,)):
        with pytest.raises(TypeError):
            e.process_outcome(bad, o.confirmed_at)
    for bad in (None, "now", 5, o.confirmed_at.date()):
        with pytest.raises(TypeError):
            e.process_outcome(o, bad)
    assert state(e) == ((), ())
 
 
def test_rejects_an_outcome_from_another_timeframe():
    for etf in (M5, M15, H1):
        for otf in (M5, M15, H1):
            o, _ = found_outcome(tf=otf)
            e = ZoneEngine(etf)
            if etf is otf:
                assert run(e, o) is not None
            else:
                with pytest.raises(ValueError):
                    run(e, o)
                assert state(e) == ((), ())
 
 
def test_now_may_not_precede_bos_confirmation():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    for delta in (timedelta(seconds=1), timedelta(minutes=5), timedelta(days=1)):
        with pytest.raises(ValueError):
            e.process_outcome(o, o.confirmed_at - delta)
    assert state(e) == ((), ())
    assert e.process_outcome(o, o.confirmed_at) is not None          # exactly at created_at is fine
 
 
def test_now_after_confirmation_is_fine():
    o, _ = found_outcome()
    assert ZoneEngine(M5).process_outcome(o, o.confirmed_at + timedelta(days=3)) is not None
 
 
def test_outcomes_must_arrive_in_non_decreasing_confirmed_at_order():
    s = Scenario()
    s.add("BEAR")
    o1 = s.bos(BULL)
    s.add("BEAR")
    o2 = s.bos(BULL)
    e = ZoneEngine(M5)
    run(e, o2)
    before = state(e)
    with pytest.raises(ValueError):
        run(e, o1)                                                    # earlier BOS after a later one
    assert state(e) == before
    assert o1.confirmed_at < o2.confirmed_at
 
 
def test_equal_confirmed_at_is_accepted_for_distinct_outcomes():
    o1, _ = found_outcome(BULL)
    swing2 = Swing(HIGH, M5, o1.bos.broken_swing.price - 0.5, o1.bos.broken_swing.candle_time,
                   o1.bos.broken_swing.confirmed_at, 1)
    bos2 = BOS(M5, BULL, swing2, o1.bos.candle_time, o1.bos.close_price)
    o2 = PivotOutcome(bos2, PivotStatus.NO_PIVOT_FOUND)
    assert o2.identity != o1.identity and o2.confirmed_at == o1.confirmed_at
    e = ZoneEngine(M5)
    run(e, o1)
    run(e, o2)
    assert len(e.outcomes) == 2
 
 
def test_rejected_calls_change_nothing_and_the_engine_stays_healthy():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    with pytest.raises(ValueError):
        e.process_outcome(o, o.confirmed_at - timedelta(seconds=1))
    with pytest.raises(TypeError):
        e.process_outcome("x", o.confirmed_at)
    assert state(e) == ((), ())
    assert run(e, o) is not None and len(e.zones) == 1
 
 
def test_mixing_naive_and_aware_datetimes_raises_and_changes_nothing():
    o, _ = found_outcome()
    e = ZoneEngine(M5)
    aware_now = datetime(2027, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(TypeError):
        e.process_outcome(o, aware_now)
    assert state(e) == ((), ())
    aware_outcome, _ = found_outcome(t0=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc))
    z = run(e, aware_outcome, aware_outcome.confirmed_at)
    assert z.created_at.tzinfo is timezone.utc
 
 
# ------------------------------------------------------------------ persistence / no lifecycle
def test_zones_persist_and_history_is_append_only():
    s = Scenario()
    s.add("BEAR")
    outcomes = [s.bos(BULL)]
    for _ in range(4):
        s.add("BEAR")
        outcomes.append(s.bos(BULL))
    e = ZoneEngine(M5)
    previous = ()
    for o in outcomes:
        run(e, o)
        assert e.zones[: len(previous)] == previous
        previous = e.zones
    assert len(e.zones) == 5 and isinstance(e.zones, tuple)
 
 
def test_a_zone_survives_later_opposite_direction_zones_and_gaps():
    s = Scenario()
    s.add("BEAR")
    demand_o = s.bos(BULL)
    s.add("BULL")
    s.add("DOJI")
    supply_o = s.bos(BEAR)                      # market "changes direction"
    e = ZoneEngine(M5)
    demand = run(e, demand_o)
    supply = run(e, supply_o)
    assert demand_o.confirmed_at < supply_o.confirmed_at
    assert e.zones == (demand, supply)
    assert (demand.zone_type, supply.zone_type) == (ZoneType.DEMAND, ZoneType.SUPPLY)
    assert (demand.high, demand.low) == (110.0, 99.0)           # boundaries unchanged
 
 
def test_overlapping_zones_are_kept_separate_and_never_merged():
    s = Scenario()
    s.add("BIGBEAR")
    o1 = s.bos(BULL)
    s.add("BEAR")
    o2 = s.bos(BULL)
    e = ZoneEngine(M5)
    z1, z2 = run(e, o1), run(e, o2)
    assert len(e.zones) == 2 and z1 != z2
    assert (z1.low, z1.high) == (100.0, 160.0) and (z2.low, z2.high) == (99.0, 110.0)
    assert z2.low < z1.high and z1.low < z2.high
    assert e.zone_for(o1.identity) is z1 and e.zone_for(o2.identity) is z2
 
 
def test_multiple_zones_of_the_same_type_coexist():
    s = Scenario()
    s.add("BEAR")
    o1 = s.bos(BULL)
    s.add("BEAR")
    o2 = s.bos(BULL)
    e = ZoneEngine(M5)
    run(e, o1)
    run(e, o2)
    assert [z.zone_type for z in e.zones] == [ZoneType.DEMAND, ZoneType.DEMAND]
 
 
def test_provenance_survives_inside_the_engine():
    o, s = found_outcome(BULL)
    z = run(ZoneEngine(M5), o)
    assert z.pivot is o.pivot and z.bos is o.bos and z.broken_swing is o.bos.broken_swing
    assert z.identity == o.identity
 
 
def test_engine_has_no_deletion_invalidation_or_other_logic():
    e = ZoneEngine(M5)
    for attr in ("delete", "remove", "invalidate", "mitigate", "merge", "rank", "score",
                 "touch", "process_candle", "process_price", "consume", "mark_consumed",
                 "expire", "clear", "liquidity", "entry", "sl", "tp", "setup"):
        assert not hasattr(e, attr)
    for name in ("PivotEngine", "BOSEngine", "StructureEngine", "Candle", "SwingEngine"):
        assert not hasattr(engine_module, name)
 
 
# ------------------------------------------------------------------ timeframes
def test_each_timeframe_has_its_own_independent_engine():
    engines = {tf: ZoneEngine(tf) for tf in (M5, M15, H1)}
    zones = {}
    for tf, e in engines.items():
        o, _ = found_outcome(BULL, tf=tf)
        zones[tf] = run(e, o)
    for tf, e in engines.items():
        assert e.zones == (zones[tf],) and zones[tf].timeframe is tf
    assert len({z.identity for z in zones.values()}) == 3
 
 
# ------------------------------------------------------------------ zones_known_at (temporal rule)
def test_zones_known_at_respects_created_at_not_origin_time():
    o, _ = found_outcome(BULL)
    e = ZoneEngine(M5)
    z = run(e, o)
    assert e.zones_known_at(z.origin_time) == ()
    assert e.zones_known_at(z.origin_time + M5.duration) == ()
    assert e.zones_known_at(z.created_at - timedelta(seconds=1)) == ()
    assert e.zones_known_at(z.created_at) == (z,)
    assert e.zones_known_at(z.created_at + timedelta(days=9)) == (z,)
 
 
def test_zones_known_at_returns_only_zones_already_created_in_order():
    s = Scenario()
    s.add("BEAR")
    o1 = s.bos(BULL)
    s.add("BEAR")
    o2 = s.bos(BULL)
    e = ZoneEngine(M5)
    z1, z2 = run(e, o1), run(e, o2)
    assert e.zones_known_at(z1.created_at) == (z1,)
    assert e.zones_known_at(z2.created_at - timedelta(seconds=1)) == (z1,)
    assert e.zones_known_at(z2.created_at) == (z1, z2)
 
 
def test_zones_known_at_validates_and_ignores_no_pivot_outcomes():
    e = ZoneEngine(M5)
    o = empty_outcome()
    run(e, o)
    assert e.zones_known_at(o.confirmed_at + timedelta(days=1)) == ()
    for bad in (None, "now", 5):
        with pytest.raises(TypeError):
            e.zones_known_at(bad)
 
 
# ------------------------------------------------------------------ randomized integration with a real PivotEngine
def test_random_integration_with_a_real_pivot_engine():
    kinds = list(SPECS)
    total_zones = 0
    for seed in range(40):
        rng = random.Random(seed)
        tf = rng.choice([M5, M15, H1])
        pe, ze = PivotEngine(tf), ZoneEngine(tf)
        candles, outcomes = [], []
        for i in range(60):
            c = mk(i, rng.choice(kinds), tf)
            bos = bos_for(c, rng.choice([BULL, BEAR])) if i >= 1 and rng.random() < 0.25 else None
            o = pe.process_candle(c, c.close_time, bos)
            candles.append(c)
            if o is not None:
                z = ze.process_outcome(o, c.close_time)
                outcomes.append((o, z))
                if rng.random() < 0.3:
                    assert ze.process_outcome(o, c.close_time) is z       # replay
        assert len(ze.zones) == sum(1 for o, _ in outcomes if o.is_found)
        assert len(ze.outcomes) == len(outcomes)
        total_zones += len(ze.zones)
        for o, z in outcomes:
            if not o.is_found:
                assert z is None and ze.zone_for(o.identity) is None
                continue
            pivot_candle = next(c for c in candles if c.open_time == o.pivot.candle_time)
            bos_candle = next(c for c in candles if c.open_time == o.bos.candle_time)
            assert (z.high, z.low) == (pivot_candle.high, pivot_candle.low)
            assert z.zone_type is (ZoneType.DEMAND if o.bos.direction is BULL else ZoneType.SUPPLY)
            assert z.created_at == bos_candle.close_time and z.origin_time == pivot_candle.open_time
            assert z.origin_time < z.created_at
            assert z.pivot is o.pivot and z.bos is o.bos and z.broken_swing is o.bos.broken_swing
            assert ze.zone_for(o.identity) is z
    assert total_zones > 0
 
 
def test_deterministic_replay_of_the_same_inputs():
    def run_once():
        s = Scenario()
        s.add("BEAR")
        os_ = [s.bos(BULL)]
        s.add("BULL")
        os_.append(s.bos(BEAR))
        e = ZoneEngine(M5)
        for o in os_:
            run(e, o)
        return e.zones, e.outcomes
    assert run_once() == run_once()