import dataclasses
import random
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.models.zone as zone_module
from koffie.strategy.engines.pivot_engine import PivotEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.pivot import BOSIdentity, Pivot, PivotOutcome, PivotStatus
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import ZONE_TYPE_FOR_DIRECTION, Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
# (open, high, low, close)
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
 
 
def feed(engine, candle, bos=None):
    return engine.process_candle(candle, candle.close_time, bos)
 
 
def make_pivot(direction=BULL, tf=M5, pivot_kind=None, gap=0, t0=T0):
    """Real Pivot produced by the real PivotEngine. Returns (pivot, pivot_candle, bos_candle)."""
    pivot_kind = pivot_kind or ("BEAR" if direction is BULL else "BULL")
    e = PivotEngine(tf)
    pivot_candle = mk(0, pivot_kind, tf, t0)
    feed(e, pivot_candle)
    bos_candle = mk(2 + gap, "DOJI", tf, t0)
    feed(e, mk(1, "DOJI", tf, t0))
    outcome = feed(e, bos_candle, bos_for(bos_candle, direction))
    assert outcome.is_found
    return outcome.pivot, pivot_candle, bos_candle
 
 
# ------------------------------------------------------------------ type
def test_zone_types_are_exactly_demand_and_supply():
    assert {t.name for t in ZoneType} == {"DEMAND", "SUPPLY"}
 
 
def test_bullish_bos_creates_a_demand_zone():
    pivot, _, _ = make_pivot(BULL)
    assert Zone(pivot).zone_type is ZoneType.DEMAND
 
 
def test_bearish_bos_creates_a_supply_zone():
    pivot, _, _ = make_pivot(BEAR)
    assert Zone(pivot).zone_type is ZoneType.SUPPLY
 
 
def test_direction_to_type_mapping_is_exactly_the_specified_one():
    assert ZONE_TYPE_FOR_DIRECTION == {BULL: ZoneType.DEMAND, BEAR: ZoneType.SUPPLY}
 
 
def test_zone_type_follows_the_pivot_candle_colour():
    demand, _, _ = make_pivot(BULL)
    supply, _, _ = make_pivot(BEAR)
    assert demand.candle_class is CandleClass.BEARISH_DECISIVE
    assert supply.candle_class is CandleClass.BULLISH_DECISIVE
 
 
# ------------------------------------------------------------------ single source of truth
def test_zone_stores_exactly_one_field_the_pivot():
    assert [f.name for f in dataclasses.fields(Zone)] == ["pivot"]
 
 
def test_zone_is_immutable_including_derived_values():
    pivot, _, _ = make_pivot()
    z = Zone(pivot)
    for name in ("pivot", "high", "low", "origin_time", "created_at", "zone_type", "timeframe"):
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            setattr(z, name, None)
 
 
# ------------------------------------------------------------------ price range
def test_range_is_the_full_wick_to_wick_pivot_range_with_no_buffer():
    pivot, pivot_candle, _ = make_pivot(BULL, pivot_kind="BEAR")
    z = Zone(pivot)
    assert (z.high, z.low) == (pivot_candle.high, pivot_candle.low) == (110.0, 99.0)
    body_top, body_bottom = max(pivot_candle.open, pivot_candle.close), min(pivot_candle.open, pivot_candle.close)
    assert z.high != body_top and z.low != body_bottom        # wicks included, not body-only
 
 
def test_range_comes_from_the_pivot_not_from_other_candles():
    pivot, _, _ = make_pivot(BULL, pivot_kind="BIGBEAR")
    z = Zone(pivot)
    assert (z.high, z.low) == (160.0, 100.0)
    assert z.high == pivot.high and z.low == pivot.low
 
 
def test_range_matches_for_supply_zones_too():
    pivot, pivot_candle, _ = make_pivot(BEAR, pivot_kind="BIGBULL")
    z = Zone(pivot)
    assert (z.high, z.low) == (pivot_candle.high, pivot_candle.low) == (160.0, 90.0)
 
 
def test_boundaries_do_not_change_across_reads():
    pivot, _, _ = make_pivot()
    z = Zone(pivot)
    assert [(z.high, z.low) for _ in range(3)] == [(z.high, z.low)] * 3
 
 
# ------------------------------------------------------------------ timeframe
def test_timeframe_comes_from_the_bos_on_every_timeframe():
    for tf in (M5, M15, H1):
        pivot, _, _ = make_pivot(BULL, tf=tf)
        z = Zone(pivot)
        assert z.timeframe is tf and z.bos.timeframe is tf
 
 
# ------------------------------------------------------------------ the two times
def test_origin_time_is_the_pivot_candle_open_time():
    pivot, pivot_candle, _ = make_pivot()
    assert Zone(pivot).origin_time == pivot_candle.open_time == pivot.candle_time
 
 
def test_created_at_is_the_bos_confirmation_close_time():
    pivot, _, bos_candle = make_pivot()
    z = Zone(pivot)
    assert z.created_at == bos_candle.close_time == pivot.bos.confirmed_at == pivot.confirmed_at
 
 
def test_origin_time_is_strictly_earlier_than_created_at():
    for gap in (0, 1, 50):
        pivot, _, _ = make_pivot(gap=gap)
        z = Zone(pivot)
        assert z.origin_time < z.created_at
        assert z.created_at - z.origin_time >= 2 * M5.duration
 
 
def test_the_two_times_are_different_even_for_the_same_range():
    near, _, _ = make_pivot(gap=0)
    far, _, _ = make_pivot(gap=40)
    zn, zf = Zone(near), Zone(far)
    assert (zn.high, zn.low) == (zf.high, zf.low)
    assert zn.origin_time == zf.origin_time and zn.created_at != zf.created_at
 
 
def test_zone_is_known_only_from_created_at_onward():
    pivot, _, _ = make_pivot()
    z = Zone(pivot)
    assert not z.is_known_at(z.origin_time)                      # the Pivot candle already existed
    assert not z.is_known_at(z.origin_time + M5.duration)        # ...even after it closed
    assert not z.is_known_at(z.created_at - timedelta(seconds=1))
    assert z.is_known_at(z.created_at)
    assert z.is_known_at(z.created_at + timedelta(days=30))
 
 
def test_is_known_at_validates_its_argument():
    z = Zone(make_pivot()[0])
    for bad in (None, "2026-01-01", 5, z.created_at.date()):
        with pytest.raises(TypeError):
            z.is_known_at(bad)
    with pytest.raises(TypeError):
        z.is_known_at(datetime(2026, 1, 1, tzinfo=timezone.utc))   # naive zone vs aware moment
 
 
def test_timezone_aware_zone_works():
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    pivot, _, bos_candle = make_pivot(t0=t0)
    z = Zone(pivot)
    assert z.created_at == bos_candle.close_time and z.created_at.tzinfo is timezone.utc
    assert z.is_known_at(z.created_at) and not z.is_known_at(z.created_at - timedelta(seconds=1))
 
 
# ------------------------------------------------------------------ provenance
def test_full_provenance_chain_is_exposed_and_consistent():
    pivot, pivot_candle, bos_candle = make_pivot(BULL)
    z = Zone(pivot)
    assert z.pivot is pivot
    assert z.bos is pivot.bos
    assert z.broken_swing is pivot.bos.broken_swing
    assert z.bos.candle_time == bos_candle.open_time
    assert z.pivot.candle_time == pivot_candle.open_time
    assert z.broken_swing.swing_type is HIGH                      # bullish BOS broke a swing high
    chain_back = Zone(z.pivot).pivot.bos.broken_swing
    assert chain_back is z.broken_swing
 
 
def test_bearish_zone_provenance_points_at_a_swing_low():
    z = Zone(make_pivot(BEAR)[0])
    assert z.broken_swing.swing_type is LOW and z.bos.direction is BEAR
 
 
def test_identity_is_the_bos_identity():
    pivot, _, _ = make_pivot()
    z = Zone(pivot)
    assert isinstance(z.identity, BOSIdentity)
    assert z.identity == BOSIdentity.from_bos(pivot.bos) == PivotOutcome(pivot.bos, PivotStatus.FOUND, pivot).identity
 
 
def test_identity_is_stable_for_an_equal_rebuilt_zone():
    a, _, _ = make_pivot()
    b, _, _ = make_pivot()
    assert a is not b and Zone(a).identity == Zone(b).identity
 
 
def test_identity_differs_between_zones_from_different_bos_events():
    a, _, _ = make_pivot(gap=0)
    b, _, _ = make_pivot(gap=3)
    assert Zone(a).identity != Zone(b).identity
    c, _, _ = make_pivot(BULL, tf=M15)
    assert Zone(a).identity != Zone(c).identity
 
 
def test_derived_fields_always_agree_with_the_pivot_on_random_pivots():
    rng = random.Random(7)
    for _ in range(200):
        tf = rng.choice([M5, M15, H1])
        direction = rng.choice([BULL, BEAR])
        kind = rng.choice(["BIGBEAR", "BEAR"] if direction is BULL else ["BIGBULL", "BULL"])
        pivot, pivot_candle, bos_candle = make_pivot(direction, tf, kind, gap=rng.randint(0, 20))
        z = Zone(pivot)
        assert z.timeframe is tf is pivot.timeframe
        assert (z.high, z.low) == (pivot.high, pivot.low) == (pivot_candle.high, pivot_candle.low)
        assert z.origin_time == pivot.candle_time and z.created_at == pivot.confirmed_at
        assert z.zone_type is (ZoneType.DEMAND if direction is BULL else ZoneType.SUPPLY)
        assert z.origin_time < z.created_at == bos_candle.close_time
 
 
# ------------------------------------------------------------------ equality / overlap
def test_zone_equality_and_hash_follow_the_pivot():
    a, _, _ = make_pivot()
    b, _, _ = make_pivot()
    assert Zone(a) == Zone(b) and hash(Zone(a)) == hash(Zone(b))
    assert len({Zone(a), Zone(b)}) == 1
    c, _, _ = make_pivot(gap=3)
    assert Zone(a) != Zone(c) and len({Zone(a), Zone(c)}) == 2
 
 
def test_overlapping_zones_remain_separate_records_with_their_own_chains():
    e = PivotEngine(M5)
    feed(e, mk(0, "BIGBEAR"))
    c1 = mk(1, "DOJI")
    o1 = feed(e, c1, bos_for(c1, BULL))
    feed(e, mk(2, "BEAR"))
    c2 = mk(3, "DOJI")
    o2 = feed(e, c2, bos_for(c2, BULL))
    z1, z2 = Zone(o1.pivot), Zone(o2.pivot)
    assert (z1.low, z1.high) == (100.0, 160.0) and (z2.low, z2.high) == (99.0, 110.0)
    assert z2.low < z1.high and z1.low < z2.high                # they overlap
    assert z1 != z2 and z1.identity != z2.identity
    assert z1.pivot is not z2.pivot and z1.bos is not z2.bos
    assert (z1.low, z1.high) == (100.0, 160.0)                  # unchanged by the overlap
 
 
# ------------------------------------------------------------------ validation
def test_zone_requires_a_pivot():
    pivot, _, _ = make_pivot()
    outcome = PivotOutcome(pivot.bos, PivotStatus.FOUND, pivot)
    class Lookalike:
        bos, candle_time, high, low = pivot.bos, pivot.candle_time, pivot.high, pivot.low
        candle_class = pivot.candle_class
    for bad in (None, outcome, pivot.bos, pivot.high, "pivot", {"high": 1}, Lookalike()):
        with pytest.raises(ValueError):
            Zone(bad)
 
 
# ------------------------------------------------------------------ scope
def test_zone_has_no_lifecycle_setup_or_trading_attributes():
    z = Zone(make_pivot()[0])
    for attr in ("consumed_for_entry", "consumed", "invalidated", "mitigated", "touched",
                 "touch_count", "end_time", "expires_at", "status", "buffer", "strength",
                 "score", "rank", "fresh", "merge", "liquidity", "entry", "sl", "tp",
                 "setup", "mid", "midpoint", "contains", "update", "delete", "invalidate"):
        assert not hasattr(z, attr)
 
 
def test_zone_module_contains_no_engine_or_other_strategy_logic():
    for name in ("ZoneEngine", "PivotEngine", "BOSEngine", "StructureEngine", "SetupEngine",
                 "LiquidityEngine", "Candle"):
        assert not hasattr(zone_module, name)