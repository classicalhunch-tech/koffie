import dataclasses
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.models.liquidity as liquidity_module
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import CandleClass, Timeframe
from koffie.strategy.models.liquidity import LiquidityIdentity, LiquidityLevel, LiquiditySide
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
 
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
 
 
# ------------------------------------------------------------------ construction
def test_builds_from_a_real_swing():
    s = mk_swing(LOW, 98.0)
    level = LiquidityLevel(s)
    assert level.swing is s
 
 
def test_rejects_anything_that_is_not_a_swing():
    s = mk_swing(LOW, 98.0)
 
    class Lookalike:
        swing_type, timeframe, price = s.swing_type, s.timeframe, s.price
        candle_time, confirmed_at, sequence = s.candle_time, s.confirmed_at, s.sequence
 
    for bad in (None, "swing", 5, (s,), s.price, Lookalike()):
        with pytest.raises(ValueError):
            LiquidityLevel(bad)
 
 
def test_the_swing_is_the_only_stored_field():
    assert [f.name for f in dataclasses.fields(LiquidityLevel)] == ["swing"]
 
 
# ------------------------------------------------------------------ derived values
def test_side_is_below_for_a_low_and_above_for_a_high():
    assert LiquidityLevel(mk_swing(LOW, 98.0)).side is LiquiditySide.BELOW
    assert LiquidityLevel(mk_swing(HIGH, 111.0)).side is LiquiditySide.ABOVE
 
 
def test_price_timeframe_and_sequence_come_from_the_swing():
    s = mk_swing(HIGH, 111.5, i=3, seq=7)
    level = LiquidityLevel(s)
    assert level.price == 111.5
    assert level.timeframe is M5
    assert level.sequence == 7
 
 
def test_the_two_times_are_distinct_and_come_from_the_swing():
    s = mk_swing(LOW, 98.0, i=4)
    level = LiquidityLevel(s)
    assert level.origin_time == s.candle_time == T0 + 4 * M5.duration
    assert level.known_at == s.confirmed_at == T0 + 6 * M5.duration
    assert level.origin_time < level.known_at
 
 
# ------------------------------------------------------------------ is_known_at
def test_is_known_at_boundaries():
    level = LiquidityLevel(mk_swing(LOW, 98.0))
    assert level.is_known_at(level.origin_time) is False
    assert level.is_known_at(level.origin_time + M5.duration) is False
    assert level.is_known_at(level.known_at - timedelta(seconds=1)) is False
    assert level.is_known_at(level.known_at) is True
    assert level.is_known_at(level.known_at + timedelta(days=9)) is True
 
 
def test_is_known_at_requires_a_datetime():
    level = LiquidityLevel(mk_swing(LOW, 98.0))
    for bad in (None, "now", 5, level.known_at.date()):
        with pytest.raises(TypeError):
            level.is_known_at(bad)
 
 
def test_is_known_at_mixing_naive_and_aware_raises():
    level = LiquidityLevel(mk_swing(LOW, 98.0))
    with pytest.raises(TypeError):
        level.is_known_at(datetime(2027, 1, 1, tzinfo=timezone.utc))
    aware_t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    aware = LiquidityLevel(mk_swing(LOW, 98.0, t0=aware_t0))
    assert aware.known_at.tzinfo is timezone.utc
    assert aware.is_known_at(aware.known_at) is True
    with pytest.raises(TypeError):
        aware.is_known_at(datetime(2027, 1, 1))
 
 
# ------------------------------------------------------------------ identity
def test_identity_fields_come_from_the_swing():
    s = mk_swing(LOW, 98.0, i=2, seq=5)
    ident = LiquidityLevel(s).identity
    assert ident == LiquidityIdentity(M5, LOW, 5, s.candle_time)
    assert LiquidityIdentity.from_swing(s) == ident
 
 
def test_equal_swings_rebuilt_have_equal_identity_and_equal_levels():
    a, b = mk_swing(LOW, 98.0, i=2, seq=5), mk_swing(LOW, 98.0, i=2, seq=5)
    assert a is not b
    assert LiquidityLevel(a) == LiquidityLevel(b)
    assert LiquidityLevel(a).identity == LiquidityLevel(b).identity
 
 
def test_identity_differs_when_sequence_time_type_or_timeframe_differ():
    base = LiquidityLevel(mk_swing(LOW, 98.0, i=2, seq=5)).identity
    assert LiquidityLevel(mk_swing(LOW, 98.0, i=2, seq=6)).identity != base
    assert LiquidityLevel(mk_swing(LOW, 98.0, i=3, seq=5)).identity != base
    assert LiquidityLevel(mk_swing(HIGH, 98.0, i=2, seq=5)).identity != base
    assert LiquidityLevel(mk_swing(LOW, 98.0, i=2, seq=5, tf=M15)).identity != base
 
 
def test_identity_ignores_price_so_a_different_price_is_a_conflict_not_a_new_level():
    a = LiquidityLevel(mk_swing(LOW, 98.0, i=2, seq=5))
    b = LiquidityLevel(mk_swing(LOW, 97.0, i=2, seq=5))
    assert a.identity == b.identity and a != b
 
 
def test_identity_from_swing_validates_its_argument():
    for bad in (None, "swing", 5):
        with pytest.raises(TypeError):
            LiquidityIdentity.from_swing(bad)
 
 
# ------------------------------------------------------------------ immutability
def test_level_is_frozen_and_every_value_is_read_only():
    level = LiquidityLevel(mk_swing(LOW, 98.0))
    for name in ("swing", "side", "price", "timeframe", "sequence", "identity", "origin_time", "known_at"):
        with pytest.raises(AttributeError):
            setattr(level, name, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        level.swing = mk_swing(LOW, 97.0)
 
 
# ------------------------------------------------------------------ relevance: DEMAND
def test_demand_a_swing_low_strictly_below_the_zone_low_is_relevant():
    zone = mk_zone(BULL, low=99.0, high=110.0)
    assert zone.zone_type is ZoneType.DEMAND
    assert LiquidityLevel(mk_swing(LOW, 98.5)).is_relevant_to(zone) is True
    assert LiquidityLevel(mk_swing(LOW, 50.0)).is_relevant_to(zone) is True      # no distance limit
 
 
def test_demand_a_swing_low_equal_to_or_above_the_zone_low_is_not_relevant():
    zone = mk_zone(BULL, low=99.0, high=110.0)
    assert LiquidityLevel(mk_swing(LOW, 99.0)).is_relevant_to(zone) is False     # exact equality: not strictly below
    assert LiquidityLevel(mk_swing(LOW, 99.5)).is_relevant_to(zone) is False
    assert LiquidityLevel(mk_swing(LOW, 105.0)).is_relevant_to(zone) is False
    assert LiquidityLevel(mk_swing(LOW, 120.0)).is_relevant_to(zone) is False
 
 
def test_demand_a_swing_high_is_never_relevant():
    zone = mk_zone(BULL, low=99.0, high=110.0)
    for price in (50.0, 98.0, 99.0, 105.0, 120.0):
        assert LiquidityLevel(mk_swing(HIGH, price)).is_relevant_to(zone) is False
 
 
def test_there_is_no_tolerance_a_hair_below_is_relevant_and_the_boundary_is_not():
    zone = mk_zone(BULL, low=99.0, high=110.0)
    assert LiquidityLevel(mk_swing(LOW, 99.0 - 1e-9)).is_relevant_to(zone) is True
    assert LiquidityLevel(mk_swing(LOW, 99.0)).is_relevant_to(zone) is False
 
 
# ------------------------------------------------------------------ relevance: SUPPLY
def test_supply_a_swing_high_strictly_above_the_zone_high_is_relevant():
    zone = mk_zone(BEAR, low=99.0, high=110.0)
    assert zone.zone_type is ZoneType.SUPPLY
    assert LiquidityLevel(mk_swing(HIGH, 110.5)).is_relevant_to(zone) is True
    assert LiquidityLevel(mk_swing(HIGH, 500.0)).is_relevant_to(zone) is True    # no distance limit
 
 
def test_supply_a_swing_high_equal_to_or_below_the_zone_high_is_not_relevant():
    zone = mk_zone(BEAR, low=99.0, high=110.0)
    assert LiquidityLevel(mk_swing(HIGH, 110.0)).is_relevant_to(zone) is False   # exact equality: not strictly above
    assert LiquidityLevel(mk_swing(HIGH, 109.5)).is_relevant_to(zone) is False
    assert LiquidityLevel(mk_swing(HIGH, 100.0)).is_relevant_to(zone) is False
    assert LiquidityLevel(mk_swing(HIGH, 50.0)).is_relevant_to(zone) is False
 
 
def test_supply_a_swing_low_is_never_relevant():
    zone = mk_zone(BEAR, low=99.0, high=110.0)
    for price in (50.0, 99.0, 110.0, 111.0, 500.0):
        assert LiquidityLevel(mk_swing(LOW, price)).is_relevant_to(zone) is False
 
 
# ------------------------------------------------------------------ relevance: other cases
def test_a_5m_level_is_relevant_to_zones_of_every_timeframe():
    for tf in (M5, M15, H1):
        demand = mk_zone(BULL, low=99.0, high=110.0, tf=tf)
        supply = mk_zone(BEAR, low=99.0, high=110.0, tf=tf)
        assert demand.timeframe is tf and supply.timeframe is tf
        assert LiquidityLevel(mk_swing(LOW, 95.0)).is_relevant_to(demand) is True
        assert LiquidityLevel(mk_swing(HIGH, 115.0)).is_relevant_to(supply) is True
 
 
def test_relevance_does_not_depend_on_when_the_zone_was_created():
    early = mk_zone(BULL)
    level = LiquidityLevel(mk_swing(LOW, 95.0, i=50))            # known long after the zone was created
    assert level.known_at > early.created_at
    assert level.is_relevant_to(early) is True
    older = LiquidityLevel(mk_swing(LOW, 95.0, i=-50))           # known long before the zone existed
    assert older.known_at < early.created_at
    assert older.is_relevant_to(early) is True
 
 
def test_is_relevant_to_requires_a_zone():
    level = LiquidityLevel(mk_swing(LOW, 98.0))
    zone = mk_zone(BULL)
 
    class Lookalike:
        zone_type, low, high = zone.zone_type, zone.low, zone.high
 
    for bad in (None, "zone", 5, zone.pivot, zone.bos, Lookalike()):
        with pytest.raises(TypeError):
            level.is_relevant_to(bad)
 
 
def test_is_relevant_to_changes_nothing():
    s = mk_swing(LOW, 98.0)
    level = LiquidityLevel(s)
    zone = mk_zone(BULL)
    before = (level, zone)
    for _ in range(3):
        level.is_relevant_to(zone)
    assert (level, zone) == before and level.swing is s
 
 
# ------------------------------------------------------------------ scope
def test_model_has_no_sweep_touch_setup_entry_or_lifecycle_logic():
    level = LiquidityLevel(mk_swing(LOW, 98.0))
    for attr in ("swept", "sweep", "is_swept", "touched", "touch", "consumed", "consume",
                 "mark_consumed", "setup", "entry", "sl", "tp", "expired", "expire",
                 "invalidate", "mitigate", "rank", "score", "merge", "cluster", "equal",
                 "tolerance", "distance"):
        assert not hasattr(level, attr)
    for name in ("ZoneEngine", "SwingEngine", "PivotEngine", "LiquidityEngine", "Candle"):
        assert not hasattr(liquidity_module, name)