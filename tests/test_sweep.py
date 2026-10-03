import dataclasses
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.models.sweep as sweep_module
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import BOSIdentity, Pivot
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.sweep import Sweep, SweepIdentity, penetrates
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
 
# ------------------------------------------------------------------ helpers
def mk_swing(swing_type, price, i=0, seq=0, tf=M5, t0=T0):
    candle_time = t0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_level(swing_type, price, i=0, seq=0, tf=M5, t0=T0):
    return LiquidityLevel(mk_swing(swing_type, price, i, seq, tf, t0))
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5):
    """A Zone built directly from a BOS and a Pivot (no engines involved).
 
    For the default 5M zone: created_at = T0 + 11 candles (55 minutes).
    """
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=2, seq=0, tf=tf)
    close_price = 101.0 if direction is BULL else 99.0
    bos = BOS(tf, direction, broken, T0 + 10 * tf.duration, close_price)
    pivot = Pivot(bos, T0 + 5 * tf.duration, high, low, pivot_class)
    return Zone(pivot)
 
 
def cndl(open_time, low, high, open_=None, close=None, tf=M5):
    mid = (low + high) / 2.0
    return Candle(tf, open_time, mid if open_ is None else open_, high, low, mid if close is None else close)
 
 
def at(i, tf=M5):
    return T0 + i * tf.duration
 
 
# Default scenario: DEMAND zone low 99 / high 110, liquidity LOW at 98 (known T0+10min),
# candle 11 opens exactly when the zone becomes known (T0+55min).
def demand_setup():
    return mk_zone(BULL), mk_level(LOW, 98.0)
 
 
def supply_setup():
    return mk_zone(BEAR), mk_level(HIGH, 111.0)
 
 
# ------------------------------------------------------------------ construction
def test_demand_sweep_builds_from_real_objects():
    zone, level = demand_setup()
    candle = cndl(at(11), 97.5, 100.0)
    s = Sweep(zone, level, candle)
    assert s.zone is zone and s.level is level and s.candle is candle
 
 
def test_supply_sweep_builds_from_real_objects():
    zone, level = supply_setup()
    candle = cndl(at(11), 109.0, 112.0)
    s = Sweep(zone, level, candle)
    assert s.zone is zone and s.level is level and s.candle is candle
 
 
def test_the_zone_level_and_candle_are_the_only_stored_fields():
    assert [f.name for f in dataclasses.fields(Sweep)] == ["zone", "level", "candle"]
 
 
def test_a_wick_is_enough_and_where_the_candle_closes_does_not_matter():
    zone, level = demand_setup()
    closes_back_above_level = Candle(M5, at(11), 100.0, 101.0, 97.0, 100.0)
    closes_inside_zone = Candle(M5, at(12), 98.5, 105.0, 97.0, 104.0)
    closes_beyond_level = Candle(M5, at(13), 98.2, 98.5, 97.0, 97.2)
    for c in (closes_back_above_level, closes_inside_zone, closes_beyond_level):
        assert Sweep(zone, level, c).candle is c
 
 
def test_a_candle_that_only_equals_the_level_is_not_a_sweep():
    zone, level = demand_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 98.0, 100.0))
    zone, level = supply_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 109.0, 111.0))
 
 
def test_a_candle_that_does_not_reach_the_level_is_not_a_sweep():
    zone, level = demand_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 98.5, 100.0))
    zone, level = supply_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 109.0, 110.5))
 
 
def test_only_the_wick_on_the_liquidity_side_counts():
    zone, level = demand_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 99.5, 130.0))        # high spike does not sweep DEMAND liquidity
    zone, level = supply_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 50.0, 110.5))        # low spike does not sweep SUPPLY liquidity
 
 
def test_a_level_that_is_not_relevant_to_the_zone_is_rejected():
    demand = mk_zone(BULL)
    candle = cndl(at(11), 90.0, 100.0)
    for bad in (mk_level(LOW, 99.0), mk_level(LOW, 99.5), mk_level(HIGH, 95.0)):
        assert bad.is_relevant_to(demand) is False
        with pytest.raises(ValueError):
            Sweep(demand, bad, candle)
 
 
def test_only_5m_candles_and_5m_levels_are_accepted():
    zone, level = demand_setup()
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(11), 97.5, 100.0, tf=M15))
    level_15m = mk_level(LOW, 98.0, i=0, tf=M15)                       # known T0+30min
    assert level_15m.is_relevant_to(zone) is True
    with pytest.raises(ValueError):
        Sweep(zone, level_15m, cndl(at(11), 97.5, 100.0))
 
 
def test_the_level_must_be_known_when_the_candle_opens():
    zone = mk_zone(BULL)
    late_level = mk_level(LOW, 98.0, i=12)                              # known at T0+70min
    assert late_level.known_at == at(14)
    with pytest.raises(ValueError):
        Sweep(zone, late_level, cndl(at(13), 97.0, 100.0))              # opens before it is known
    assert Sweep(zone, late_level, cndl(at(14), 97.0, 100.0)).level is late_level   # opens exactly at known_at
 
 
def test_the_zone_must_be_known_when_the_candle_opens():
    zone, level = demand_setup()
    assert zone.created_at == at(11)
    with pytest.raises(ValueError):
        Sweep(zone, level, cndl(at(10), 97.0, 100.0))
    assert Sweep(zone, level, cndl(at(11), 97.0, 100.0)).zone is zone   # opens exactly at created_at
 
 
def test_rejects_wrong_field_types():
    zone, level = demand_setup()
    candle = cndl(at(11), 97.5, 100.0)
    for bad in (None, "zone", 5, zone.pivot):
        with pytest.raises(ValueError):
            Sweep(bad, level, candle)
    for bad in (None, "level", 5, level.swing):
        with pytest.raises(ValueError):
            Sweep(zone, bad, candle)
    for bad in (None, "candle", 5, (candle,)):
        with pytest.raises(ValueError):
            Sweep(zone, level, bad)
 
 
def test_zones_of_every_timeframe_can_be_swept_by_5m_liquidity():
    for tf in (M5, M15, H1):
        for direction, level in ((BULL, mk_level(LOW, 98.0)), (BEAR, mk_level(HIGH, 111.0))):
            zone = mk_zone(direction, tf=tf)
            candle = cndl(zone.created_at, 97.0, 112.0)                 # wicks through both sides
            s = Sweep(zone, level, candle)
            assert s.zone.timeframe is tf and s.candle.timeframe is M5
 
 
def test_mixing_naive_and_aware_datetimes_raises():
    zone, level = demand_setup()
    aware_candle = cndl(datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), 97.0, 100.0)
    with pytest.raises(TypeError):
        Sweep(zone, level, aware_candle)
 
 
# ------------------------------------------------------------------ derived values
def test_derived_values():
    zone, level = demand_setup()
    candle = cndl(at(12), 97.5, 100.0)
    s = Sweep(zone, level, candle)
    assert s.zone_type is ZoneType.DEMAND
    assert s.swept_price == 98.0
    assert s.candle_time == candle.open_time == at(12)
    assert s.known_at == candle.close_time == at(13)
    zone2, level2 = supply_setup()
    assert Sweep(zone2, level2, cndl(at(11), 109.0, 112.0)).zone_type is ZoneType.SUPPLY
 
 
def test_a_sweep_is_known_only_once_its_candle_has_closed():
    zone, level = demand_setup()
    s = Sweep(zone, level, cndl(at(11), 97.0, 100.0))
    assert s.is_known_at(s.candle_time) is False
    assert s.is_known_at(s.known_at - timedelta(seconds=1)) is False
    assert s.is_known_at(s.known_at) is True
    assert s.is_known_at(s.known_at + timedelta(days=5)) is True
 
 
def test_is_known_at_requires_a_datetime_and_rejects_mixed_datetimes():
    zone, level = demand_setup()
    s = Sweep(zone, level, cndl(at(11), 97.0, 100.0))
    for bad in (None, "now", 5, s.known_at.date()):
        with pytest.raises(TypeError):
            s.is_known_at(bad)
    with pytest.raises(TypeError):
        s.is_known_at(datetime(2027, 1, 1, tzinfo=timezone.utc))
 
 
# ------------------------------------------------------------------ identity
def test_identity_is_the_zone_identity_plus_the_liquidity_identity():
    zone, level = demand_setup()
    s = Sweep(zone, level, cndl(at(11), 97.0, 100.0))
    assert s.identity == SweepIdentity(zone.identity, level.identity)
    assert isinstance(s.identity.zone_identity, BOSIdentity)
    assert SweepIdentity.from_parts(zone, level) == s.identity
 
 
def test_rebuilt_equal_objects_have_equal_identity_and_equal_sweeps():
    a = Sweep(*demand_setup(), cndl(at(11), 97.0, 100.0))
    b = Sweep(*demand_setup(), cndl(at(11), 97.0, 100.0))
    assert a is not b and a == b and a.identity == b.identity
 
 
def test_the_candle_is_not_part_of_the_identity():
    zone, level = demand_setup()
    first = Sweep(zone, level, cndl(at(11), 97.0, 100.0))
    later = Sweep(zone, level, cndl(at(15), 96.0, 100.0))
    assert first.identity == later.identity and first != later
 
 
def test_identity_differs_for_another_level_or_another_zone():
    zone, level = demand_setup()
    candle = cndl(at(11), 90.0, 100.0)
    base = Sweep(zone, level, candle).identity
    other_level = mk_level(LOW, 97.0, i=1, seq=1)
    assert Sweep(zone, other_level, candle).identity != base
    other_zone = mk_zone(BULL, tf=M15)
    assert Sweep(other_zone, level, cndl(other_zone.created_at, 90.0, 100.0)).identity != base
 
 
def test_identity_from_parts_validates_its_arguments():
    zone, level = demand_setup()
    for bad in (None, "zone", 5, zone.pivot):
        with pytest.raises(TypeError):
            SweepIdentity.from_parts(bad, level)
    for bad in (None, "level", 5, level.swing):
        with pytest.raises(TypeError):
            SweepIdentity.from_parts(zone, bad)
 
 
# ------------------------------------------------------------------ immutability
def test_sweep_is_frozen_and_every_value_is_read_only():
    zone, level = demand_setup()
    s = Sweep(zone, level, cndl(at(11), 97.0, 100.0))
    for name in ("zone", "level", "candle", "zone_type", "swept_price", "candle_time", "known_at", "identity"):
        with pytest.raises(AttributeError):
            setattr(s, name, None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.candle = cndl(at(12), 96.0, 100.0)
 
 
# ------------------------------------------------------------------ penetrates()
def test_penetrates_is_strict_for_demand_and_supply():
    zone, level = demand_setup()
    assert penetrates(zone, level, cndl(at(11), 97.99, 100.0)) is True
    assert penetrates(zone, level, cndl(at(11), 98.0, 100.0)) is False
    assert penetrates(zone, level, cndl(at(11), 98.01, 100.0)) is False
    zone, level = supply_setup()
    assert penetrates(zone, level, cndl(at(11), 100.0, 111.01)) is True
    assert penetrates(zone, level, cndl(at(11), 100.0, 111.0)) is False
    assert penetrates(zone, level, cndl(at(11), 100.0, 110.99)) is False
 
 
def test_penetrates_is_a_pure_price_check_and_has_no_tolerance():
    zone, level = demand_setup()
    assert penetrates(zone, level, cndl(at(0), 98.0 - 1e-9, 100.0)) is True     # times are not looked at
    assert penetrates(zone, level, cndl(at(0), 98.0, 100.0)) is False
 
 
def test_penetrates_validates_its_arguments():
    zone, level = demand_setup()
    candle = cndl(at(11), 97.0, 100.0)
    for bad in (None, "zone", 5):
        with pytest.raises(TypeError):
            penetrates(bad, level, candle)
        with pytest.raises(TypeError):
            penetrates(zone, bad, candle)
        with pytest.raises(TypeError):
            penetrates(zone, level, bad)
 
 
# ------------------------------------------------------------------ scope
def test_model_has_no_touch_setup_entry_sl_tp_or_structure_logic():
    zone, level = demand_setup()
    s = Sweep(zone, level, cndl(at(11), 97.0, 100.0))
    for attr in ("touch", "touched", "setup", "confirm", "confirmation", "entry", "sl", "tp",
                 "stop_loss", "take_profit", "risk", "bos", "choch", "swept", "consumed", "consume",
                 "expired", "expire", "invalidate", "rank", "score", "tolerance", "distance", "depth"):
        if attr == "bos":
            continue                                                      # `zone.bos` exists on Zone, not on Sweep
        assert not hasattr(s, attr)
    for name in ("ZoneEngine", "SwingEngine", "PivotEngine", "LiquidityEngine", "SweepEngine",
                 "BOSEngine", "CHOCHEngine", "StructureEngine"):
        assert not hasattr(sweep_module, name)