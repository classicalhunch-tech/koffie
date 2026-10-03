"""Confirmation / Invalidation model tests (data only; the sequence is tested in test_confirmation_engine.py)."""
import dataclasses
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.models.confirmation as confirmation_module
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.confirmation import (
    CONFIRMING_CLASS_FOR, WRONG_CLASS_FOR, Confirmation, ConfirmationStatus, Invalidation,
    InvalidationReason, closes_beyond_boundary,
)
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.sweep import Sweep, SweepIdentity
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
BULL_DEC, BEAR_DEC = CandleClass.BULLISH_DECISIVE, CandleClass.BEARISH_DECISIVE
BOUNDARY, SECOND = InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY, InvalidationReason.SECOND_WRONG_DIRECTION_DECISIVE
 
 
# ------------------------------------------------------------------ helpers
def at(i, tf=M5):
    return T0 + i * tf.duration
 
 
def mk_swing(swing_type, price, i=0, seq=0, tf=M5):
    candle_time = T0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5):
    """Zone 99..110, known at slot 11 (for 5M)."""
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=2, seq=0, tf=tf)
    bos = BOS(tf, direction, broken, T0 + 10 * tf.duration, 101.0 if direction is BULL else 99.0)
    return Zone(Pivot(bos, T0 + 5 * tf.duration, high, low, pivot_class))
 
 
def ohlc(i, o, h, l, c, tf=M5):
    return Candle(tf, at(i, tf), o, h, l, c)
 
 
def cls_of(candle):
    return candle.classify(candle.close_time)
 
 
def make_sweep(direction, sweep_candle=None):
    zone = mk_zone(direction)
    if direction is BULL:
        level = LiquidityLevel(mk_swing(LOW, 98.0))
        candle = sweep_candle or ohlc(11, 98.5, 100.0, 97.5, 98.8)         # NEUTRAL, low 97.5 < 98
    else:
        level = LiquidityLevel(mk_swing(HIGH, 111.0))
        candle = sweep_candle or ohlc(11, 110.5, 112.0, 109.5, 110.8)      # NEUTRAL, high 112 > 111
    return Sweep(zone, level, candle)
 
 
# DEMAND zone 99..110 candles
D_CONFIRM = ohlc(12, 100.0, 105.0, 99.8, 104.8)           # BULLISH_DECISIVE, close inside the zone
D_CONFIRM_BELOW = ohlc(12, 95.0, 98.9, 94.9, 98.8)        # BULLISH_DECISIVE, closes BELOW the zone: still fine
D_WRONG = ohlc(12, 104.8, 105.0, 100.0, 100.2)            # BEARISH_DECISIVE, close inside the zone
D_WRONG_BOUNDARY = ohlc(12, 99.5, 99.6, 97.0, 97.1)       # BEARISH_DECISIVE, close 97.1 < zone.low 99
D_WRONG_WICK = ohlc(12, 103.0, 103.2, 98.0, 99.2)         # BEARISH_DECISIVE, wick below 99 but close 99.2
D_WRONG_EQUAL = ohlc(12, 103.0, 103.1, 98.9, 99.0)        # BEARISH_DECISIVE, close exactly on zone.low
# SUPPLY zone 99..110 candles
S_CONFIRM = ohlc(12, 104.8, 105.0, 100.0, 100.2)          # BEARISH_DECISIVE
S_CONFIRM_ABOVE = ohlc(12, 118.0, 118.2, 111.0, 111.3)    # BEARISH_DECISIVE, closes ABOVE the zone: still fine
S_WRONG = ohlc(12, 100.0, 105.0, 99.8, 104.8)             # BULLISH_DECISIVE, close inside the zone
S_WRONG_BOUNDARY = ohlc(12, 110.5, 113.0, 110.4, 112.9)   # BULLISH_DECISIVE, close 112.9 > zone.high 110
S_WRONG_WICK = ohlc(12, 106.0, 111.0, 105.8, 109.8)       # BULLISH_DECISIVE, wick above 110 but close 109.8
S_WRONG_EQUAL = ohlc(12, 106.0, 110.1, 105.9, 110.0)      # BULLISH_DECISIVE, close exactly on zone.high
 
 
def test_the_fixtures_have_the_classes_the_tests_assume():
    for c in (D_CONFIRM, D_CONFIRM_BELOW, S_WRONG, S_WRONG_BOUNDARY, S_WRONG_WICK, S_WRONG_EQUAL):
        assert cls_of(c) is BULL_DEC
    for c in (D_WRONG, D_WRONG_BOUNDARY, D_WRONG_WICK, D_WRONG_EQUAL, S_CONFIRM, S_CONFIRM_ABOVE):
        assert cls_of(c) is BEAR_DEC
 
 
# ------------------------------------------------------------------ enums and mappings
def test_status_has_exactly_the_four_locked_states():
    assert [s.value for s in ConfirmationStatus] == [
        "WAITING_FOR_DECISIVE", "WAITING_FOR_SECOND_DECISIVE", "CONFIRMED", "INVALIDATED"]
 
 
def test_invalidation_reason_has_exactly_the_two_locked_values():
    assert {r.value for r in InvalidationReason} == {
        "CLOSED_BEYOND_ZONE_BOUNDARY", "SECOND_WRONG_DIRECTION_DECISIVE"}
 
 
def test_direction_mappings():
    assert CONFIRMING_CLASS_FOR == {ZoneType.DEMAND: BULL_DEC, ZoneType.SUPPLY: BEAR_DEC}
    assert WRONG_CLASS_FOR == {ZoneType.DEMAND: BEAR_DEC, ZoneType.SUPPLY: BULL_DEC}
 
 
# ------------------------------------------------------------------ closes_beyond_boundary
def test_boundary_is_a_strict_close_not_a_wick():
    demand, supply = mk_zone(BULL), mk_zone(BEAR)
    assert closes_beyond_boundary(ZoneType.DEMAND, demand, D_WRONG_BOUNDARY) is True
    assert closes_beyond_boundary(ZoneType.DEMAND, demand, D_WRONG_WICK) is False      # wick below, close above
    assert closes_beyond_boundary(ZoneType.DEMAND, demand, D_WRONG_EQUAL) is False     # close exactly on zone.low
    assert closes_beyond_boundary(ZoneType.SUPPLY, supply, S_WRONG_BOUNDARY) is True
    assert closes_beyond_boundary(ZoneType.SUPPLY, supply, S_WRONG_WICK) is False
    assert closes_beyond_boundary(ZoneType.SUPPLY, supply, S_WRONG_EQUAL) is False
 
 
def test_boundary_only_looks_at_the_close_on_the_invalidating_side():
    demand, supply = mk_zone(BULL), mk_zone(BEAR)
    above = ohlc(12, 111.0, 118.0, 110.5, 117.5)
    below = ohlc(12, 95.0, 98.9, 94.9, 98.8)
    assert closes_beyond_boundary(ZoneType.DEMAND, demand, above) is False             # upside is not DEMAND's boundary
    assert closes_beyond_boundary(ZoneType.SUPPLY, supply, below) is False             # downside is not SUPPLY's boundary
 
 
def test_boundary_validates_its_arguments():
    zone, candle = mk_zone(BULL), D_WRONG
    for bad in (None, "x", 5):
        with pytest.raises(TypeError):
            closes_beyond_boundary(bad, zone, candle)
        with pytest.raises(TypeError):
            closes_beyond_boundary(ZoneType.DEMAND, bad, candle)
        with pytest.raises(TypeError):
            closes_beyond_boundary(ZoneType.DEMAND, zone, bad)
 
 
# ------------------------------------------------------------------ Confirmation
@pytest.mark.parametrize("direction, candle", [(BULL, D_CONFIRM), (BEAR, S_CONFIRM)])
def test_confirmation_builds_and_derives_its_values(direction, candle):
    sweep = make_sweep(direction)
    c = Confirmation(sweep, candle)
    assert c.sweep is sweep and c.candle is candle
    assert c.zone is sweep.zone and c.zone_type is sweep.zone_type
    assert c.candle_class is cls_of(candle) is CONFIRMING_CLASS_FOR[sweep.zone_type]
    assert c.candle_time == candle.open_time == at(12)
    assert c.known_at == candle.close_time == at(13)
    assert c.identity == sweep.identity and isinstance(c.identity, SweepIdentity)
 
 
def test_the_sweep_candle_itself_may_be_the_confirmation_candle():
    demand = make_sweep(BULL, ohlc(11, 97.1, 100.0, 97.0, 99.9))               # bullish decisive, low 97 < 98
    assert cls_of(demand.candle) is BULL_DEC
    assert Confirmation(demand, demand.candle).candle is demand.candle
    supply = make_sweep(BEAR, ohlc(11, 112.0, 112.4, 105.0, 105.2))            # bearish decisive, high 112.4 > 111
    assert cls_of(supply.candle) is BEAR_DEC
    assert Confirmation(supply, supply.candle).candle is supply.candle
 
 
@pytest.mark.parametrize("direction, candle", [(BULL, D_CONFIRM_BELOW), (BEAR, S_CONFIRM_ABOVE)])
def test_a_confirming_candle_may_close_outside_the_zone(direction, candle):
    sweep = make_sweep(direction)
    assert Confirmation(sweep, candle).candle is candle
 
 
def test_confirmation_stores_exactly_two_fields_and_no_prices():
    assert [f.name for f in dataclasses.fields(Confirmation)] == ["sweep", "candle"]
 
 
@pytest.mark.parametrize("direction, wrong", [(BULL, D_WRONG), (BEAR, S_WRONG)])
def test_the_wrong_direction_decisive_candle_does_not_confirm(direction, wrong):
    with pytest.raises(ValueError):
        Confirmation(make_sweep(direction), wrong)
 
 
def test_neutral_and_zero_range_candles_do_not_confirm():
    neutral = ohlc(12, 100.0, 104.0, 99.0, 101.0)
    zero = ohlc(12, 100.0, 100.0, 100.0, 100.0)
    exactly_050 = ohlc(12, 100.0, 110.0, 100.0, 105.0)
    for direction in (BULL, BEAR):
        for c in (neutral, zero, exactly_050):
            with pytest.raises(ValueError):
                Confirmation(make_sweep(direction), c)
 
 
def test_brr_just_above_050_confirms():
    assert Confirmation(make_sweep(BULL), ohlc(12, 100.0, 110.0, 100.0, 105.01)).candle_class is BULL_DEC
    assert Confirmation(make_sweep(BEAR), ohlc(12, 105.01, 110.0, 100.0, 100.0)).candle_class is BEAR_DEC
 
 
def test_a_non_5m_candle_does_not_confirm():
    sweep = make_sweep(BULL)
    with pytest.raises(ValueError):
        Confirmation(sweep, ohlc(12, 100.0, 105.0, 99.8, 104.8, tf=M15))
 
 
def test_a_candle_earlier_than_the_sweep_candle_does_not_confirm():
    sweep = make_sweep(BULL)
    with pytest.raises(ValueError):
        Confirmation(sweep, ohlc(10, 100.0, 105.0, 99.8, 104.8))
 
 
def test_confirmation_rejects_wrong_field_types():
    sweep = make_sweep(BULL)
    for bad in (None, "sweep", 5, sweep.zone):
        with pytest.raises(ValueError):
            Confirmation(bad, D_CONFIRM)
    for bad in (None, "candle", 5, (D_CONFIRM,)):
        with pytest.raises(ValueError):
            Confirmation(sweep, bad)
 
 
def test_mixing_naive_and_aware_datetimes_raises():
    sweep = make_sweep(BULL)
    aware = Candle(M5, datetime(2026, 1, 1, 11, 5, tzinfo=timezone.utc), 100.0, 105.0, 99.8, 104.8)
    with pytest.raises(TypeError):
        Confirmation(sweep, aware)
 
 
def test_confirmation_is_known_only_once_its_candle_has_closed():
    c = Confirmation(make_sweep(BULL), D_CONFIRM)
    assert c.is_known_at(c.candle_time) is False
    assert c.is_known_at(c.known_at - timedelta(seconds=1)) is False
    assert c.is_known_at(c.known_at) is True
    assert c.is_known_at(c.known_at + timedelta(days=5)) is True
    for bad in (None, "now", 5, c.known_at.date()):
        with pytest.raises(TypeError):
            c.is_known_at(bad)
    with pytest.raises(TypeError):
        c.is_known_at(datetime(2027, 1, 1, tzinfo=timezone.utc))
 
 
def test_confirmation_is_immutable_and_value_comparable():
    sweep = make_sweep(BULL)
    a, b = Confirmation(sweep, D_CONFIRM), Confirmation(make_sweep(BULL), ohlc(12, 100.0, 105.0, 99.8, 104.8))
    assert a is not b and a == b and a.identity == b.identity
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.candle = D_CONFIRM_BELOW
    for name in ("zone", "zone_type", "candle_class", "candle_time", "known_at", "identity"):
        with pytest.raises(AttributeError):
            setattr(a, name, None)
 
 
# ------------------------------------------------------------------ Invalidation
@pytest.mark.parametrize("direction, candle", [(BULL, D_WRONG_BOUNDARY), (BEAR, S_WRONG_BOUNDARY)])
def test_invalidation_by_a_close_beyond_the_boundary(direction, candle):
    sweep = make_sweep(direction)
    inv = Invalidation(sweep, candle, BOUNDARY)
    assert inv.sweep is sweep and inv.candle is candle and inv.reason is BOUNDARY
    assert inv.zone is sweep.zone and inv.zone_type is sweep.zone_type
    assert inv.candle_class is WRONG_CLASS_FOR[sweep.zone_type]
    assert inv.candle_time == candle.open_time and inv.known_at == candle.close_time
    assert inv.identity == sweep.identity
 
 
@pytest.mark.parametrize("direction, candle", [(BULL, D_WRONG), (BEAR, S_WRONG)])
def test_invalidation_by_a_second_wrong_direction_candle_inside_the_zone(direction, candle):
    inv = Invalidation(make_sweep(direction), candle, SECOND)
    assert inv.reason is SECOND
 
 
@pytest.mark.parametrize("direction, wick, equal", [(BULL, D_WRONG_WICK, D_WRONG_EQUAL), (BEAR, S_WRONG_WICK, S_WRONG_EQUAL)])
def test_a_wick_or_an_exact_boundary_close_is_not_a_boundary_invalidation(direction, wick, equal):
    sweep = make_sweep(direction)
    for c in (wick, equal):
        with pytest.raises(ValueError):
            Invalidation(sweep, c, BOUNDARY)
        assert Invalidation(sweep, c, SECOND).reason is SECOND
 
 
@pytest.mark.parametrize("direction, candle", [(BULL, D_WRONG_BOUNDARY), (BEAR, S_WRONG_BOUNDARY)])
def test_when_both_reasons_apply_the_boundary_label_is_required(direction, candle):
    sweep = make_sweep(direction)
    with pytest.raises(ValueError):
        Invalidation(sweep, candle, SECOND)
    assert Invalidation(sweep, candle, BOUNDARY).reason is BOUNDARY
 
 
def test_only_a_wrong_direction_decisive_candle_can_invalidate():
    for direction, confirming in ((BULL, D_CONFIRM), (BEAR, S_CONFIRM)):
        sweep = make_sweep(direction)
        for reason in InvalidationReason:
            with pytest.raises(ValueError):
                Invalidation(sweep, confirming, reason)
    neutral, zero = ohlc(12, 100.0, 104.0, 99.0, 101.0), ohlc(12, 100.0, 100.0, 100.0, 100.0)
    for direction in (BULL, BEAR):
        for c in (neutral, zero):
            with pytest.raises(ValueError):
                Invalidation(make_sweep(direction), c, SECOND)
 
 
def test_the_sweep_candle_itself_may_invalidate():
    demand = make_sweep(BULL, ohlc(11, 99.5, 99.6, 97.0, 97.1))                # bearish decisive, close below 99
    assert Invalidation(demand, demand.candle, BOUNDARY).candle is demand.candle
    supply = make_sweep(BEAR, ohlc(11, 110.2, 112.0, 110.1, 111.9))            # bullish decisive, close above 110
    assert Invalidation(supply, supply.candle, BOUNDARY).candle is supply.candle
 
 
def test_invalidation_stores_exactly_three_fields():
    assert [f.name for f in dataclasses.fields(Invalidation)] == ["sweep", "candle", "reason"]
 
 
def test_invalidation_rejects_wrong_types_timeframes_and_earlier_candles():
    sweep = make_sweep(BULL)
    for bad in (None, "r", 5, "CLOSED_BEYOND_ZONE_BOUNDARY"):
        with pytest.raises(ValueError):
            Invalidation(sweep, D_WRONG, bad)
    for bad in (None, "sweep", 5):
        with pytest.raises(ValueError):
            Invalidation(bad, D_WRONG, SECOND)
    for bad in (None, "candle", 5):
        with pytest.raises(ValueError):
            Invalidation(sweep, bad, SECOND)
    with pytest.raises(ValueError):
        Invalidation(sweep, ohlc(12, 104.8, 105.0, 100.0, 100.2, tf=M15), SECOND)
    with pytest.raises(ValueError):
        Invalidation(sweep, ohlc(10, 104.8, 105.0, 100.0, 100.2), SECOND)
 
 
def test_invalidation_is_known_only_once_its_candle_has_closed_and_is_immutable():
    inv = Invalidation(make_sweep(BULL), D_WRONG, SECOND)
    assert inv.is_known_at(inv.known_at - timedelta(seconds=1)) is False
    assert inv.is_known_at(inv.known_at) is True
    with pytest.raises(TypeError):
        inv.is_known_at("now")
    with pytest.raises(dataclasses.FrozenInstanceError):
        inv.reason = BOUNDARY
 
 
# ------------------------------------------------------------------ scope
def test_models_have_no_setup_entry_sl_tp_risk_or_filter_logic():
    sweep = make_sweep(BULL)
    for obj in (Confirmation(sweep, D_CONFIRM), Invalidation(sweep, D_WRONG, SECOND)):
        for attr in ("setup", "entry", "sl", "tp", "stop_loss", "take_profit", "risk", "news", "execute",
                     "choch", "tolerance", "distance", "atr", "displacement", "session", "expire", "expired",
                     "timeout", "consumed", "active", "valid", "price", "zone_high", "zone_low",
                     "swept_price", "liquidity_price", "entry_price"):
            assert not hasattr(obj, attr), attr
    for name in ("ConfirmationEngine", "SweepEngine", "ZoneEngine", "LiquidityEngine", "PivotEngine",
                 "BOSEngine", "CHOCHEngine", "StructureEngine", "SwingEngine"):
        assert not hasattr(confirmation_module, name)
