import dataclasses
from datetime import datetime, timedelta, timezone
 
import pytest
 
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.swing import Swing, SwingType
 
T0 = datetime(2026, 1, 1, 10, 0)
 
 
def mk(swing_type=SwingType.HIGH, tf=Timeframe.M5, price=12.0, candle_time=T0,
       confirmed_at=None, sequence=0):
    if confirmed_at is None:
        confirmed_at = candle_time + 2 * tf.duration
    return Swing(swing_type, tf, price, candle_time, confirmed_at, sequence)
 
 
# --- types ---------------------------------------------------------------
def test_only_high_and_low_swing_types_exist():
    assert {t.name for t in SwingType} == {"HIGH", "LOW"}
    assert SwingType.HIGH.value == "HIGH" and SwingType.LOW.value == "LOW"
 
 
# --- construction ----------------------------------------------------------
def test_holds_all_required_fields():
    s = mk(SwingType.LOW, Timeframe.M15, 7.5, T0, T0 + timedelta(minutes=30), 4)
    assert s.swing_type is SwingType.LOW
    assert s.timeframe is Timeframe.M15
    assert s.price == 7.5
    assert s.candle_time == T0
    assert s.confirmed_at == T0 + timedelta(minutes=30)
    assert s.sequence == 4
 
 
def test_direction_helpers():
    assert mk(SwingType.HIGH).is_high and not mk(SwingType.HIGH).is_low
    assert mk(SwingType.LOW).is_low and not mk(SwingType.LOW).is_high
 
 
def test_model_has_exactly_the_specified_fields():
    names = [f.name for f in dataclasses.fields(Swing)]
    assert names == ["swing_type", "timeframe", "price", "candle_time",
                     "confirmed_at", "sequence"]
 
 
def test_integer_price_allowed_and_negative_price_allowed():
    assert mk(price=10).price == 10
    assert mk(price=-3.2).price == -3.2   # no rule forbids it; instruments vary
 
 
# --- confirmed, not a candidate ---------------------------------------------
def test_confirmation_time_is_required():
    with pytest.raises(TypeError):
        Swing(SwingType.HIGH, Timeframe.M5, 12.0, T0, sequence=0)
 
 
def test_confirmed_at_may_equal_minimum_close_of_confirmation_candle():
    for tf in Timeframe:
        s = mk(tf=tf, confirmed_at=T0 + 2 * tf.duration)
        assert s.confirmed_at == T0 + 2 * tf.duration
 
 
def test_confirmed_at_may_be_later_across_a_gap():
    s = mk(confirmed_at=T0 + timedelta(days=2))
    assert s.confirmed_at > T0 + 2 * Timeframe.M5.duration
 
 
def test_confirmed_at_before_confirmation_candle_close_is_rejected():
    for tf in Timeframe:
        with pytest.raises(ValueError):
            mk(tf=tf, confirmed_at=T0 + 2 * tf.duration - timedelta(seconds=1))
        with pytest.raises(ValueError):
            mk(tf=tf, confirmed_at=T0 + tf.duration)   # only the swing candle's close
        with pytest.raises(ValueError):
            mk(tf=tf, confirmed_at=T0)                  # same time as swing candle
        with pytest.raises(ValueError):
            mk(tf=tf, confirmed_at=T0 - timedelta(hours=1))
 
 
def test_compatible_with_candle_model_contiguous_candles():
    for tf in Timeframe:
        n = Candle(tf, T0, 10, 12, 8, 11)
        n_plus_1 = Candle(tf, n.close_time, 11, 11.5, 9, 10)
        s = Swing(SwingType.HIGH, tf, n.high, n.open_time, n_plus_1.close_time, 0)
        assert s.price == n.high
        assert s.candle_time == n.open_time
        assert s.confirmed_at == n_plus_1.close_time
        assert s.timeframe is n.timeframe
 
 
def test_compatible_with_candle_model_low_uses_candle_low():
    n = Candle(Timeframe.M5, T0, 10, 12, 8, 11)
    n1 = Candle(Timeframe.M5, n.close_time, 11, 13, 9, 12)
    s = Swing(SwingType.LOW, Timeframe.M5, n.low, n.open_time, n1.close_time, 0)
    assert s.price == 8
 
 
def test_compatible_with_candle_model_across_a_gap():
    n = Candle(Timeframe.H1, T0, 10, 12, 8, 11)
    n1 = Candle(Timeframe.H1, T0 + timedelta(days=2), 11, 11.5, 9, 10)
    s = Swing(SwingType.HIGH, Timeframe.H1, n.high, n.open_time, n1.close_time, 0)
    assert s.confirmed_at == n1.close_time
 
 
# --- validation ------------------------------------------------------------
def test_rejects_wrong_types():
    with pytest.raises(ValueError):
        mk(swing_type="HIGH")
    with pytest.raises(ValueError):
        mk(tf="5M", confirmed_at=T0 + timedelta(minutes=10))
    with pytest.raises(ValueError):
        mk(price="12")
    with pytest.raises(ValueError):
        mk(price=True)
    with pytest.raises(ValueError):
        mk(candle_time="2026-01-01", confirmed_at=T0 + timedelta(minutes=10))
    with pytest.raises(ValueError):
        Swing(SwingType.HIGH, Timeframe.M5, 12.0, T0, "later", 0)
 
 
def test_rejects_non_finite_price():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            mk(price=bad)
 
 
def test_sequence_validation():
    assert mk(sequence=0).sequence == 0
    assert mk(sequence=10_000).sequence == 10_000
    with pytest.raises(ValueError):
        mk(sequence=-1)
    with pytest.raises(ValueError):
        mk(sequence=1.0)
    with pytest.raises(ValueError):
        mk(sequence=True)
    with pytest.raises(ValueError):
        mk(sequence="0")
 
 
def test_rejects_mixing_naive_and_aware_datetimes():
    aware = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        Swing(SwingType.HIGH, Timeframe.M5, 12.0, T0, aware + timedelta(hours=1), 0)
    with pytest.raises(ValueError):
        Swing(SwingType.HIGH, Timeframe.M5, 12.0, aware, T0 + timedelta(hours=1), 0)
 
 
def test_timezone_aware_datetimes_work():
    t = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    s = Swing(SwingType.LOW, Timeframe.M5, 7.0, t, t + timedelta(minutes=10), 1)
    assert s.confirmed_at.tzinfo is timezone.utc
 
 
# --- immutability / equality ------------------------------------------------
def test_swing_is_immutable():
    s = mk()
    for field, value in (("price", 99.0), ("sequence", 5), ("swing_type", SwingType.LOW),
                         ("confirmed_at", T0), ("candle_time", T0), ("timeframe", Timeframe.H1)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(s, field, value)
 
 
def test_no_superseded_or_other_strategy_fields():
    s = mk()
    for attr in ("superseded", "trend", "bos", "choch", "zone", "liquidity", "brr"):
        assert not hasattr(s, attr)
 
 
def test_value_equality_and_hashing():
    assert mk() == mk()
    assert hash(mk()) == hash(mk())
    assert mk() != mk(sequence=1)
    assert mk() != mk(price=13.0)
    assert mk(SwingType.HIGH) != mk(SwingType.LOW)
    assert len({mk(), mk(), mk(sequence=1)}) == 2
 
 
def test_sequence_orders_chronologically():
    swings = [mk(sequence=i, candle_time=T0 + timedelta(minutes=5 * i)) for i in (2, 0, 1)]
    assert [s.sequence for s in sorted(swings, key=lambda s: s.sequence)] == [0, 1, 2]
 