import dataclasses
from datetime import datetime, timedelta
 
import pytest
 
from koffie.strategy.models.candle import Candle, Timeframe
 
T0 = datetime(2026, 1, 1, 10, 0)
M5 = Timeframe.M5
 
 
def mk(o=10, h=12, l=8, c=11, tf=M5, t=T0, v=0.0):
    return Candle(tf, t, o, h, l, c, v)
 
 
def test_timeframe_durations():
    assert Timeframe.M5.duration == timedelta(minutes=5)
    assert Timeframe.M15.duration == timedelta(minutes=15)
    assert Timeframe.H1.duration == timedelta(hours=1)
 
 
def test_close_time_per_timeframe():
    for tf in Timeframe:
        assert mk(tf=tf).close_time == T0 + tf.duration
 
 
def test_closed_only_after_full_timeframe_elapsed():
    cd = mk()
    assert not cd.is_closed_at(T0)
    assert not cd.is_closed_at(T0 + timedelta(minutes=4, seconds=59))
    assert cd.is_closed_at(T0 + timedelta(minutes=5))
    assert cd.is_closed_at(T0 + timedelta(minutes=6))
 
 
def test_h1_and_m15_not_closed_early():
    assert not mk(tf=Timeframe.M15).is_closed_at(T0 + timedelta(minutes=14))
    assert mk(tf=Timeframe.M15).is_closed_at(T0 + timedelta(minutes=15))
    assert not mk(tf=Timeframe.H1).is_closed_at(T0 + timedelta(minutes=59))
    assert mk(tf=Timeframe.H1).is_closed_at(T0 + timedelta(hours=1))
 
 
def test_range_and_body():
    cd = mk(o=10, h=12, l=8, c=11)
    assert cd.range == 4
    assert cd.body == 1
    assert mk(o=11, h=12, l=8, c=10).body == 1   # bearish: abs()
 
 
def test_direction():
    assert mk(o=10, c=11).is_bullish and not mk(o=10, c=11).is_bearish
    assert mk(o=11, c=10).is_bearish and not mk(o=11, c=10).is_bullish
    d = mk(o=10, c=10)
    assert d.is_doji and not d.is_bullish and not d.is_bearish
 
 
def test_zero_range_candle_is_valid():
    cd = mk(o=10, h=10, l=10, c=10)
    assert cd.range == 0 and cd.body == 0 and cd.is_doji
 
 
def test_rejects_inconsistent_ohlc():
    with pytest.raises(ValueError):
        mk(h=7, l=8)                # high < low
    with pytest.raises(ValueError):
        mk(o=13, h=12, l=8, c=11)   # open above high
    with pytest.raises(ValueError):
        mk(o=10, h=12, l=8, c=13)   # close above high
    with pytest.raises(ValueError):
        mk(o=7, h=12, l=8, c=11)    # open below low
    with pytest.raises(ValueError):
        mk(o=10, h=12, l=8, c=7)    # close below low
 
 
def test_rejects_nan_and_inf():
    nan, inf = float("nan"), float("inf")
    for kwargs in ({"o": nan}, {"h": inf}, {"l": -inf}, {"c": nan}):
        with pytest.raises(ValueError):
            mk(**kwargs)
 
 
def test_volume_validation():
    assert mk(v=0).volume == 0 and mk(v=123.5).volume == 123.5
    with pytest.raises(ValueError):
        mk(v=-1)
    with pytest.raises(ValueError):
        mk(v=float("nan"))
 
 
def test_rejects_wrong_types():
    with pytest.raises(ValueError):
        Candle("5M", T0, 10, 12, 8, 11)
    with pytest.raises(ValueError):
        Candle(M5, "2026-01-01", 10, 12, 8, 11)
 
 
def test_candle_is_immutable_and_value_equal():
    cd = mk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cd.high = 99
    assert mk() == mk()
    assert mk() != mk(c=10)