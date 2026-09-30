import dataclasses
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.models.bos as bos_module
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.swing import Swing, SwingType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
HIGH, LOW = SwingType.HIGH, SwingType.LOW
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
 
 
def sw(kind, price, tf=M5, ct=T0, seq=0):
    return Swing(kind, tf, float(price), ct, ct + 2 * tf.duration, seq)
 
 
def bos(direction=BULL, swing=None, tf=M5, ct=None, close=None):
    if swing is None:
        swing = sw(HIGH if direction is BULL else LOW, 100, tf)
    if ct is None:
        ct = T0 + 10 * tf.duration
    if close is None:
        close = 101 if direction is BULL else 99
    return BOS(tf, direction, swing, ct, close)
 
 
# ------------------------------------------------------------- enum / fields
def test_direction_enum_has_exactly_bullish_and_bearish():
    assert {d.name for d in BOSDirection} == {"BULLISH", "BEARISH"}
 
 
def test_model_has_exactly_the_specified_fields():
    assert [f.name for f in dataclasses.fields(BOS)] == [
        "timeframe", "direction", "broken_swing", "candle_time", "close_price"]
 
 
def test_model_is_close_based_and_has_no_wick_fields():
    names = {f.name for f in dataclasses.fields(BOS)}
    assert not names & {"high", "low", "wick", "candle_high", "candle_low"}
 
 
# ------------------------------------------------------------- construction
def test_bullish_bos_holds_its_data():
    swing = sw(HIGH, 100)
    b = BOS(M5, BULL, swing, T0 + timedelta(hours=1), 100.5)
    assert b.timeframe is M5 and b.direction is BULL
    assert b.broken_swing is swing
    assert b.candle_time == T0 + timedelta(hours=1) and b.close_price == 100.5
 
 
def test_bearish_bos_holds_its_data():
    swing = sw(LOW, 100)
    b = BOS(M5, BEAR, swing, T0 + timedelta(hours=1), 99.5)
    assert b.direction is BEAR and b.broken_swing is swing
 
 
def test_confirmed_at_is_the_breaking_candles_close_time():
    for tf in (M5, M15, H1):
        b = bos(tf=tf)
        assert b.confirmed_at == b.candle_time + tf.duration
 
 
def test_integer_close_price_is_accepted():
    assert bos(close=101).close_price == 101
 
 
# ------------------------------------------------------------- direction vs swing type
def test_bullish_bos_must_break_a_swing_high():
    with pytest.raises(ValueError):
        BOS(M5, BULL, sw(LOW, 100), T0 + timedelta(hours=1), 101)
 
 
def test_bearish_bos_must_break_a_swing_low():
    with pytest.raises(ValueError):
        BOS(M5, BEAR, sw(HIGH, 100), T0 + timedelta(hours=1), 99)
 
 
# ------------------------------------------------------------- the close must be beyond the level
def test_bullish_close_must_be_strictly_above_the_swing_high():
    for bad in (100.0, 99.99, 50, -1):
        with pytest.raises(ValueError):
            bos(BULL, close=bad)
    assert bos(BULL, close=100.0001).close_price == 100.0001
 
 
def test_bearish_close_must_be_strictly_below_the_swing_low():
    for bad in (100.0, 100.01, 150, 1e9):
        with pytest.raises(ValueError):
            bos(BEAR, close=bad)
    assert bos(BEAR, close=99.9999).close_price == 99.9999
 
 
# ------------------------------------------------------------- timeframe and causality
def test_broken_swing_must_share_the_timeframe():
    for stf in (M5, M15, H1):
        for btf in (M5, M15, H1):
            swing = sw(HIGH, 100, tf=stf)
            if stf is btf:
                BOS(btf, BULL, swing, T0 + 10 * btf.duration, 101)
            else:
                with pytest.raises(ValueError):
                    BOS(btf, BULL, swing, T0 + 10 * btf.duration, 101)
 
 
def test_swing_confirmed_before_the_break_is_allowed():
    swing = sw(HIGH, 100)                                  # confirmed at T0 + 10 min
    BOS(M5, BULL, swing, T0 + timedelta(hours=1), 101)
 
 
def test_swing_confirmed_by_the_same_candle_is_allowed():
    swing = sw(HIGH, 100)                                  # confirmed at T0 + 10 min
    breaking_open = swing.confirmed_at - M5.duration       # candle closing exactly then
    b = BOS(M5, BULL, swing, breaking_open, 101)
    assert b.confirmed_at == swing.confirmed_at
 
 
def test_swing_confirmed_after_the_break_is_rejected():
    swing = sw(HIGH, 100)
    too_early = swing.confirmed_at - 2 * M5.duration       # BOS confirmed before the swing
    with pytest.raises(ValueError):
        BOS(M5, BULL, swing, too_early, 101)
 
 
# ------------------------------------------------------------- datetimes
def test_rejects_mixing_naive_and_aware_datetimes():
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        BOS(M5, BULL, sw(HIGH, 100), aware, 101)            # naive swing, aware candle
    aware_swing = sw(HIGH, 100, ct=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        BOS(M5, BULL, aware_swing, T0 + timedelta(hours=1), 101)
 
 
def test_timezone_aware_bos_works():
    t = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    b = BOS(M5, BULL, sw(HIGH, 100, ct=t), t + timedelta(hours=1), 101)
    assert b.confirmed_at == t + timedelta(hours=1, minutes=5)
 
 
# ------------------------------------------------------------- type validation
def test_rejects_wrong_types():
    swing = sw(HIGH, 100)
    ct = T0 + timedelta(hours=1)
    with pytest.raises(ValueError):
        BOS("5M", BULL, swing, ct, 101)
    with pytest.raises(ValueError):
        BOS(M5, "BULLISH", swing, ct, 101)
    with pytest.raises(ValueError):
        BOS(M5, BULL, "swing", ct, 101)
    with pytest.raises(ValueError):
        BOS(M5, BULL, swing, "2026-01-01", 101)
    for bad in ("101", None, True, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            BOS(M5, BULL, swing, ct, bad)
 
 
def test_rejects_swing_lookalikes():
    class Lookalike:
        swing_type, timeframe, price = HIGH, M5, 100.0
        candle_time, confirmed_at, sequence = T0, T0 + timedelta(minutes=10), 0
 
    with pytest.raises(ValueError):
        BOS(M5, BULL, Lookalike(), T0 + timedelta(hours=1), 101)
 
 
# ------------------------------------------------------------- immutability / equality
def test_bos_is_immutable_hashable_and_value_equal():
    b = bos()
    for name, value in (("close_price", 200.0), ("direction", BEAR), ("timeframe", M15)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(b, name, value)
    assert bos() == bos() and hash(bos()) == hash(bos())
    assert bos() != bos(close=102)
    assert len({bos(), bos(), bos(close=102)}) == 2
 
 
# ------------------------------------------------------------- compatibility with Candle / Swing
def test_can_be_built_from_a_real_closed_candle_and_swing():
    for tf in (M5, M15, H1):
        swing = sw(HIGH, 100, tf=tf)
        candle = Candle(tf, T0 + 10 * tf.duration, 99, 103, 98, 102)   # closes above 100
        b = BOS(candle.timeframe, BULL, swing, candle.open_time, candle.close)
        assert b.confirmed_at == candle.close_time
        assert b.close_price == candle.close
 
 
def test_a_wick_above_the_level_with_a_close_below_is_not_a_bos():
    swing = sw(HIGH, 100)
    candle = Candle(M5, T0 + timedelta(hours=1), 98, 103, 97, 99)       # wick to 103, close 99
    with pytest.raises(ValueError):
        BOS(M5, BULL, swing, candle.open_time, candle.close)
 
 
def test_a_wick_below_the_level_with_a_close_above_is_not_a_bearish_bos():
    swing = sw(LOW, 100)
    candle = Candle(M5, T0 + timedelta(hours=1), 102, 104, 96, 101)     # wick to 96, close 101
    with pytest.raises(ValueError):
        BOS(M5, BEAR, swing, candle.open_time, candle.close)
 
 
# ------------------------------------------------------------- scope
def test_model_contains_no_detection_or_other_strategy_logic():
    for name in ("BOSEngine", "detect_bos", "process_candle", "CHOCH", "StructureEngine",
                 "Zone", "Liquidity"):
        assert not hasattr(bos_module, name)
    b = bos()
    for attr in ("choch", "zone", "pivot", "entry", "sl", "tp", "state", "update", "process"):
        assert not hasattr(b, attr)