import dataclasses
from datetime import datetime, timedelta, timezone

import pytest

import koffie.strategy.models.choch as choch_module
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.choch import CHOCH, CHOCHDirection
from koffie.strategy.models.structure import StructureSnapshot, StructureState
from koffie.strategy.models.swing import Swing, SwingType

T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
HIGH, LOW = SwingType.HIGH, SwingType.LOW
BEAR, BULL = CHOCHDirection.BEARISH, CHOCHDirection.BULLISH
S = StructureState


def sw(kind, price, tf=M5, ct=T0, seq=0):
    return Swing(kind, tf, float(price), ct, ct + 2 * tf.duration, seq)


def choch(direction=BEAR, swing=None, tf=M5, ct=None, price=None):
    if swing is None:
        swing = sw(LOW if direction is BEAR else HIGH, 100, tf)
    if ct is None:
        ct = T0 + 10 * tf.duration
    if price is None:
        price = 99 if direction is BEAR else 101
    return CHOCH(tf, direction, swing, ct, price)


# ------------------------------------------------------------- enum / fields
def test_direction_enum_has_exactly_bearish_and_bullish():
    assert {d.name for d in CHOCHDirection} == {"BEARISH", "BULLISH"}


def test_model_has_exactly_the_specified_fields():
    assert [f.name for f in dataclasses.fields(CHOCH)] == [
        "timeframe", "direction", "protected_swing", "candle_time", "break_price"]


# ------------------------------------------------------------- construction
def test_bearish_choch_holds_its_data():
    swing = sw(LOW, 100)
    c = CHOCH(M5, BEAR, swing, T0 + timedelta(hours=1), 98.5)
    assert c.timeframe is M5 and c.direction is BEAR
    assert c.protected_swing is swing
    assert c.candle_time == T0 + timedelta(hours=1) and c.break_price == 98.5


def test_bullish_choch_holds_its_data():
    swing = sw(HIGH, 100)
    c = CHOCH(M5, BULL, swing, T0 + timedelta(hours=1), 101.5)
    assert c.direction is BULL and c.protected_swing is swing


def test_confirmed_at_is_the_breaking_candles_close_time():
    for tf in (M5, M15, H1):
        c = choch(tf=tf)
        assert c.confirmed_at == c.candle_time + tf.duration


def test_prior_state_is_the_trend_that_was_broken():
    assert choch(BEAR).prior_state is S.BULLISH
    assert choch(BULL).prior_state is S.BEARISH


def test_integer_break_price_is_accepted():
    assert choch(price=99).break_price == 99


# ------------------------------------------------------------- direction vs swing type
def test_bearish_choch_must_break_a_protected_low():
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, sw(HIGH, 100), T0 + timedelta(hours=1), 99)


def test_bullish_choch_must_break_a_protected_high():
    with pytest.raises(ValueError):
        CHOCH(M5, BULL, sw(LOW, 100), T0 + timedelta(hours=1), 101)


# ------------------------------------------------------------- strictly beyond the level
def test_bearish_break_price_must_be_strictly_below_the_protected_low():
    for bad in (100.0, 100.01, 150, 1e9):
        with pytest.raises(ValueError):
            choch(BEAR, price=bad)
    assert choch(BEAR, price=99.9999).break_price == 99.9999


def test_bullish_break_price_must_be_strictly_above_the_protected_high():
    for bad in (100.0, 99.99, 50, -1):
        with pytest.raises(ValueError):
            choch(BULL, price=bad)
    assert choch(BULL, price=100.0001).break_price == 100.0001


# ------------------------------------------------------------- the wick counts
def test_a_wick_through_the_protected_low_is_enough_even_if_the_close_is_back_inside():
    swing = sw(LOW, 100)
    candle = Candle(M5, T0 + timedelta(hours=1), 102, 104, 96, 103)   # low 96 < 100, close 103
    c = CHOCH(M5, BEAR, swing, candle.open_time, candle.low)
    assert c.break_price == 96 and candle.close > swing.price


def test_a_wick_through_the_protected_high_is_enough_even_if_the_close_is_back_inside():
    swing = sw(HIGH, 100)
    candle = Candle(M5, T0 + timedelta(hours=1), 98, 104, 97, 99)     # high 104 > 100, close 99
    c = CHOCH(M5, BULL, swing, candle.open_time, candle.high)
    assert c.break_price == 104 and candle.close < swing.price


def test_a_candle_that_only_touches_the_level_is_not_a_choch():
    swing_low = sw(LOW, 100)
    touch_low = Candle(M5, T0 + timedelta(hours=1), 102, 104, 100, 103)   # low == level
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, swing_low, touch_low.open_time, touch_low.low)
    swing_high = sw(HIGH, 100)
    touch_high = Candle(M5, T0 + timedelta(hours=1), 98, 100, 97, 99)     # high == level
    with pytest.raises(ValueError):
        CHOCH(M5, BULL, swing_high, touch_high.open_time, touch_high.high)


# ------------------------------------------------------------- timeframe and causality
def test_protected_swing_must_share_the_timeframe():
    for stf in (M5, M15, H1):
        for ctf in (M5, M15, H1):
            swing = sw(LOW, 100, tf=stf)
            if stf is ctf:
                CHOCH(ctf, BEAR, swing, T0 + 10 * ctf.duration, 99)
            else:
                with pytest.raises(ValueError):
                    CHOCH(ctf, BEAR, swing, T0 + 10 * ctf.duration, 99)


def test_swing_confirmed_before_the_break_is_allowed():
    CHOCH(M5, BEAR, sw(LOW, 100), T0 + timedelta(hours=1), 99)


def test_swing_confirmed_by_the_same_candle_is_allowed():
    swing = sw(LOW, 100)
    breaking_open = swing.confirmed_at - M5.duration
    c = CHOCH(M5, BEAR, swing, breaking_open, 99)
    assert c.confirmed_at == swing.confirmed_at


def test_swing_confirmed_after_the_break_is_rejected():
    swing = sw(LOW, 100)
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, swing, swing.confirmed_at - 2 * M5.duration, 99)


# ------------------------------------------------------------- datetimes
def test_rejects_mixing_naive_and_aware_datetimes():
    aware = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, sw(LOW, 100), aware, 99)
    aware_swing = sw(LOW, 100, ct=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc))
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, aware_swing, T0 + timedelta(hours=1), 99)


def test_timezone_aware_choch_works():
    t = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    c = CHOCH(M5, BEAR, sw(LOW, 100, ct=t), t + timedelta(hours=1), 99)
    assert c.confirmed_at == t + timedelta(hours=1, minutes=5)


# ------------------------------------------------------------- type validation
def test_rejects_wrong_types():
    swing = sw(LOW, 100)
    ct = T0 + timedelta(hours=1)
    with pytest.raises(ValueError):
        CHOCH("5M", BEAR, swing, ct, 99)
    with pytest.raises(ValueError):
        CHOCH(M5, "BEARISH", swing, ct, 99)
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, "swing", ct, 99)
    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, swing, "2026-01-01", 99)
    for bad in ("99", None, True, float("nan"), float("-inf")):
        with pytest.raises(ValueError):
            CHOCH(M5, BEAR, swing, ct, bad)


def test_rejects_swing_lookalikes():
    class Lookalike:
        swing_type, timeframe, price = LOW, M5, 100.0
        candle_time, confirmed_at, sequence = T0, T0 + timedelta(minutes=10), 0

    with pytest.raises(ValueError):
        CHOCH(M5, BEAR, Lookalike(), T0 + timedelta(hours=1), 99)


# ------------------------------------------------------------- immutability / equality
def test_choch_is_immutable_hashable_and_value_equal():
    c = choch()
    for name, value in (("break_price", 1.0), ("direction", BULL), ("timeframe", M15)):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(c, name, value)
    assert choch() == choch() and hash(choch()) == hash(choch())
    assert choch() != choch(price=98)
    assert len({choch(), choch(), choch(price=98)}) == 2


# ------------------------------------------------------------- compatibility
def test_can_be_built_from_real_candles_and_swings_on_every_timeframe():
    for tf in (M5, M15, H1):
        low_swing = sw(LOW, 100, tf=tf)
        candle = Candle(tf, T0 + 10 * tf.duration, 101, 103, 97, 102)
        c = CHOCH(tf, BEAR, low_swing, candle.open_time, candle.low)
        assert c.confirmed_at == candle.close_time and c.break_price == candle.low
        high_swing = sw(HIGH, 100, tf=tf)
        candle2 = Candle(tf, T0 + 10 * tf.duration, 99, 103, 98, 99.5)
        c2 = CHOCH(tf, BULL, high_swing, candle2.open_time, candle2.high)
        assert c2.break_price == candle2.high


def test_protected_swing_matches_the_structure_snapshot_definition():
    ph, pl = sw(HIGH, 10, seq=0), sw(LOW, 4, seq=1, ct=T0 + M5.duration)
    ah, al = sw(HIGH, 12, seq=2, ct=T0 + 2 * M5.duration), sw(LOW, 6, seq=3, ct=T0 + 3 * M5.duration)
    bullish = StructureSnapshot(M5, S.BULLISH, ah, ph, al, pl)
    c = CHOCH(M5, BEAR, bullish.protected_swing, T0 + timedelta(hours=1), 5.5)
    assert c.protected_swing is al and c.prior_state is bullish.state

    ah2, ph2 = sw(HIGH, 8, seq=2, ct=T0 + 2 * M5.duration), sw(HIGH, 10, seq=0)
    al2, pl2 = sw(LOW, 3, seq=3, ct=T0 + 3 * M5.duration), sw(LOW, 4, seq=1, ct=T0 + M5.duration)
    bearish = StructureSnapshot(M5, S.BEARISH, ah2, ph2, al2, pl2)
    c2 = CHOCH(M5, BULL, bearish.protected_swing, T0 + timedelta(hours=1), 8.5)
    assert c2.protected_swing is ah2 and c2.prior_state is bearish.state


# ------------------------------------------------------------- scope
def test_model_contains_no_detection_or_other_strategy_logic():
    for name in ("CHOCHEngine", "detect_choch", "process_candle", "BOS", "StructureEngine",
                 "enter_revaluating", "Zone", "Liquidity"):
        assert not hasattr(choch_module, name)
    c = choch()
    for attr in ("bos", "zone", "pivot", "entry", "sl", "tp", "state", "update",
                 "process", "enter_revaluating", "trade", "signal"):
        assert not hasattr(c, attr)