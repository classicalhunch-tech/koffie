from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
 
import pytest
 
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import CandleClass, Timeframe
from koffie.strategy.models.pivot import (
    PIVOT_CANDLE_CLASS,
    BOSIdentity,
    Pivot,
    PivotOutcome,
    PivotStatus,
)
from koffie.strategy.models.swing import Swing, SwingType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
BD, RD = CandleClass.BULLISH_DECISIVE, CandleClass.BEARISH_DECISIVE
 
 
def make_bos(direction=BULL, tf=M5, seq=0, bos_open=None, close=112.0, swing_price=None):
    bos_open = bos_open if bos_open is not None else T0 + 10 * tf.duration
    kind = HIGH if direction is BULL else LOW
    price = swing_price if swing_price is not None else (close - 1.0 if direction is BULL else close + 1.0)
    swing = Swing(kind, tf, price, bos_open - 4 * tf.duration, bos_open - 2 * tf.duration, seq)
    return BOS(tf, direction, swing, bos_open, close)
 
 
def make_pivot(bos=None, offset=3, high=110.0, low=99.0, cls=None):
    bos = bos or make_bos()
    cls = cls or PIVOT_CANDLE_CLASS[bos.direction]
    return Pivot(bos, bos.candle_time - offset * bos.timeframe.duration, high, low, cls)
 
 
# ------------------------------------------------------------------ constants
def test_pivot_class_mapping_is_opposite_colour_decisive():
    assert PIVOT_CANDLE_CLASS[BULL] is RD
    assert PIVOT_CANDLE_CLASS[BEAR] is BD
    assert len(PIVOT_CANDLE_CLASS) == 2
 
 
def test_status_values():
    assert {s.value for s in PivotStatus} == {"FOUND", "NO_PIVOT_FOUND"}
 
 
# ------------------------------------------------------------------ BOSIdentity
def test_identity_is_derived_from_bos_fields():
    bos = make_bos(seq=7)
    ident = BOSIdentity.from_bos(bos)
    assert ident.timeframe is M5 and ident.direction is BULL
    assert ident.broken_swing_type is HIGH and ident.broken_swing_sequence == 7
    assert ident.broken_swing_candle_time == bos.broken_swing.candle_time
    assert ident.candle_time == bos.candle_time
 
 
def test_identity_is_equal_for_equal_bos_built_separately():
    a, b = make_bos(), make_bos()
    assert a is not b and a == b
    assert BOSIdentity.from_bos(a) == BOSIdentity.from_bos(b)
    assert hash(BOSIdentity.from_bos(a)) == hash(BOSIdentity.from_bos(b))
 
 
def test_identity_differs_when_any_identifying_field_differs():
    base = BOSIdentity.from_bos(make_bos())
    assert BOSIdentity.from_bos(make_bos(direction=BEAR)) != base
    assert BOSIdentity.from_bos(make_bos(seq=1)) != base
    assert BOSIdentity.from_bos(make_bos(bos_open=T0 + 11 * M5.duration)) != base
    assert BOSIdentity.from_bos(make_bos(tf=M15)) != base
 
 
def test_identity_ignores_prices_so_conflicts_are_detectable():
    a, b = make_bos(close=112.0), make_bos(close=113.0)
    assert a != b
    assert BOSIdentity.from_bos(a) == BOSIdentity.from_bos(b)
 
 
def test_identity_requires_a_bos():
    for bad in (None, "bos", object()):
        with pytest.raises(TypeError):
            BOSIdentity.from_bos(bad)
 
 
def test_identity_is_frozen():
    ident = BOSIdentity.from_bos(make_bos())
    with pytest.raises(FrozenInstanceError):
        ident.candle_time = T0
 
 
# ------------------------------------------------------------------ Pivot
def test_pivot_preserves_bos_range_and_class():
    bos = make_bos()
    p = make_pivot(bos, offset=3, high=110.0, low=99.0)
    assert p.bos is bos and p.high == 110.0 and p.low == 99.0
    assert p.candle_class is RD
    assert p.candle_time == bos.candle_time - 3 * M5.duration
 
 
def test_pivot_derives_direction_timeframe_and_confirmed_at_from_bos():
    for direction in (BULL, BEAR):
        bos = make_bos(direction=direction)
        p = make_pivot(bos)
        assert p.direction is direction and p.timeframe is M5
        assert p.confirmed_at == bos.confirmed_at
        assert p.candle_close_time == p.candle_time + M5.duration
 
 
def test_pivot_class_must_be_opposite_colour_decisive():
    bull_bos, bear_bos = make_bos(direction=BULL), make_bos(direction=BEAR)
    for bad in (BD, CandleClass.NEUTRAL, CandleClass.ZERO_RANGE_ANOMALY):
        with pytest.raises(ValueError):
            make_pivot(bull_bos, cls=bad)
    for bad in (RD, CandleClass.NEUTRAL, CandleClass.ZERO_RANGE_ANOMALY):
        with pytest.raises(ValueError):
            make_pivot(bear_bos, cls=bad)
    assert make_pivot(bear_bos, cls=BD).candle_class is BD
 
 
def test_pivot_must_be_strictly_before_the_bos_candle():
    bos = make_bos()
    with pytest.raises(ValueError):
        Pivot(bos, bos.candle_time, 110.0, 99.0, RD)                       # the BOS candle itself
    with pytest.raises(ValueError):
        Pivot(bos, bos.candle_time + M5.duration, 110.0, 99.0, RD)         # later than the BOS
    assert make_pivot(bos, offset=1).candle_time < bos.candle_time
 
 
def test_pivot_validates_fields():
    bos = make_bos()
    t = bos.candle_time - M5.duration
    for bad_bos in (None, "bos", object()):
        with pytest.raises(ValueError):
            Pivot(bad_bos, t, 110.0, 99.0, RD)
    with pytest.raises(ValueError):
        Pivot(bos, "t", 110.0, 99.0, RD)
    for bad in (True, "1", None, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            Pivot(bos, t, bad, 99.0, RD)
        with pytest.raises(ValueError):
            Pivot(bos, t, 110.0, bad, RD)
    with pytest.raises(ValueError):
        Pivot(bos, t, 99.0, 110.0, RD)                                     # high < low
    with pytest.raises(ValueError):
        Pivot(bos, t, 110.0, 99.0, "BEARISH_DECISIVE")
    assert Pivot(bos, t, 100.0, 100.0, RD).high == 100.0                   # equal high/low is a valid record
 
 
def test_pivot_rejects_naive_aware_mix():
    bos = make_bos()
    aware = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        Pivot(bos, aware, 110.0, 99.0, RD)
 
 
def test_pivot_is_frozen_and_hashable():
    p = make_pivot()
    with pytest.raises(FrozenInstanceError):
        p.high = 1.0
    assert p == make_pivot(p.bos) and hash(p) == hash(make_pivot(p.bos))
 
 
# ------------------------------------------------------------------ PivotOutcome
def test_found_outcome_requires_a_pivot_of_the_same_bos():
    bos = make_bos()
    pivot = make_pivot(bos)
    out = PivotOutcome(bos, PivotStatus.FOUND, pivot)
    assert out.is_found and out.pivot is pivot and out.bos is bos
    assert out.identity == BOSIdentity.from_bos(bos) and out.confirmed_at == bos.confirmed_at
    with pytest.raises(ValueError):
        PivotOutcome(bos, PivotStatus.FOUND)
    with pytest.raises(ValueError):
        PivotOutcome(bos, PivotStatus.FOUND, None)
    with pytest.raises(ValueError):
        PivotOutcome(bos, PivotStatus.FOUND, make_pivot(make_bos(seq=5)))  # pivot of another BOS
 
 
def test_no_pivot_found_outcome_must_not_carry_a_pivot():
    bos = make_bos()
    out = PivotOutcome(bos, PivotStatus.NO_PIVOT_FOUND)
    assert not out.is_found and out.pivot is None
    with pytest.raises(ValueError):
        PivotOutcome(bos, PivotStatus.NO_PIVOT_FOUND, make_pivot(bos))
 
 
def test_outcome_validates_types_and_is_frozen():
    bos = make_bos()
    with pytest.raises(ValueError):
        PivotOutcome("bos", PivotStatus.NO_PIVOT_FOUND)
    with pytest.raises(ValueError):
        PivotOutcome(bos, "FOUND", make_pivot(bos))
    out = PivotOutcome(bos, PivotStatus.NO_PIVOT_FOUND)
    with pytest.raises(FrozenInstanceError):
        out.status = PivotStatus.FOUND
 
 
def test_models_hold_no_trading_logic():
    import koffie.strategy.models.pivot as module
    for name in ("StructureEngine", "BOSEngine", "CHOCH", "SwingEngine", "Zone", "Liquidity"):
        assert not hasattr(module, name)