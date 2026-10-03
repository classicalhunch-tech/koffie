"""Setup model tests (data only; the lifecycle is tested in test_setup_engine.py)."""
import dataclasses
from datetime import datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.models.setup as setup_model_module
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import BOSIdentity, Pivot
from koffie.strategy.models.setup import (
    TIE_BREAK_RANK, ConfirmationEventKey, Setup, SetupIdentity, SetupStatus, SetupTransition,
    is_touch, tie_break_order,
)
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = SetupStatus
 
 
def at(i, tf=M5):
    return T0 + i * tf.duration
 
 
def mk_swing(swing_type, price, i=0, seq=0, tf=M5):
    candle_time = T0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5, bos_i=10, broken_i=2):
    """Zone 99..110; a 5M zone is created (known) at slot 11."""
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=broken_i, seq=0, tf=tf)
    bos = BOS(tf, direction, broken, T0 + bos_i * tf.duration, 101.0 if direction is BULL else 99.0)
    return Zone(Pivot(bos, T0 + 5 * tf.duration, high, low, pivot_class))
 
 
def ohlc(i, o, h, l, c, tf=M5):
    return Candle(tf, at(i, tf), o, h, l, c)
 
 
# ------------------------------------------------------------------ SetupStatus
def test_status_has_exactly_the_locked_lifecycle_members():
    assert [s.value for s in SetupStatus] == [
        "NO_SETUP", "ZONE_TOUCHED", "WAITING_FOR_SWEEP", "SWEPT", "WAITING_FOR_CONFIRMATION",
        "CONFIRMED", "TRADE_CREATED", "INVALIDATED"]
 
 
def test_active_and_resolved_partition_the_real_setup_states():
    assert {s for s in S if s.is_active} == {S.ZONE_TOUCHED, S.WAITING_FOR_SWEEP, S.SWEPT, S.WAITING_FOR_CONFIRMATION}
    assert {s for s in S if s.is_resolved} == {S.CONFIRMED, S.TRADE_CREATED, S.INVALIDATED}
    assert not S.NO_SETUP.is_active and not S.NO_SETUP.is_resolved
    assert all(s.is_active != s.is_resolved for s in S if s is not S.NO_SETUP)
 
 
# ------------------------------------------------------------------ is_touch
def test_touch_is_a_close_inside_the_zone_with_inclusive_boundaries():
    for zone in (mk_zone(BULL), mk_zone(BEAR)):
        assert is_touch(zone, ohlc(11, 101, 104, 100, 102)) is True             # strictly inside
        assert is_touch(zone, ohlc(11, 100, 101, 99, 99)) is True               # close == zone.low
        assert is_touch(zone, ohlc(11, 105, 110.5, 104, 110)) is True           # close == zone.high
        assert is_touch(zone, ohlc(11, 100, 101, 98, 98.99)) is False           # just below the low
        assert is_touch(zone, ohlc(11, 105, 111, 104, 110.01)) is False         # just above the high
 
 
def test_a_wick_into_the_zone_with_the_close_outside_is_not_a_touch():
    zone = mk_zone(BULL)
    assert is_touch(zone, ohlc(11, 97.5, 100.0, 97.0, 98.5)) is False           # wick up into the zone, close below
    assert is_touch(zone, ohlc(11, 112.0, 113.0, 105.0, 111.0)) is False        # wick down into the zone, close above
    assert is_touch(zone, ohlc(11, 95.0, 115.0, 94.0, 96.0)) is False           # wick across the whole zone
 
 
def test_the_open_is_irrelevant_to_a_touch():
    zone = mk_zone(BULL)
    for o in (90.0, 99.0, 104.0, 110.0, 120.0):
        c = Candle(M5, at(11), o, max(o, 105.0), min(o, 100.0), 102.0)
        assert is_touch(zone, c) is True
    for o in (90.0, 104.0, 120.0):
        c = Candle(M5, at(11), o, max(o, 111.0), min(o, 98.0), 98.0)
        assert is_touch(zone, c) is False
 
 
def test_touch_validates_its_arguments():
    zone, candle = mk_zone(BULL), ohlc(11, 101, 104, 100, 102)
    for bad in (None, "zone", 5, zone.pivot):
        with pytest.raises(TypeError):
            is_touch(bad, candle)
    for bad in (None, "candle", 5, (1, 2)):
        with pytest.raises(TypeError):
            is_touch(zone, bad)
 
 
# ------------------------------------------------------------------ Setup
def test_setup_stores_exactly_two_fields():
    assert [f.name for f in dataclasses.fields(Setup)] == ["zone", "touch_candle"]
 
 
def test_setup_builds_and_derives_its_values():
    for direction, zt in ((BULL, ZoneType.DEMAND), (BEAR, ZoneType.SUPPLY)):
        zone, candle = mk_zone(direction), ohlc(11, 101, 104, 100, 102)
        s = Setup(zone, candle)
        assert s.zone is zone and s.touch_candle is candle and s.zone_type is zt
        assert s.touch_time == candle.open_time == at(11)
        assert s.known_at == candle.close_time == at(12)
        assert s.identity == SetupIdentity.from_parts(zone, candle)
 
 
def test_a_candle_opening_exactly_at_the_zone_creation_is_accepted():
    zone = mk_zone(BULL)
    assert zone.created_at == at(11)
    assert Setup(zone, ohlc(11, 101, 104, 100, 102)).touch_time == zone.created_at
 
 
def test_setup_rejects_a_candle_that_is_not_a_touch():
    zone = mk_zone(BULL)
    for c in (ohlc(11, 97.5, 100.0, 97.0, 98.5), ohlc(11, 112.0, 113.0, 105.0, 111.0)):
        with pytest.raises(ValueError):
            Setup(zone, c)
 
 
def test_setup_rejects_a_candle_before_the_zone_existed():
    zone = mk_zone(BULL)
    for i in (0, 5, 10):
        with pytest.raises(ValueError):
            Setup(zone, ohlc(i, 101, 104, 100, 102))
    with pytest.raises(ValueError):
        Setup(zone, Candle(M5, zone.created_at - timedelta(minutes=1), 101, 104, 100, 102))
 
 
def test_setup_rejects_non_5m_candles_and_wrong_types():
    zone = mk_zone(BULL)
    for tf in (M15, H1):
        with pytest.raises(ValueError):
            Setup(zone, Candle(tf, zone.created_at, 101, 104, 100, 102))
    for bad in (None, "zone", 5, zone.pivot):
        with pytest.raises(ValueError):
            Setup(bad, ohlc(11, 101, 104, 100, 102))
    for bad in (None, "candle", 5):
        with pytest.raises(ValueError):
            Setup(zone, bad)
 
 
def test_setup_rejects_mixed_naive_and_aware_datetimes():
    zone = mk_zone(BULL)
    aware = Candle(M5, datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc), 101, 104, 100, 102)
    with pytest.raises(TypeError):
        Setup(zone, aware)
 
 
def test_setup_is_known_only_once_the_touch_candle_has_closed():
    s = Setup(mk_zone(BULL), ohlc(11, 101, 104, 100, 102))
    assert s.is_known_at(s.touch_time) is False
    assert s.is_known_at(s.known_at - timedelta(seconds=1)) is False
    assert s.is_known_at(s.known_at) is True
    for bad in (None, "now", 5, s.known_at.date()):
        with pytest.raises(TypeError):
            s.is_known_at(bad)
    with pytest.raises(TypeError):
        s.is_known_at(datetime(2027, 1, 1, tzinfo=timezone.utc))
 
 
def test_setup_is_immutable_value_comparable_and_hashable():
    a = Setup(mk_zone(BULL), ohlc(11, 101, 104, 100, 102))
    b = Setup(mk_zone(BULL), ohlc(11, 101, 104, 100, 102))
    assert a is not b and a == b and hash(a) == hash(b) and a.identity == b.identity
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.touch_candle = ohlc(12, 101, 104, 100, 102)
    for name in ("zone_type", "touch_time", "known_at", "identity"):
        with pytest.raises(AttributeError):
            setattr(a, name, None)
 
 
def test_setup_holds_no_sweep_confirmation_status_or_prices():
    s = Setup(mk_zone(BULL), ohlc(11, 101, 104, 100, 102))
    for attr in ("sweep", "confirmation", "invalidation", "status", "state", "consumed", "entry", "sl", "tp",
                 "stop_loss", "take_profit", "risk", "news", "execute", "timeout", "expires_at", "price",
                 "zone_high", "zone_low", "swept_price", "trade"):
        assert not hasattr(s, attr), attr
 
 
# ------------------------------------------------------------------ identity
def test_identity_is_the_zone_identity_plus_the_touch_open_time():
    zone, candle = mk_zone(BULL), ohlc(12, 101, 104, 100, 102)
    ident = SetupIdentity.from_parts(zone, candle)
    assert ident.zone_identity == zone.identity and isinstance(ident.zone_identity, BOSIdentity)
    assert ident.touch_time == candle.open_time
 
 
def test_the_same_zone_has_distinct_identities_for_distinct_touches():
    zone = mk_zone(BULL)
    first = Setup(zone, ohlc(11, 101, 104, 100, 102))
    second = Setup(zone, ohlc(20, 101, 104, 100, 102))
    assert first.zone == second.zone and first.identity != second.identity
    assert first.identity.zone_identity == second.identity.zone_identity
 
 
def test_identity_differs_between_zones_and_is_value_based():
    candle = ohlc(15, 101, 104, 100, 102)
    a, b = mk_zone(BULL, bos_i=10, broken_i=2), mk_zone(BULL, bos_i=14, broken_i=3)
    assert SetupIdentity.from_parts(a, candle) != SetupIdentity.from_parts(b, candle)
    assert SetupIdentity.from_parts(mk_zone(BULL), candle) == SetupIdentity.from_parts(mk_zone(BULL), candle)
    assert hash(SetupIdentity.from_parts(a, candle)) == hash(SetupIdentity.from_parts(mk_zone(BULL, bos_i=10, broken_i=2), candle))
 
 
def test_identity_ignores_everything_but_the_touch_time():
    zone = mk_zone(BULL)
    a = SetupIdentity.from_parts(zone, ohlc(12, 101, 104, 100, 102))
    b = SetupIdentity.from_parts(zone, ohlc(12, 105, 110, 104, 109))
    assert a == b
 
 
def test_identity_validates_its_arguments():
    zone, candle = mk_zone(BULL), ohlc(12, 101, 104, 100, 102)
    for bad in (None, "zone", 5):
        with pytest.raises(TypeError):
            SetupIdentity.from_parts(bad, candle)
    for bad in (None, "candle", 5):
        with pytest.raises(TypeError):
            SetupIdentity.from_parts(zone, bad)
 
 
# ------------------------------------------------------------------ SetupTransition and ConfirmationEventKey
def test_transition_validates_and_is_frozen():
    t = SetupTransition(S.WAITING_FOR_SWEEP, at(12))
    assert t.status is S.WAITING_FOR_SWEEP and t.at == at(12)
    for status in S:
        if status is S.NO_SETUP:
            with pytest.raises(ValueError):
                SetupTransition(status, at(12))
        else:
            assert SetupTransition(status, at(12)).status is status
    for bad in ("WAITING_FOR_SWEEP", None, 5):
        with pytest.raises(ValueError):
            SetupTransition(bad, at(12))
    for bad in (None, "now", 5):
        with pytest.raises(ValueError):
            SetupTransition(S.SWEPT, bad)
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.status = S.SWEPT
 
 
def test_confirmation_event_key_is_a_value():
    ident = LiquidityLevel(mk_swing(LOW, 98.0)).identity
    a, b = ConfirmationEventKey(at(13), ident), ConfirmationEventKey(at(13), LiquidityLevel(mk_swing(LOW, 98.0)).identity)
    assert a == b and hash(a) == hash(b)
    assert a != ConfirmationEventKey(at(14), ident)
    assert a != ConfirmationEventKey(at(13), LiquidityLevel(mk_swing(LOW, 97.0, seq=1)).identity)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.candle_time = at(1)
 
 
# ------------------------------------------------------------------ tie-break
def test_the_tie_break_order_is_1h_then_15m_then_5m():
    assert TIE_BREAK_RANK == {H1: 0, M15: 1, M5: 2}
    s5 = Setup(mk_zone(BULL, tf=M5), Candle(M5, mk_zone(BULL, tf=M5).created_at, 101, 104, 100, 102))
    z15 = mk_zone(BULL, tf=M15)
    s15 = Setup(z15, Candle(M5, z15.created_at, 101, 104, 100, 102))
    z1h = mk_zone(BULL, tf=H1)
    s1h = Setup(z1h, Candle(M5, z1h.created_at, 101, 104, 100, 102))
    assert tie_break_order([s5, s15, s1h]) == (s1h, s15, s5)
    assert tie_break_order([s1h, s5, s15]) == (s1h, s15, s5)
    assert tie_break_order([]) == ()
 
 
def test_the_tie_break_is_stable_for_equal_timeframes_and_never_changes_the_input():
    a = Setup(mk_zone(BULL, bos_i=10, broken_i=2), ohlc(15, 101, 104, 100, 102))
    b = Setup(mk_zone(BULL, bos_i=14, broken_i=3), ohlc(15, 101, 104, 100, 102))
    items = [b, a]
    assert tie_break_order(items) == (b, a) and items == [b, a]
    assert isinstance(tie_break_order(items), tuple)
 
 
def test_the_tie_break_validates_its_items():
    s = Setup(mk_zone(BULL), ohlc(11, 101, 104, 100, 102))
    for bad in ([None], ["setup"], [s, 5], [s.zone]):
        with pytest.raises(TypeError):
            tie_break_order(bad)
 
 
# ------------------------------------------------------------------ scope
def test_module_contains_no_engine_or_other_strategy_logic():
    for name in ("SetupEngine", "SweepEngine", "ConfirmationEngine", "LiquidityEngine", "ZoneEngine",
                 "PivotEngine", "BOSEngine", "StructureEngine", "Sweep", "Confirmation", "Invalidation"):
        assert not hasattr(setup_model_module, name), name
