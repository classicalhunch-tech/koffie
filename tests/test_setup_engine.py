"""Engine-level tests for SetupEngine: touch, the Setup lifecycle, same-candle events, independent
Setups, pre-touch Sweeps, trade creation / zone consumption, overlapping zones, replay and integrity.
 
Only real project objects are used (Zone, Pivot, BOS, Swing, LiquidityEngine, SweepEngine,
ConfirmationEngine, Candle). The only test double is a LiquidityEngine SUBCLASS that records the
arguments of required_liquidity_for and then delegates to the real implementation.
 
Every scenario runs for LONG (DEMAND zone, liquidity below) and SHORT (SUPPLY zone, liquidity above);
the SHORT candles are the exact price mirror of the LONG ones (p -> 209 - p around the zone 99..110).
"""
import ast
import inspect
import random
from datetime import date, datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.setup_engine as engine_module
from koffie.strategy.engines.confirmation_engine import ConfirmationEngine
from koffie.strategy.engines.liquidity_engine import LiquidityEngine
from koffie.strategy.engines.setup_engine import SetupEngine
from koffie.strategy.engines.sweep_engine import SweepEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.confirmation import Confirmation, Invalidation, InvalidationReason
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.setup import (
    ConfirmationEventKey, Setup, SetupIdentity, SetupStatus, SetupTransition, is_touch, tie_break_order,
)
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.sweep import Sweep, SweepIdentity
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = SetupStatus
 
 
# ------------------------------------------------------------------ helpers
def at(i, tf=M5):
    return T0 + i * tf.duration
 
 
def mk_swing(swing_type, price, i=0, seq=0, tf=M5):
    candle_time = T0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5, bos_i=10, broken_i=2):
    """Zone 99..110. A 5M zone is created (known) at slot 11."""
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=broken_i, seq=0, tf=tf)
    bos = BOS(tf, direction, broken, T0 + bos_i * tf.duration, 101.0 if direction is BULL else 99.0)
    return Zone(Pivot(bos, T0 + 5 * tf.duration, high, low, pivot_class))
 
 
def conflicting_zone(zone):
    """A DIFFERENT Zone with the SAME identity: same BOS, different Pivot range."""
    p = zone.pivot
    clash = Zone(Pivot(p.bos, p.candle_time - zone.timeframe.duration, 120.0, 95.0, p.candle_class))
    assert clash != zone and clash.identity == zone.identity
    return clash
 
 
def ohlc(i, o, h, l, c, tf=M5):
    return Candle(tf, at(i, tf), o, h, l, c)
 
 
def classify(candle):
    return candle.classify(candle.close_time)
 
 
# LONG candle specs (open, high, low, close) for the DEMAND zone 99..110 with liquidity LOW 98.
SPEC = {
    "TOUCH_ONLY": (101.0, 104.0, 100.0, 102.0),            # closes inside, no sweep
    "CLOSE_LOW": (100.0, 101.0, 99.0, 99.0),               # close == zone.low
    "CLOSE_HIGH": (105.0, 110.5, 104.0, 110.0),            # close == zone.high
    "OPEN_OUTSIDE": (112.0, 112.5, 101.0, 102.0),          # open far outside, closes inside
    "WICK_ONLY": (105.0, 106.0, 98.5, 98.9),               # opens inside, wick inside, closes OUTSIDE
    "FAR_OUTSIDE": (112.0, 115.0, 111.0, 113.0),           # entirely outside
    "SWEEP_NO_TOUCH": (97.5, 100.0, 97.0, 98.5),           # sweeps 98 but closes outside the zone
    "TOUCH_SWEEP": (100.0, 104.0, 97.0, 101.0),            # closes inside AND sweeps, neutral
    "TOUCH_SWEEP_CONFIRM": (97.2, 104.8, 97.0, 104.5),     # closes inside, sweeps, BULLISH_DECISIVE
    "TOUCH_SWEEP_WRONG": (104.0, 104.1, 97.0, 99.0),       # closes ON the low, sweeps, BEARISH_DECISIVE
    "CONFIRM": (100.0, 105.0, 99.8, 104.8),                # BULLISH_DECISIVE, closes inside
    "WRONG_INSIDE": (104.8, 105.0, 100.0, 100.2),          # BEARISH_DECISIVE, closes inside
    "WRONG_BOUNDARY": (99.5, 99.6, 97.0, 97.1),            # BEARISH_DECISIVE, closes below the zone
    "DOJI": (100.0, 104.0, 99.0, 101.0),                   # neutral, closes inside
}
 
 
class Kit:
    """One direction's scenario kit."""
 
    def __init__(self, name):
        self.name = name
        self.long = name == "LONG"
        self.direction = BULL if self.long else BEAR
        self.zone_type = ZoneType.DEMAND if self.long else ZoneType.SUPPLY
 
    def zone(self, **kw):
        return mk_zone(self.direction, **kw)
 
    def level_swing(self, price_shift=0.0, i=0, seq=0):
        return mk_swing(LOW, 98.0 - price_shift, i, seq) if self.long else mk_swing(HIGH, 111.0 + price_shift, i, seq)
 
    def c(self, i, key):
        o, h, l, c = SPEC[key]
        if self.long:
            return ohlc(i, o, h, l, c)
        return ohlc(i, 209.0 - o, 209.0 - l, 209.0 - h, 209.0 - c)
 
    @property
    def confirming_class(self):
        return CandleClass.BULLISH_DECISIVE if self.long else CandleClass.BEARISH_DECISIVE
 
    @property
    def wrong_class(self):
        return CandleClass.BEARISH_DECISIVE if self.long else CandleClass.BULLISH_DECISIVE
 
 
KITS = (Kit("LONG"), Kit("SHORT"))
 
 
def make_liquidity(*swings):
    liq = LiquidityEngine(M5)
    for s in swings:
        liq.process_swing(s, s.confirmed_at)
    return liq
 
 
def env(kit, zone=None):
    zone = zone or kit.zone()
    liq = make_liquidity(kit.level_swing())
    return SetupEngine(liq), liq, zone
 
 
def run(engine, zone, candle, now=None):
    return engine.process_candle(zone, candle, candle.close_time if now is None else now)
 
 
def snap(engine, *zones):
    setups = engine.setups
    return (setups,
            tuple(engine.status_for(s) for s in setups),
            tuple(engine.transitions_for(s) for s in setups),
            tuple(engine.status_for_zone(z) for z in zones),
            tuple(engine.is_zone_consumed(z) for z in zones))
 
 
def statuses(engine, setup):
    return [t.status for t in engine.transitions_for(setup)]
 
 
def aware(dt):
    return dt.replace(tzinfo=timezone.utc)
 
 
class RecordingLiquidity(LiquidityEngine):
    """Real LiquidityEngine that also records the arguments of required_liquidity_for."""
 
    def __init__(self, timeframe):
        super().__init__(timeframe)
        self.calls = []
 
    def required_liquidity_for(self, zone, moment):
        self.calls.append((zone, moment))
        return super().required_liquidity_for(zone, moment)
 
 
def test_the_fixtures_have_the_classes_and_geometry_the_tests_assume():
    for kit in KITS:
        zone = kit.zone()
        for key in ("TOUCH_ONLY", "CLOSE_LOW", "CLOSE_HIGH", "OPEN_OUTSIDE", "TOUCH_SWEEP", "TOUCH_SWEEP_CONFIRM",
                    "TOUCH_SWEEP_WRONG", "CONFIRM", "WRONG_INSIDE", "DOJI"):
            assert is_touch(zone, kit.c(11, key)), (kit.name, key)
        for key in ("WICK_ONLY", "FAR_OUTSIDE", "SWEEP_NO_TOUCH", "WRONG_BOUNDARY"):
            assert not is_touch(zone, kit.c(11, key)), (kit.name, key)
        assert classify(kit.c(11, "TOUCH_SWEEP_CONFIRM")) is kit.confirming_class
        assert classify(kit.c(11, "CONFIRM")) is kit.confirming_class
        for key in ("TOUCH_SWEEP_WRONG", "WRONG_INSIDE", "WRONG_BOUNDARY"):
            assert classify(kit.c(11, key)) is kit.wrong_class, key
        for key in ("TOUCH_ONLY", "DOJI", "TOUCH_SWEEP", "SWEEP_NO_TOUCH"):
            assert classify(kit.c(11, key)) is CandleClass.NEUTRAL, key
        swing = kit.level_swing()
        penetrating = ("SWEEP_NO_TOUCH", "TOUCH_SWEEP", "TOUCH_SWEEP_CONFIRM", "TOUCH_SWEEP_WRONG", "WRONG_BOUNDARY")
        for key in SPEC:
            c = kit.c(11, key)
            hit = c.low < swing.price if kit.long else c.high > swing.price
            assert hit == (key in penetrating), (kit.name, key)
        assert zone.zone_type is kit.zone_type
 
 
# ================================================================== construction
def test_constructor_requires_a_liquidity_engine_and_is_m5():
    engine = SetupEngine(LiquidityEngine(M5))
    assert engine.timeframe is M5 and engine.setups == ()
    for bad in (None, "liquidity", 5, M5, [], object()):
        with pytest.raises(TypeError):
            SetupEngine(bad)
    assert SetupEngine(RecordingLiquidity(M5)).timeframe is M5
    with pytest.raises(AttributeError):
        engine.timeframe = H1
 
 
# ================================================================== Zone Touch
def test_a_close_strictly_inside_the_zone_starts_a_setup_waiting_for_sweep():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_ONLY")
        setup = run(engine, zone, candle)
        assert isinstance(setup, Setup) and setup.zone is zone and setup.touch_candle is candle
        assert engine.setups == (setup,)
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        assert engine.active_setup_for(zone) is setup and engine.status_for_zone(zone) is S.WAITING_FOR_SWEEP
        assert statuses(engine, setup) == [S.ZONE_TOUCHED, S.WAITING_FOR_SWEEP]
        assert engine.sweep_for(setup) is None and engine.confirmation_for(setup) is None
 
 
def test_a_close_exactly_at_zone_low_or_zone_high_is_a_touch():
    for kit in KITS:
        for key in ("CLOSE_LOW", "CLOSE_HIGH"):
            engine, liq, zone = env(kit)
            candle = kit.c(11, key)
            assert candle.close in (zone.low, zone.high)
            setup = run(engine, zone, candle)
            assert setup is not None and engine.status_for(setup) is S.WAITING_FOR_SWEEP, (kit.name, key)
 
 
def test_a_close_outside_the_zone_is_not_a_touch():
    for kit in KITS:
        for key in ("FAR_OUTSIDE", "WICK_ONLY", "SWEEP_NO_TOUCH", "WRONG_BOUNDARY"):
            engine, liq, zone = env(kit)
            assert run(engine, zone, kit.c(11, key)) is None, (kit.name, key)
            assert engine.setups == () and engine.status_for_zone(zone) is S.NO_SETUP
 
 
def test_a_wick_into_the_zone_with_the_close_outside_does_not_qualify():
    for kit in KITS:
        engine, liq, zone = env(kit)
        wick = kit.c(11, "WICK_ONLY")
        inside = (zone.low <= wick.open <= zone.high) and (zone.low <= wick.high <= zone.high or zone.low <= wick.low <= zone.high)
        assert inside                                                  # the wick/body really is in the zone
        assert run(engine, zone, wick) is None and engine.setups == ()
 
 
def test_the_open_price_is_irrelevant_to_the_touch():
    for kit in KITS:
        engine, liq, zone = env(kit)
        assert run(engine, zone, kit.c(11, "OPEN_OUTSIDE")) is not None          # open far outside, close inside
        engine, liq, zone = env(kit)
        assert run(engine, zone, kit.c(11, "WICK_ONLY")) is None                 # open inside, close outside
 
 
def test_a_candle_opening_exactly_when_the_zone_is_created_is_accepted():
    for kit in KITS:
        engine, liq, zone = env(kit)
        assert zone.created_at == at(11)
        assert run(engine, zone, kit.c(11, "TOUCH_ONLY")) is not None
 
 
def test_a_candle_before_the_zone_was_created_is_rejected_even_if_it_would_touch():
    for kit in KITS:
        engine, liq, zone = env(kit)
        before = snap(engine, zone)
        for i in (0, 5, 10):
            with pytest.raises(ValueError):
                run(engine, zone, kit.c(i, "TOUCH_ONLY"))
        with pytest.raises(ValueError):
            run(engine, zone, Candle(M5, zone.created_at - timedelta(minutes=1), 101, 104, 100, 102))
        assert snap(engine, zone) == before
        assert run(engine, zone, kit.c(11, "TOUCH_ONLY")) is not None            # slot 11 is still free
 
 
def test_only_completed_5m_candles_can_start_a_setup():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_ONLY")
        for now in (candle.open_time, candle.open_time + timedelta(minutes=3), candle.close_time - timedelta(seconds=1)):
            with pytest.raises(ValueError):
                engine.process_candle(zone, candle, now)
        for tf in (M15, H1):
            with pytest.raises(ValueError):
                run(engine, zone, Candle(tf, zone.created_at, 101, 104, 100, 102))
        assert engine.setups == ()
        assert engine.process_candle(zone, candle, candle.close_time) is not None  # exactly at the close is fine
        far = env(kit)
        c2 = kit.c(11, "TOUCH_ONLY")
        assert far[0].process_candle(far[2], c2, c2.close_time + timedelta(days=30)) is not None
 
 
def test_there_is_no_distance_age_or_other_eligibility_filter():
    for kit in KITS:
        engine, liq, zone = env(kit)
        assert run(engine, zone, kit.c(5000, "TOUCH_ONLY")) is not None           # ~17 days after creation
        odd_hour = datetime(2026, 3, 7, 3, 35)                                    # a Saturday, 03:35
        engine2, _, zone2 = env(kit)
        candle = Candle(M5, odd_hour, *(SPEC["TOUCH_ONLY"] if kit.long else (108.0, 109.0, 105.0, 107.0)))
        assert run(engine2, zone2, candle) is not None
 
 
# ================================================================== same-candle events
def test_touch_and_sweep_on_the_same_candle():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_SWEEP")
        setup = run(engine, zone, candle)
        sweep = engine.sweep_for(setup)
        assert isinstance(sweep, Sweep) and sweep.candle is candle and sweep.zone is zone
        assert sweep.level is liq.levels[0]
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        assert statuses(engine, setup) == [S.ZONE_TOUCHED, S.WAITING_FOR_SWEEP, S.SWEPT, S.WAITING_FOR_CONFIRMATION]
        assert {t.at for t in engine.transitions_for(setup)} == {candle.close_time}   # all in one candle, no delay
        assert engine.confirmation_for(setup) is None and engine.invalidation_for(setup) is None
 
 
def test_touch_sweep_and_confirmation_on_the_same_candle():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_SWEEP_CONFIRM")
        setup = run(engine, zone, candle)
        confirmation = engine.confirmation_for(setup)
        assert isinstance(confirmation, Confirmation) and confirmation.candle is candle
        assert confirmation.sweep is engine.sweep_for(setup)
        assert engine.status_for(setup) is S.CONFIRMED
        assert statuses(engine, setup) == [S.ZONE_TOUCHED, S.WAITING_FOR_SWEEP, S.SWEPT,
                                           S.WAITING_FOR_CONFIRMATION, S.CONFIRMED]
        assert {t.at for t in engine.transitions_for(setup)} == {candle.close_time}
        assert engine.active_setup_for(zone) is None and engine.status_for_zone(zone) is S.NO_SETUP
 
 
def test_a_touching_sweep_candle_can_never_invalidate_on_that_same_candle():
    """Invalidation of a first candle needs a close strictly beyond the zone boundary, which is
    exactly what a touch (close inside, boundaries inclusive) excludes. So touch + Sweep +
    Invalidation on ONE candle cannot occur; the nearest case is a wrong-direction touch that waits."""
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_SWEEP_WRONG")
        assert classify(candle) is kit.wrong_class
        setup = run(engine, zone, candle)
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        assert engine.invalidation_for(setup) is None
        assert S.INVALIDATED not in statuses(engine, setup)
    rng = random.Random(11)
    for kit in KITS:
        for _ in range(1500):
            low = rng.uniform(90.0, 112.0)
            high = low + rng.uniform(0.0, 25.0)
            a, b = rng.uniform(low, high), rng.uniform(low, high)
            raw = Candle(M5, at(11), a, high, low, b)
            candle = raw if kit.long else Candle(M5, at(11), 209.0 - a, 209.0 - low, 209.0 - high, 209.0 - b)
            engine, liq, zone = env(kit)
            result = run(engine, zone, candle)
            if result is not None:                                    # the candle touched the zone
                assert engine.status_for(result) is not S.INVALIDATED
                assert engine.invalidation_for(result) is None
 
 
def test_a_sweep_and_invalidation_across_candles():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        result = run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))                  # closes beyond the boundary
        assert result is setup and engine.status_for(setup) is S.INVALIDATED
        invalidation = engine.invalidation_for(setup)
        assert isinstance(invalidation, Invalidation) and invalidation.reason is InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY
        assert statuses(engine, setup)[-1] is S.INVALIDATED and engine.confirmation_for(setup) is None
 
 
def test_two_wrong_direction_candles_invalidate_a_setup():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_WRONG"))              # first wrong-direction decisive
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        assert run(engine, zone, kit.c(12, "WRONG_INSIDE")) is setup           # second wrong-direction decisive
        assert engine.status_for(setup) is S.INVALIDATED
        assert engine.invalidation_for(setup).reason is InvalidationReason.SECOND_WRONG_DIRECTION_DECISIVE
 
 
# ================================================================== lifecycle
def test_full_lifecycle_touch_then_sweep_then_confirmation_on_separate_candles():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        assert run(engine, zone, kit.c(12, "DOJI")) is None                    # nothing changes
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        swept = run(engine, zone, kit.c(13, "SWEEP_NO_TOUCH"))                  # a sweep needs no touch of its own
        assert swept is setup and engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        assert engine.sweep_for(setup).candle == kit.c(13, "SWEEP_NO_TOUCH")
        assert run(engine, zone, kit.c(14, "DOJI")) is None
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        confirmed = run(engine, zone, kit.c(15, "CONFIRM"))
        assert confirmed is setup and engine.status_for(setup) is S.CONFIRMED
        assert statuses(engine, setup) == [S.ZONE_TOUCHED, S.WAITING_FOR_SWEEP, S.SWEPT,
                                           S.WAITING_FOR_CONFIRMATION, S.CONFIRMED]
        times = [t.at for t in engine.transitions_for(setup)]
        assert times == [at(12), at(12), at(14), at(14), at(16)]
 
 
def test_every_status_is_distinct_and_in_the_locked_order():
    order = [S.ZONE_TOUCHED, S.WAITING_FOR_SWEEP, S.SWEPT, S.WAITING_FOR_CONFIRMATION, S.CONFIRMED, S.TRADE_CREATED]
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        engine.record_trade_created(setup, at(20))
        assert statuses(engine, setup) == order
        assert len(set(order)) == len(order)
 
 
def test_an_invalidated_setup_never_revives():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        invalidation = engine.invalidation_for(setup)
        frozen = (engine.status_for(setup), engine.transitions_for(setup), invalidation)
        for i, key in enumerate(("CONFIRM", "TOUCH_SWEEP_CONFIRM", "DOJI", "TOUCH_SWEEP", "WRONG_INSIDE", "CONFIRM")):
            run(engine, zone, kit.c(13 + i, key))
            assert (engine.status_for(setup), engine.transitions_for(setup), engine.invalidation_for(setup)) == frozen
        assert engine.status_for(setup) is S.INVALIDATED and engine.confirmation_for(setup) is None
        assert engine.setups[0] is setup
 
 
def test_a_confirmed_setup_stays_confirmed_whatever_comes_later():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        confirmation = engine.confirmation_for(setup)
        frozen = (engine.status_for(setup), engine.transitions_for(setup))
        for i, key in enumerate(("WRONG_BOUNDARY", "WRONG_INSIDE", "WRONG_INSIDE", "SWEEP_NO_TOUCH", "DOJI")):
            run(engine, zone, kit.c(12 + i, key))
        assert (engine.status_for(setup), engine.transitions_for(setup)) == frozen
        assert engine.confirmation_for(setup) is confirmation and engine.invalidation_for(setup) is None
 
 
def test_there_is_no_timeout_while_waiting_for_the_sweep():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        for i in range(12, 3012):                                              # ~10 days of 5M candles
            assert run(engine, zone, kit.c(i, "DOJI")) is None
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        assert run(engine, zone, kit.c(9000, "SWEEP_NO_TOUCH")) is setup       # after a long gap too
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
 
 
def test_there_is_no_timeout_while_waiting_for_the_confirmation():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        for i in range(12, 2012):
            run(engine, zone, kit.c(i, "DOJI"))
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
        assert run(engine, zone, kit.c(5000, "CONFIRM")) is setup and engine.status_for(setup) is S.CONFIRMED
 
 
def test_nothing_but_the_confirmation_outcome_invalidates_a_setup_waiting_for_the_sweep():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        newer_zone = kit.zone(bos_i=14, broken_i=3)                              # a newer zone appears
        run(engine, newer_zone, kit.c(15, "TOUCH_ONLY"))
        newer_level = kit.level_swing(price_shift=8.0, i=12, seq=1)              # newer, different liquidity
        liq.process_swing(newer_level, newer_level.confirmed_at)
        for i, key in enumerate(("TOUCH_ONLY", "FAR_OUTSIDE", "WRONG_BOUNDARY", "WICK_ONLY", "CLOSE_LOW")):
            run(engine, zone, kit.c(13 + i, key))                                # another touch, price moves away, ...
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        assert engine.invalidation_for(setup) is None
 
 
# ================================================================== independent setups
def test_an_active_setup_ignores_later_touches():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        for i, key in enumerate(("TOUCH_ONLY", "CLOSE_LOW", "CLOSE_HIGH", "DOJI", "OPEN_OUTSIDE")):
            assert run(engine, zone, kit.c(12 + i, key)) is None
            assert engine.setups == (setup,)
        assert engine.active_setup_for(zone) is setup and engine.status_for(setup) is S.WAITING_FOR_SWEEP
 
 
def test_an_active_setup_waiting_for_confirmation_ignores_later_touches_too():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        assert run(engine, zone, kit.c(12, "DOJI")) is None                      # a touch, ignored
        assert engine.setups == (setup,) and engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
 
 
def test_an_invalidated_setup_does_not_block_a_new_independent_setup_on_the_same_zone():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first_touch, second_touch = kit.c(11, "TOUCH_SWEEP"), kit.c(20, "TOUCH_SWEEP")
        first = run(engine, zone, first_touch)
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        assert engine.status_for(first) is S.INVALIDATED and engine.active_setup_for(zone) is None
        second = run(engine, zone, second_touch)
        assert isinstance(second, Setup) and second is not first
        assert engine.setups == (first, second) and engine.setups_for_zone(zone) == (first, second)
        assert engine.status_for(first) is S.INVALIDATED                          # the old one is untouched
        assert engine.status_for(second) is S.WAITING_FOR_CONFIRMATION
        assert engine.active_setup_for(zone) is second
 
 
def test_the_same_zone_has_distinct_setup_identities_and_the_setups_are_never_merged():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        second = run(engine, zone, kit.c(20, "TOUCH_SWEEP"))
        assert first.zone == second.zone and first.identity != second.identity
        assert first.identity == SetupIdentity.from_parts(zone, kit.c(11, "TOUCH_SWEEP"))
        assert second.identity == SetupIdentity.from_parts(zone, kit.c(20, "TOUCH_SWEEP"))
        assert engine.transitions_for(first) != engine.transitions_for(second)
        assert len({s.identity for s in engine.setups}) == 2
 
 
def test_the_new_setup_gets_its_own_new_causal_sweep_of_the_same_liquidity_level():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        second = run(engine, zone, kit.c(20, "TOUCH_SWEEP"))
        s1, s2 = engine.sweep_for(first), engine.sweep_for(second)
        assert s1 is not None and s2 is not None and s1 != s2
        assert s1.level is s2.level                                              # the SAME liquidity level
        assert s1.candle == kit.c(11, "TOUCH_SWEEP") and s2.candle == kit.c(20, "TOUCH_SWEEP")
        assert s1.identity == s2.identity                                        # (zone, level) alone cannot tell them apart
        assert s2.candle.open_time > s1.candle.open_time                          # a genuinely new, later causal event
        assert s2.known_at == kit.c(20, "TOUCH_SWEEP").close_time
 
 
def test_the_second_setup_runs_its_own_confirmation_chain():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        second = run(engine, zone, kit.c(20, "TOUCH_SWEEP"))
        assert run(engine, zone, kit.c(21, "CONFIRM")) is second
        assert engine.status_for(second) is S.CONFIRMED and engine.status_for(first) is S.INVALIDATED
        assert engine.confirmation_for(second).sweep is engine.sweep_for(second)
        assert engine.confirmation_for(first) is None and engine.invalidation_for(second) is None
 
 
def test_the_candle_that_resolves_a_setup_cannot_also_start_the_next_one():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP_WRONG"))
        resolving = kit.c(12, "WRONG_INSIDE")                                    # invalidates AND closes inside the zone
        assert is_touch(zone, resolving)
        assert run(engine, zone, resolving) is first
        assert engine.status_for(first) is S.INVALIDATED and engine.setups == (first,)
        second = run(engine, zone, kit.c(13, "TOUCH_SWEEP"))                      # the NEXT touch starts the next setup
        assert second is not None and engine.setups == (first, second)
 
 
def test_a_confirmed_setup_does_not_block_a_later_independent_setup_while_the_zone_is_unconsumed():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        second = run(engine, zone, kit.c(20, "TOUCH_ONLY"))
        assert second is not None and engine.setups == (first, second)
        assert engine.status_for(first) is S.CONFIRMED and engine.status_for(second) is S.WAITING_FOR_SWEEP
 
 
# ================================================================== the SweepEngine / ConfirmationEngine conflict (A1)
def test_existing_sweep_engine_records_only_one_sweep_per_zone_and_level_pair():
    """Documents the upstream behaviour that makes per-Setup engines necessary."""
    for kit in KITS:
        zone = kit.zone()
        liq = make_liquidity(kit.level_swing())
        shared = SweepEngine(liq)
        first = shared.process_candle(zone, kit.c(11, "TOUCH_SWEEP"), kit.c(11, "TOUCH_SWEEP").close_time)
        assert isinstance(first, Sweep)
        again = shared.process_candle(zone, kit.c(20, "TOUCH_SWEEP"), kit.c(20, "TOUCH_SWEEP").close_time)
        assert again is None                                                     # same pair: no new Sweep, ever
        assert shared.sweeps == (first,)
 
 
def test_existing_confirmation_engine_rejects_a_second_sweep_with_the_same_identity():
    for kit in KITS:
        zone = kit.zone()
        liq = make_liquidity(kit.level_swing())
        level = liq.levels[0]
        sweep1 = Sweep(zone, level, kit.c(11, "TOUCH_SWEEP"))
        sweep2 = Sweep(zone, level, kit.c(20, "TOUCH_SWEEP"))
        assert sweep1.identity == sweep2.identity and sweep1 != sweep2
        shared = ConfirmationEngine()
        shared.process_candle(sweep1, sweep1.candle, sweep1.candle.close_time)
        with pytest.raises(ValueError):
            shared.process_candle(sweep2, sweep2.candle, sweep2.candle.close_time)
 
 
def test_per_setup_engines_resolve_the_conflict_without_changing_any_upstream_engine():
    for kit in KITS:
        engine, liq, zone = env(kit)
        for k in range(3):                                                       # three independent setups, one level
            base = 11 + 10 * k
            setup = run(engine, zone, kit.c(base, "TOUCH_SWEEP"))
            assert engine.sweep_for(setup).candle == kit.c(base, "TOUCH_SWEEP")
            run(engine, zone, kit.c(base + 1, "WRONG_BOUNDARY"))
            assert engine.status_for(setup) is S.INVALIDATED
        assert len(engine.setups) == 3
        assert len({engine.sweep_for(s).candle.open_time for s in engine.setups}) == 3
        assert len({s.identity for s in engine.setups}) == 3
 
 
# ================================================================== a Sweep before the touch does not belong to a Setup
def test_a_sweep_before_the_first_touch_does_not_belong_to_a_later_setup():
    for kit in KITS:
        engine, liq, zone = env(kit)
        early = kit.c(11, "SWEEP_NO_TOUCH")                                      # sweeps the level, no touch
        assert run(engine, zone, early) is None and engine.setups == ()
        setup = run(engine, zone, kit.c(12, "TOUCH_ONLY"))                       # the touch starts the Setup
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        assert engine.sweep_for(setup) is None                                   # the earlier sweep does not count
        assert run(engine, zone, kit.c(13, "DOJI")) is None
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
 
 
def test_the_earlier_sweep_does_not_use_up_the_level_for_the_later_setup():
    for kit in KITS:
        engine, liq, zone = env(kit)
        run(engine, zone, kit.c(11, "SWEEP_NO_TOUCH"))
        setup = run(engine, zone, kit.c(12, "TOUCH_ONLY"))
        later = run(engine, zone, kit.c(13, "SWEEP_NO_TOUCH"))                   # the same level is swept again, after the touch
        assert later is setup and engine.sweep_for(setup).candle == kit.c(13, "SWEEP_NO_TOUCH")
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
 
 
def test_setup_feeds_the_sweep_engine_only_from_the_qualifying_touch_onward():
    for kit in KITS:
        zone = kit.zone()
        liq = RecordingLiquidity(M5)
        swing = kit.level_swing()
        liq.process_swing(swing, swing.confirmed_at)
        engine = SetupEngine(liq)
        for i, key in enumerate(("SWEEP_NO_TOUCH", "FAR_OUTSIDE", "WICK_ONLY", "SWEEP_NO_TOUCH")):
            run(engine, zone, kit.c(11 + i, key))
        assert liq.calls == []                                                   # no sweep evaluation before a touch
        touch = kit.c(15, "TOUCH_ONLY")
        run(engine, zone, touch)
        assert liq.calls == [(zone, touch.open_time)]                             # the touch candle itself is fed
        run(engine, zone, kit.c(16, "DOJI"))
        assert [m for _, m in liq.calls] == [at(15), at(16)]
 
 
def test_the_touch_candle_itself_is_fed_so_it_can_also_be_the_sweep_candle():
    for kit in KITS:
        zone = kit.zone()
        liq = RecordingLiquidity(M5)
        swing = kit.level_swing()
        liq.process_swing(swing, swing.confirmed_at)
        engine = SetupEngine(liq)
        candle = kit.c(11, "TOUCH_SWEEP")
        setup = run(engine, zone, candle)
        assert liq.calls == [(zone, candle.open_time)]
        assert engine.sweep_for(setup).candle == candle
 
 
# ================================================================== trade creation and zone consumption
def test_an_invalidated_setup_does_not_consume_the_zone():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        assert engine.is_zone_consumed(zone) is False and engine.consumed_by(zone) is None
        with pytest.raises(ValueError):
            engine.record_trade_created(setup, at(30))                            # no trade can follow an invalidation
        assert engine.is_zone_consumed(zone) is False
        assert run(engine, zone, kit.c(20, "TOUCH_ONLY")) is not None             # the zone still creates setups
 
 
def test_a_confirmed_setup_does_not_consume_the_zone():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        assert engine.status_for(setup) is S.CONFIRMED
        assert engine.is_zone_consumed(zone) is False and engine.consumed_by(zone) is None
        assert engine.can_create_trade(setup) is True
        assert engine.confirmed_setups_known_at(at(12)) == (setup,)
 
 
def test_actual_trade_creation_is_the_consumption_point():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        assert engine.record_trade_created(setup, at(15)) is None
        assert engine.status_for(setup) is S.TRADE_CREATED
        assert engine.transitions_for(setup)[-1] == SetupTransition(S.TRADE_CREATED, at(15))
        assert engine.is_zone_consumed(zone) is True and engine.consumed_by(zone) is setup
        assert engine.can_create_trade(setup) is False
        assert engine.confirmed_setups_known_at(at(100)) == ()
 
 
def test_trade_creation_is_idempotent_for_the_same_setup():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        engine.record_trade_created(setup, at(15))
        before = snap(engine, zone)
        assert engine.record_trade_created(setup, at(40)) is None
        assert snap(engine, zone) == before and len(engine.transitions_for(setup)) == 6
 
 
def test_a_consumed_zone_cannot_produce_another_entry():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        second = run(engine, zone, kit.c(20, "TOUCH_SWEEP_CONFIRM"))              # a second independent confirmed setup
        assert engine.status_for(second) is S.CONFIRMED and first.identity != second.identity
        engine.record_trade_created(first, at(25))
        assert engine.can_create_trade(second) is False
        before = snap(engine, zone)
        with pytest.raises(ValueError):
            engine.record_trade_created(second, at(26))
        assert snap(engine, zone) == before
        assert engine.status_for(second) is S.CONFIRMED                           # not deleted, just not entry-eligible
        assert engine.confirmed_setups_known_at(at(100)) == ()
 
 
def test_setup_history_stays_intact_after_consumption():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        second = run(engine, zone, kit.c(20, "TOUCH_SWEEP_CONFIRM"))
        third = run(engine, zone, kit.c(30, "TOUCH_SWEEP"))                       # still waiting for confirmation
        history = (engine.setups, engine.transitions_for(second), engine.confirmation_for(second))
        engine.record_trade_created(first, at(35))
        assert engine.setups == (first, second, third) == history[0]
        assert engine.setups_for_zone(zone) == (first, second, third)
        assert engine.transitions_for(second) == history[1] and engine.confirmation_for(second) is history[2]
        assert engine.status_for(second) is S.CONFIRMED and engine.status_for(third) is S.WAITING_FOR_CONFIRMATION
 
 
def test_a_consumed_zone_starts_no_new_setup_but_active_setups_keep_their_lifecycle():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        waiting = run(engine, zone, kit.c(20, "TOUCH_SWEEP"))                     # active, waiting for confirmation
        engine.record_trade_created(first, at(25))
        assert run(engine, zone, kit.c(30, "CONFIRM")) is waiting                  # the active setup still advances...
        assert engine.status_for(waiting) is S.CONFIRMED
        assert engine.can_create_trade(waiting) is False                           # ...but cannot take another entry
        before = engine.setups
        assert run(engine, zone, kit.c(40, "TOUCH_SWEEP_CONFIRM")) is None        # no new setup on a consumed zone
        assert engine.setups == before
 
 
def test_trade_creation_requires_a_confirmed_tracked_setup_and_valid_arguments():
    for kit in KITS:
        engine, liq, zone = env(kit)
        waiting = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        before = snap(engine, zone)
        with pytest.raises(ValueError):
            engine.record_trade_created(waiting, at(30))                          # not confirmed yet
        stranger = Setup(kit.zone(bos_i=14, broken_i=3), kit.c(15, "TOUCH_ONLY"))
        with pytest.raises(ValueError):
            engine.record_trade_created(stranger, at(30))                         # never seen by the engine
        for bad in (None, "setup", 5, zone):
            with pytest.raises(TypeError):
                engine.record_trade_created(bad, at(30))
        for bad in (None, "now", 5, date(2026, 1, 1)):
            with pytest.raises(TypeError):
                engine.record_trade_created(waiting, bad)
        assert snap(engine, zone) == before
 
 
def test_a_trade_cannot_be_created_before_the_confirmation_is_known():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        confirmation = engine.confirmation_for(setup)
        before = snap(engine, zone)
        for now in (confirmation.candle_time, confirmation.known_at - timedelta(seconds=1)):
            with pytest.raises(ValueError):
                engine.record_trade_created(setup, now)
        with pytest.raises(TypeError):
            engine.record_trade_created(setup, aware(confirmation.known_at))
        assert snap(engine, zone) == before
        engine.record_trade_created(setup, confirmation.known_at)                  # exactly when it became known is fine
        assert engine.status_for(setup) is S.TRADE_CREATED
 
 
def test_setup_never_fabricates_trade_creation():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        for i in range(12, 60):
            run(engine, zone, kit.c(i, "DOJI"))
        assert engine.status_for(setup) is S.CONFIRMED and S.TRADE_CREATED not in statuses(engine, setup)
        assert engine.is_zone_consumed(zone) is False
 
 
# ================================================================== overlapping zones
def overlapping_env(kit):
    """A 5M zone and a 15M zone, both 99..110, sharing one liquidity level; both exist at slot 33."""
    liq = make_liquidity(kit.level_swing())
    z5, z15 = kit.zone(tf=M5), kit.zone(tf=M15)
    assert z15.created_at == at(33) and z5.created_at == at(11) and z5.identity != z15.identity
    return SetupEngine(liq), liq, z5, z15
 
 
def test_overlapping_zones_stay_separate_objects_with_separate_setups():
    for kit in KITS:
        engine, liq, z5, z15 = overlapping_env(kit)
        candle = kit.c(33, "TOUCH_SWEEP_CONFIRM")
        s5, s15 = run(engine, z5, candle), run(engine, z15, candle)
        assert s5 is not s15 and s5.zone == z5 and s15.zone == z15
        assert engine.setups_for_zone(z5) == (s5,) and engine.setups_for_zone(z15) == (s15,)
        assert s5.identity != s15.identity
        assert engine.sweep_for(s5) is not engine.sweep_for(s15)
        assert (z5.low, z5.high) == (z15.low, z15.high) == (99.0, 110.0)           # neither zone changed
        assert engine.status_for(s5) is S.CONFIRMED and engine.status_for(s15) is S.CONFIRMED
 
 
def test_one_causal_event_produces_at_most_one_trade_across_overlapping_zones():
    for kit in KITS:
        engine, liq, z5, z15 = overlapping_env(kit)
        candle = kit.c(33, "TOUCH_SWEEP_CONFIRM")
        s5, s15 = run(engine, z5, candle), run(engine, z15, candle)
        assert engine.confirmation_event_key_for(s5) == engine.confirmation_event_key_for(s15)
        assert isinstance(engine.confirmation_event_key_for(s5), ConfirmationEventKey)
        engine.record_trade_created(s15, at(40))
        before = snap(engine, z5, z15)
        assert engine.can_create_trade(s5) is False
        with pytest.raises(ValueError):
            engine.record_trade_created(s5, at(41))                                 # a second trade from the same event
        assert snap(engine, z5, z15) == before
        assert engine.status_for(s5) is S.CONFIRMED
 
 
def test_only_the_causal_zone_is_consumed_and_the_other_zone_stays_available():
    for kit in KITS:
        engine, liq, z5, z15 = overlapping_env(kit)
        candle = kit.c(33, "TOUCH_SWEEP_CONFIRM")
        s5, s15 = run(engine, z5, candle), run(engine, z15, candle)
        engine.record_trade_created(s5, at(40))                                     # the 5M zone is the causal one here
        assert engine.is_zone_consumed(z5) is True and engine.is_zone_consumed(z15) is False
        assert engine.consumed_by(z5) is s5 and engine.consumed_by(z15) is None
        assert z15 in {s.zone for s in engine.setups}                                # still recorded
        later = run(engine, z15, kit.c(50, "TOUCH_SWEEP"))                          # and still takes part in future setups
        assert later is not None and later is not s15 and engine.status_for(later) is S.WAITING_FOR_CONFIRMATION
        assert run(engine, z5, kit.c(50, "TOUCH_SWEEP")) is None                    # the consumed zone does not
 
 
def test_a_clearly_causal_lower_timeframe_zone_is_not_replaced_by_a_higher_timeframe_zone():
    for kit in KITS:
        # Only the 5M zone contains this candle's close; the 15M zone (higher timeframe) is not touched,
        # so it is not part of this causal event and the 5M zone stays the causal one.
        narrow5 = mk_zone(kit.direction, low=100.0, high=105.0, tf=M5)
        narrow15 = mk_zone(kit.direction, low=105.0 + 0.1, high=110.0, tf=M15)
        eng = SetupEngine(make_liquidity(kit.level_swing()))
        candle = kit.c(33, "TOUCH_SWEEP_CONFIRM")
        assert is_touch(narrow5, candle) is True and is_touch(narrow15, candle) is False
        s_causal, s_other = run(eng, narrow5, candle), run(eng, narrow15, candle)
        assert s_causal is not None and s_other is None
        assert eng.status_for(s_causal) is S.CONFIRMED
        assert eng.can_create_trade(s_causal) is True
        eng.record_trade_created(s_causal, at(40))                                  # allowed whatever its timeframe
        assert eng.consumed_by(narrow5) is s_causal and eng.is_zone_consumed(narrow15) is False
 
 
def test_the_tie_break_is_only_a_deterministic_ordering_1h_15m_5m():
    for kit in KITS:
        engine, liq, z5, z15 = overlapping_env(kit)
        z1h = kit.zone(tf=H1)
        liq.process_swing(kit.level_swing(price_shift=0.0, i=0, seq=0), at(2))      # replay of the same swing: no-op
        c1h = Candle(M5, z1h.created_at, *(SPEC["TOUCH_SWEEP_CONFIRM"] if kit.long else (
            209.0 - 97.2, 209.0 - 97.0, 209.0 - 104.8, 209.0 - 104.5)))
        s1h = run(engine, z1h, c1h)
        s5 = run(engine, z5, kit.c(33, "TOUCH_SWEEP_CONFIRM"))
        s15 = run(engine, z15, kit.c(33, "TOUCH_SWEEP_CONFIRM"))
        ordered = tie_break_order([s5, s15, s1h])
        assert [s.zone.timeframe for s in ordered] == [H1, M15, M5]
        # the engine itself never picks a zone by timeframe
        for attr in ("tie_break", "select_zone", "best_setup", "preferred_setup", "choose"):
            assert not hasattr(engine, attr)
        engine.record_trade_created(s5, at(500))                                    # nothing forces the higher timeframe first
        assert engine.consumed_by(z5) is s5
 
 
# ================================================================== replay and integrity
def test_replaying_the_latest_candle_returns_the_same_result_without_mutation():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_ONLY")
        setup = run(engine, zone, candle)
        before = snap(engine, zone)
        assert run(engine, zone, candle) is setup
        assert run(engine, zone, kit.c(11, "TOUCH_ONLY")) is setup                  # an equal, rebuilt candle
        assert snap(engine, zone) == before and len(engine.setups) == 1
        quiet = kit.c(12, "DOJI")
        assert run(engine, zone, quiet) is None
        before = snap(engine, zone)
        assert run(engine, zone, quiet) is None and snap(engine, zone) == before
 
 
def test_replaying_an_earlier_candle_that_changed_a_setup_returns_it_without_mutation():
    for kit in KITS:
        engine, liq, zone = env(kit)
        touch = kit.c(11, "TOUCH_ONLY")
        setup = run(engine, zone, touch)
        sweep_candle = kit.c(13, "SWEEP_NO_TOUCH")
        run(engine, zone, sweep_candle)
        run(engine, zone, kit.c(14, "DOJI"))
        run(engine, zone, kit.c(15, "CONFIRM"))
        before = snap(engine, zone)
        assert run(engine, zone, touch) is setup                                    # the touch candle
        assert run(engine, zone, sweep_candle) is setup                             # the sweep candle
        assert run(engine, zone, kit.c(15, "CONFIRM")) is setup                     # the confirming candle (also the latest)
        assert snap(engine, zone) == before
 
 
def test_replaying_an_earlier_candle_that_changed_nothing_is_out_of_order():
    for kit in KITS:
        engine, liq, zone = env(kit)
        run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        quiet = kit.c(12, "DOJI")
        run(engine, zone, quiet)
        run(engine, zone, kit.c(13, "DOJI"))
        before = snap(engine, zone)
        with pytest.raises(ValueError):
            run(engine, zone, quiet)
        assert snap(engine, zone) == before
 
 
def test_replay_with_a_later_now_or_an_equal_rebuilt_zone_changes_nothing():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_SWEEP_CONFIRM")
        setup = run(engine, zone, candle)
        assert engine.process_candle(zone, candle, candle.close_time + timedelta(days=3)) is setup
        rebuilt = kit.zone()
        assert rebuilt is not zone and rebuilt == zone
        assert run(engine, rebuilt, candle) is setup
        assert len(engine.setups) == 1
 
 
def test_a_conflicting_zone_with_the_same_identity_is_rejected():
    for kit in KITS:
        engine, liq, zone = env(kit)
        run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        before = snap(engine, zone)
        clash = conflicting_zone(zone)
        with pytest.raises(ValueError):
            run(engine, clash, kit.c(12, "DOJI"))
        with pytest.raises(ValueError):
            run(engine, clash, kit.c(11, "TOUCH_ONLY"))                             # even for a replayed candle
        assert snap(engine, zone) == before
        assert engine.setups_for_zone(clash) == ()
        assert run(engine, zone, kit.c(12, "DOJI")) is None
 
 
REJECTIONS = {
    "wrong zone type": lambda e, z, k: e.process_candle("zone", k.c(13, "SWEEP_NO_TOUCH"), at(14)),
    "wrong candle type": lambda e, z, k: e.process_candle(z, "candle", at(14)),
    "wrong now type": lambda e, z, k: e.process_candle(z, k.c(13, "SWEEP_NO_TOUCH"), None),
    "non-5m candle": lambda e, z, k: e.process_candle(z, Candle(M15, at(13), 101, 104, 100, 102), at(900)),
    "unclosed candle": lambda e, z, k: e.process_candle(z, k.c(13, "SWEEP_NO_TOUCH"), at(13)),
    "before the zone existed": lambda e, z, k: e.process_candle(z, k.c(5, "TOUCH_ONLY"), at(30)),
    "overlapping candle": lambda e, z, k: e.process_candle(
        z, Candle(M5, at(12) + timedelta(minutes=2), 101, 104, 100, 102), at(14)),
    "different candle in a used slot": lambda e, z, k: e.process_candle(z, k.c(12, "TOUCH_SWEEP"), at(14)),
    "conflicting zone": lambda e, z, k: e.process_candle(conflicting_zone(z), k.c(13, "SWEEP_NO_TOUCH"), at(14)),
    "mixed naive and aware": lambda e, z, k: e.process_candle(
        z, Candle(M5, aware(at(13)), 101, 104, 100, 102), at(14)),
}
 
 
def test_a_rejected_call_changes_nothing_and_the_engine_stays_healthy():
    for name in REJECTIONS:
        for kit in KITS:
            engine, liq, zone = env(kit)
            setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
            assert run(engine, zone, kit.c(12, "DOJI")) is None                     # established: waiting, last candle = slot 12
            before = snap(engine, zone)
            levels = liq.levels
            with pytest.raises((TypeError, ValueError)):
                REJECTIONS[name](engine, zone, kit)
            assert snap(engine, zone) == before, name
            assert liq.levels == levels
            probe = run(engine, zone, kit.c(13, "SWEEP_NO_TOUCH"))                  # slot 13 is still next and still sweeps
            assert probe is setup and engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION, name
 
 
def test_wrong_argument_types_are_rejected():
    for kit in KITS:
        engine, liq, zone = env(kit)
        candle = kit.c(11, "TOUCH_ONLY")
        for bad in (None, "zone", 5, 1.5, [], object(), zone.pivot):
            with pytest.raises(TypeError):
                engine.process_candle(bad, candle, candle.close_time)
        for bad in (None, "candle", 5, (1, 2), [], object()):
            with pytest.raises(TypeError):
                engine.process_candle(zone, bad, candle.close_time)
        for bad in (None, "now", 5, 1.5, [], date(2026, 1, 1)):
            with pytest.raises(TypeError):
                engine.process_candle(zone, candle, bad)
        assert engine.setups == ()
 
 
def test_mixed_naive_and_aware_datetimes_raise_type_error_and_change_nothing():
    for kit in KITS:
        engine, liq, zone = env(kit)
        good = kit.c(11, "TOUCH_ONLY")
        aware_candle = Candle(M5, aware(at(11)), 101, 104, 100, 102)
        for args in ((zone, aware_candle, at(12)), (zone, good, aware(good.close_time)),
                     (zone, aware_candle, aware(aware_candle.close_time))):
            with pytest.raises(TypeError):
                engine.process_candle(*args)
        assert engine.setups == ()
        assert run(engine, zone, good) is not None                                  # still healthy
 
 
def test_chronology_is_per_zone_not_global():
    for kit in KITS:
        zone_a, zone_b = kit.zone(bos_i=10, broken_i=2), kit.zone(bos_i=14, broken_i=3)
        liq = make_liquidity(kit.level_swing())
        engine = SetupEngine(liq)
        run(engine, zone_a, kit.c(40, "TOUCH_ONLY"))                               # zone A is far ahead in time
        assert run(engine, zone_b, kit.c(15, "TOUCH_ONLY")) is not None             # zone B may still receive an earlier candle
        with pytest.raises(ValueError):
            run(engine, zone_a, kit.c(20, "DOJI"))                                  # but A itself may not go back
 
 
def test_back_to_back_and_gapped_candles_are_accepted():
    for kit in KITS:
        engine, liq, zone = env(kit)
        run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        run(engine, zone, kit.c(12, "DOJI"))                                         # opens exactly when the previous closed
        run(engine, zone, kit.c(40, "DOJI"))                                         # a gap
        assert run(engine, zone, kit.c(900, "SWEEP_NO_TOUCH")) is not None           # a very large gap
 
 
def test_zones_of_every_timeframe_are_processed_with_5m_candles():
    for kit in KITS:
        for tf in (M5, M15, H1):
            zone = kit.zone(tf=tf)
            liq = make_liquidity(kit.level_swing())
            engine = SetupEngine(liq)
            setup = run(engine, zone, Candle(M5, zone.created_at, *(
                SPEC["TOUCH_SWEEP_CONFIRM"] if kit.long else (209.0 - 97.2, 209.0 - 97.0, 209.0 - 104.8, 209.0 - 104.5))))
            assert setup is not None and setup.zone.timeframe is tf and setup.touch_candle.timeframe is M5
            assert engine.status_for(setup) is S.CONFIRMED
 
 
def test_no_lookahead_a_level_known_only_when_the_candle_closes_cannot_be_swept_by_it():
    for kit in KITS:
        zone = kit.zone()
        late = kit.level_swing(i=11)                                               # known at slot 13
        liq = make_liquidity(late)
        assert late.confirmed_at == at(13)
        engine = SetupEngine(liq)
        setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        assert run(engine, zone, kit.c(12, "SWEEP_NO_TOUCH")) is None               # candle closes exactly when the level becomes known
        assert engine.status_for(setup) is S.WAITING_FOR_SWEEP
        assert run(engine, zone, kit.c(13, "SWEEP_NO_TOUCH")) is setup              # a later candle may use it
        assert engine.status_for(setup) is S.WAITING_FOR_CONFIRMATION
 
 
def test_no_lookahead_results_depend_only_on_candles_already_processed():
    keys = ("TOUCH_ONLY", "DOJI", "SWEEP_NO_TOUCH", "TOUCH_SWEEP", "CONFIRM", "WRONG_BOUNDARY",
            "WRONG_INSIDE", "TOUCH_SWEEP_CONFIRM", "CLOSE_LOW", "FAR_OUTSIDE", "TOUCH_SWEEP_WRONG")
    for kit in KITS:
        for seed in range(25):
            rng = random.Random(seed)
            slot, stream = 11, []
            for _ in range(40):
                stream.append(kit.c(slot, rng.choice(keys)))
                slot += rng.choice((1, 1, 1, 2, 5))
            engine, liq, zone = env(kit)
            history = []
            for candle in stream:
                run(engine, zone, candle)
                history.append(snap(engine, zone))
            for k in (3, 10, 25, 39):
                eng, _, z = env(kit)
                for candle in stream[:k + 1]:
                    run(eng, z, candle)
                assert snap(eng, z) == history[k], (kit.name, seed, k)
 
 
def test_invariants_hold_on_random_streams():
    keys = ("TOUCH_ONLY", "DOJI", "SWEEP_NO_TOUCH", "TOUCH_SWEEP", "CONFIRM", "WRONG_BOUNDARY",
            "WRONG_INSIDE", "TOUCH_SWEEP_CONFIRM", "CLOSE_LOW", "CLOSE_HIGH", "FAR_OUTSIDE", "TOUCH_SWEEP_WRONG")
    created = resolved_seen = 0
    for kit in KITS:
        for seed in range(40):
            rng = random.Random(1000 + seed)
            engine, liq, zone = env(kit)
            slot, prev_setups, frozen = 11, (), {}
            for _ in range(60):
                run(engine, zone, kit.c(slot, rng.choice(keys)))
                slot += rng.choice((1, 1, 2, 7))
                setups = engine.setups
                assert setups[:len(prev_setups)] == prev_setups                      # append-only
                prev_setups = setups
                active = [s for s in setups if engine.status_for(s).is_active]
                assert len(active) <= 1                                              # one active setup per zone
                for s in setups:
                    status = engine.status_for(s)
                    if status.is_resolved:
                        key = (s.identity)
                        state = (status, engine.transitions_for(s), engine.confirmation_for(s), engine.invalidation_for(s))
                        assert frozen.setdefault(key, state) == state                 # a resolved setup never changes
                    assert statuses(engine, s)[0] is S.ZONE_TOUCHED
                    assert (engine.confirmation_for(s) is None) or (engine.invalidation_for(s) is None)
                    assert (status is S.INVALIDATED) == (engine.invalidation_for(s) is not None)
                    assert (status in (S.CONFIRMED, S.TRADE_CREATED)) == (engine.confirmation_for(s) is not None)
            created += len(engine.setups)
            resolved_seen += len(frozen)
    assert created > 0 and resolved_seen > 0
 
 
def test_no_upstream_object_is_modified_and_the_liquidity_engine_is_only_read():
    for kit in KITS:
        engine, liq, zone = env(kit)
        level = liq.levels[0]
        before = (repr(zone), repr(level), liq.levels)
 
        def forbidden(*args, **kwargs):
            raise AssertionError("SetupEngine must only READ the LiquidityEngine")
 
        liq.process_swing = forbidden
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        run(engine, zone, kit.c(20, "TOUCH_SWEEP_CONFIRM"))
        engine.record_trade_created(engine.setups[1], at(30))
        assert (repr(zone), repr(level), liq.levels) == before
        assert engine.sweep_for(setup).zone is zone and engine.sweep_for(setup).level is level
        assert zone == kit.zone() and level == liq.levels[0]
 
 
def test_every_upstream_result_is_a_real_upstream_object():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))
        assert type(engine.sweep_for(setup)) is Sweep and type(engine.confirmation_for(setup)) is Confirmation
        assert engine.confirmation_for(setup).sweep is engine.sweep_for(setup)
        assert engine.sweep_for(setup).identity == SweepIdentity.from_parts(zone, liq.levels[0])
 
 
# ================================================================== read-only queries
def test_queries_for_unseen_inputs_are_empty_and_change_nothing():
    for kit in KITS:
        engine, liq, zone = env(kit)
        stranger = Setup(kit.zone(bos_i=14, broken_i=3), kit.c(15, "TOUCH_ONLY"))
        before = snap(engine, zone)
        assert engine.status_for(stranger) is None and engine.transitions_for(stranger) == ()
        assert engine.sweep_for(stranger) is None and engine.confirmation_for(stranger) is None
        assert engine.invalidation_for(stranger) is None and engine.confirmation_event_key_for(stranger) is None
        assert engine.can_create_trade(stranger) is False
        assert engine.setups_for_zone(zone) == () and engine.active_setup_for(zone) is None
        assert engine.status_for_zone(zone) is S.NO_SETUP
        assert engine.is_zone_consumed(zone) is False and engine.consumed_by(zone) is None
        assert engine.confirmed_setups_known_at(at(1000)) == ()
        assert snap(engine, zone) == before
        assert run(engine, zone, kit.c(11, "TOUCH_ONLY")) is not None               # queries registered nothing
 
 
def test_queries_validate_their_arguments():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_ONLY"))
        for name in ("status_for", "sweep_for", "confirmation_for", "invalidation_for", "transitions_for",
                     "confirmation_event_key_for", "can_create_trade"):
            for bad in (None, "x", 5, zone, setup.touch_candle):
                with pytest.raises(TypeError):
                    getattr(engine, name)(bad)
        for name in ("setups_for_zone", "active_setup_for", "status_for_zone", "is_zone_consumed", "consumed_by"):
            for bad in (None, "x", 5, zone.pivot, setup):
                with pytest.raises(TypeError):
                    getattr(engine, name)(bad)
        for bad in (None, "now", 5, date(2026, 1, 1)):
            with pytest.raises(TypeError):
                engine.confirmed_setups_known_at(bad)
 
 
def test_confirmed_setups_known_at_respects_the_confirmation_close_time():
    for kit in KITS:
        engine, liq, zone = env(kit)
        setup = run(engine, zone, kit.c(11, "TOUCH_SWEEP_CONFIRM"))                  # known when slot 11 closes (slot 12)
        assert engine.confirmed_setups_known_at(at(11)) == ()
        assert engine.confirmed_setups_known_at(at(12) - timedelta(seconds=1)) == ()
        assert engine.confirmed_setups_known_at(at(12)) == (setup,)
        assert engine.confirmed_setups_known_at(at(9999)) == (setup,)
        with pytest.raises(TypeError):
            engine.confirmed_setups_known_at(aware(at(12)))
 
 
def test_setups_are_returned_as_immutable_append_only_tuples():
    for kit in KITS:
        engine, liq, zone = env(kit)
        first = run(engine, zone, kit.c(11, "TOUCH_SWEEP"))
        older = engine.setups
        run(engine, zone, kit.c(12, "WRONG_BOUNDARY"))
        second = run(engine, zone, kit.c(20, "TOUCH_ONLY"))
        assert isinstance(engine.setups, tuple) and older == (first,) and engine.setups == (first, second)
        assert engine.setups[:1] == older
        assert isinstance(engine.transitions_for(first), tuple)
 
 
def test_several_zones_are_tracked_independently():
    for kit in KITS:
        zone_a, zone_b = kit.zone(bos_i=10, broken_i=2), kit.zone(bos_i=14, broken_i=3)
        liq = make_liquidity(kit.level_swing())
        engine = SetupEngine(liq)
        sa = run(engine, zone_a, kit.c(15, "TOUCH_SWEEP_CONFIRM"))
        assert engine.setups_for_zone(zone_b) == () and engine.status_for_zone(zone_b) is S.NO_SETUP
        sb = run(engine, zone_b, kit.c(15, "TOUCH_ONLY"))
        assert engine.status_for(sa) is S.CONFIRMED and engine.status_for(sb) is S.WAITING_FOR_SWEEP
        engine.record_trade_created(sa, at(20))
        assert engine.is_zone_consumed(zone_a) and not engine.is_zone_consumed(zone_b)
        assert engine.setups == (sa, sb)
 
 
def test_demand_and_supply_zones_in_one_engine_use_their_own_liquidity():
    liq = make_liquidity(mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(HIGH, 111.0, i=1, seq=1))
    demand, supply = mk_zone(BULL, bos_i=10, broken_i=2), mk_zone(BEAR, bos_i=14, broken_i=3)
    engine = SetupEngine(liq)
    sd = run(engine, demand, Kit("LONG").c(15, "TOUCH_SWEEP"))
    ss = run(engine, supply, Kit("SHORT").c(15, "TOUCH_SWEEP"))
    assert engine.sweep_for(sd).level.swing.swing_type is LOW
    assert engine.sweep_for(ss).level.swing.swing_type is HIGH
 
 
# ================================================================== scope
def test_the_engine_exposes_no_entry_risk_execution_or_upstream_logic():
    for kit in KITS:
        engine, liq, zone = env(kit)
        for attr in ("entry", "sl", "tp", "stop_loss", "take_profit", "risk", "news", "execute", "execution",
                     "bos", "choch", "tolerance", "distance", "atr", "displacement", "session", "timeout",
                     "expire", "expired", "delete", "remove", "clear", "reset", "invalidate", "revive",
                     "process_swing", "process_zone", "process_sweep", "process_confirmation", "merge_zones"):
            assert not hasattr(engine, attr), attr
        for name in ("StructureEngine", "BOSEngine", "CHOCHEngine", "PivotEngine", "ZoneEngine", "SwingEngine"):
            assert not hasattr(engine_module, name), name
 
 
def test_the_engine_does_not_repeat_confirmation_or_sweep_rules():
    source = inspect.getsource(engine_module)
    assert "abs(" not in source and "BRR" not in source and "classify" not in source
    assert "penetrates" not in source and "closes_beyond_boundary" not in source
    assert "CandleClass" not in source and "InvalidationReason" not in source
 
 
def test_the_engine_module_imports_only_the_expected_components():
    tree = ast.parse(inspect.getsource(engine_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= {
        "__future__", "datetime", "typing",
        "koffie.strategy.engines.confirmation_engine",
        "koffie.strategy.engines.liquidity_engine",
        "koffie.strategy.engines.sweep_engine",
        "koffie.strategy.models.candle",
        "koffie.strategy.models.confirmation",
        "koffie.strategy.models.pivot",
        "koffie.strategy.models.setup",
        "koffie.strategy.models.sweep",
        "koffie.strategy.models.zone",
    }, imported