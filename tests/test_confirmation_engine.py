"""Engine-level tests for ConfirmationEngine: the locked confirmation sequence, replay,
chronology, terminal behaviour, independence of Sweeps and input validation.
 
Only real project objects are used (Zone, Pivot, BOS, LiquidityLevel, Sweep, Candle). The
sequence is tested for LONG (DEMAND sweep) and SHORT (SUPPLY sweep) with the same scenarios.
"""
import ast
import inspect
from datetime import date, datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.confirmation_engine as engine_module
from koffie.strategy.engines.confirmation_engine import ConfirmationEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.confirmation import (
    Confirmation, ConfirmationStatus, Invalidation, InvalidationReason,
)
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.sweep import Sweep
from koffie.strategy.models.zone import ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
S = ConfirmationStatus
BOUNDARY, SECOND = InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY, InvalidationReason.SECOND_WRONG_DIRECTION_DECISIVE
 
 
# ------------------------------------------------------------------ helpers
def at(i, tf=M5):
    return T0 + i * tf.duration
 
 
def mk_swing(swing_type, price, i=0, seq=0, tf=M5):
    candle_time = T0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5, bos_i=10, broken_i=2):
    """Zone 99..110. Default 5M zone is created (known) at slot 11."""
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=broken_i, seq=0, tf=tf)
    bos = BOS(tf, direction, broken, T0 + bos_i * tf.duration, 101.0 if direction is BULL else 99.0)
    return Zone_from(bos, T0 + 5 * tf.duration, high, low, pivot_class)
 
 
def Zone_from(bos, pivot_time, high, low, pivot_class):
    from koffie.strategy.models.zone import Zone
    return Zone(Pivot(bos, pivot_time, high, low, pivot_class))
 
 
def ohlc(i, o, h, l, c, tf=M5):
    return Candle(tf, at(i, tf), o, h, l, c)
 
 
def classify(candle):
    return candle.classify(candle.close_time)
 
 
def go(engine, sweep, candle, now=None):
    return engine.process_candle(sweep, candle, candle.close_time if now is None else now)
 
 
def snap(engine, *sweeps):
    return (engine.confirmations, engine.invalidations, tuple(engine.status_for(s) for s in sweeps))
 
 
class Side:
    """One direction's scenario kit. Candle slots are chosen by the caller (default after the sweep at slot 11)."""
 
    def __init__(self, name):
        self.name = name
        self.long = name == "LONG"
        self.direction = BULL if self.long else BEAR
        self.zone_type = ZoneType.DEMAND if self.long else ZoneType.SUPPLY
        self.zone = mk_zone(self.direction)
        self.level = LiquidityLevel(mk_swing(LOW, 98.0) if self.long else mk_swing(HIGH, 111.0))
        # candles that can be the SWEEP candle (slot 11)
        if self.long:      # low must be < 98
            self.sweep_neutral = ohlc(11, 98.5, 100.0, 97.5, 98.8)
            self.sweep_confirming = ohlc(11, 97.1, 100.0, 97.0, 99.9)
            self.sweep_wrong_inside = ohlc(11, 104.0, 104.0, 97.0, 99.0)       # bearish decisive, close == zone.low
            self.sweep_wrong_boundary = ohlc(11, 99.5, 99.6, 97.0, 97.1)
        else:              # high must be > 111
            self.sweep_neutral = ohlc(11, 110.5, 112.0, 109.5, 110.8)
            self.sweep_confirming = ohlc(11, 112.0, 112.4, 105.0, 105.2)
            self.sweep_wrong_inside = ohlc(11, 105.0, 111.5, 104.9, 109.9)     # bullish decisive, close <= zone.high
            self.sweep_wrong_boundary = ohlc(11, 110.2, 112.0, 110.1, 111.9)
 
    def sweep(self, candle=None):
        return Sweep(self.zone, self.level, self.sweep_neutral if candle is None else candle)
 
    # -- candles AFTER the sweep ----------------------------------------------------
    def confirm(self, i):
        return ohlc(i, 100.0, 105.0, 99.8, 104.8) if self.long else ohlc(i, 104.8, 105.0, 100.0, 100.2)
 
    def wrong(self, i):                                   # wrong-direction decisive, close inside the zone
        return ohlc(i, 104.8, 105.0, 100.0, 100.2) if self.long else ohlc(i, 100.0, 105.0, 99.8, 104.8)
 
    def wrong_boundary(self, i):                          # wrong-direction decisive closing beyond the boundary
        return ohlc(i, 99.5, 99.6, 97.0, 97.1) if self.long else ohlc(i, 110.5, 113.0, 110.4, 112.9)
 
    def wrong_wick_only(self, i):                         # wick beyond the boundary, close back inside
        return ohlc(i, 103.0, 103.2, 98.0, 99.2) if self.long else ohlc(i, 106.0, 111.0, 105.8, 109.8)
 
    def wrong_close_equal(self, i):                       # close exactly on the boundary
        return ohlc(i, 103.0, 103.1, 98.9, 99.0) if self.long else ohlc(i, 106.0, 110.1, 105.9, 110.0)
 
    def confirm_outside(self, i):                         # confirming candles closing OUTSIDE the zone
        if self.long:
            return [ohlc(i, 95.0, 98.9, 94.9, 98.8), ohlc(i, 111.0, 118.0, 110.5, 117.5)]
        return [ohlc(i, 118.0, 118.2, 111.0, 111.3), ohlc(i, 98.9, 99.0, 94.0, 94.2)]
 
    def confirm_050(self, i):                             # BRR exactly 0.50: NEUTRAL
        return ohlc(i, 100.0, 110.0, 100.0, 105.0) if self.long else ohlc(i, 105.0, 110.0, 100.0, 100.0)
 
    def confirm_0501(self, i):                            # BRR just above 0.50: decisive
        return ohlc(i, 100.0, 110.0, 100.0, 105.01) if self.long else ohlc(i, 105.01, 110.0, 100.0, 100.0)
 
    def doji(self, i):
        return ohlc(i, 100.0, 104.0, 99.0, 101.0)
 
    def zero(self, i):
        return ohlc(i, 100.0, 100.0, 100.0, 100.0)
 
    @property
    def confirming_class(self):
        return CandleClass.BULLISH_DECISIVE if self.long else CandleClass.BEARISH_DECISIVE
 
    @property
    def wrong_class(self):
        return CandleClass.BEARISH_DECISIVE if self.long else CandleClass.BULLISH_DECISIVE
 
 
@pytest.fixture(params=["LONG", "SHORT"])
def side(request):
    return Side(request.param)
 
 
def test_the_fixtures_have_the_classes_the_tests_assume(side):
    assert classify(side.confirm(12)) is side.confirming_class
    assert classify(side.wrong(12)) is side.wrong_class
    assert classify(side.wrong_boundary(12)) is side.wrong_class
    assert classify(side.wrong_wick_only(12)) is side.wrong_class
    assert classify(side.wrong_close_equal(12)) is side.wrong_class
    assert all(classify(c) is side.confirming_class for c in side.confirm_outside(12))
    assert classify(side.confirm_050(12)) is CandleClass.NEUTRAL
    assert classify(side.confirm_0501(12)) is side.confirming_class
    assert classify(side.doji(12)) is CandleClass.NEUTRAL
    assert classify(side.zero(12)) is CandleClass.ZERO_RANGE_ANOMALY
    assert classify(side.sweep_neutral) is CandleClass.NEUTRAL
    assert classify(side.sweep_confirming) is side.confirming_class
    assert classify(side.sweep_wrong_inside) is side.wrong_class
    assert classify(side.sweep_wrong_boundary) is side.wrong_class
 
 
def started(side, sweep_candle=None):
    """Engine with the sweep's own candle already processed."""
    sweep = side.sweep(sweep_candle)
    engine = ConfirmationEngine()
    result = go(engine, sweep, sweep.candle)
    return engine, sweep, result
 
 
# ================================================================== construction and shape
def test_constructor_takes_no_arguments_and_the_engine_is_m5():
    engine = ConfirmationEngine()
    assert engine.timeframe is Timeframe.M5
    assert engine.confirmations == () and engine.invalidations == ()
    with pytest.raises(TypeError):
        ConfirmationEngine(object())
    with pytest.raises(AttributeError):
        engine.timeframe = H1
 
 
# ================================================================== the locked sequence (LONG and SHORT)
def test_first_confirming_decisive_candle_confirms(side):
    engine, sweep, first = started(side)
    assert first is None and engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
    candle = side.confirm(12)
    result = go(engine, sweep, candle)
    assert isinstance(result, Confirmation) and result.sweep is sweep and result.candle is candle
    assert result.zone_type is side.zone_type and result.known_at == candle.close_time
    assert engine.status_for(sweep) is S.CONFIRMED
    assert engine.confirmations == (result,) and engine.invalidations == ()
    assert engine.confirmation_for(sweep) is result and engine.invalidation_for(sweep) is None
 
 
def test_dojis_keep_waiting(side):
    engine, sweep, _ = started(side)
    for k in range(12, 40):
        assert go(engine, sweep, side.doji(k)) is None
        assert engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
    assert engine.confirmations == () and engine.invalidations == ()
    assert isinstance(go(engine, sweep, side.confirm(40)), Confirmation)       # the sequence was never consumed
 
 
def test_zero_range_candles_keep_waiting(side):
    engine, sweep, _ = started(side)
    for k in range(12, 30):
        assert go(engine, sweep, side.zero(k)) is None
        assert engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
    assert isinstance(go(engine, sweep, side.confirm(30)), Confirmation)
 
 
def test_first_wrong_direction_decisive_waits_for_a_second_decisive(side):
    engine, sweep, _ = started(side)
    assert go(engine, sweep, side.wrong(12)) is None
    assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
    assert engine.confirmations == () and engine.invalidations == ()
 
 
def test_wrong_then_confirming_decisive_confirms(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    result = go(engine, sweep, side.confirm(13))
    assert isinstance(result, Confirmation) and result.candle_time == at(13)
    assert engine.status_for(sweep) is S.CONFIRMED
 
 
def test_wrong_then_wrong_again_invalidates(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    candle = side.wrong(13)
    result = go(engine, sweep, candle)
    assert isinstance(result, Invalidation) and result.candle is candle
    assert result.reason is SECOND                                       # closes inside the zone: not the boundary label
    assert engine.status_for(sweep) is S.INVALIDATED
    assert engine.invalidations == (result,) and engine.confirmations == ()
    assert engine.invalidation_for(sweep) is result and engine.confirmation_for(sweep) is None
 
 
def test_neutral_candles_do_not_consume_the_second_decisive_state(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    for k in range(13, 20):
        assert go(engine, sweep, side.doji(k) if k % 2 else side.zero(k)) is None
        assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
    assert isinstance(go(engine, sweep, side.confirm(20)), Confirmation)
 
 
def test_a_wrong_direction_decisive_closing_beyond_the_boundary_invalidates_immediately(side):
    engine, sweep, _ = started(side)
    candle = side.wrong_boundary(12)
    result = go(engine, sweep, candle)                                   # the FIRST wrong-direction decisive candle
    assert isinstance(result, Invalidation) and result.reason is BOUNDARY
    assert engine.status_for(sweep) is S.INVALIDATED
 
 
def test_a_second_wrong_direction_candle_that_also_closes_beyond_the_boundary_uses_the_boundary_label(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    result = go(engine, sweep, side.wrong_boundary(13))
    assert isinstance(result, Invalidation) and result.reason is BOUNDARY
 
 
def test_a_wick_beyond_the_boundary_without_the_close_does_not_invalidate(side):
    engine, sweep, _ = started(side)
    assert go(engine, sweep, side.wrong_wick_only(12)) is None
    assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE     # a normal first wrong-direction candle
    assert engine.invalidations == ()
    result = go(engine, sweep, side.confirm(13))
    assert isinstance(result, Confirmation)
 
 
def test_a_close_exactly_on_the_boundary_does_not_invalidate(side):
    engine, sweep, _ = started(side)
    assert go(engine, sweep, side.wrong_close_equal(12)) is None
    assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
 
 
def test_a_second_wick_only_wrong_candle_invalidates_with_the_second_label(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    result = go(engine, sweep, side.wrong_wick_only(13))
    assert isinstance(result, Invalidation) and result.reason is SECOND
 
 
@pytest.mark.parametrize("which", [0, 1])
def test_a_confirming_candle_may_close_outside_the_zone_and_remain_valid(side, which):
    candle = side.confirm_outside(12)[which]
    engine, sweep, _ = started(side)
    result = go(engine, sweep, candle)
    assert isinstance(result, Confirmation) and result.candle is candle
    # also as the SECOND decisive candle
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    assert isinstance(go(engine, sweep, side.confirm_outside(13)[which]), Confirmation)
 
 
def test_brr_exactly_050_is_neutral_and_just_above_is_decisive(side):
    engine, sweep, _ = started(side)
    assert go(engine, sweep, side.confirm_050(12)) is None
    assert engine.status_for(sweep) is S.WAITING_FOR_DECISIVE            # exactly 0.70 did not confirm or consume
    assert isinstance(go(engine, sweep, side.confirm_0501(13)), Confirmation)
 
 
# ================================================================== the sweep candle itself (D1, Q2)
def test_the_sweep_candle_can_confirm(side):
    engine, sweep, result = started(side, side.sweep_confirming)
    assert isinstance(result, Confirmation) and result.candle is sweep.candle
    assert engine.status_for(sweep) is S.CONFIRMED
 
 
def test_the_sweep_candle_can_enter_waiting_for_second_decisive(side):
    engine, sweep, result = started(side, side.sweep_wrong_inside)
    assert result is None and engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
    assert isinstance(go(engine, sweep, side.confirm(12)), Confirmation)
 
 
def test_the_sweep_candle_can_invalidate(side):
    engine, sweep, result = started(side, side.sweep_wrong_boundary)
    assert isinstance(result, Invalidation) and result.candle is sweep.candle and result.reason is BOUNDARY
    assert engine.status_for(sweep) is S.INVALIDATED
 
 
def test_a_neutral_sweep_candle_starts_the_waiting(side):
    engine, sweep, result = started(side)
    assert result is None and engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
 
 
def test_the_first_candle_must_be_the_sweeps_own_candle(side):
    sweep, engine = side.sweep(), ConfirmationEngine()
    before = snap(engine, sweep)
    for later in (side.confirm(12), side.confirm(13), side.wrong(12)):
        with pytest.raises(ValueError):
            go(engine, sweep, later)
    different_same_slot = ohlc(11, side.sweep_neutral.open + 0.01, side.sweep_neutral.high,
                               side.sweep_neutral.low, side.sweep_neutral.close)
    with pytest.raises(ValueError):
        go(engine, sweep, different_same_slot)
    assert snap(engine, sweep) == before and engine.status_for(sweep) is None
    assert go(engine, sweep, sweep.candle) is None                       # the sweep candle starts it
    assert engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
 
 
def test_evaluation_cannot_silently_start_later(side):
    # a bearish-decisive close below the boundary on the sweep candle must not be skippable
    sweep = side.sweep(side.sweep_wrong_boundary)
    engine = ConfirmationEngine()
    with pytest.raises(ValueError):
        go(engine, sweep, side.confirm(12))                              # would wrongly "confirm" if allowed
    assert engine.confirmations == ()
    assert isinstance(go(engine, sweep, sweep.candle), Invalidation)
 
 
# ================================================================== terminal behaviour (D5, tests 23, 25)
def test_later_candles_never_alter_a_confirmed_result(side):
    engine, sweep, _ = started(side)
    confirmation = go(engine, sweep, side.confirm(12))
    before = snap(engine, sweep)
    for k, candle in enumerate((side.confirm(13), side.wrong(14), side.wrong_boundary(15), side.doji(16),
                                side.zero(17), side.confirm(18))):
        assert go(engine, sweep, candle) is None
    assert snap(engine, sweep) == before
    assert engine.status_for(sweep) is S.CONFIRMED and engine.confirmation_for(sweep) is confirmation
 
 
def test_later_candles_never_alter_an_invalidated_result(side):
    engine, sweep, _ = started(side)
    invalidation = go(engine, sweep, side.wrong_boundary(12))
    before = snap(engine, sweep)
    for candle in (side.confirm(13), side.confirm_outside(14)[0], side.wrong(15), side.doji(16)):
        assert go(engine, sweep, candle) is None
    assert snap(engine, sweep) == before
    assert engine.status_for(sweep) is S.INVALIDATED and engine.invalidation_for(sweep) is invalidation
    assert engine.confirmations == ()
 
 
def test_one_sweep_cannot_produce_multiple_confirmations_or_both_outcomes(side):
    engine, sweep, _ = started(side)
    for k in range(12, 40):
        go(engine, sweep, side.confirm(k))
    assert len(engine.confirmations) == 1 and engine.invalidations == ()
    engine, sweep, _ = started(side)
    for k in range(12, 40):
        go(engine, sweep, side.wrong(k))
    assert len(engine.invalidations) == 1 and engine.confirmations == ()
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong_boundary(12))
    go(engine, sweep, side.confirm(13))
    assert len(engine.invalidations) == 1 and engine.confirmations == ()
 
 
def test_after_a_terminal_state_candles_are_still_checked_for_validity(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.confirm(12))
    go(engine, sweep, side.doji(20))
    before = snap(engine, sweep)
    with pytest.raises(ValueError):
        go(engine, sweep, side.doji(15))                                  # earlier than the last candle
    with pytest.raises(ValueError):
        go(engine, sweep, ohlc(20, 100.0, 104.0, 99.0, 101.5))            # same slot, different data
    with pytest.raises(ValueError):
        go(engine, sweep, ohlc(21, 100.0, 105.0, 99.8, 104.8, tf=M15), now=at(500))   # not 5M
    with pytest.raises(ValueError):
        engine.process_candle(sweep, side.doji(25), at(25))               # not closed yet
    assert snap(engine, sweep) == before
 
 
def test_there_is_no_timeout(side):
    engine, sweep, _ = started(side)
    for k in range(12, 2012):                                             # ~7 days of 5M candles
        assert go(engine, sweep, side.doji(k) if k % 2 else side.zero(k)) is None
    assert engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
    assert isinstance(go(engine, sweep, side.confirm(5000)), Confirmation)    # after a long gap too
 
 
def test_there_is_no_timeout_in_the_second_decisive_state(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.wrong(12))
    for k in range(13, 1500):
        go(engine, sweep, side.doji(k))
    assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
    assert isinstance(go(engine, sweep, side.confirm(3000)), Confirmation)
 
 
# ================================================================== independence (tests 26, Q8, Q11)
def test_separate_sweeps_of_one_zone_are_independent():
    side = Side("LONG")
    zone = side.zone
    l1, l2 = LiquidityLevel(mk_swing(LOW, 98.0, i=0, seq=0)), LiquidityLevel(mk_swing(LOW, 97.0, i=12, seq=1))
    s1 = Sweep(zone, l1, ohlc(11, 98.5, 100.0, 97.5, 98.8))               # sweep of the first level
    s2 = Sweep(zone, l2, ohlc(14, 97.5, 100.0, 96.5, 97.8))               # a later, different level
    assert s1.identity != s2.identity and s1.zone == s2.zone
    engine = ConfirmationEngine()
    go(engine, s1, s1.candle)
    go(engine, s1, side.wrong_boundary(12))                               # s1 invalidated
    assert engine.status_for(s1) is S.INVALIDATED
    assert go(engine, s2, s2.candle) is None                              # s2 starts fresh and is unaffected
    assert engine.status_for(s2) is S.WAITING_FOR_DECISIVE
    result = go(engine, s2, side.confirm(15))
    assert isinstance(result, Confirmation) and result.sweep is s2
    assert engine.status_for(s1) is S.INVALIDATED and engine.status_for(s2) is S.CONFIRMED
    assert len(engine.confirmations) == 1 and len(engine.invalidations) == 1
 
 
def test_one_candle_can_advance_several_sweeps_independently():
    long_, short_ = Side("LONG"), Side("SHORT")
    sl, ss = long_.sweep(), short_.sweep()
    engine = ConfirmationEngine()
    go(engine, sl, sl.candle)
    go(engine, ss, ss.candle)
    candle = ohlc(12, 100.0, 105.0, 99.8, 104.8)                          # BULLISH_DECISIVE
    assert isinstance(go(engine, sl, candle), Confirmation)               # confirms the LONG sweep
    assert go(engine, ss, candle) is None                                 # first wrong-direction for the SHORT sweep
    assert engine.status_for(sl) is S.CONFIRMED and engine.status_for(ss) is S.WAITING_FOR_SECOND_DECISIVE
 
 
@pytest.mark.parametrize("tf", [M5, M15, H1])
@pytest.mark.parametrize("name", ["LONG", "SHORT"])
def test_zones_of_every_timeframe_are_confirmed_on_5m_candles(tf, name):
    long_ = name == "LONG"
    direction = BULL if long_ else BEAR
    zone = mk_zone(direction, tf=tf)
    level = LiquidityLevel(mk_swing(LOW, 98.0) if long_ else mk_swing(HIGH, 111.0))
    first = zone.created_at                                               # a 5M candle opening when the zone is created
    sweep_candle = Candle(M5, first, 98.5, 100.0, 97.5, 98.8) if long_ else Candle(M5, first, 110.5, 112.0, 109.5, 110.8)
    sweep = Sweep(zone, level, sweep_candle)
    engine = ConfirmationEngine()
    assert go(engine, sweep, sweep.candle) is None
    confirm = Candle(M5, first + M5.duration, 100.0, 105.0, 99.8, 104.8) if long_ else \
        Candle(M5, first + M5.duration, 104.8, 105.0, 100.0, 100.2)
    result = go(engine, sweep, confirm)
    assert isinstance(result, Confirmation) and result.zone.timeframe is tf and result.candle.timeframe is M5
 
 
def test_the_logic_does_not_depend_on_the_zone_timeframe():
    outcomes = []
    for tf in (M5, M15, H1):
        zone = mk_zone(BULL, tf=tf)
        level = LiquidityLevel(mk_swing(LOW, 98.0))
        t = zone.created_at
        sweep = Sweep(zone, level, Candle(M5, t, 98.5, 100.0, 97.5, 98.8))
        engine = ConfirmationEngine()
        go(engine, sweep, sweep.candle)
        steps = [Candle(M5, t + M5.duration, 104.8, 105.0, 100.0, 100.2),        # wrong
                 Candle(M5, t + 2 * M5.duration, 100.0, 104.0, 99.0, 101.0),     # doji
                 Candle(M5, t + 3 * M5.duration, 100.0, 105.0, 99.8, 104.8)]     # confirm
        outcomes.append([type(go(engine, sweep, c)).__name__ for c in steps] + [engine.status_for(sweep)])
    assert outcomes[0] == outcomes[1] == outcomes[2] == ["NoneType", "NoneType", "Confirmation", S.CONFIRMED]
 
 
# ================================================================== replay
def test_replaying_the_latest_candle_returns_the_same_result_without_mutation(side):
    engine, sweep, _ = started(side)
    waiting = side.wrong(12)
    assert go(engine, sweep, waiting) is None
    before = snap(engine, sweep)
    assert go(engine, sweep, waiting) is None
    assert go(engine, sweep, ohlc(12, waiting.open, waiting.high, waiting.low, waiting.close)) is None
    assert snap(engine, sweep) == before
    terminal = side.confirm(13)
    event = go(engine, sweep, terminal)
    before = snap(engine, sweep)
    assert go(engine, sweep, terminal) is event
    assert snap(engine, sweep) == before
 
 
def test_replaying_the_sweep_candle_after_it_was_processed(side):
    engine, sweep, first = started(side, side.sweep_confirming)
    assert isinstance(first, Confirmation)
    before = snap(engine, sweep)
    assert go(engine, sweep, sweep.candle) is first                       # latest candle
    go(engine, sweep, side.doji(12))
    go(engine, sweep, side.doji(13))
    assert go(engine, sweep, sweep.candle) is first                       # the terminal-producing candle, not the latest
    assert snap(engine, sweep)[:2] == before[:2]
 
 
def test_replaying_the_terminal_candle_after_later_candles_returns_the_existing_event(side):
    engine, sweep, _ = started(side)
    terminal_candle = side.wrong_boundary(12)
    event = go(engine, sweep, terminal_candle)
    for k in range(13, 18):
        go(engine, sweep, side.doji(k))
    before = snap(engine, sweep)
    assert go(engine, sweep, terminal_candle) is event
    assert go(engine, sweep, ohlc(12, 99.5 if side.long else 110.5, 99.6 if side.long else 113.0,
                                  97.0 if side.long else 110.4, 97.1 if side.long else 112.9)) is event
    assert snap(engine, sweep) == before and len(engine.invalidations) == 1
 
 
def test_replaying_an_earlier_non_terminal_candle_after_progression_is_rejected(side):
    engine, sweep, _ = started(side)
    quiet = side.doji(12)
    go(engine, sweep, quiet)
    wrong = side.wrong(13)
    go(engine, sweep, wrong)
    go(engine, sweep, side.doji(14))
    before = snap(engine, sweep)
    for earlier in (quiet, wrong, side.doji(13)):
        with pytest.raises(ValueError):
            go(engine, sweep, earlier)
    assert snap(engine, sweep) == before
    assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
    assert isinstance(go(engine, sweep, side.confirm(15)), Confirmation)  # state was not rewound
 
 
def test_replaying_a_terminal_candle_does_not_rewind_the_state(side):
    engine, sweep, _ = started(side)
    terminal = side.confirm(12)
    event = go(engine, sweep, terminal)
    latest = side.doji(14)
    go(engine, sweep, latest)
    assert go(engine, sweep, terminal) is event
    assert go(engine, sweep, latest) is None                              # the latest candle is still the latest one
    with pytest.raises(ValueError):
        go(engine, sweep, side.doji(13))
 
 
def test_replay_with_a_later_now_changes_nothing(side):
    engine, sweep, _ = started(side)
    candle = side.confirm(12)
    event = go(engine, sweep, candle)
    assert engine.process_candle(sweep, candle, candle.close_time + timedelta(days=3)) is event
    assert len(engine.confirmations) == 1
 
 
def test_an_equal_rebuilt_sweep_is_the_same_sweep(side):
    engine, sweep, _ = started(side)
    rebuilt = side.sweep()
    assert rebuilt is not sweep and rebuilt == sweep
    event = go(engine, rebuilt, side.confirm(12))
    assert isinstance(event, Confirmation) and engine.status_for(sweep) is S.CONFIRMED
    assert engine.status_for(rebuilt) is S.CONFIRMED
 
 
# ================================================================== sweep identity integrity
def test_a_different_sweep_with_the_same_identity_is_rejected(side):
    engine, sweep, _ = started(side)
    other = Sweep(side.zone, side.level, side.sweep_confirming if side.sweep_confirming != sweep.candle else side.sweep_wrong_inside)
    assert other.identity == sweep.identity and other != sweep
    before = snap(engine, sweep)
    with pytest.raises(ValueError):
        go(engine, other, other.candle)
    with pytest.raises(ValueError):
        go(engine, other, side.confirm(12))
    assert snap(engine, sweep) == before
    assert engine.status_for(other) is None                               # the conflicting sweep is not the tracked one
    assert isinstance(go(engine, sweep, side.confirm(12)), Confirmation)  # the original sweep is unaffected
 
 
def test_a_conflicting_sweep_is_rejected_even_after_the_first_became_terminal(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.confirm(12))
    other = Sweep(side.zone, side.level, side.sweep_wrong_inside)
    with pytest.raises(ValueError):
        go(engine, other, other.candle)
    assert len(engine.confirmations) == 1 and engine.invalidations == ()
 
 
# ================================================================== chronology
def test_back_to_back_and_gapped_candles_are_accepted(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.doji(12))
    go(engine, sweep, side.doji(13))                                      # opens exactly when the previous closed
    go(engine, sweep, side.doji(40))                                      # a gap
    assert isinstance(go(engine, sweep, side.confirm(900)), Confirmation)
 
 
def test_overlapping_and_earlier_candles_are_rejected(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.doji(14))
    before = snap(engine, sweep)
    overlapping = ohlc(14, 100.0, 105.0, 99.8, 104.8)
    overlapping = Candle(M5, at(14) + timedelta(minutes=2), 100.0, 105.0, 99.8, 104.8)
    for bad in (overlapping, side.confirm(13), side.confirm(12), side.confirm(14)):
        with pytest.raises(ValueError):
            go(engine, sweep, bad)
    assert snap(engine, sweep) == before
 
 
def test_a_rejected_candle_does_not_move_the_chronology_forward(side):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.doji(12))
    with pytest.raises(ValueError):
        engine.process_candle(sweep, side.confirm(13), at(13) - timedelta(seconds=1))   # unclosed
    assert isinstance(go(engine, sweep, side.confirm(13)), Confirmation)  # slot 13 is still next
 
 
# ================================================================== validation (tests 21, 27)
def test_unclosed_candles_cannot_be_used(side):
    engine, sweep, _ = started(side)
    candle = side.confirm(12)                                             # would confirm if it were closed
    before = snap(engine, sweep)
    for now in (candle.open_time, candle.open_time + timedelta(minutes=3), candle.close_time - timedelta(seconds=1)):
        with pytest.raises(ValueError):
            engine.process_candle(sweep, candle, now)
    assert snap(engine, sweep) == before
    assert isinstance(engine.process_candle(sweep, candle, candle.close_time), Confirmation)   # exactly at the close
 
 
def test_an_unclosed_sweep_candle_cannot_start_the_sequence(side):
    sweep, engine = side.sweep(side.sweep_confirming), ConfirmationEngine()
    with pytest.raises(ValueError):
        engine.process_candle(sweep, sweep.candle, sweep.candle.close_time - timedelta(seconds=1))
    assert engine.status_for(sweep) is None and engine.confirmations == ()
 
 
def test_non_5m_candles_are_rejected(side):
    engine, sweep, _ = started(side)
    before = snap(engine, sweep)
    for tf in (M15, H1):
        c = Candle(tf, at(12), 100.0, 105.0, 99.8, 104.8) if side.long else Candle(tf, at(12), 104.8, 105.0, 100.0, 100.2)
        with pytest.raises(ValueError):
            go(engine, sweep, c)
    assert snap(engine, sweep) == before
 
 
@pytest.mark.parametrize("bad", [None, "sweep", 5, 1.5, [], object()])
def test_wrong_sweep_types_are_rejected(side, bad):
    engine = ConfirmationEngine()
    with pytest.raises(TypeError):
        engine.process_candle(bad, side.confirm(12), at(13))
 
 
def test_a_sweep_component_is_not_accepted_as_a_sweep(side):
    engine, sweep = ConfirmationEngine(), side.sweep()
    for bad in (sweep.zone, sweep.level, sweep.candle, sweep.identity):
        with pytest.raises(TypeError):
            engine.process_candle(bad, side.confirm(12), at(13))
 
 
@pytest.mark.parametrize("bad", [None, "candle", 5, (1, 2), [], object()])
def test_wrong_candle_types_are_rejected(side, bad):
    engine, sweep, _ = started(side)
    with pytest.raises(TypeError):
        engine.process_candle(sweep, bad, at(13))
 
 
@pytest.mark.parametrize("bad", [None, "now", 5, 1.5, [], date(2026, 1, 1)])
def test_wrong_now_types_are_rejected(side, bad):
    engine, sweep, _ = started(side)
    with pytest.raises(TypeError):
        engine.process_candle(sweep, side.confirm(12), bad)
 
 
def aware(dt):
    return dt.replace(tzinfo=timezone.utc)
 
 
def test_mixed_naive_and_aware_datetimes_raise_type_error(side):
    engine, sweep, _ = started(side)
    before = snap(engine, sweep)
    aware_candle = Candle(M5, aware(at(12)), 100.0, 105.0, 99.8, 104.8)
    with pytest.raises(TypeError):
        engine.process_candle(sweep, aware_candle, at(13))
    with pytest.raises(TypeError):
        engine.process_candle(sweep, side.confirm(12), aware(at(13)))
    with pytest.raises(TypeError):
        engine.process_candle(sweep, aware_candle, aware(at(13)))         # aware candle against the naive sweep
    assert snap(engine, sweep) == before
    assert isinstance(go(engine, sweep, side.confirm(12)), Confirmation)
 
 
REJECTIONS = {
    "wrong sweep type": lambda e, s, d: e.process_candle("sweep", d.confirm(13), at(14)),
    "wrong candle type": lambda e, s, d: e.process_candle(s, "candle", at(14)),
    "wrong now type": lambda e, s, d: e.process_candle(s, d.confirm(13), None),
    "non-5m candle": lambda e, s, d: e.process_candle(s, Candle(M15, at(13), 100.0, 105.0, 99.8, 104.8), at(500)),
    "unclosed candle": lambda e, s, d: e.process_candle(s, d.confirm(13), at(13)),
    "overlapping candle": lambda e, s, d: e.process_candle(
        s, Candle(M5, at(12) + timedelta(minutes=2), 100.0, 105.0, 99.8, 104.8), at(14)),
    "earlier candle": lambda e, s, d: e.process_candle(s, d.confirm(11), at(12)),
    "conflicting sweep": lambda e, s, d: e.process_candle(
        Sweep(d.zone, d.level, d.sweep_confirming), d.sweep_confirming, d.sweep_confirming.close_time),
    "mixed naive and aware": lambda e, s, d: e.process_candle(
        s, Candle(M5, aware(at(13)), 100.0, 105.0, 99.8, 104.8), at(14)),
}
 
 
@pytest.mark.parametrize("name", list(REJECTIONS))
def test_a_rejected_call_changes_nothing(side, name):
    engine, sweep, _ = started(side)
    assert go(engine, sweep, side.doji(12)) is None                       # established: waiting, last candle = slot 12
    before = snap(engine, sweep)
    with pytest.raises((TypeError, ValueError)):
        REJECTIONS[name](engine, sweep, side)
    assert snap(engine, sweep) == before
    result = go(engine, sweep, side.confirm(13))                          # slot 13 is still next and still confirms
    assert isinstance(result, Confirmation) and result.candle_time == at(13)
 
 
# ================================================================== append-only order and queries
def test_confirmations_and_invalidations_are_append_only_in_acceptance_order():
    side = Side("LONG")
    zone = side.zone
    levels = [LiquidityLevel(mk_swing(LOW, 98.0 - k, i=k * 4, seq=k)) for k in range(3)]
    sweeps = [Sweep(zone, levels[0], ohlc(11, 98.5, 100.0, 97.5, 98.8)),
              Sweep(zone, levels[1], ohlc(20, 97.5, 100.0, 96.5, 97.8)),
              Sweep(zone, levels[2], ohlc(30, 96.5, 100.0, 95.5, 96.8))]
    engine = ConfirmationEngine()
    for s in sweeps:
        go(engine, s, s.candle)
    c1 = go(engine, sweeps[1], side.confirm(21))
    seen = engine.confirmations
    i0 = go(engine, sweeps[0], side.wrong_boundary(22))
    c2 = go(engine, sweeps[2], side.confirm(31))
    assert engine.confirmations == (c1, c2) and engine.invalidations == (i0,)
    assert engine.confirmations[:1] == seen == (c1,)                      # earlier entries never move
    assert isinstance(engine.confirmations, tuple) and isinstance(engine.invalidations, tuple)
    go(engine, sweeps[1], side.doji(25))
    assert engine.confirmations == (c1, c2)                               # nothing removed or replaced
 
 
def test_status_for_reports_every_state_and_none_for_unseen(side):
    sweep, engine = side.sweep(), ConfirmationEngine()
    assert engine.status_for(sweep) is None
    go(engine, sweep, sweep.candle)
    assert engine.status_for(sweep) is S.WAITING_FOR_DECISIVE
    go(engine, sweep, side.wrong(12))
    assert engine.status_for(sweep) is S.WAITING_FOR_SECOND_DECISIVE
    go(engine, sweep, side.confirm(13))
    assert engine.status_for(sweep) is S.CONFIRMED
 
 
@pytest.mark.parametrize("bad", [None, "x", 5])
def test_status_and_lookups_validate_their_argument(side, bad):
    engine, sweep, _ = started(side)
    for fn in (engine.status_for, engine.confirmation_for, engine.invalidation_for):
        with pytest.raises(TypeError):
            fn(bad)
        with pytest.raises(TypeError):
            fn(sweep.zone)
 
 
def test_queries_do_not_record_or_change_anything(side):
    sweep, engine = side.sweep(), ConfirmationEngine()
    assert engine.status_for(sweep) is None and engine.confirmation_for(sweep) is None
    assert engine.invalidation_for(sweep) is None
    assert engine.confirmations_known_at(at(1000)) == () and engine.invalidations_known_at(at(1000)) == ()
    assert go(engine, sweep, sweep.candle) is None                        # the sweep was never "registered" by the queries
 
 
def test_known_at_queries_respect_the_event_candle_close_time(side):
    engine, sweep, _ = started(side)
    event = go(engine, sweep, side.confirm(12))                           # known when slot 12 closes (slot 13)
    assert engine.confirmations_known_at(at(12)) == ()
    assert engine.confirmations_known_at(event.known_at - timedelta(seconds=1)) == ()
    assert engine.confirmations_known_at(event.known_at) == (event,)
    assert engine.confirmations_known_at(at(9999)) == (event,)
    other = Side(side.name)
    sweep2 = Sweep(other.zone, LiquidityLevel(mk_swing(LOW, 97.0, i=12, seq=1) if side.long else mk_swing(HIGH, 112.0, i=12, seq=1)),
                   ohlc(14, 97.5, 100.0, 96.5, 97.8) if side.long else ohlc(14, 110.5, 113.0, 109.5, 110.8))
    go(engine, sweep2, sweep2.candle)
    inv = go(engine, sweep2, side.wrong_boundary(15))
    assert isinstance(inv, Invalidation)
    assert engine.invalidations_known_at(inv.known_at - timedelta(seconds=1)) == ()
    assert engine.invalidations_known_at(inv.known_at) == (inv,)
    assert engine.confirmations_known_at(inv.known_at) == (event,)        # the two kinds are separate
 
 
@pytest.mark.parametrize("bad", [None, "now", 5, 1.5, date(2026, 1, 1)])
def test_known_at_queries_require_a_datetime(side, bad):
    engine, sweep, _ = started(side)
    go(engine, sweep, side.confirm(12))
    for fn in (engine.confirmations_known_at, engine.invalidations_known_at):
        with pytest.raises(TypeError):
            fn(bad)
 
 
def test_known_at_queries_reject_mixed_datetimes_when_events_exist(side):
    engine, sweep, _ = started(side)
    event = go(engine, sweep, side.confirm(12))
    with pytest.raises(TypeError):
        engine.confirmations_known_at(aware(event.known_at))
 
 
# ================================================================== the engine does not touch anything else (test 28)
def test_the_engine_does_not_modify_the_sweep_or_upstream_objects(side):
    sweep = side.sweep()
    zone, level, candle, identity = sweep.zone, sweep.level, sweep.candle, sweep.identity
    snapshot = (repr(sweep), repr(zone), repr(level), repr(candle), repr(identity))
    engine = ConfirmationEngine()
    go(engine, sweep, sweep.candle)
    go(engine, sweep, side.wrong(12))
    go(engine, sweep, side.confirm(13))
    go(engine, sweep, side.doji(14))
    assert (repr(sweep), repr(zone), repr(level), repr(candle), repr(identity)) == snapshot
    assert sweep.zone is zone and sweep.level is level and sweep.candle is candle
    assert sweep == side.sweep()                                          # still equal to a freshly built one
 
 
def test_the_engine_holds_no_other_engine_and_exposes_no_downstream_behaviour():
    engine = ConfirmationEngine()
    for value in vars(engine).values():
        assert type(value).__module__ == "builtins"                       # only plain containers; no engine instances
    for attr in ("setup", "entry", "sl", "tp", "stop_loss", "take_profit", "risk", "news", "execute", "execution",
                 "bos", "choch", "timeout", "expire", "expired", "tolerance", "distance", "atr", "displacement",
                 "session", "delete", "remove", "clear", "reset", "process_swing", "process_zone"):
        assert not hasattr(engine, attr), attr
 
 
def test_the_engine_module_imports_no_other_engine_or_strategy_component():
    tree = ast.parse(inspect.getsource(engine_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert imported <= {
        "__future__", "datetime", "typing",
        "koffie.strategy.models.candle",
        "koffie.strategy.models.confirmation",
        "koffie.strategy.models.sweep",
    }, imported
 
 
def test_brr_is_not_reimplemented_in_the_engine():
    source = inspect.getsource(engine_module)
    assert "abs(" not in source and "BRR" not in source.replace("BRR rule is not repeated", "")
    assert ".classify(" in source