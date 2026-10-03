"""Engine-level tests for SweepEngine (M5 execution infrastructure).
 
The Sweep MODEL is covered by tests/test_sweep.py. This file covers SweepEngine
itself and its interaction with the real LiquidityEngine: as-of-candle-open liquidity
selection, strict wick penetration, one sweep per (zone, liquidity) pair, per-zone
chronology, replay, zone-identity integrity, append-only sweeps, the read-only
queries, and the read-only dependency boundary.
 
Only real project objects are used (Zone, Pivot, BOS, LiquidityEngine, Swing, Candle).
The only test double is a LiquidityEngine SUBCLASS that records the arguments of
required_liquidity_for and then delegates to the real implementation.
"""
import ast
import inspect
from datetime import date, datetime, timedelta, timezone
 
import pytest
 
import koffie.strategy.engines.sweep_engine as engine_module
from koffie.strategy.engines.liquidity_engine import LiquidityEngine
from koffie.strategy.engines.sweep_engine import SweepEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.sweep import Sweep, SweepIdentity
from koffie.strategy.models.zone import Zone, ZoneType
 
T0 = datetime(2026, 1, 1, 10, 0)
M5, M15, H1 = Timeframe.M5, Timeframe.M15, Timeframe.H1
BULL, BEAR = BOSDirection.BULLISH, BOSDirection.BEARISH
HIGH, LOW = SwingType.HIGH, SwingType.LOW
 
 
# ------------------------------------------------------------------ helpers
def at(i, tf=M5):
    return T0 + i * tf.duration
 
 
def mk_swing(swing_type, price, i=0, seq=0, tf=M5, t0=T0):
    """Swing whose candle opens at slot i; it is confirmed (known) at slot i + 2."""
    candle_time = t0 + i * tf.duration
    return Swing(swing_type, tf, price, candle_time, candle_time + 2 * tf.duration, seq)
 
 
def mk_zone(direction, low=99.0, high=110.0, tf=M5, bos_i=10, broken_i=2):
    """A Zone built directly from a BOS and a Pivot (no engines involved).
 
    Default 5M zone: created_at = slot 11 (T0 + 55 minutes).
    """
    broken_type = HIGH if direction is BULL else LOW
    pivot_class = CandleClass.BEARISH_DECISIVE if direction is BULL else CandleClass.BULLISH_DECISIVE
    broken = mk_swing(broken_type, 100.0, i=broken_i, seq=0, tf=tf)
    close_price = 101.0 if direction is BULL else 99.0
    bos = BOS(tf, direction, broken, T0 + bos_i * tf.duration, close_price)
    pivot = Pivot(bos, T0 + 5 * tf.duration, high, low, pivot_class)
    return Zone(pivot)
 
 
def conflicting_zone(zone):
    """A DIFFERENT Zone with the SAME identity: same BOS, different Pivot range."""
    pivot = zone.pivot
    other = Pivot(pivot.bos, pivot.candle_time - zone.timeframe.duration, 120.0, 95.0, pivot.candle_class)
    clash = Zone(other)
    assert clash != zone and clash.identity == zone.identity
    return clash
 
 
def cndl(open_time, low, high, open_=None, close=None, tf=M5):
    mid = (low + high) / 2.0
    return Candle(tf, open_time, mid if open_ is None else open_, high, low, mid if close is None else close)
 
 
def liquidity_with(*swings):
    liq = LiquidityEngine(M5)
    for s in swings:
        liq.process_swing(s, s.confirmed_at)
    return liq
 
 
def make_engine(*swings):
    liq = liquidity_with(*swings)
    return SweepEngine(liq), liq
 
 
def run(engine, zone, candle, now=None):
    return engine.process_candle(zone, candle, candle.close_time if now is None else now)
 
 
def snap(engine, *zones):
    return engine.sweeps, tuple(engine.sweeps_for_zone(z) for z in zones)
 
 
def demand_engine():
    """DEMAND zone 99..110 (known at slot 11) and one relevant liquidity LOW at 98 (known at slot 2)."""
    zone = mk_zone(BULL)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    return engine, liq, zone
 
 
def supply_engine():
    """SUPPLY zone 99..110 (known at slot 11) and one relevant liquidity HIGH at 111 (known at slot 2)."""
    zone = mk_zone(BEAR)
    engine, liq = make_engine(mk_swing(HIGH, 111.0))
    return engine, liq, zone
 
 
class RecordingLiquidity(LiquidityEngine):
    """Real LiquidityEngine that also records the arguments of required_liquidity_for."""
 
    def __init__(self, timeframe):
        super().__init__(timeframe)
        self.calls = []
 
    def required_liquidity_for(self, zone, moment):
        self.calls.append((zone, moment))
        return super().required_liquidity_for(zone, moment)
 
 
# ================================================================== 1-2. constructor and timeframe
def test_constructor_accepts_a_liquidity_engine_and_is_m5():
    engine = SweepEngine(LiquidityEngine(M5))
    assert engine.timeframe is M5
    assert engine.sweeps == ()
 
 
@pytest.mark.parametrize("bad", [None, "liquidity", 5, M5, [], object()])
def test_constructor_rejects_anything_that_is_not_a_liquidity_engine(bad):
    with pytest.raises(TypeError):
        SweepEngine(bad)
 
 
def test_constructor_accepts_a_liquidity_engine_subclass():
    assert SweepEngine(RecordingLiquidity(M5)).timeframe is M5
 
 
def test_timeframe_is_m5_regardless_of_zone_timeframes_and_is_read_only():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(11), 97.0, 100.0))
    assert engine.timeframe is M5
    with pytest.raises(AttributeError):
        engine.timeframe = H1
 
 
# ================================================================== 3-4. creating sweeps
def test_demand_sweep_is_created_through_the_engine():
    engine, liq, zone = demand_engine()
    candle = cndl(at(11), 97.5, 100.0)
    sweep = run(engine, zone, candle)
    assert isinstance(sweep, Sweep)
    assert sweep.zone is zone and sweep.candle is candle
    assert sweep.level is liq.levels[0] and sweep.zone_type is ZoneType.DEMAND
    assert sweep.swept_price == 98.0
    assert engine.sweeps == (sweep,)
 
 
def test_supply_sweep_is_created_through_the_engine():
    engine, liq, zone = supply_engine()
    candle = cndl(at(11), 109.0, 112.0)
    sweep = run(engine, zone, candle)
    assert isinstance(sweep, Sweep)
    assert sweep.zone is zone and sweep.candle is candle
    assert sweep.level is liq.levels[0] and sweep.zone_type is ZoneType.SUPPLY
    assert sweep.swept_price == 111.0
    assert engine.sweeps == (sweep,)
 
 
def test_the_sweep_identity_is_the_zone_and_liquidity_pair():
    engine, liq, zone = demand_engine()
    sweep = run(engine, zone, cndl(at(11), 97.5, 100.0))
    assert sweep.identity == SweepIdentity.from_parts(zone, liq.levels[0])
 
 
# ================================================================== 5. no relevant liquidity
def test_no_sweep_when_no_liquidity_exists_at_all():
    zone = mk_zone(BULL)
    engine = SweepEngine(LiquidityEngine(M5))
    assert run(engine, zone, cndl(at(11), 50.0, 100.0)) is None
    assert engine.sweeps == ()
 
 
def test_no_sweep_when_the_only_liquidity_is_not_relevant_to_the_zone():
    zone = mk_zone(BULL)                                              # needs a swing LOW strictly below 99
    for irrelevant in (mk_swing(LOW, 99.0), mk_swing(LOW, 99.5), mk_swing(HIGH, 95.0), mk_swing(HIGH, 120.0)):
        engine, liq = make_engine(irrelevant)
        assert run(engine, zone, cndl(at(11), 50.0, 130.0)) is None   # wicks through everything
        assert engine.sweeps == ()
    zone = mk_zone(BEAR)                                              # needs a swing HIGH strictly above 110
    for irrelevant in (mk_swing(HIGH, 110.0), mk_swing(HIGH, 105.0), mk_swing(LOW, 120.0), mk_swing(LOW, 90.0)):
        engine, liq = make_engine(irrelevant)
        assert run(engine, zone, cndl(at(11), 50.0, 130.0)) is None
        assert engine.sweeps == ()
 
 
# ================================================================== 6-7. strict penetration
def test_a_wick_that_only_equals_the_liquidity_is_not_a_sweep():
    engine, liq, zone = demand_engine()
    assert run(engine, zone, cndl(at(11), 98.0, 100.0)) is None
    assert engine.sweeps == ()
    engine, liq, zone = supply_engine()
    assert run(engine, zone, cndl(at(11), 109.0, 111.0)) is None
    assert engine.sweeps == ()
 
 
def test_an_equal_wick_does_not_consume_the_pair_a_later_penetration_still_sweeps():
    engine, liq, zone = demand_engine()
    assert run(engine, zone, cndl(at(11), 98.0, 100.0)) is None
    sweep = run(engine, zone, cndl(at(12), 97.99, 100.0))
    assert sweep is not None and sweep.candle.open_time == at(12)
 
 
def test_a_wick_that_does_not_penetrate_is_not_a_sweep():
    engine, liq, zone = demand_engine()
    assert run(engine, zone, cndl(at(11), 98.5, 100.0)) is None
    engine, liq, zone = supply_engine()
    assert run(engine, zone, cndl(at(11), 109.0, 110.5)) is None
 
 
# ================================================================== 8. a wick is enough
@pytest.mark.parametrize("open_, close", [
    (100.0, 100.0),      # closes back above the level
    (98.5, 104.0),       # closes inside the zone
    (98.2, 97.2),        # closes beyond the level
    (97.5, 97.5),        # opens and closes beyond the level
])
def test_demand_wick_penetration_is_sufficient_regardless_of_the_close(open_, close):
    engine, liq, zone = demand_engine()
    candle = Candle(M5, at(11), open_, 105.0, 97.0, close)
    sweep = run(engine, zone, candle)
    assert sweep is not None and sweep.candle is candle
 
 
@pytest.mark.parametrize("open_, close", [
    (110.0, 110.0),      # closes back below the level
    (111.5, 105.0),      # closes inside the zone
    (111.5, 112.5),      # closes beyond the level
    (112.2, 112.2),
])
def test_supply_wick_penetration_is_sufficient_regardless_of_the_close(open_, close):
    engine, liq, zone = supply_engine()
    candle = Candle(M5, at(11), open_, 113.0, 105.0, close)
    sweep = run(engine, zone, candle)
    assert sweep is not None and sweep.candle is candle
 
 
# ================================================================== 9. only the correct side
def test_only_the_liquidity_side_of_the_zone_can_be_swept():
    # DEMAND with liquidity on BOTH sides: only the LOW below the zone is relevant.
    zone = mk_zone(BULL)
    engine, liq = make_engine(mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(HIGH, 111.0, i=1, seq=1))
    spike_up = cndl(at(11), 100.0, 130.0)
    assert run(engine, zone, spike_up) is None                        # a high spike never sweeps DEMAND liquidity
    sweep = run(engine, zone, cndl(at(12), 97.0, 100.0))
    assert sweep is not None and sweep.level.swing.swing_type is LOW
    # SUPPLY with liquidity on both sides: only the HIGH above the zone is relevant.
    zone = mk_zone(BEAR)
    engine, liq = make_engine(mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(HIGH, 111.0, i=1, seq=1))
    spike_down = cndl(at(11), 50.0, 109.0)
    assert run(engine, zone, spike_down) is None                      # a low spike never sweeps SUPPLY liquidity
    sweep = run(engine, zone, cndl(at(12), 100.0, 112.0))
    assert sweep is not None and sweep.level.swing.swing_type is HIGH
 
 
# ================================================================== 10-12. as-of the candle OPEN time
def test_liquidity_is_selected_as_of_the_candle_open_time_not_close_or_now():
    zone = mk_zone(BULL)
    liq = RecordingLiquidity(M5)
    liq.process_swing(mk_swing(LOW, 98.0), at(2))
    engine = SweepEngine(liq)
    candle = cndl(at(12), 97.0, 100.0)
    later_now = candle.close_time + timedelta(hours=3)
    run(engine, zone, candle, now=later_now)
    assert liq.calls == [(zone, candle.open_time)]
    moment = liq.calls[0][1]
    assert moment == at(12) and moment != candle.close_time and moment != later_now
 
 
def test_a_level_known_only_when_the_candle_closes_cannot_be_swept_by_that_candle():
    zone = mk_zone(BULL)
    late = mk_swing(LOW, 98.0, i=11)                                  # candle slot 11, confirmed (known) at slot 13
    assert late.confirmed_at == at(13)
    engine, liq = make_engine(late)
    candle = cndl(at(12), 97.0, 100.0)                                # opens slot 12, closes slot 13 == known_at
    assert candle.close_time == late.confirmed_at
    assert run(engine, zone, candle) is None                          # now == close_time == known_at: still not eligible
    assert engine.sweeps == ()
 
 
def test_a_level_known_between_candle_open_and_now_is_still_not_eligible():
    zone = mk_zone(BULL)
    late = mk_swing(LOW, 98.0, i=11)                                  # known at slot 13
    engine, liq = make_engine(late)
    candle = cndl(at(12), 97.0, 100.0)
    assert run(engine, zone, candle, now=at(500)) is None             # `now` is far after known_at; open time decides
    assert engine.sweeps == ()
 
 
def test_a_later_candle_can_use_the_newly_known_level():
    zone = mk_zone(BULL)
    late = mk_swing(LOW, 98.0, i=11)                                  # known at slot 13
    engine, liq = make_engine(late)
    assert run(engine, zone, cndl(at(12), 97.0, 100.0)) is None       # too early
    sweep = run(engine, zone, cndl(at(13), 97.0, 100.0))              # opens exactly at known_at
    assert sweep is not None and sweep.level.known_at == at(13) and sweep.candle_time == at(13)
 
 
def test_a_level_known_exactly_at_the_candle_open_is_eligible():
    zone = mk_zone(BULL)
    engine, liq = make_engine(mk_swing(LOW, 98.0, i=10))              # known at slot 12
    assert run(engine, zone, cndl(at(12), 97.0, 100.0)) is not None
 
 
def test_a_level_in_the_liquidity_engine_that_becomes_known_later_is_ignored_by_earlier_candles():
    zone = mk_zone(BULL)
    early, future = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 90.0, i=20, seq=1)   # future known at slot 22
    engine, liq = make_engine(early, future)                          # the engine already contains the future level
    sweep = run(engine, zone, cndl(at(15), 89.0, 100.0))              # wicks through BOTH levels
    assert sweep.level.price == 98.0                                  # only what was known at slot 15
    assert engine.sweeps_for_zone(zone) == (sweep,)
 
 
# ================================================================== required (newest) level only
def test_only_the_required_newest_level_is_considered_not_every_known_level():
    zone = mk_zone(BULL)
    older, newer = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 97.0, i=4, seq=1)   # known slots 2 and 6
    engine, liq = make_engine(older, newer)
    assert liq.required_liquidity_for(zone, at(11)).price == 97.0
    assert run(engine, zone, cndl(at(11), 97.5, 100.0)) is None       # penetrates 98 (older) but NOT 97 (required)
    assert engine.sweeps == ()
    sweep = run(engine, zone, cndl(at(12), 96.5, 100.0))
    assert sweep is not None and sweep.level.price == 97.0
 
 
# ================================================================== 13-14. one sweep per (zone, liquidity)
def test_a_swept_pair_cannot_generate_a_duplicate():
    engine, liq, zone = demand_engine()
    first = run(engine, zone, cndl(at(11), 97.5, 100.0))
    assert first is not None
    for k in range(12, 22):
        assert run(engine, zone, cndl(at(k), 97.0 - k, 100.0)) is None   # ever deeper wicks, same level
    assert engine.sweeps == (first,)
    assert engine.sweep_for(zone, liq.levels[0]) is first
    assert first.candle_time == at(11)                                # the FIRST penetrating candle owns it
 
 
def test_a_swept_pair_cannot_generate_a_duplicate_for_supply():
    engine, liq, zone = supply_engine()
    first = run(engine, zone, cndl(at(11), 109.0, 111.5))
    for k in range(12, 20):
        assert run(engine, zone, cndl(at(k), 109.0, 111.0 + k)) is None
    assert engine.sweeps == (first,)
 
 
def test_a_different_newer_level_generates_a_new_sweep_for_the_same_zone():
    zone = mk_zone(BULL)
    l1, l2, l3 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 97.0, i=12, seq=1), mk_swing(LOW, 95.0, i=16, seq=2)
    engine, liq = make_engine(l1, l2, l3)                             # known at slots 2, 14, 18
    s1 = run(engine, zone, cndl(at(12), 97.5, 100.0))                 # required = l1 (98)
    assert s1.level.price == 98.0
    assert run(engine, zone, cndl(at(13), 96.0, 100.0)) is None       # l2 not known yet; l1 already swept
    s2 = run(engine, zone, cndl(at(14), 96.5, 100.0))                 # l2 (97) is now required and penetrated
    assert s2.level.price == 97.0 and s2.identity != s1.identity
    assert run(engine, zone, cndl(at(15), 80.0, 100.0)) is None       # l2 swept, l3 not known yet
    s3 = run(engine, zone, cndl(at(18), 94.0, 100.0))
    assert s3.level.price == 95.0
    assert engine.sweeps == (s1, s2, s3)
    assert engine.sweeps_for_zone(zone) == (s1, s2, s3)
    assert engine.sweep_for(zone, l2_level(liq, 97.0)) is s2
 
 
def l2_level(liq, price):
    return next(level for level in liq.levels if level.price == price)
 
 
def test_a_newer_level_that_is_not_penetrated_creates_no_sweep_and_the_older_one_stays_swept():
    zone = mk_zone(BULL)
    l1, l2 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 90.0, i=12, seq=1)       # l2 known at slot 14
    engine, liq = make_engine(l1, l2)
    s1 = run(engine, zone, cndl(at(12), 97.0, 100.0))
    assert run(engine, zone, cndl(at(14), 97.0, 100.0)) is None       # l2 (90) is required: 97 does not reach it
    assert engine.sweeps == (s1,)
 
 
# ================================================================== 15-16. several zones
def test_multiple_zones_are_tracked_independently():
    zone_a = mk_zone(BULL, bos_i=10, broken_i=2)
    zone_b = mk_zone(BULL, bos_i=14, broken_i=3)                      # different BOS identity, known at slot 15
    assert zone_a.identity != zone_b.identity
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    sa = run(engine, zone_a, cndl(at(12), 97.0, 100.0))
    assert sa is not None and engine.sweeps_for_zone(zone_b) == ()
    sb = run(engine, zone_b, cndl(at(15), 97.0, 100.0))               # same level, its own pair
    assert sb is not None and sb is not sa and sb.level is sa.level
    assert sa.identity != sb.identity
    assert engine.sweeps_for_zone(zone_a) == (sa,) and engine.sweeps_for_zone(zone_b) == (sb,)
    assert engine.sweeps == (sa, sb)
 
 
def test_one_zone_being_swept_does_not_affect_another_zone():
    zone_a = mk_zone(BULL, bos_i=10, broken_i=2)
    zone_b = mk_zone(BULL, bos_i=14, broken_i=3)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    run(engine, zone_a, cndl(at(12), 97.0, 100.0))
    run(engine, zone_a, cndl(at(13), 90.0, 100.0))                    # no duplicate for A
    assert run(engine, zone_b, cndl(at(15), 97.5, 100.0)) is not None  # B is untouched by A's sweep
 
 
def test_demand_and_supply_zones_in_one_engine_use_their_own_liquidity():
    demand, supply = mk_zone(BULL, bos_i=10, broken_i=2), mk_zone(BEAR, bos_i=10, broken_i=2)
    assert demand.identity != supply.identity
    engine, liq = make_engine(mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(HIGH, 111.0, i=1, seq=1))
    candle = cndl(at(12), 97.0, 112.0)                                # wicks through both liquidity levels
    sd, ss = run(engine, demand, candle), run(engine, supply, candle)
    assert sd.level.swing.swing_type is LOW and ss.level.swing.swing_type is HIGH
    assert sd.candle is ss.candle is candle and sd != ss
    assert engine.sweeps == (sd, ss)
 
 
def test_chronology_is_per_zone_not_global():
    zone_a = mk_zone(BULL, bos_i=10, broken_i=2)
    zone_b = mk_zone(BULL, bos_i=14, broken_i=3)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    run(engine, zone_a, cndl(at(40), 99.5, 100.0))                    # zone A is far ahead in time
    assert run(engine, zone_b, cndl(at(15), 97.0, 100.0)) is not None  # zone B may still receive an earlier candle
    with pytest.raises(ValueError):
        run(engine, zone_a, cndl(at(20), 99.5, 100.0))                # but A itself may not go back
 
 
@pytest.mark.parametrize("tf", [M5, M15, H1])
@pytest.mark.parametrize("direction", [BULL, BEAR])
def test_zones_of_every_timeframe_are_processed_with_5m_candles(tf, direction):
    zone = mk_zone(direction, tf=tf)
    swing = mk_swing(LOW, 98.0) if direction is BULL else mk_swing(HIGH, 111.0)
    engine, liq = make_engine(swing)
    candle = cndl(zone.created_at, 97.0, 112.0)                       # a 5M candle opening when the zone is created
    sweep = run(engine, zone, candle)
    assert sweep is not None and sweep.zone.timeframe is tf and sweep.candle.timeframe is M5
 
 
def test_1h_15m_and_5m_zones_in_one_engine_are_independent():
    zones = {tf: mk_zone(BULL, tf=tf) for tf in (M5, M15, H1)}
    assert len({z.identity for z in zones.values()}) == 3
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    candle = cndl(max(z.created_at for z in zones.values()), 97.0, 100.0)
    results = {tf: run(engine, z, candle) for tf, z in zones.items()}
    assert all(r is not None for r in results.values())
    assert len(engine.sweeps) == 3 and len({s.identity for s in engine.sweeps}) == 3
    for tf, z in zones.items():
        assert engine.sweeps_for_zone(z) == (results[tf],)
 
 
# ================================================================== 17. zone created_at timing
@pytest.mark.parametrize("low", [97.0, 99.5])          # penetrating and NOT penetrating the required level
def test_a_candle_opening_before_the_zone_existed_is_rejected(low):
    engine, liq, zone = demand_engine()
    assert zone.created_at == at(11)
    before = snap(engine, zone)
    for early in (at(0), at(10), zone.created_at - timedelta(minutes=1)):
        with pytest.raises(ValueError):
            run(engine, zone, cndl(early, low, 100.0))
    assert snap(engine, zone) == before
    assert run(engine, zone, cndl(at(11), 97.0, 100.0)) is not None
 
 
def test_a_candle_opening_exactly_at_created_at_is_accepted():
    engine, liq, zone = demand_engine()
    assert run(engine, zone, cndl(zone.created_at, 97.0, 100.0)) is not None
 
 
@pytest.mark.parametrize("tf", [M15, H1])
def test_created_at_is_enforced_for_higher_timeframe_zones(tf):
    zone = mk_zone(BULL, tf=tf)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    too_early = cndl(zone.created_at - M5.duration, 97.0, 100.0)
    with pytest.raises(ValueError):
        run(engine, zone, too_early)
    assert run(engine, zone, cndl(zone.created_at, 97.0, 100.0)) is not None
 
 
# ================================================================== 18-20. candle and argument validation
@pytest.mark.parametrize("low", [97.0, 99.5])
def test_unclosed_candles_are_rejected(low):
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), low, 100.0)
    before = snap(engine, zone)
    for now in (candle.open_time, candle.open_time + timedelta(minutes=3), candle.close_time - timedelta(seconds=1)):
        with pytest.raises(ValueError):
            engine.process_candle(zone, candle, now)
    assert snap(engine, zone) == before
    result = engine.process_candle(zone, candle, candle.close_time)               # exactly at close_time is fine
    assert (result is not None) == (low < 98.0)
 
 
def test_a_far_later_now_is_fine():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    assert engine.process_candle(zone, candle, candle.close_time + timedelta(days=30)) is not None
 
 
@pytest.mark.parametrize("low", [97.0, 99.5])          # penetrating and NOT penetrating the required level
@pytest.mark.parametrize("tf", [M15, H1])
def test_non_5m_candles_are_rejected(tf, low):
    engine, liq, zone = demand_engine()
    candle = cndl(zone.created_at, low, 100.0, tf=tf)
    before = snap(engine, zone)
    with pytest.raises(ValueError):
        run(engine, zone, candle)
    assert snap(engine, zone) == before
    assert run(engine, zone, cndl(zone.created_at, 97.0, 100.0)) is not None   # the slot is still free
 
 
@pytest.mark.parametrize("bad", [None, "zone", 5, 1.5, [], object()])
def test_wrong_zone_types_are_rejected(bad):
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    with pytest.raises(TypeError):
        engine.process_candle(bad, candle, candle.close_time)
 
 
def test_a_zone_component_or_level_is_not_accepted_as_a_zone():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    for bad in (zone.pivot, zone.bos, liq.levels[0], liq.levels[0].swing):
        with pytest.raises(TypeError):
            engine.process_candle(bad, candle, candle.close_time)
 
 
@pytest.mark.parametrize("bad", [None, "candle", 5, (1, 2), [], object()])
def test_wrong_candle_types_are_rejected(bad):
    engine, liq, zone = demand_engine()
    with pytest.raises(TypeError):
        engine.process_candle(zone, bad, at(13))
 
 
@pytest.mark.parametrize("bad", [None, "now", 5, 1.5, [], date(2026, 1, 1)])
def test_wrong_now_types_are_rejected(bad):
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    with pytest.raises(TypeError):
        engine.process_candle(zone, candle, bad)
 
 
def test_wrong_argument_types_do_not_change_state():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(12), 97.0, 100.0))
    before = snap(engine, zone)
    candle = cndl(at(13), 90.0, 100.0)
    for args in ((None, candle, candle.close_time), (zone, None, candle.close_time), (zone, candle, None)):
        with pytest.raises(TypeError):
            engine.process_candle(*args)
    assert snap(engine, zone) == before
 
 
# ================================================================== 21. naive / aware datetimes
def aware(dt):
    return dt.replace(tzinfo=timezone.utc)
 
 
def test_an_aware_candle_with_a_naive_now_raises_type_error():
    engine, liq, zone = demand_engine()
    candle = Candle(M5, aware(at(12)), 99.0, 100.0, 97.0, 99.0)
    with pytest.raises(TypeError):
        engine.process_candle(zone, candle, at(13))
    assert engine.sweeps == ()
 
 
def test_a_naive_candle_with_an_aware_now_raises_type_error():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    with pytest.raises(TypeError):
        engine.process_candle(zone, candle, aware(candle.close_time))
    assert engine.sweeps == ()
 
 
def test_an_aware_candle_and_now_against_a_naive_zone_raises_type_error():
    engine, liq, zone = demand_engine()
    candle = Candle(M5, aware(at(12)), 99.0, 100.0, 97.0, 99.0)
    with pytest.raises(TypeError):
        engine.process_candle(zone, candle, aware(candle.close_time))
    assert engine.sweeps == ()
 
 
def test_mixed_datetimes_leave_the_engine_usable_and_unchanged():
    engine, liq, zone = demand_engine()
    bad = Candle(M5, aware(at(12)), 99.0, 100.0, 97.0, 99.0)
    with pytest.raises(TypeError):
        engine.process_candle(zone, bad, at(13))
    assert run(engine, zone, cndl(at(12), 97.0, 100.0)) is not None   # the same slot is still free
 
 
def test_mixed_datetimes_raise_in_the_time_based_query_when_sweeps_exist():
    engine, liq, zone = demand_engine()
    sweep = run(engine, zone, cndl(at(12), 97.0, 100.0))
    with pytest.raises(TypeError):
        engine.sweeps_known_at(aware(sweep.known_at))
 
 
# ================================================================== 22-23. chronological order per zone
def test_back_to_back_and_gapped_candles_are_accepted():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(11), 99.5, 100.0))
    run(engine, zone, cndl(at(12), 99.5, 100.0))                      # opens exactly when the previous closed
    run(engine, zone, cndl(at(20), 99.5, 100.0))                      # a gap (e.g. a pause in trading)
    assert run(engine, zone, cndl(at(900), 97.0, 100.0)) is not None  # a very large gap
 
 
def test_an_overlapping_candle_is_rejected():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(12), 99.5, 100.0))
    overlapping = cndl(at(12) + timedelta(minutes=2), 97.0, 100.0)    # opens before slot 12 closes
    before = snap(engine, zone)
    with pytest.raises(ValueError):
        run(engine, zone, overlapping)
    assert snap(engine, zone) == before
 
 
def test_an_earlier_candle_is_rejected():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(20), 99.5, 100.0))
    before = snap(engine, zone)
    for earlier in (at(19), at(15), at(11)):
        with pytest.raises(ValueError):
            run(engine, zone, cndl(earlier, 97.0, 100.0))
    assert snap(engine, zone) == before
 
 
def test_a_different_candle_with_the_same_open_time_is_rejected():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(12), 99.5, 100.0))
    before = snap(engine, zone)
    with pytest.raises(ValueError):
        run(engine, zone, cndl(at(12), 97.0, 100.0))                  # same slot, different data: not a replay
    assert snap(engine, zone) == before
 
 
def test_a_rejected_out_of_order_candle_does_not_move_the_chronology_forward():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(12), 99.5, 100.0))
    with pytest.raises(ValueError):
        run(engine, zone, cndl(at(12) + timedelta(minutes=1), 97.0, 100.0))
    assert run(engine, zone, cndl(at(13), 97.0, 100.0)) is not None   # slot 13 is still next in line
 
 
# ================================================================== 24-25. replay
def test_replaying_the_latest_non_sweeping_candle_returns_the_same_result_without_mutation():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 99.5, 100.0)
    assert run(engine, zone, candle) is None
    before = snap(engine, zone)
    assert run(engine, zone, candle) is None
    assert run(engine, zone, cndl(at(12), 99.5, 100.0)) is None       # an equal, rebuilt candle object
    assert snap(engine, zone) == before == ((), ((),))
 
 
def test_replaying_the_latest_sweeping_candle_returns_the_same_sweep_without_mutation():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    first = run(engine, zone, candle)
    before = snap(engine, zone)
    assert run(engine, zone, candle) is first
    assert run(engine, zone, cndl(at(12), 97.0, 100.0)) is first
    assert snap(engine, zone) == before and len(engine.sweeps) == 1
 
 
def test_replaying_the_candle_that_produced_a_stored_sweep_returns_it_even_after_later_candles():
    engine, liq, zone = demand_engine()
    sweeping = cndl(at(12), 97.0, 100.0)
    first = run(engine, zone, sweeping)
    run(engine, zone, cndl(at(13), 90.0, 100.0))
    latest = cndl(at(14), 99.5, 100.0)
    run(engine, zone, latest)
    before = snap(engine, zone)
    assert run(engine, zone, sweeping) is first                       # the existing Sweep, not a new one
    assert run(engine, zone, cndl(at(12), 97.0, 100.0)) is first
    assert snap(engine, zone) == before and engine.sweeps == (first,)
 
 
def test_replaying_an_old_sweeping_candle_does_not_rewind_the_zone_state():
    engine, liq, zone = demand_engine()
    sweeping = cndl(at(12), 97.0, 100.0)
    first = run(engine, zone, sweeping)
    latest = cndl(at(14), 99.5, 100.0)
    run(engine, zone, latest)
    assert run(engine, zone, sweeping) is first
    assert run(engine, zone, latest) is None                          # the latest candle is still the latest one
    with pytest.raises(ValueError):
        run(engine, zone, cndl(at(13), 90.0, 100.0))                  # slot 13 is still behind
    assert run(engine, zone, cndl(at(15), 99.5, 100.0)) is None
 
 
def test_replaying_an_old_non_sweeping_candle_is_out_of_order_and_rejected():
    engine, liq, zone = demand_engine()
    quiet = cndl(at(12), 99.5, 100.0)
    run(engine, zone, quiet)
    run(engine, zone, cndl(at(13), 99.5, 100.0))
    before = snap(engine, zone)
    with pytest.raises(ValueError):
        run(engine, zone, quiet)                                      # neither the latest candle nor a sweeping one
    assert snap(engine, zone) == before
 
 
def test_replay_with_a_later_now_changes_nothing():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    first = run(engine, zone, candle)
    assert engine.process_candle(zone, candle, candle.close_time + timedelta(days=2)) is first
    assert len(engine.sweeps) == 1
 
 
def test_replay_with_an_equal_rebuilt_zone_is_accepted():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    first = run(engine, zone, candle)
    rebuilt = mk_zone(BULL)
    assert rebuilt is not zone and rebuilt == zone
    assert run(engine, rebuilt, candle) is first
    assert len(engine.sweeps) == 1
 
 
def test_replay_never_creates_a_duplicate_for_the_same_pair_across_many_replays():
    engine, liq, zone = demand_engine()
    candle = cndl(at(12), 97.0, 100.0)
    results = [run(engine, zone, candle) for _ in range(5)]
    assert all(r is results[0] for r in results) and len(engine.sweeps) == 1
 
 
# ================================================================== 26. rejected calls never mutate state
REJECTIONS = {
    "wrong zone type": lambda e, z: e.process_candle("zone", cndl(at(13), 97.0, 100.0), at(14)),
    "wrong candle type": lambda e, z: e.process_candle(z, "candle", at(14)),
    "wrong now type": lambda e, z: e.process_candle(z, cndl(at(13), 97.0, 100.0), None),
    "non-5m candle": lambda e, z: e.process_candle(z, cndl(at(13), 97.0, 100.0, tf=M15), at(500)),
    "unclosed candle": lambda e, z: e.process_candle(z, cndl(at(13), 97.0, 100.0), at(13)),
    "before the zone existed": lambda e, z: e.process_candle(z, cndl(at(10), 97.0, 100.0), at(11)),
    "overlapping candle": lambda e, z: e.process_candle(
        z, cndl(at(12) + timedelta(minutes=2), 97.0, 100.0), at(13) + timedelta(minutes=2)),
    "earlier candle": lambda e, z: e.process_candle(z, cndl(at(11), 97.0, 100.0), at(12)),
    "conflicting zone": lambda e, z: e.process_candle(conflicting_zone(z), cndl(at(13), 97.0, 100.0), at(14)),
    "mixed naive and aware": lambda e, z: e.process_candle(
        z, Candle(M5, aware(at(13)), 99.0, 100.0, 97.0, 99.0), at(14)),
}
 
 
@pytest.mark.parametrize("name", list(REJECTIONS))
def test_a_rejected_call_changes_nothing(name):
    engine, liq, zone = demand_engine()
    assert run(engine, zone, cndl(at(12), 99.5, 100.0)) is None       # established state: slot 12 processed, no sweep
    before = snap(engine, zone)
    levels_before = liq.levels
    with pytest.raises((TypeError, ValueError)):
        REJECTIONS[name](engine, zone)
    assert snap(engine, zone) == before
    assert liq.levels == levels_before
    # the engine behaves as if the rejected call never happened: slot 13 is still next and sweeps normally
    probe = run(engine, zone, cndl(at(13), 97.0, 100.0))
    assert probe is not None and probe.candle_time == at(13)
    assert engine.sweeps == (probe,)
 
 
# ================================================================== 27. zone identity integrity
def test_a_different_zone_with_the_same_identity_is_rejected():
    engine, liq, zone = demand_engine()
    clash = conflicting_zone(zone)
    run(engine, zone, cndl(at(12), 99.5, 100.0))
    before = snap(engine, zone)
    with pytest.raises(ValueError):
        run(engine, clash, cndl(at(13), 97.0, 100.0))
    assert snap(engine, zone) == before
    assert engine.sweeps_for_zone(clash) == ()
 
 
def test_the_conflicting_zone_is_rejected_even_if_the_first_zone_never_swept():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(12), 99.5, 100.0))                      # no sweep: the zone was still recorded as seen
    with pytest.raises(ValueError):
        run(engine, conflicting_zone(zone), cndl(at(13), 97.0, 100.0))
 
 
def test_a_conflicting_zone_cannot_replace_the_first_zone_even_when_supplied_first_afterwards():
    engine, liq, zone = demand_engine()
    clash = conflicting_zone(zone)
    run(engine, zone, cndl(at(12), 99.5, 100.0))
    with pytest.raises(ValueError):
        run(engine, clash, cndl(at(13), 97.0, 100.0))
    # the original zone is still the recorded one
    assert run(engine, zone, cndl(at(13), 97.0, 100.0)) is not None
 
 
def test_the_conflicting_zone_is_rejected_on_replay_of_a_sweeping_candle_too():
    engine, liq, zone = demand_engine()
    sweeping = cndl(at(12), 97.0, 100.0)
    run(engine, zone, sweeping)
    before = snap(engine, zone)
    with pytest.raises(ValueError):
        run(engine, conflicting_zone(zone), sweeping)
    assert snap(engine, zone) == before
 
 
def test_zone_identity_is_by_value_so_distinct_but_equal_zones_are_the_same_zone():
    engine, liq, zone = demand_engine()
    rebuilt = mk_zone(BULL)
    assert rebuilt is not zone and rebuilt == zone and rebuilt.identity == zone.identity
    run(engine, zone, cndl(at(12), 99.5, 100.0))
    sweep = run(engine, rebuilt, cndl(at(13), 97.0, 100.0))           # accepted as the same zone
    assert sweep is not None and engine.sweeps_for_zone(zone) == engine.sweeps_for_zone(rebuilt) == (sweep,)
 
 
def test_different_zones_with_different_identities_are_never_in_conflict():
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    a, b = mk_zone(BULL, bos_i=10, broken_i=2), mk_zone(BULL, bos_i=14, broken_i=3, low=99.0, high=120.0)
    assert a.identity != b.identity
    assert run(engine, a, cndl(at(12), 97.0, 100.0)) is not None
    assert run(engine, b, cndl(at(15), 97.0, 100.0)) is not None
 
 
# ================================================================== 28. append-only and acceptance order
def test_sweeps_is_append_only_and_preserves_acceptance_order_across_zones():
    zone_a = mk_zone(BULL, bos_i=10, broken_i=2)
    zone_b = mk_zone(BEAR, bos_i=14, broken_i=3)
    l1, h1 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(HIGH, 111.0, i=1, seq=1)
    engine, liq = make_engine(l1, h1)
    seen = []
    s1 = run(engine, zone_a, cndl(at(12), 97.0, 100.0)); seen.append(s1)
    assert engine.sweeps == tuple(seen)
    s2 = run(engine, zone_b, cndl(at(15), 100.0, 112.0)); seen.append(s2)
    assert engine.sweeps == tuple(seen)
    run(engine, zone_a, cndl(at(16), 80.0, 100.0))                    # same pair again: nothing is added
    assert engine.sweeps == tuple(seen)
    older_snapshot = engine.sweeps
    l2 = mk_swing(LOW, 90.0, i=20, seq=2)                              # known at slot 22
    liq.process_swing(l2, l2.confirmed_at)
    s3 = run(engine, zone_a, cndl(at(23), 89.0, 100.0)); seen.append(s3)
    assert engine.sweeps == (s1, s2, s3)
    assert engine.sweeps[:2] == older_snapshot                        # earlier entries never move or disappear
    assert older_snapshot == (s1, s2)                                  # a previously returned tuple never changes
    assert isinstance(engine.sweeps, tuple)
 
 
def test_sweeps_are_never_removed_expired_or_replaced():
    engine, liq, zone = demand_engine()
    first = run(engine, zone, cndl(at(12), 97.0, 100.0))
    for k in range(13, 60):
        run(engine, zone, cndl(at(k), 99.5, 100.0))
    far_future = at(5000)
    assert engine.sweeps == (first,)
    assert engine.sweeps_known_at(far_future) == (first,)
    assert engine.sweep_for(zone, liq.levels[0]) is first
 
 
# ================================================================== 29. sweep_for
def test_sweep_for_returns_the_sweep_or_none():
    zone = mk_zone(BULL)
    l1, l2 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 90.0, i=40, seq=1)
    engine, liq = make_engine(l1, l2)
    level1, level2 = liq.levels
    assert engine.sweep_for(zone, level1) is None                     # nothing processed yet
    s1 = run(engine, zone, cndl(at(12), 97.0, 100.0))
    assert engine.sweep_for(zone, level1) is s1
    assert engine.sweep_for(zone, level2) is None                     # a different level: not swept
    assert engine.sweep_for(mk_zone(BULL, bos_i=14, broken_i=3), level1) is None   # a different zone: not swept
 
 
def test_sweep_for_uses_value_identity_not_object_identity():
    engine, liq, zone = demand_engine()
    sweep = run(engine, zone, cndl(at(12), 97.0, 100.0))
    assert engine.sweep_for(mk_zone(BULL), LiquidityLevel(mk_swing(LOW, 98.0))) is sweep
 
 
def test_sweep_for_does_not_record_or_change_anything():
    engine, liq, zone = demand_engine()
    before = snap(engine, zone)
    assert engine.sweep_for(zone, liq.levels[0]) is None
    assert snap(engine, zone) == before
 
 
@pytest.mark.parametrize("bad", [None, "x", 5])
def test_sweep_for_validates_its_arguments(bad):
    engine, liq, zone = demand_engine()
    with pytest.raises(TypeError):
        engine.sweep_for(bad, liq.levels[0])
    with pytest.raises(TypeError):
        engine.sweep_for(zone, bad)
 
 
def test_sweep_for_rejects_components_in_place_of_the_zone_or_level():
    engine, liq, zone = demand_engine()
    with pytest.raises(TypeError):
        engine.sweep_for(zone.pivot, liq.levels[0])
    with pytest.raises(TypeError):
        engine.sweep_for(zone, liq.levels[0].swing)
 
 
# ================================================================== 30. sweeps_for_zone
def test_sweeps_for_zone_isolates_sweeps_by_zone():
    zone_a = mk_zone(BULL, bos_i=10, broken_i=2)
    zone_b = mk_zone(BULL, bos_i=14, broken_i=3)
    zone_c = mk_zone(BULL, bos_i=18, broken_i=4)
    l1, l2 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 90.0, i=14, seq=1)   # l2 known at slot 16
    engine, liq = make_engine(l1, l2)
    a1 = run(engine, zone_a, cndl(at(12), 97.0, 100.0))
    b1 = run(engine, zone_b, cndl(at(15), 97.0, 100.0))
    a2 = run(engine, zone_a, cndl(at(17), 89.0, 100.0))
    assert engine.sweeps == (a1, b1, a2)
    assert engine.sweeps_for_zone(zone_a) == (a1, a2)                 # acceptance order, only A's
    assert engine.sweeps_for_zone(zone_b) == (b1,)
    assert engine.sweeps_for_zone(zone_c) == ()                       # never seen
 
 
def test_sweeps_for_zone_of_an_unseen_zone_is_empty_and_changes_nothing():
    engine, liq, zone = demand_engine()
    before = snap(engine)
    assert engine.sweeps_for_zone(zone) == ()
    assert snap(engine) == before
    run(engine, zone, cndl(at(12), 97.0, 100.0))                      # the zone is still usable afterwards
    assert len(engine.sweeps_for_zone(zone)) == 1
 
 
def test_sweeps_for_zone_is_not_shared_with_a_conflicting_zone_of_the_same_identity():
    engine, liq, zone = demand_engine()
    sweep = run(engine, zone, cndl(at(12), 97.0, 100.0))
    assert engine.sweeps_for_zone(zone) == (sweep,)
    assert engine.sweeps_for_zone(conflicting_zone(zone)) == ()
 
 
@pytest.mark.parametrize("bad", [None, "zone", 5, [], object()])
def test_sweeps_for_zone_validates_its_argument(bad):
    engine, liq, zone = demand_engine()
    with pytest.raises(TypeError):
        engine.sweeps_for_zone(bad)
    with pytest.raises(TypeError):
        engine.sweeps_for_zone(zone.pivot)
 
 
# ================================================================== 31. sweeps_known_at
def test_sweeps_known_at_respects_the_sweep_candle_close_time():
    engine, liq, zone = demand_engine()
    sweep = run(engine, zone, cndl(at(12), 97.0, 100.0))              # opens slot 12, closes (known) slot 13
    assert sweep.known_at == at(13)
    assert engine.sweeps_known_at(at(0)) == ()
    assert engine.sweeps_known_at(sweep.candle_time) == ()            # the candle's own open time: not yet known
    assert engine.sweeps_known_at(sweep.known_at - timedelta(seconds=1)) == ()
    assert engine.sweeps_known_at(sweep.known_at) == (sweep,)         # known exactly at the close
    assert engine.sweeps_known_at(sweep.known_at + timedelta(days=9)) == (sweep,)
 
 
def test_sweeps_known_at_returns_only_the_sweeps_known_by_then_in_acceptance_order():
    zone = mk_zone(BULL)
    l1, l2 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 90.0, i=14, seq=1)
    engine, liq = make_engine(l1, l2)
    s1 = run(engine, zone, cndl(at(12), 97.0, 100.0))                 # known slot 13
    s2 = run(engine, zone, cndl(at(17), 89.0, 100.0))                 # known slot 18
    assert engine.sweeps_known_at(at(12)) == ()
    assert engine.sweeps_known_at(at(13)) == (s1,)
    assert engine.sweeps_known_at(at(17)) == (s1,)
    assert engine.sweeps_known_at(at(18)) == (s1, s2)
 
 
def test_sweeps_known_at_spans_zones_and_does_not_mutate():
    zone_a, zone_b = mk_zone(BULL, bos_i=10, broken_i=2), mk_zone(BULL, bos_i=14, broken_i=3)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    sa = run(engine, zone_a, cndl(at(12), 97.0, 100.0))
    sb = run(engine, zone_b, cndl(at(15), 97.0, 100.0))
    before = snap(engine, zone_a, zone_b)
    assert engine.sweeps_known_at(at(14)) == (sa,)
    assert engine.sweeps_known_at(at(16)) == (sa, sb)
    assert snap(engine, zone_a, zone_b) == before
 
 
@pytest.mark.parametrize("bad", [None, "now", 5, 1.5, date(2026, 1, 1)])
def test_sweeps_known_at_requires_a_datetime(bad):
    engine, liq, zone = demand_engine()
    with pytest.raises(TypeError):
        engine.sweeps_known_at(bad)
    run(engine, zone, cndl(at(12), 97.0, 100.0))
    with pytest.raises(TypeError):
        engine.sweeps_known_at(bad)
 
 
def test_sweeps_known_at_with_no_sweeps_is_empty():
    engine, liq, zone = demand_engine()
    assert engine.sweeps_known_at(at(1000)) == ()
 
 
# ================================================================== 32. the engine only READS the LiquidityEngine
def test_the_sweep_engine_does_not_mutate_the_liquidity_engine():
    zone = mk_zone(BULL)
    l1, l2 = mk_swing(LOW, 98.0, i=0, seq=0), mk_swing(LOW, 90.0, i=14, seq=1)
    engine, liq = make_engine(l1, l2)
    levels_before = liq.levels
    attrs_before = sorted(vars(liq))
    required_before = liq.required_liquidity_for(zone, at(30))
    run(engine, zone, cndl(at(12), 97.0, 100.0))                      # a sweep is recorded
    run(engine, zone, cndl(at(17), 89.0, 100.0))                      # and another
    run(engine, zone, cndl(at(18), 50.0, 100.0))                      # and nothing new
    assert liq.levels == levels_before and all(a is b for a, b in zip(liq.levels, levels_before))
    assert sorted(vars(liq)) == attrs_before
    assert liq.required_liquidity_for(zone, at(30)) is required_before    # a swept level is not consumed or removed
    assert liq.levels_known_at(at(30)) == levels_before
 
 
def test_no_sweep_state_is_stored_inside_the_liquidity_engine():
    engine, liq, zone = demand_engine()
    run(engine, zone, cndl(at(12), 97.0, 100.0))
    for name in dir(liq):
        assert "sweep" not in name.lower()
    for value in vars(liq).values():
        assert not any(isinstance(item, Sweep) for item in (value if isinstance(value, (list, tuple)) else [value]))
 
 
def test_the_sweep_engine_never_feeds_the_liquidity_engine(monkeypatch):
    engine, liq, zone = demand_engine()
 
    def forbidden(*args, **kwargs):
        raise AssertionError("SweepEngine must only READ the LiquidityEngine")
 
    monkeypatch.setattr(liq, "process_swing", forbidden)
    run(engine, zone, cndl(at(12), 97.0, 100.0))
    run(engine, zone, cndl(at(13), 99.5, 100.0))
    engine.sweeps_known_at(at(40))
 
 
def test_a_rejected_call_does_not_touch_the_liquidity_engine():
    liq = RecordingLiquidity(M5)
    liq.process_swing(mk_swing(LOW, 98.0), at(2))
    engine = SweepEngine(liq)
    zone = mk_zone(BULL)
    levels_before = liq.levels
    with pytest.raises(ValueError):
        run(engine, zone, cndl(at(5), 97.0, 100.0))                   # before the zone existed
    assert liq.levels == levels_before
 
 
# ================================================================== scope: no extra rules
def test_there_is_no_tolerance_a_tiny_penetration_sweeps():
    engine, liq, zone = demand_engine()
    assert run(engine, zone, cndl(at(11), 98.0 - 1e-9, 100.0)) is not None
    engine, liq, zone = supply_engine()
    assert run(engine, zone, cndl(at(11), 100.0, 111.0 + 1e-9)) is not None
 
 
def test_there_is_no_distance_filter():
    zone = mk_zone(BULL)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    assert run(engine, zone, cndl(at(11), 0.5, 100.0)) is not None    # a very deep wick still sweeps
 
 
def test_there_is_no_distance_between_the_level_and_the_zone_limit():
    zone = mk_zone(BULL)
    engine, liq = make_engine(mk_swing(LOW, 1.0))                      # far below the zone
    assert run(engine, zone, cndl(at(11), 0.5, 100.0)) is not None
 
 
def test_there_is_no_time_or_session_filter():
    zone = mk_zone(BULL)
    engine, liq = make_engine(mk_swing(LOW, 98.0))
    odd_hour = datetime(2026, 1, 3, 3, 35)                             # a Saturday, 03:35
    assert run(engine, zone, cndl(odd_hour, 97.0, 100.0)) is not None
 
 
@pytest.mark.parametrize("open_, close", [(98.0, 98.0), (97.0, 99.9), (99.9, 97.0), (97.0, 97.0)])
def test_the_candle_shape_does_not_matter_only_the_wick(open_, close):
    engine, liq, zone = demand_engine()
    candle = Candle(M5, at(11), open_, max(open_, close, 99.9), min(open_, close, 96.5), close)
    assert run(engine, zone, candle) is not None
 
 
def test_the_engine_exposes_no_touch_setup_entry_risk_or_structure_behaviour():
    engine, liq, zone = demand_engine()
    for attr in ("touch", "touched", "setup", "confirm", "confirmation", "entry", "sl", "tp", "stop_loss",
                 "take_profit", "risk", "news", "execute", "execution", "bos", "choch", "tolerance",
                 "distance", "atr", "displacement", "session", "consume", "consumed", "expire", "expired",
                 "invalidate", "delete", "remove", "clear", "reset"):
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
        "koffie.strategy.engines.liquidity_engine",
        "koffie.strategy.models.candle",
        "koffie.strategy.models.liquidity",
        "koffie.strategy.models.pivot",
        "koffie.strategy.models.sweep",
        "koffie.strategy.models.zone",
    }, imported