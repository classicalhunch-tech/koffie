"""Tests for the Strategy 1 coordinator (koffie/strategy/strategy_1.py).
 
Only the REAL engines and models are used; there are no mocks. Every scenario is fed
through `Strategy1.process_candle` with candles merged in (close time, timeframe) order.
 
The synthetic market (BUY version; the SELL version is its exact mirror, p -> 200 - p):
 
  H1   ten candles whose confirmed swings make HH + HL        -> LONG_ONLY
  M15  the same shape plus a bearish decisive candle (the Pivot) and a close above the
       active swing high (BOS)                                -> DEMAND zone 100.0 .. 103.6,
       created when the BOS candle closes (Z)
  M5   p0..p2  three candles just before Z with a swing low at 98.0 (the required liquidity)
       a0      touch   : closes inside the zone
       a1      sweep   : wick below 98.0 (neutral candle)
       a2      confirm : bullish decisive candle closing at 103.2
  BUY  entry 103.2, SL 97.5 (sweep candle low), R 5.7, TP 114.6 (= entry + 2R)
"""
from datetime import datetime, timedelta
 
import pytest
 
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.confirmation import InvalidationReason
from koffie.strategy.models.entry import EntryDirection
from koffie.strategy.models.setup import SetupStatus
from koffie.strategy.models.structure import DirectionalPermission
from koffie.strategy.models.zone import ZoneType
from koffie.strategy.strategy_1 import (
    SkipReason,
    Strategy1,
    Strategy1Result,
    Strategy1Trade,
    permission_allows,
)
 
H1, M15, M5 = Timeframe.H1, Timeframe.M15, Timeframe.M5
LONG_ONLY = DirectionalPermission.LONG_ONLY
SHORT_ONLY = DirectionalPermission.SHORT_ONLY
NO_TRADE = DirectionalPermission.NO_TRADE_PERMITTED
RANK = {H1: 0, M15: 1, M5: 2}
 
H1_START = datetime(2026, 1, 5, 0, 0)
M15_START = datetime(2026, 1, 6, 0, 0)
Z = M15_START + 11 * M15.duration                   # the zone is created when the M15 BOS candle closes
 
# (low, high) of the H1 / M15 swing structure: low@1, high@3, low@5 (HL), high@7 (HH)
SWING_ROWS = [(95, 96), (90, 95), (93, 99), (97, 100), (96, 98), (95, 99), (97, 103), (100, 105), (99, 104)]
 
 
# ------------------------------------------------------------------ data builders
def mirror(row):
    """(o, h, l, c) -> the mirror image around 100 (BUY market -> SELL market)."""
    o, h, l, c = row
    return (200.0 - o, 200.0 - l, 200.0 - h, 200.0 - c)
 
 
def at(tf, start, i):
    return start + i * tf.duration
 
 
def candle(tf, open_time, row, flip=False):
    o, h, l, c = mirror(row) if flip else row
    return Candle(tf, open_time, float(o), float(h), float(l), float(c))
 
 
def h1_candles(flip=False, start=H1_START):
    rows = [(lo, hi, lo, hi) for lo, hi in SWING_ROWS] + [(101, 103, 101, 103)]
    return [candle(H1, at(H1, start, i), r, flip) for i, r in enumerate(rows)]
 
 
def m15_candles(flip=False, with_bos=True, with_pivot=True):
    rows = [(lo, hi, lo, hi) for lo, hi in SWING_ROWS]
    rows.append((103.5, 103.6, 100.0, 100.4) if with_pivot else (100.5, 103.6, 100.0, 103.2))   # Pivot candle
    if with_bos:
        rows.append((101.5, 107.0, 101.0, 106.5))                                                # BOS candle
    return [candle(M15, at(M15, M15_START, i), r, flip) for i, r in enumerate(rows)]
 
 
PRE = [(101, 104, 99, 101), (100, 104, 98, 100), (101, 104, 99.5, 101)]    # swing low 98 (known at Z)
TOUCH = (102, 103, 100.5, 101)                                              # neutral, closes inside the zone
SWEEP = (99.5, 101, 97.5, 99.8)                                             # neutral, wick below 98
CONFIRM = (99.0, 103.5, 98.8, 103.2)                                        # bullish decisive
 
 
def m5_candles(rows, flip=False, first=0):
    """M5 candles from slot `first` (slot 0 opens exactly when the zone is created)."""
    return [candle(M5, Z + (first + k) * M5.duration, r, flip) for k, r in enumerate(rows)]
 
 
def m5_pre(flip=False):
    return m5_candles(PRE, flip, first=-3)
 
 
def key(c):
    return (c.close_time, RANK[c.timeframe])
 
 
def feed(strategy, *series):
    """Merge candles by (close time, timeframe) and process them with now = close time."""
    merged = sorted((c for s in series for c in s), key=key)
    return [(c, strategy.process_candle(c, c.close_time)) for c in merged]
 
 
def buy_market(strategy, post_rows, h1=True):
    """H1 + M15 + pre-zone M5 + the given post-zone M5 rows, all merged."""
    return feed(strategy, h1_candles() if h1 else [], m15_candles(), m5_pre(), m5_candles(post_rows))
 
 
def last_setup(strategy):
    return strategy.setup_engine.setups[-1]
 
 
def near(a, b):
    return abs(a - b) < 1e-9
 
 
# ------------------------------------------------------------------ the market itself behaves as designed
def test_the_synthetic_h1_market_gives_long_only_and_its_mirror_short_only():
    s = Strategy1()
    feed(s, h1_candles())
    assert s.permission is LONG_ONLY
    s = Strategy1()
    feed(s, h1_candles(flip=True))
    assert s.permission is SHORT_ONLY
 
 
def test_the_synthetic_m15_market_creates_one_demand_zone_and_its_mirror_one_supply_zone():
    s = Strategy1()
    feed(s, m15_candles())
    (zone,) = s.zones
    assert zone.zone_type is ZoneType.DEMAND and (zone.low, zone.high) == (100.0, 103.6) and zone.created_at == Z
    s = Strategy1()
    feed(s, m15_candles(flip=True))
    (zone,) = s.zones
    assert zone.zone_type is ZoneType.SUPPLY and (zone.low, zone.high) == (96.4, 100.0) and zone.created_at == Z
 
 
# ------------------------------------------------------------------ 1. directional permission failure
def test_no_h1_conclusion_means_no_trade_permitted_and_the_confirmed_setup_is_skipped():
    s = Strategy1()
    assert s.permission is NO_TRADE
    out = buy_market(s, [TOUCH, SWEEP, CONFIRM], h1=False)
    result = out[-1][1]
    assert s.permission is NO_TRADE and result.permission is NO_TRADE
    assert s.trades == () and result.trades == ()
    (skip,) = result.skipped
    assert skip.reason is SkipReason.DIRECTION_NOT_PERMITTED
    assert s.setup_engine.status_for(skip.setup) is SetupStatus.CONFIRMED          # the engines still ran
    assert s.entry_engine.entries == () and s.sl_tp_engine.sl_tps == ()
    assert not s.setup_engine.is_zone_consumed(skip.setup.zone)
 
 
def test_a_bearish_h1_does_not_allow_a_buy_from_a_demand_zone():
    s = Strategy1()
    out = feed(s, h1_candles(flip=True), m15_candles(), m5_pre(), m5_candles([TOUCH, SWEEP, CONFIRM]))
    assert s.permission is SHORT_ONLY
    assert s.trades == ()
    (skip,) = out[-1][1].skipped
    assert skip.reason is SkipReason.DIRECTION_NOT_PERMITTED and skip.setup.zone_type is ZoneType.DEMAND
 
 
def test_a_bullish_h1_does_not_allow_a_sell_from_a_supply_zone():
    s = Strategy1()
    out = feed(s, h1_candles(), m15_candles(flip=True), m5_pre(flip=True),
               m5_candles([TOUCH, SWEEP, CONFIRM], flip=True))
    assert s.permission is LONG_ONLY
    assert s.trades == ()
    (skip,) = out[-1][1].skipped
    assert skip.reason is SkipReason.DIRECTION_NOT_PERMITTED and skip.setup.zone_type is ZoneType.SUPPLY
 
 
def test_permission_allows_maps_each_zone_type_to_its_own_h1_permission():
    assert permission_allows(LONG_ONLY, ZoneType.DEMAND) is True
    assert permission_allows(LONG_ONLY, ZoneType.SUPPLY) is False
    assert permission_allows(SHORT_ONLY, ZoneType.SUPPLY) is True
    assert permission_allows(SHORT_ONLY, ZoneType.DEMAND) is False
    assert permission_allows(NO_TRADE, ZoneType.DEMAND) is False
    assert permission_allows(NO_TRADE, ZoneType.SUPPLY) is False
    with pytest.raises(TypeError):
        permission_allows("LONG_ONLY", ZoneType.DEMAND)
    with pytest.raises(TypeError):
        permission_allows(LONG_ONLY, "DEMAND")
 
 
# ------------------------------------------------------------------ 2. no valid zone
def test_no_zone_exists_without_an_m15_bos():
    s = Strategy1()
    feed(s, h1_candles(), m15_candles(with_bos=False), m5_pre(), m5_candles([TOUCH, SWEEP, CONFIRM]))
    assert s.zones == () and s.setup_engine.setups == () and s.trades == ()
 
 
def test_a_bos_without_a_pivot_creates_no_zone_and_no_setup():
    s = Strategy1()
    feed(s, h1_candles(), m15_candles(with_pivot=False), m5_pre(), m5_candles([TOUCH, SWEEP, CONFIRM]))
    (outcome,) = s.zone_engine.outcomes
    assert not outcome.is_found
    assert s.zones == () and s.setup_engine.setups == () and s.trades == ()
 
 
# ------------------------------------------------------------------ 3. no M5 touch
def test_no_setup_starts_when_no_m5_candle_closes_inside_the_zone():
    s = Strategy1()
    outside = [(108, 111, 107.5, 110), (108, 109, 97.0, 108.5), (108, 112, 107, 111)]   # one wick even dives below 98
    buy_market(s, outside)
    assert len(s.zones) == 1
    assert s.setup_engine.setups == () and s.trades == ()
 
 
def test_a_candle_that_closes_inside_the_zone_exactly_when_it_is_created_is_not_a_touch():
    s = Strategy1()
    feed(s, h1_candles(), m15_candles(), m5_pre())            # the last pre candle closes inside the zone at Z
    assert s.zones[0].created_at == Z
    assert s.setup_engine.setups == ()                        # it opened before the zone existed
    out = feed(s, m5_candles([TOUCH]))
    assert len(s.setup_engine.setups) == 1 and out[-1][1].setup_changes == s.setup_engine.setups
 
 
# ------------------------------------------------------------------ 4. no required liquidity
def test_a_touch_without_required_liquidity_waits_for_a_sweep():
    s = Strategy1()
    deep = (101, 101.5, 95, 100.9)
    feed(s, h1_candles(), m15_candles(),
         m5_candles([TOUCH, deep, (101, 102, 100.8, 101.5), (101, 102, 99, 101.2)]))
    setup = last_setup(s)
    zone = s.zones[0]
    assert s.setup_engine.status_for(setup) is SetupStatus.WAITING_FOR_SWEEP
    assert s.liquidity_engine.required_liquidity_for(zone, Z + 2 * M5.duration) is None   # nothing known when `deep` opened
    assert s.setup_engine.sweep_for(setup) is None and s.trades == ()
 
 
# ------------------------------------------------------------------ 5. no sweep
def test_required_liquidity_that_is_never_penetrated_gives_no_sweep():
    s = Strategy1()
    buy_market(s, [TOUCH, (101, 103, 99, 102), (101, 103, 98.5, 102), (101, 103, 98.2, 102)])
    setup = last_setup(s)
    zone = s.zones[0]
    assert s.liquidity_engine.required_liquidity_for(zone, Z + 3 * M5.duration).price == 98.0
    assert s.setup_engine.status_for(setup) is SetupStatus.WAITING_FOR_SWEEP
    assert s.setup_engine.sweep_for(setup) is None and s.trades == ()
 
 
def test_a_wick_exactly_on_the_liquidity_is_not_a_sweep():
    s = Strategy1()
    buy_market(s, [TOUCH, (101, 103, 98.0, 102)])
    assert s.setup_engine.status_for(last_setup(s)) is SetupStatus.WAITING_FOR_SWEEP
 
 
# ------------------------------------------------------------------ 6. confirmation waiting
def test_a_sweep_followed_by_a_neutral_candle_keeps_waiting_for_confirmation():
    s = Strategy1()
    out = buy_market(s, [TOUCH, SWEEP, (100, 101, 99.5, 100.4)])
    setup = last_setup(s)
    assert s.setup_engine.status_for(setup) is SetupStatus.WAITING_FOR_CONFIRMATION
    assert s.setup_engine.sweep_for(setup).swept_price == 98.0
    assert s.setup_engine.confirmation_for(setup) is None and s.trades == () and out[-1][1].skipped == ()
 
 
def test_the_first_wrong_direction_decisive_candle_inside_the_zone_only_waits():
    s = Strategy1()
    out = buy_market(s, [TOUCH, SWEEP, (103, 103.2, 100.2, 100.4)])        # bearish decisive, close >= zone low
    setup = last_setup(s)
    assert s.setup_engine.status_for(setup) is SetupStatus.WAITING_FOR_CONFIRMATION
    assert s.setup_engine.invalidation_for(setup) is None
    assert s.trades == () and out[-1][1].trades == () and out[-1][1].skipped == ()
 
 
# ------------------------------------------------------------------ 7. confirmation invalidation
def test_a_wrong_direction_decisive_close_beyond_the_zone_boundary_invalidates():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP, (101, 101.2, 98.5, 99.0)])                # bearish decisive, close below zone low
    setup = last_setup(s)
    assert s.setup_engine.status_for(setup) is SetupStatus.INVALIDATED
    assert s.setup_engine.invalidation_for(setup).reason is InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY
    assert s.trades == () and s.entry_engine.entries == ()
 
 
def test_a_second_wrong_direction_decisive_candle_invalidates():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP, (103, 103.2, 100.2, 100.4), (103, 103.2, 100.3, 100.5)])
    setup = last_setup(s)
    assert s.setup_engine.status_for(setup) is SetupStatus.INVALIDATED
    assert s.setup_engine.invalidation_for(setup).reason is InvalidationReason.SECOND_WRONG_DIRECTION_DECISIVE
    assert s.trades == ()
 
 
def test_a_confirmation_after_one_wrong_decisive_candle_still_confirms():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP, (103, 103.2, 100.2, 100.4), (100.5, 104, 100.4, 103.8)])
    assert s.setup_engine.status_for(last_setup(s)) is SetupStatus.TRADE_CREATED
    assert len(s.trades) == 1
 
 
# ------------------------------------------------------------------ 8-9. successful setup and entry (BUY)
def test_a_complete_buy_chain_creates_the_setup_the_entry_and_the_trade():
    s = Strategy1()
    out = buy_market(s, [TOUCH, SWEEP, CONFIRM])
 
    (zone,) = s.zones
    assert zone.zone_type is ZoneType.DEMAND
    (setup,) = s.setup_engine.setups
    assert setup.zone == zone and setup.touch_candle.open_time == Z
    sweep = s.setup_engine.sweep_for(setup)
    assert sweep.swept_price == 98.0 and sweep.candle.open_time == Z + M5.duration
    confirmation = s.setup_engine.confirmation_for(setup)
    assert confirmation.candle.open_time == Z + 2 * M5.duration
 
    (trade,) = s.trades
    assert isinstance(trade, Strategy1Trade)
    assert trade.setup == setup and trade.zone == zone and trade.direction is EntryDirection.LONG
    assert s.entry_engine.entries == (trade.entry,) and s.sl_tp_engine.sl_tps == (trade.sl_tp,)
    assert s.setup_engine.status_for(setup) is SetupStatus.TRADE_CREATED
    assert s.setup_engine.is_zone_consumed(zone)
 
    result = out[-1][1]
    assert isinstance(result, Strategy1Result) and result.trades == (trade,) and result.skipped == ()
    assert result.permission is LONG_ONLY and result.candle == out[-1][0]
    assert all(r.trades == () for _, r in out[:-1])                        # only the confirming candle trades
 
 
def test_the_entry_price_is_the_close_of_the_confirming_candle_and_is_known_when_it_closes():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP, CONFIRM])
    (trade,) = s.trades
    assert trade.entry_price == 103.2 == trade.entry.candle.close
    assert trade.known_at == trade.entry.candle.close_time == Z + 3 * M5.duration
    assert trade.sl_tp.is_known_at(trade.known_at) and not trade.sl_tp.is_known_at(trade.known_at - timedelta(seconds=1))
 
 
# ------------------------------------------------------------------ 10. BUY SL / TP
def test_buy_sl_is_the_sweep_candle_low_and_tp_is_exactly_two_r():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP, CONFIRM])
    (trade,) = s.trades
    assert trade.stop_loss == 97.5 == trade.sl_tp.sweep_candle.low
    assert near(trade.sl_tp.risk, 103.2 - 97.5)
    assert near(trade.take_profit, 103.2 + 2 * (103.2 - 97.5)) and near(trade.take_profit, 114.6)
    assert near(trade.sl_tp.rr_ratio, 2.0)
    assert trade.stop_loss < trade.entry_price < trade.take_profit
 
 
def test_a_buy_with_the_stop_on_the_wrong_side_creates_no_entry_and_does_not_consume_the_zone():
    s = Strategy1()
    wrong_side = (94.0, 97.3, 93.9, 97.2)                                   # bullish decisive, closes below the sweep low 97.5
    out = buy_market(s, [TOUCH, SWEEP, wrong_side])
    (skip,) = out[-1][1].skipped
    assert skip.reason is SkipReason.INVALID_GEOMETRY
    assert s.setup_engine.status_for(skip.setup) is SetupStatus.CONFIRMED
    assert s.trades == () and s.entry_engine.entries == () and s.sl_tp_engine.sl_tps == ()
    assert not s.setup_engine.is_zone_consumed(skip.setup.zone)
 
 
# ------------------------------------------------------------------ 11. SELL chain and SL / TP
def sell_market(strategy, post_rows, h1=True):
    return feed(strategy, h1_candles(flip=True) if h1 else [], m15_candles(flip=True),
                m5_pre(flip=True), m5_candles(post_rows, flip=True))
 
 
def test_a_complete_sell_chain_creates_a_supply_zone_trade_with_mirrored_sl_and_tp():
    s = Strategy1()
    sell_market(s, [TOUCH, SWEEP, CONFIRM])
    (zone,) = s.zones
    assert zone.zone_type is ZoneType.SUPPLY
    (trade,) = s.trades
    assert trade.direction is EntryDirection.SHORT and trade.zone == zone
    assert near(trade.entry_price, 96.8)
    assert near(trade.stop_loss, 102.5) and trade.stop_loss == trade.sl_tp.sweep_candle.high
    assert near(trade.sl_tp.risk, 102.5 - 96.8)
    assert near(trade.take_profit, 96.8 - 2 * (102.5 - 96.8)) and near(trade.take_profit, 85.4)
    assert near(trade.sl_tp.rr_ratio, 2.0)
    assert trade.take_profit < trade.entry_price < trade.stop_loss
    assert s.setup_engine.is_zone_consumed(zone)
 
 
def test_sell_confirmation_waiting_and_invalidation_mirror_the_buy_rules():
    s = Strategy1()
    sell_market(s, [TOUCH, SWEEP, (103, 103.2, 100.2, 100.4)])
    assert s.setup_engine.status_for(last_setup(s)) is SetupStatus.WAITING_FOR_CONFIRMATION and s.trades == ()
    s = Strategy1()
    sell_market(s, [TOUCH, SWEEP, (101, 101.2, 98.5, 99.0)])
    setup = last_setup(s)
    assert s.setup_engine.status_for(setup) is SetupStatus.INVALIDATED
    assert s.setup_engine.invalidation_for(setup).reason is InvalidationReason.CLOSED_BEYOND_ZONE_BOUNDARY
    assert s.trades == ()
 
 
def test_a_zone_that_produced_a_trade_cannot_produce_another():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP, CONFIRM])
    feed(s, m5_candles([TOUCH], first=3))                                   # the zone is touched again
    assert len(s.setup_engine.setups) == 1 and len(s.trades) == 1
 
 
# ------------------------------------------------------------------ 12. causal / lookahead protection
def test_a_level_confirmed_by_a_candle_cannot_be_swept_by_that_candle_or_the_next_one_to_close_it():
    s = Strategy1()
    a1 = (99.5, 101, 97.5, 99.8)                 # creates a swing low at 97.5 ...
    a2 = (99.8, 100.5, 98.8, 100.0)              # ... confirmed when this candle closes
    a3 = (99, 100, 97.0, 99.5)                   # opens exactly when the level becomes known; wick below it
    feed(s, h1_candles(), m15_candles(), m5_candles([TOUCH, a1, a2]))       # no earlier liquidity at all
    setup = last_setup(s)
    assert s.setup_engine.status_for(setup) is SetupStatus.WAITING_FOR_SWEEP        # a1 did not sweep its own swing
    level = s.liquidity_engine.levels[-1]
    assert level.price == 97.5 and level.known_at == Z + 3 * M5.duration
    feed(s, m5_candles([a3], first=3))
    assert s.setup_engine.status_for(setup) is SetupStatus.WAITING_FOR_CONFIRMATION
    sweep = s.setup_engine.sweep_for(setup)
    assert sweep.swept_price == 97.5 and sweep.candle.open_time == Z + 3 * M5.duration
 
 
def test_a_permission_that_only_appears_after_the_confirmation_never_creates_a_late_trade():
    s = Strategy1()
    late_h1 = h1_candles(start=H1_START + timedelta(hours=19))              # its last swing is confirmed after the M5 confirmation
    out = feed(s, late_h1, m15_candles(), m5_pre(), m5_candles([TOUCH, SWEEP, CONFIRM]))
    confirming = [r for c, r in out if c.timeframe is M5 and r.skipped]
    assert len(confirming) == 1 and confirming[0].skipped[0].reason is SkipReason.DIRECTION_NOT_PERMITTED
    assert s.permission is LONG_ONLY                                         # H1 turned bullish afterwards ...
    feed(s, m5_candles([(101, 102, 100.5, 101.5)], first=3 * 12))            # ... and later M5 candles keep coming
    assert s.trades == () and s.entry_engine.entries == ()                   # ... but the old confirmation is never traded
    assert s.setup_engine.status_for(confirming[0].skipped[0].setup) is SetupStatus.CONFIRMED
 
 
def test_the_h1_permission_used_is_the_one_known_when_the_confirming_candle_closes():
    s = Strategy1()
    out = feed(s, h1_candles(), m15_candles(), m5_pre(), m5_candles([TOUCH, SWEEP, CONFIRM]))
    assert [r.permission for c, r in out if c.timeframe is M5][-1] is LONG_ONLY
    assert len(s.trades) == 1
 
 
def test_candles_that_go_backwards_are_rejected_and_change_nothing():
    s = Strategy1()
    buy_market(s, [TOUCH, SWEEP])
    before = (s.setup_engine.setups, s.trades, s.liquidity_engine.levels, s.zones)
    for old in m5_candles([TOUCH], first=0) + m5_candles([(101, 103, 99, 102)], first=-2):
        with pytest.raises(ValueError):
            s.process_candle(old, old.close_time)
    older_h1 = h1_candles()[0]
    with pytest.raises(ValueError):
        s.process_candle(older_h1, older_h1.close_time)
    assert (s.setup_engine.setups, s.trades, s.liquidity_engine.levels, s.zones) == before
    feed(s, m5_candles([CONFIRM], first=2))                                  # the strategy is still healthy
    assert len(s.trades) == 1
 
 
def test_an_unclosed_candle_is_rejected_and_changes_nothing():
    s = Strategy1()
    buy_market(s, [TOUCH])
    c = m5_candles([SWEEP], first=1)[0]
    before = (s.setup_engine.setups, s.liquidity_engine.levels)
    for now in (c.open_time, c.close_time - timedelta(seconds=1)):
        with pytest.raises(ValueError):
            s.process_candle(c, now)
    assert (s.setup_engine.setups, s.liquidity_engine.levels) == before
    s.process_candle(c, c.close_time)
    assert s.setup_engine.status_for(last_setup(s)) is SetupStatus.WAITING_FOR_CONFIRMATION
 
 
def test_at_the_same_instant_higher_timeframes_must_be_fed_before_lower_ones():
    s = Strategy1()
    m5 = Candle(M5, datetime(2026, 1, 6, 0, 10), 100.0, 101.0, 99.0, 100.0)
    m15 = Candle(M15, datetime(2026, 1, 6, 0, 0), 100.0, 101.0, 99.0, 100.0)    # closes at the same instant as `m5`
    s.process_candle(m5, m5.close_time)
    with pytest.raises(ValueError):
        s.process_candle(m15, m15.close_time)
    s2 = Strategy1()
    s2.process_candle(m15, m15.close_time)
    s2.process_candle(m5, m5.close_time)                                         # the other order is fine
 
 
def test_overlapping_or_repeated_candles_of_one_timeframe_are_rejected():
    s = Strategy1()
    a = Candle(M5, datetime(2026, 1, 6, 0, 0), 100.0, 101.0, 99.0, 100.0)
    s.process_candle(a, a.close_time)
    with pytest.raises(ValueError):
        s.process_candle(a, a.close_time)                                        # repeated
    overlap = Candle(M5, a.open_time + timedelta(minutes=2), 100.0, 101.0, 99.0, 100.0)
    with pytest.raises(ValueError):
        s.process_candle(overlap, overlap.close_time)                            # overlapping
 
 
def test_replaying_the_confirming_candle_never_duplicates_a_trade():
    s = Strategy1()
    out = buy_market(s, [TOUCH, SWEEP, CONFIRM])
    confirming = out[-1][0]
    with pytest.raises(ValueError):
        s.process_candle(confirming, confirming.close_time)
    assert len(s.trades) == 1 and len(s.entry_engine.entries) == 1 and len(s.sl_tp_engine.sl_tps) == 1
 
 
def test_gaps_between_candles_are_allowed():
    s = Strategy1()
    feed(s, h1_candles(), m15_candles(), m5_pre(), m5_candles([TOUCH, SWEEP]))
    feed(s, m5_candles([CONFIRM], first=2 + 40))                                 # a long pause before the confirming candle
    assert len(s.trades) == 1
 
 
# ------------------------------------------------------------------ argument validation
def test_wrong_argument_types_are_rejected():
    s = Strategy1()
    c = m5_candles([TOUCH])[0]
    for bad in (None, "candle", 5, (c,)):
        with pytest.raises(TypeError):
            s.process_candle(bad, c.close_time)
    for bad in (None, "now", 5, c.close_time.date()):
        with pytest.raises(TypeError):
            s.process_candle(c, bad)
 
 
def test_mixed_naive_and_aware_datetimes_raise_and_change_nothing():
    from datetime import timezone
 
    s = Strategy1()
    c = m5_candles([TOUCH])[0]
    with pytest.raises(TypeError):
        s.process_candle(c, c.close_time.replace(tzinfo=timezone.utc))
    assert s.setup_engine.setups == () and s.liquidity_engine.levels == ()
    s.process_candle(c, c.close_time)
 
 
# ------------------------------------------------------------------ scope: the coordinator adds no rules or engines of its own
def test_result_fields_of_each_timeframe_are_only_filled_where_they_apply():
    s = Strategy1()
    out = feed(s, h1_candles(), m15_candles(), m5_pre(), m5_candles([TOUCH, SWEEP, CONFIRM]))
    for c, r in out:
        if c.timeframe is H1:
            assert r.new_zones == () and r.setup_changes == () and r.trades == () and r.skipped == ()
        elif c.timeframe is M15:
            assert r.setup_changes == () and r.trades == () and r.skipped == ()
        else:
            assert r.new_zones == ()
    assert sum(len(r.new_zones) for _, r in out) == 1
    assert sum(len(r.trades) for _, r in out) == 1
 
 
def test_the_coordinator_exposes_no_forbidden_behaviour():
    s = Strategy1()
    for attr in ("h4", "spread", "order_flow", "volume", "risk", "position_size", "news", "execute",
                 "flip_zones", "strategy_2", "tolerance", "atr"):
        assert not hasattr(s, attr), attr
 
 
def test_the_coordinator_composes_the_existing_engines_instead_of_copying_their_rules():
    s = Strategy1()
    from koffie.strategy.engines.entry_engine import EntryEngine
    from koffie.strategy.engines.liquidity_engine import LiquidityEngine
    from koffie.strategy.engines.setup_engine import SetupEngine
    from koffie.strategy.engines.sl_tp_engine import SLTPEngine
    from koffie.strategy.engines.structure_processor import StructureProcessor
    from koffie.strategy.engines.zone_engine import ZoneEngine
 
    assert isinstance(s.h1, StructureProcessor) and s.h1.timeframe is H1
    assert isinstance(s.m15, StructureProcessor) and s.m15.timeframe is M15
    assert isinstance(s.zone_engine, ZoneEngine) and s.zone_engine.timeframe is M15
    assert isinstance(s.liquidity_engine, LiquidityEngine) and s.liquidity_engine.timeframe is M5
    assert isinstance(s.setup_engine, SetupEngine)
    assert isinstance(s.entry_engine, EntryEngine) and s.entry_engine.setup_engine is s.setup_engine
    assert isinstance(s.sl_tp_engine, SLTPEngine) and s.sl_tp_engine.entry_engine is s.entry_engine