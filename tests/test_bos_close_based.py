"""BOS is close-based, independent of CHOCH, and emitted once per level."""
from koffie.strategy.engines.bos_engine import BOSEngine
from koffie.strategy.models.bos import BOSDirection
from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.structure import StructureState
from tests._structure_fixtures import (
    bearish_prefix, bullish_prefix, candle, mirror, run,
)

# After the bullish prefix: active high = 115, protected low = 94.


def test_bullish_close_above_high_is_bos_and_trend_stays_bullish():
    c9 = candle(9, 118, 100, open_=102, close=116)
    sp, res = run(bullish_prefix() + [c9])
    bos = res[-1].bos
    assert bos is not None and bos.direction is BOSDirection.BULLISH
    assert bos.broken_swing.price == 115 and bos.close_price == 116
    assert res[-1].choch is None
    assert sp.structure.state is StructureState.BULLISH


def test_bearish_close_below_low_is_bos_and_trend_stays_bearish():
    c9 = mirror(candle(9, 118, 100, open_=102, close=116), 9)   # close 84 < active low 85
    sp, res = run(bearish_prefix() + [c9])
    bos = res[-1].bos
    assert bos is not None and bos.direction is BOSDirection.BEARISH
    assert bos.broken_swing.price == 85 and bos.close_price == 84
    assert res[-1].choch is None
    assert sp.structure.state is StructureState.BEARISH


def test_wick_above_high_without_close_above_is_not_bos():
    c9 = candle(9, 118, 100, open_=102, close=114)              # high 118 > 115, close 114
    _, res = run(bullish_prefix() + [c9])
    assert res[-1].bos is None


def test_wick_below_low_without_close_below_is_not_bos():
    c9 = mirror(candle(9, 118, 100, open_=102, close=114), 9)
    _, res = run(bearish_prefix() + [c9])
    assert res[-1].bos is None


def test_close_exactly_on_level_is_not_bos():
    c9 = candle(9, 118, 100, open_=102, close=115)
    _, res = run(bullish_prefix() + [c9])
    assert res[-1].bos is None


def test_no_bos_outside_bullish_or_bearish():
    sp, res = run(bullish_prefix()[:5] + [candle(5, 200, 100, open_=101, close=199)])
    assert sp.structure.state is StructureState.UNDETERMINED
    assert all(r.bos is None for r in res)


def test_opposite_wick_exclusion_removed():
    # Closes above the active high AND wicks below the protected low (92 < 94).
    # Old behaviour: suppressed (CHOCH-only). Locked rule: BOS, and no CHOCH (close 117 > 94).
    c9 = candle(9, 120, 92, open_=100, close=117)
    sp, res = run(bullish_prefix() + [c9])
    assert res[-1].bos is not None and res[-1].bos.direction is BOSDirection.BULLISH
    assert res[-1].choch is None
    assert sp.structure.state is StructureState.BULLISH


def test_opposite_wick_exclusion_removed_bearish():
    c9 = mirror(candle(9, 120, 92, open_=100, close=117), 9)
    sp, res = run(bearish_prefix() + [c9])
    assert res[-1].bos is not None and res[-1].bos.direction is BOSDirection.BEARISH
    assert res[-1].choch is None


def test_same_unchanged_level_emits_only_one_bos():
    c9 = candle(9, 118, 100, open_=102, close=116)              # first close above 115 -> BOS
    c10 = candle(10, 119, 112, open_=113, close=116.5)          # still above 115; c9 stays a non-swing
    sp, res = run(bullish_prefix() + [c9, c10])
    assert res[9].bos is not None
    assert res[10].bos is None
    assert res[10].snapshot.active_swing_high.price == 115      # level really unchanged
    assert len(sp.bos_engine.history) == 1


def test_bearish_same_unchanged_level_emits_only_one_bos():
    c9 = mirror(candle(9, 118, 100, open_=102, close=116), 9)
    c10 = mirror(candle(10, 119, 112, open_=113, close=116.5), 10)
    sp, res = run(bearish_prefix() + [c9, c10])
    assert res[9].bos is not None and res[10].bos is None
    assert res[10].snapshot.active_swing_low.price == 85         # level really unchanged
    assert len(sp.bos_engine.history) == 1


def test_new_structural_level_permits_new_bos():
    candles = bullish_prefix() + [
        candle(9, 118, 100, open_=102, close=116),              # BOS of 115
        candle(10, 119, 110, open_=112, close=117),             # still 115: no 2nd BOS
        candle(11, 117, 108, open_=110, close=110),             # confirms swing high 119 at c10
        candle(12, 122, 112, open_=113, close=121),             # close above NEW level 119
    ]
    sp, res = run(candles)
    assert res[9].bos is not None and res[9].bos.broken_swing.price == 115
    assert res[10].bos is None
    assert res[11].snapshot.active_swing_high.price == 119       # new relevant level
    assert res[11].bos is None                                    # close 110 not above it
    assert res[12].bos is not None and res[12].bos.broken_swing.price == 119
    assert [b.broken_swing.price for b in sp.bos_engine.history] == [115, 119]
    assert sp.structure.state is StructureState.BULLISH


def test_new_bearish_structural_level_permits_new_bos():
    candles = bullish_prefix() + [
        candle(9, 118, 100, open_=102, close=116),
        candle(10, 119, 110, open_=112, close=117),
        candle(11, 117, 108, open_=110, close=110),
        candle(12, 122, 112, open_=113, close=121),
    ]
    mirrored = [mirror(c, i) for i, c in enumerate(candles)]
    sp, res = run(mirrored)
    assert [b.broken_swing.price for b in sp.bos_engine.history] == [85, 81]
    assert all(b.direction is BOSDirection.BEARISH for b in sp.bos_engine.history)


def test_rejected_candle_does_not_update_dedupe_memory():
    sp, _ = run(bullish_prefix())
    eng = sp.bos_engine
    snap = sp.structure.snapshot
    bad = candle(9, 118, 100, open_=102, close=116)
    try:
        eng.process_candle(bad, bad.open_time, snap)            # not closed yet
    except ValueError:
        pass
    assert eng.history == ()
    ok = eng.process_candle(bad, bad.close_time, snap)
    assert ok is not None                                        # memory untouched by the rejection


def test_dedupe_survives_revaluating_and_restoration():
    candles = bullish_prefix() + [
        candle(9, 118, 100, open_=102, close=116),     # BOS of 115
        candle(10, 119, 88, open_=110, close=90),      # close below protected 94 -> CHOCH
        candle(11, 120, 92, open_=95, close=100),      # confirms low 88 (LL): mixed, stays REVALUATING
        candle(12, 121, 90, open_=100, close=108),
        candle(13, 122, 95, open_=108, close=112),     # confirms low 90 (HL vs 88): HH+HL -> BULLISH again
        candle(14, 123, 110, open_=112, close=120),    # closes above the SAME unchanged 115
    ]
    sp, res = run(candles)
    assert res[9].bos is not None and res[9].bos.broken_swing.price == 115
    assert res[10].choch is not None
    assert res[10].snapshot.state is StructureState.REVALUATING
    assert res[11].snapshot.state is StructureState.REVALUATING
    assert res[13].snapshot.state is StructureState.BULLISH            # restored
    assert res[13].snapshot.active_swing_high.price == 115             # same unchanged level
    assert res[14].bos is None                                          # no duplicate after restoration
    assert len(sp.bos_engine.history) == 1