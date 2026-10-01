"""CHOCH is close-based (locked Strategy 1 rule)."""
import pytest

from koffie.strategy.models.candle import Timeframe
from koffie.strategy.models.choch import CHOCH, CHOCHDirection
from koffie.strategy.models.structure import StructureState
from koffie.strategy.models.swing import SwingType
from tests._structure_fixtures import (
    bearish_prefix, bullish_prefix, candle, mirror, run,
)


def test_prefixes_establish_expected_structure():
    sp, _ = run(bullish_prefix())
    assert sp.structure.state is StructureState.BULLISH
    assert sp.structure.protected_swing.price == 94
    sp, _ = run(bearish_prefix())
    assert sp.structure.state is StructureState.BEARISH
    assert sp.structure.protected_swing.price == 106


# ---- bullish -> REVALUATING ------------------------------------------------
def test_bullish_close_below_protected_low_is_bearish_choch_and_revaluating():
    c9 = candle(9, 100, 91, open_=96, close=93)           # close 93 < 94
    sp, res = run(bullish_prefix() + [c9])
    r = res[-1]
    assert r.choch is not None
    assert r.choch.direction is CHOCHDirection.BEARISH
    assert r.choch.protected_swing.price == 94
    assert r.choch.break_price == 93                       # the CLOSE
    assert r.choch.confirmed_at == c9.close_time
    assert r.revaluating is not None
    assert sp.structure.state is StructureState.REVALUATING
    assert r.bos is None


def test_bullish_wick_below_protected_low_but_close_above_is_not_choch():
    c9 = candle(9, 107, 92, open_=100, close=95)           # low 92 < 94, close 95 > 94
    sp, res = run(bullish_prefix() + [c9])
    assert res[-1].choch is None
    assert res[-1].revaluating is None
    assert sp.structure.state is StructureState.BULLISH
    assert sp.choch_engine.history == ()


def test_bullish_close_exactly_on_protected_low_is_not_choch():
    c9 = candle(9, 107, 92, open_=100, close=94)
    sp, res = run(bullish_prefix() + [c9])
    assert res[-1].choch is None
    assert sp.structure.state is StructureState.BULLISH


# ---- bearish -> REVALUATING ------------------------------------------------
def test_bearish_close_above_protected_high_is_bullish_choch_and_revaluating():
    c9 = mirror(candle(9, 100, 91, open_=96, close=93), 9)  # close 107 > 106
    sp, res = run(bearish_prefix() + [c9])
    r = res[-1]
    assert r.choch is not None
    assert r.choch.direction is CHOCHDirection.BULLISH
    assert r.choch.protected_swing.price == 106
    assert r.choch.break_price == 107
    assert sp.structure.state is StructureState.REVALUATING
    assert r.bos is None


def test_bearish_wick_above_protected_high_but_close_below_is_not_choch():
    c9 = mirror(candle(9, 107, 92, open_=100, close=95), 9)  # high 108 > 106, close 105 < 106
    sp, res = run(bearish_prefix() + [c9])
    assert res[-1].choch is None
    assert sp.structure.state is StructureState.BEARISH


# ---- no manufactured CHOCH outside a trend --------------------------------
def test_no_choch_when_revaluating():
    c9 = candle(9, 100, 91, open_=96, close=93)
    c10 = candle(10, 99, 85, open_=93, close=86)           # still far below 94
    sp, res = run(bullish_prefix() + [c9, c10])
    assert res[9].choch is not None
    assert res[10].choch is None                           # REVALUATING: no protected swing
    assert len(sp.choch_engine.history) == 1


def test_no_choch_when_undetermined():
    sp, res = run(bullish_prefix()[:5] + [candle(5, 90, 50, open_=80, close=52)])
    assert sp.structure.state is StructureState.UNDETERMINED
    assert all(r.choch is None for r in res)


# ---- CHOCH candle does not automatically become a swing ---------------------
def test_choch_candle_does_not_become_a_swing():
    c9 = candle(9, 100, 91, open_=96, close=93)            # CHOCH candle
    c10 = candle(10, 97, 88, open_=92, close=90)           # lower low: c9 is NOT a swing low
    sp, res = run(bullish_prefix() + [c9, c10])
    assert res[9].choch is not None
    assert all(sw.candle_time != c9.open_time for sw in res[9].swings)   # CHOCH candle itself is no swing
    choch_open = c9.open_time
    assert all(s.candle_time != choch_open for s in sp.swing_engine.history)
    assert sp.structure.state is StructureState.REVALUATING


def test_choch_candle_becomes_swing_only_if_three_candle_rule_met_later():
    c9 = candle(9, 100, 91, open_=96, close=93)            # CHOCH candle, low 91
    c10 = candle(10, 99, 93, open_=94, close=98)           # higher low, so c9 low 91 < 97 and < 93
    sp, res = run(bullish_prefix() + [c9, c10])
    assert all(sw.candle_time != c9.open_time for sw in res[9].swings)   # not at the CHOCH candle itself
    lows = [s for s in res[10].swings if s.swing_type is SwingType.LOW]
    assert len(lows) == 1 and lows[0].candle_time == c9.open_time   # one candle later, by the normal rule


# ---- model ------------------------------------------------------------------
def test_choch_model_requires_break_price_beyond_level():
    sp, res = run(bullish_prefix())
    protected = sp.structure.protected_swing
    t = bullish_prefix()[-1].open_time
    with pytest.raises(ValueError):
        CHOCH(Timeframe.M5, CHOCHDirection.BEARISH, protected, t, protected.price)
    ok = CHOCH(Timeframe.M5, CHOCHDirection.BEARISH, protected, t, protected.price - 0.5)
    assert ok.break_price == protected.price - 0.5