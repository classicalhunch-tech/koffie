from datetime import datetime, timedelta

import pytest

from koffie.strategy.models.candle import (
    BRR_DECISIVE_THRESHOLD, Candle, CandleClass, Timeframe,
)

T = datetime(2026, 1, 5, 0, 0)
TF = Timeframe.M5
NOW = T + TF.duration


def mk(o, h, l, c):
    return Candle(TF, T, o, h, l, c)


def test_threshold_locked():
    assert BRR_DECISIVE_THRESHOLD == 0.50


def test_brr_above_threshold_is_decisive():
    c = mk(100, 110, 100, 108)            # BRR 0.8
    assert c.brr == pytest.approx(0.8)
    assert c.classify(NOW).is_decisive


def test_brr_below_threshold_is_neutral():
    c = mk(100, 110, 100, 104)            # BRR 0.4
    assert c.classify(NOW) is CandleClass.NEUTRAL
    assert not c.classify(NOW).is_decisive


def test_brr_exactly_threshold_is_neutral():
    c = mk(100, 110, 100, 105)            # 5/10 == 0.50 exactly`r`n    assert c.brr == 0.50
    assert c.classify(NOW) is CandleClass.NEUTRAL


def test_bullish_decisive():
    assert mk(100, 110, 100, 109).classify(NOW) is CandleClass.BULLISH_DECISIVE


def test_bearish_decisive():
    assert mk(109, 110, 100, 100).classify(NOW) is CandleClass.BEARISH_DECISIVE


def test_close_equals_open_is_neutral():
    c = mk(105, 110, 100, 105)
    assert c.brr == 0.0
    assert c.classify(NOW) is CandleClass.NEUTRAL


def test_doji_with_wide_body_but_not_close_equals_open_is_neutral():
    # close != open, so is_doji is False, yet Strategy 1 treats BRR <= 0.50 as NEUTRAL.
    c = mk(100, 110, 100, 105)
    assert not c.is_doji
    assert c.classify(NOW) is CandleClass.NEUTRAL


def test_zero_range_is_anomaly_and_never_decisive():
    c = mk(100, 100, 100, 100)
    assert c.brr is None
    cls = c.classify(NOW)
    assert cls is CandleClass.ZERO_RANGE_ANOMALY
    assert not cls.is_decisive


def test_unclosed_candle_is_never_classified_decisive():
    c = mk(100, 110, 100, 109)            # would be BULLISH_DECISIVE once closed
    with pytest.raises(ValueError):
        c.classify(NOW - timedelta(seconds=1))
    with pytest.raises(ValueError):
        c.classify(T)
    assert c.classify(NOW) is CandleClass.BULLISH_DECISIVE   # exactly at close


def test_now_must_be_datetime():
    with pytest.raises(TypeError):
        mk(100, 110, 100, 109).classify("now")


def test_classification_uses_only_brr_not_size_or_wicks():
    small = mk(100.00, 100.10, 100.00, 100.09)    # tiny candle, BRR 0.9
    huge = mk(100, 500, 0, 250)                   # huge candle, BRR 0.5 -> neutral
    assert small.classify(NOW) is CandleClass.BULLISH_DECISIVE
    assert huge.classify(NOW) is CandleClass.NEUTRAL


# ---- configurable threshold keeps the strict ">" comparison -----------------
def test_default_threshold_is_the_locked_value():
    c = mk(100, 110, 100, 105)            # BRR exactly 0.50
    assert c.classify(NOW) is c.classify(NOW, BRR_DECISIVE_THRESHOLD) is CandleClass.NEUTRAL


def test_custom_threshold_changes_the_cut_but_stays_strict():
    c = mk(100, 110, 100, 105)            # BRR exactly 0.50
    assert c.classify(NOW, threshold=0.5) is CandleClass.NEUTRAL            # == is not decisive
    assert c.classify(NOW, threshold=0.49) is CandleClass.BULLISH_DECISIVE
    b = mk(105, 110, 100, 100)            # BRR 0.50, bearish
    assert b.classify(NOW, threshold=0.5) is CandleClass.NEUTRAL
    assert b.classify(NOW, threshold=0.49) is CandleClass.BEARISH_DECISIVE


def test_zero_range_is_anomaly_for_any_threshold():
    c = mk(100, 100, 100, 100)
    for t in (0.0, 0.3, 0.7, 1.0):
        assert c.classify(NOW, threshold=t) is CandleClass.ZERO_RANGE_ANOMALY


def test_unclosed_candle_rejected_for_any_threshold():
    c = mk(100, 110, 100, 109)
    with pytest.raises(ValueError):
        c.classify(NOW - timedelta(seconds=1), threshold=0.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_threshold_value_rejected(bad):
    with pytest.raises(ValueError):
        mk(100, 110, 100, 109).classify(NOW, threshold=bad)


@pytest.mark.parametrize("bad", ["0.7", None, True])
def test_invalid_threshold_type_rejected(bad):
    with pytest.raises(TypeError):
        mk(100, 110, 100, 109).classify(NOW, threshold=bad)
