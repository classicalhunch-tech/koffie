"""Shared candle-series helpers for the close-based CHOCH/BOS tests."""
from datetime import datetime, timedelta

from koffie.strategy.engines.structure_processor import StructureProcessor
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.structure import StructureState

T0 = datetime(2026, 1, 5, 0, 0)
TF = Timeframe.M5


def candle(i, high, low, open_=None, close=None):
    """Candle i on M5. Defaults keep open/close strictly inside the range."""
    if open_ is None:
        open_ = low + (high - low) * 0.3
    if close is None:
        close = low + (high - low) * 0.6
    return Candle(TF, T0 + i * TF.duration, open_, high, low, close)


def mirror(c, i):
    """Price-mirror a candle around 200 so a bullish series becomes bearish."""
    return candle(i, 200 - c.low, 200 - c.high, 200 - c.open, 200 - c.close)


# (high, low) for candles 0..8. After candle 8 the structure is BULLISH:
#   highs 110 -> 115 (HH), lows 90 -> 94 (HL); protected low = 94, active high = 115.
BULLISH_PREFIX = [
    (102, 98), (110, 100), (105, 96), (101, 90), (108, 95),
    (115, 105), (109, 100), (104, 94), (106, 97),
]


def bullish_prefix():
    return [candle(i, h, l) for i, (h, l) in enumerate(BULLISH_PREFIX)]


def bearish_prefix():
    return [mirror(c, i) for i, c in enumerate(bullish_prefix())]  # protected high = 106, active low = 85


def run(candles):
    """Feed candles to a fresh StructureProcessor; return (processor, results)."""
    sp = StructureProcessor(TF)
    results = [sp.process_candle(c, c.close_time) for c in candles]
    return sp, results


def established(prefix):
    sp, results = run(prefix)
    return sp, results