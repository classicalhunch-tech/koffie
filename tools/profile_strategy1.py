"""Profile Strategy 1 on the first N M5 candles (default 3000).

Usage:  .\\.venv\\Scripts\\python.exe tools\\profile_strategy1.py 3000
Prints how long every 500 candles take (does it keep slowing down?) and the
functions where the time is spent.
"""
import cProfile
import pstats
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import Backtest_Strategy1 as bt  # noqa: E402
from koffie.strategy.models.candle import Timeframe  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000

h1 = bt.load_candles(str(ROOT / "data" / "XAUUSD_H1.csv"), Timeframe.H1)
m15 = bt.load_candles(str(ROOT / "data" / "XAUUSD_M15.csv"), Timeframe.M15)
m5 = bt.load_candles(str(ROOT / "data" / "XAUUSD_M5.csv"), Timeframe.M5)

m5 = m5[:N]
cutoff = m5[-1].close_time
h1 = [c for c in h1 if c.close_time <= cutoff]
m15 = [c for c in m15 if c.close_time <= cutoff]

merged = sorted(h1 + m15 + m5, key=lambda c: (c.close_time, bt._FEED_RANK[c.timeframe]))
print(f"[profile] H1={len(h1)} M15={len(m15)} M5={len(m5)} total={len(merged)}", flush=True)

strategy = bt.load_strategy1_class()()


def run():
    t0 = time.time()
    last = t0
    for i, candle in enumerate(merged, 1):
        strategy.process_candle(candle, candle.close_time)
        if i % 500 == 0:
            now = time.time()
            print(f"[profile] {i:>6} candles | last 500 took {now - last:6.2f}s | total {now - t0:7.1f}s", flush=True)
            last = now
    print(f"[profile] done in {time.time() - t0:.1f}s, trades={len(list(strategy.trades))}", flush=True)


profiler = cProfile.Profile()
profiler.enable()
run()
profiler.disable()

print("\n===== TOP 25 BY TIME SPENT INSIDE EACH FUNCTION =====")
pstats.Stats(profiler).strip_dirs().sort_stats("tottime").print_stats(25)
print("\n===== TOP 25 BY CUMULATIVE TIME =====")
pstats.Stats(profiler).strip_dirs().sort_stats("cumulative").print_stats(25)