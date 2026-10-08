"""Run Strategy 1 on the first N M5 candles and write a summary + every trade to a text file.
Usage: .\\.venv\\Scripts\\python.exe tools\\dump_trades.py 6000 baseline_trades.txt
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import Backtest_Strategy1 as bt  # noqa: E402
from koffie.strategy.models.candle import Timeframe  # noqa: E402

N = int(sys.argv[1])
OUT = sys.argv[2]

h1 = bt.load_candles(str(ROOT / "data" / "XAUUSD_H1.csv"), Timeframe.H1)
m15 = bt.load_candles(str(ROOT / "data" / "XAUUSD_M15.csv"), Timeframe.M15)
m5 = bt.load_candles(str(ROOT / "data" / "XAUUSD_M5.csv"), Timeframe.M5)[:N]
cutoff = m5[-1].close_time
h1 = [c for c in h1 if c.close_time <= cutoff]
m15 = [c for c in m15 if c.close_time <= cutoff]
merged = sorted(h1 + m15 + m5, key=lambda c: (c.close_time, bt._FEED_RANK[c.timeframe]))

strategy = bt.load_strategy1_class()()
t0 = time.time()
for i, candle in enumerate(merged, 1):
    strategy.process_candle(candle, candle.close_time)
    if i % 2000 == 0:
        print(f"[dump] {i}/{len(merged)} candles, {time.time() - t0:.1f}s", flush=True)

lines = [
    f"zones={len(strategy.zones)}",
    f"setups={len(strategy.setup_engine.setups)}",
    f"trades={len(strategy.trades)}",
]
for t in strategy.trades:
    direction = getattr(t.direction, "value", t.direction)
    lines.append(f"{t.known_at.isoformat()} {direction} entry={t.entry_price} sl={t.stop_loss} tp={t.take_profit}")
Path(OUT).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[dump] done in {time.time() - t0:.1f}s -> {OUT}")