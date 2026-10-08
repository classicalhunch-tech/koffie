"""Results by stop size measured in ATR (known at entry), net of an assumed spread.
Usage: .\\.venv\\Scripts\\python.exe tools\\analyze_stop_atr.py [path-to-trades.csv]
ATR = Wilder ATR(14) of the 5M candles, as of the close of the confirmation candle (entry time).
Cost = spread / stop distance per trade (edit SPREADS to your broker's real spread)."""
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import Backtest_Strategy1 as bt  # noqa: E402
from koffie.strategy.models.candle import Timeframe  # noqa: E402

TRADES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "strategy1_backtest_trades.csv"
M5 = ROOT / "data" / "XAUUSD_M5.csv"
PERIOD = 14
SPREADS = [0.0, 0.3, 0.5]
SP = 0.3
THRESHOLDS = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

atr_at = {}
atr = None
seed = []
prev_close = None
for c in bt.load_candles(str(M5), Timeframe.M5):
    tr = c.high - c.low
    if prev_close is not None:
        tr = max(tr, abs(c.high - prev_close), abs(c.low - prev_close))
    prev_close = c.close
    if atr is None:
        seed.append(tr)
        if len(seed) >= PERIOD:
            atr = sum(seed) / PERIOD
    else:
        atr = (atr * (PERIOD - 1) + tr) / PERIOD
    if atr is not None:
        atr_at[c.close_time] = atr

rows, missing = [], 0
with open(TRADES, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["outcome"] not in ("TP", "SL"):
            continue
        t = datetime.fromisoformat(r["entry_time_utc"])
        a = atr_at.get(t)
        if a is None:
            missing += 1
            continue
        dist = max(abs(float(r["entry"]) - float(r["sl"])), 1e-9)
        rows.append({"year": t.year, "dist": dist, "ratio": dist / a, "r": float(r["r_multiple"])})


def net(sub, sp):
    return sum(x["r"] - sp / x["dist"] for x in sub)


def line(sub, sp):
    n = len(sub)
    if n == 0:
        return "no trades"
    wins = sum(1 for x in sub if x["r"] > 0)
    return f"{n:>5} trades  win {wins / n * 100:5.1f}%  net {net(sub, sp):8.1f}R  avg {net(sub, sp) / n:7.3f}R"


ratios = sorted(x["ratio"] for x in rows)
n = len(ratios)
print(f"file: {TRADES.name}   trades used: {n}   without an ATR yet: {missing}")
print(f"stop / ATR:  min {ratios[0]:.2f}  q1 {ratios[n // 4]:.2f}  median {ratios[n // 2]:.2f}  q3 {ratios[3 * n // 4]:.2f}  max {ratios[-1]:.2f}")

print("\n--- Only trades with stop >= X ATR, net of cost ---")
print(f"{'min ATR':<9}{'trades':>7}" + "".join(f"   net R @ {sp:<4}" for sp in SPREADS))
for m in THRESHOLDS:
    sub = [x for x in rows if x["ratio"] >= m]
    print(f"{m:<9}{len(sub):>7}" + "".join(f"{net(sub, sp):>14.1f}" for sp in SPREADS))

print(f"\n--- Same filters at spread {SP}, by period ---")
print(f"{'min ATR':<9}  {'2024-2025':<46}{'2026':<46}")
for m in THRESHOLDS:
    a = [x for x in rows if x["ratio"] >= m and x["year"] <= 2025]
    b = [x for x in rows if x["ratio"] >= m and x["year"] >= 2026]
    print(f"{m:<9}  {line(a, SP):<46}{line(b, SP):<46}")