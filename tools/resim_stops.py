"""Re-run your EXISTING trades with a wider stop and see what really happens to the win rate.
Usage: .\\.venv\\Scripts\\python.exe tools\\resim_stops.py [path-to-trades.csv]

For each trade (same entry, same direction) the stop is moved BUFFER x ATR further away from
the entry than the original stop, and the target is kept at exactly 2R of the NEW risk.
Resolved candle by candle on M5 data; if one candle touches both, the stop counts first.
Buffer 0 must reproduce your original numbers (check it). Cost = spread / new risk per trade."""
import csv
import sys
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import Backtest_Strategy1 as bt  # noqa: E402
from koffie.strategy.models.candle import Timeframe  # noqa: E402

TRADES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "strategy1_backtest_trades.csv"
BUFFERS = [0.0, 0.25, 0.5, 1.0, 1.5]
RR = 2.0
SP = 0.3
PERIOD = 14

candles = bt.load_candles(str(ROOT / "data" / "XAUUSD_M5.csv"), Timeframe.M5)
open_times = [c.open_time for c in candles]

atr_at, atr, seed, prev_close = {}, None, [], None
for c in candles:
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

trades = []
with open(TRADES, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["outcome"] not in ("TP", "SL"):
            continue
        t = datetime.fromisoformat(r["entry_time_utc"])
        a = atr_at.get(t)
        if a is None:
            continue
        trades.append({"t": t, "long": r["direction"] == "LONG", "entry": float(r["entry"]),
                       "sl": float(r["sl"]), "atr": a, "start": bisect_left(open_times, t)})


def run(buffer):
    out = []
    for tr in trades:
        entry, sl = tr["entry"], tr["sl"]
        new_sl = sl - buffer * tr["atr"] if tr["long"] else sl + buffer * tr["atr"]
        new_sl = round(new_sl, 2)
        risk = abs(entry - new_sl)
        if risk <= 0:
            continue
        tp = round(entry + RR * risk if tr["long"] else entry - RR * risk, 2)
        result = None
        for c in candles[tr["start"]:]:
            if tr["long"]:
                hit_sl, hit_tp = c.low <= new_sl, c.high >= tp
            else:
                hit_sl, hit_tp = c.high >= new_sl, c.low <= tp
            if hit_sl:
                result = -1.0
                break
            if hit_tp:
                result = RR
                break
        if result is not None:
            out.append((tr["t"].year, risk, result))
    return out


def summary(rows):
    n = len(rows)
    if n == 0:
        return "no trades"
    wins = sum(1 for _, _, r in rows if r > 0)
    gross = sum(r for _, _, r in rows)
    net = sum(r - SP / risk for _, risk, r in rows)
    return f"{n:>5} trades  win {wins / n * 100:5.1f}%  gross {gross:7.1f}R  net@{SP} {net:8.1f}R"


print(f"file: {TRADES.name}   trades re-run: {len(trades)}   target = {RR:g}R of the new risk")
print(f"{'buffer (ATR)':<13}{'all trades':<62}")
results = {}
for b in BUFFERS:
    results[b] = run(b)
    print(f"{b:<13}{summary(results[b])}")
print("\n--- by period (net at the assumed spread) ---")
for b in BUFFERS:
    a = [x for x in results[b] if x[0] <= 2025]
    z = [x for x in results[b] if x[0] >= 2026]
    print(f"{b:<6} 2024-2025: {summary(a)}")
    print(f"{'':<6} 2026     : {summary(z)}")