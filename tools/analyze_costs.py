"""What do spread/slippage costs and a minimum stop size do to the Strategy 1 results?
Usage: .\\.venv\\Scripts\\python.exe tools\\analyze_costs.py [path-to-trades.csv]

Cost model (an assumption, edit SPREADS to your broker's real spread, in price units):
a trade pays one spread, so its cost in R is spread / stop distance.
A small stop makes the same spread a much bigger share of the risk.
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "strategy1_backtest_trades.csv"
SPREADS = [0.0, 0.2, 0.3, 0.5]
MIN_STOPS = [0, 1, 2, 3, 4, 5, 6, 8]

rows = []
with open(PATH, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["outcome"] not in ("TP", "SL"):
            continue
        rows.append({
            "year": datetime.fromisoformat(r["entry_time_utc"]).year,
            "dist": max(abs(float(r["entry"]) - float(r["sl"])), 1e-9),
            "r": float(r["r_multiple"]),
        })


def net_total(sub, spread):
    return sum(x["r"] - spread / x["dist"] for x in sub)


def line(sub, spread):
    n = len(sub)
    if n == 0:
        return "no trades"
    wins = sum(1 for x in sub if x["r"] > 0)
    total = net_total(sub, spread)
    return f"{n:>5} trades  win {wins / n * 100:5.1f}%  net {total:8.1f}R  avg {total / n:7.3f}R"


print(f"file: {PATH.name}   closed trades: {len(rows)}")
print("\n--- All trades, by assumed spread (price units) ---")
for sp in SPREADS:
    print(f"spread {sp:<4}: {line(rows, sp)}")

print("\n--- Only trades with a stop of at least X (known at entry), net of cost ---")
print(f"{'min stop':<9}{'trades':>7}" + "".join(f"   net R @ {sp:<4}" for sp in SPREADS))
for m in MIN_STOPS:
    sub = [x for x in rows if x["dist"] >= m]
    print(f"{m:<9}{len(sub):>7}" + "".join(f"{net_total(sub, sp):>14.1f}" for sp in SPREADS))

SP = 0.3
print(f"\n--- Same filters at spread {SP}, split by period (look for filters that work in BOTH) ---")
print(f"{'min stop':<9}  {'2024-2025':<46}{'2026':<46}")
for m in MIN_STOPS:
    a = [x for x in rows if x["dist"] >= m and x["year"] <= 2025]
    b = [x for x in rows if x["dist"] >= m and x["year"] >= 2026]
    print(f"{m:<9}  {line(a, SP):<46}{line(b, SP):<46}")