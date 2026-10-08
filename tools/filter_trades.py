"""Keep only the trades whose stop distance is at least MIN_STOP and show their statistics.
Usage: .\\.venv\\Scripts\\python.exe tools\\filter_trades.py 6 [path-to-trades.csv]
This is a re-count of trades that already exist (hindsight), NOT a forecast."""
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_STOP = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "strategy1_backtest_trades.csv"
OUT = ROOT / "data" / f"filtered_minstop_{MIN_STOP:g}.csv"

kept, all_rows = [], []
with open(PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    header = reader.fieldnames
    for r in reader:
        if r["outcome"] not in ("TP", "SL"):
            continue
        r["_dist"] = abs(float(r["entry"]) - float(r["sl"]))
        r["_year"] = datetime.fromisoformat(r["entry_time_utc"]).year
        all_rows.append(r)
        if r["_dist"] >= MIN_STOP:
            kept.append(r)


def show(label, rows):
    n = len(rows)
    if n == 0:
        print(f"{label:<28} no trades")
        return
    wins = sum(1 for r in rows if r["outcome"] == "TP")
    p = wins / n
    se = math.sqrt(p * (1 - p) / n) * 100
    total = sum(float(r["r_multiple"]) for r in rows)
    print(f"{label:<28}{n:>6} trades  wins {wins:>4}  win rate {p * 100:5.1f}% (+/- {se:.1f})  total {total:7.1f}R  avg {total / n:6.3f}R")


print(f"file: {PATH.name}   minimum stop: {MIN_STOP:g} price units   (break-even win rate is 33.3%)")
show("all trades", all_rows)
show(f"stop >= {MIN_STOP:g}", kept)
show("  2024-2025", [r for r in kept if r["_year"] <= 2025])
show("  2026", [r for r in kept if r["_year"] >= 2026])
with open(OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in kept:
        writer.writerow([r[h] for h in header])
print(f"kept trades written to: {OUT}")