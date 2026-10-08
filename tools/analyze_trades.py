"""Slice the Strategy 1 backtest trades. Usage: .\\.venv\\Scripts\\python.exe tools\\analyze_trades.py"""
import csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = []
with open(ROOT / "data" / "strategy1_backtest_trades.csv", newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["outcome"] not in ("TP", "SL"):
            continue
        t = datetime.fromisoformat(r["entry_time_utc"])
        x = datetime.fromisoformat(r["exit_time_utc"])
        rows.append({
            "t": t,
            "d": r["direction"],
            "dist": abs(float(r["entry"]) - float(r["sl"])),
            "r": float(r["r_multiple"]),
            "mins": (x - t).total_seconds() / 60,
        })


def report(title, keyfn):
    groups = {}
    for row in rows:
        groups.setdefault(keyfn(row), []).append(row["r"])
    print(f"\n--- {title} ---")
    print(f"{'group':<22}{'trades':>7}{'win%':>8}{'totalR':>9}{'avgR':>8}")
    for k in sorted(groups):
        v = groups[k]
        n = len(v)
        w = sum(1 for x in v if x > 0)
        print(f"{str(k):<22}{n:>7}{w / n * 100:>7.1f}%{sum(v):>9.1f}{sum(v) / n:>8.3f}")


n = len(rows)
dists = sorted(r["dist"] for r in rows)
q1, q2, q3 = dists[n // 4], dists[n // 2], dists[3 * n // 4]
print(f"trades={n}  stop distance (price units): min={dists[0]:.2f} q1={q1:.2f} median={q2:.2f} q3={q3:.2f} max={dists[-1]:.2f}")


def stop_bucket(row):
    d = row["dist"]
    if d <= q1:
        return f"1: smallest 25% (<={q1:.2f})"
    if d <= q2:
        return f"2: 25-50% (<={q2:.2f})"
    if d <= q3:
        return f"3: 50-75% (<={q3:.2f})"
    return f"4: largest 25% (>{q3:.2f})"


def hold_bucket(row):
    m = row["mins"]
    if m <= 5:
        return "1: first candle"
    if m <= 30:
        return "2: up to 30 min"
    if m <= 240:
        return "3: up to 4 hours"
    return "4: longer"


report("By year", lambda r: r["t"].year)
report("By month", lambda r: r["t"].strftime("%Y-%m"))
report("By direction", lambda r: r["d"])
report("By stop size", stop_bucket)
report("By time to resolve", hold_bucket)
report("By hour (UTC)", lambda r: f"{r['t'].hour:02d}")
report("By weekday (0=Mon)", lambda r: r["t"].weekday())