"""Koffie Strategy 1 historical backtest runner.

Replays the exported MT5 XAUUSD candles (H1 -> M15 -> M5, NO other timeframe)
through the Strategy1 coordinator in causal order, then resolves every trade
on M5 data.

Trade resolution (no intrabar path exists in candle data, so it is explicit):
- Entry happens at the close of the confirmation candle; the FIRST M5 candle
  with open_time >= entry_time is the first candle that can close the trade.
- LONG: stop if candle.low <= SL, target if candle.high >= TP.
- SHORT: stop if candle.high >= SL, target if candle.low <= TP.
- If ONE candle touches BOTH, the STOP is counted first (pessimistic).
- Trades still open at the end of the data are reported as OPEN and excluded
  from the statistics.

All results are in R multiples (TP is exactly 2R by the locked SL/TP rule).
There is no money, pip value, spread or position sizing in Strategy 1.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Make the repository importable no matter where this script is run from.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koffie.strategy.models.candle import Candle, Timeframe  # noqa: E402
from koffie.strategy.models.entry import EntryDirection  # noqa: E402


# --------------------------------------------------------------------------
# Loading CSV candles
# --------------------------------------------------------------------------

_TIME_COLUMNS = ("time", "open_time", "open time", "datetime", "date", "gmt time", "timestamp")
_OHLC_COLUMNS = ("open", "high", "low", "close")
_VOLUME_COLUMNS = ("volume", "vol", "tick_volume", "ticks", "tick vol")

# Same-instant feeding order used by the coordinator (H1, then M15, then M5).
_FEED_RANK = {Timeframe.H1: 0, Timeframe.M15: 1, Timeframe.M5: 2}


def parse_time(raw: str) -> datetime:
    """Parse a CSV time cell into a timezone-aware UTC datetime."""
    s = str(raw).strip()
    if s.lower().endswith(" utc"):
        s = s[:-4].strip()
    dt = None
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        pass
    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f",
                    "%Y.%m.%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
    if dt is None:
        try:
            dt = datetime.fromtimestamp(float(s), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
    if dt is None:
        raise ValueError(f"unparseable time value: {raw!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _column_index(header, candidates, what, path):
    lowered = [h.strip().lower() for h in header]
    for name in candidates:
        if name in lowered:
            return lowered.index(name)
    for i, cell in enumerate(lowered):
        if any(name in cell for name in candidates):
            return i
    raise SystemExit(f"{path}: cannot find a {what} column in header {list(header)}")


def load_candles(path: str, timeframe: Timeframe):
    """Load one exported CSV into Candle objects, oldest first."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row and any(cell.strip() for cell in row)]
    if len(rows) < 3:
        raise SystemExit(f"{path}: expected a header row plus candle rows, got {len(rows)} row(s)")
    header = rows[0]
    i_time = _column_index(header, _TIME_COLUMNS, "time", path)
    i_ohlc = {name: _column_index(header, (name,), name, path) for name in _OHLC_COLUMNS}
    try:
        i_vol = _column_index(header, _VOLUME_COLUMNS, "volume", path)
    except SystemExit:
        i_vol = None

    candles = []
    for row in rows[1:]:
        try:
            t = parse_time(row[i_time])
            o = float(row[i_ohlc["open"]])
            h = float(row[i_ohlc["high"]])
            l = float(row[i_ohlc["low"]])
            c = float(row[i_ohlc["close"]])
            v = float(row[i_vol]) if i_vol is not None and row[i_vol].strip() else 0.0
        except (ValueError, IndexError) as exc:
            raise SystemExit(f"{path}: bad row {row!r}: {exc}")
        candles.append(Candle(timeframe=timeframe, open_time=t, open=o, high=h, low=l, close=c, volume=v))

    for previous, current in zip(candles, candles[1:]):
        if current.open_time < previous.close_time:
            raise SystemExit(
                f"{path}: {timeframe.value} candles overlap or repeat near {current.open_time}"
            )
    return candles


# --------------------------------------------------------------------------
# Strategy 1 coordinator discovery (module name differs between checkouts)
# --------------------------------------------------------------------------

def load_strategy1_class():
    candidates = (
        "koffie.strategy.strategy1",
        "koffie.strategy.strategy_1",
        "koffie.strategy.coordinator",
        "koffie.strategy.strategy1_coordinator",
    )
    errors = []
    for name in candidates:
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            continue
        strategy1 = getattr(module, "Strategy1", None)
        if strategy1 is not None:
            return strategy1
        errors.append(f"{name}: module exists but has no Strategy1 class")
    raise SystemExit(
        "Could not import the Strategy1 coordinator. Tried:\n  " + "\n  ".join(errors)
    )


# --------------------------------------------------------------------------
# Trade resolution on M5 candles
# --------------------------------------------------------------------------

def _is_long(trade) -> bool:
    direction = trade.direction
    if direction is EntryDirection.LONG:
        return True
    if direction is EntryDirection.SHORT:
        return False
    return str(direction).upper().endswith("LONG")


def resolve_trades(trades, m5_candles):
    """Resolve every trade on the M5 series; return result records in trade order."""
    records = []
    for trade in trades:
        entry_time = trade.known_at
        entry = float(trade.entry_price)
        sl = float(trade.stop_loss)
        tp = float(trade.take_profit)
        long = _is_long(trade)
        outcome, exit_time, r_multiple = "OPEN", None, None
        for candle in m5_candles:
            if candle.open_time < entry_time:
                continue
            if long:
                hit_sl = candle.low <= sl
                hit_tp = candle.high >= tp
            else:
                hit_sl = candle.high >= sl
                hit_tp = candle.low <= tp
            if hit_sl:                       # stop first when both are touched in one candle
                outcome, exit_time, r_multiple = "SL", candle.close_time, -1.0
                break
            if hit_tp:
                outcome, exit_time, r_multiple = "TP", candle.close_time, 2.0
                break
        records.append({
            "entry_time": entry_time,
            "direction": "LONG" if long else "SHORT",
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit_time": exit_time,
            "outcome": outcome,
            "r": r_multiple,
        })
    return records


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(records):
    closed = [r for r in records if r["outcome"] in ("TP", "SL")]
    wins = [r for r in closed if r["outcome"] == "TP"]
    losses = [r for r in closed if r["outcome"] == "SL"]
    opens = [r for r in records if r["outcome"] == "OPEN"]
    n = len(closed)
    total_r = sum(r["r"] for r in closed)
    win_rate = (len(wins) / n * 100.0) if n else 0.0
    avg_r = (total_r / n) if n else 0.0
    gross_win = sum(r["r"] for r in wins)
    gross_loss = -sum(r["r"] for r in losses)
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for r in closed:
        equity += r["r"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    longs = [r for r in closed if r["direction"] == "LONG"]
    shorts = [r for r in closed if r["direction"] == "SHORT"]
    return {
        "closed": n,
        "wins": len(wins),
        "losses": len(losses),
        "opens": len(opens),
        "win_rate": win_rate,
        "total_r": total_r,
        "avg_r": avg_r,
        "profit_factor": profit_factor,
        "max_drawdown_r": max_dd,
        "longs": len(longs),
        "long_wins": sum(1 for r in longs if r["outcome"] == "TP"),
        "shorts": len(shorts),
        "short_wins": sum(1 for r in shorts if r["outcome"] == "TP"),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Koffie Strategy 1 (H1 -> M15 -> M5) on exported MT5 XAUUSD CSVs."
    )
    parser.add_argument("--h1", default=str(ROOT / "data" / "XAUUSD_H1.csv"))
    parser.add_argument("--m15", default=str(ROOT / "data" / "XAUUSD_M15.csv"))
    parser.add_argument("--m5", default=str(ROOT / "data" / "XAUUSD_M5.csv"))
    parser.add_argument("--out", default=str(ROOT / "data" / "strategy1_backtest_trades.csv"))
    args = parser.parse_args()

    Strategy1 = load_strategy1_class()

    h1 = load_candles(args.h1, Timeframe.H1)
    m15 = load_candles(args.m15, Timeframe.M15)
    m5 = load_candles(args.m5, Timeframe.M5)

    merged = sorted(h1 + m15 + m5, key=lambda c: (c.close_time, _FEED_RANK[c.timeframe]))

    strategy = Strategy1()
    skipped = Counter()
    for candle in merged:
        result = strategy.process_candle(candle, candle.close_time)
        for item in getattr(result, "skipped", ()) or ():
            skipped[getattr(item.reason, "value", str(item.reason))] += 1

    trades = list(strategy.trades)
    records = resolve_trades(trades, m5)
    stats = compute_metrics(records)

    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["entry_time_utc", "direction", "entry", "sl", "tp",
                         "exit_time_utc", "outcome", "r_multiple"])
        for r in records:
            writer.writerow([
                r["entry_time"].isoformat(), r["direction"],
                f"{r['entry']:.2f}", f"{r['sl']:.2f}", f"{r['tp']:.2f}",
                r["exit_time"].isoformat() if r["exit_time"] else "",
                r["outcome"], "" if r["r"] is None else f"{r['r']:.1f}",
            ])

    def span(candles):
        return f"{candles[0].open_time.isoformat()} -> {candles[-1].close_time.isoformat()}"

    print("================ Strategy 1 backtest ================")
    print(f"Period            : {merged[0].open_time.isoformat()} -> {merged[-1].close_time.isoformat()} (UTC)")
    print(f"Candles processed : H1={len(h1)}  M15={len(m15)}  M5={len(m5)}  total={len(merged)}")
    print(f"  H1 span         : {span(h1)}")
    print(f"  M15 span        : {span(m15)}")
    print(f"  M5 span         : {span(m5)}")
    print(f"M15 zones created : {len(getattr(strategy, 'zones', ()))}")
    print(f"Skipped setups    : {dict(skipped) if skipped else 'none'}")
    print("----------------------- Results ---------------------")
    print(f"Closed trades     : {stats['closed']}  (wins {stats['wins']} / losses {stats['losses']})")
    print(f"Win rate          : {stats['win_rate']:.1f}%")
    print(f"Total R           : {stats['total_r']:+.2f} R")
    print(f"Average R         : {stats['avg_r']:+.3f} R")
    pf = stats["profit_factor"]
    print(f"Profit factor     : {'inf' if pf == float('inf') else f'{pf:.2f}'}")
    print(f"Max drawdown      : {stats['max_drawdown_r']:.2f} R")
    print(f"LONG  trades      : {stats['longs']}  (wins {stats['long_wins']})")
    print(f"SHORT trades      : {stats['shorts']}  (wins {stats['short_wins']})")
    print(f"Open at data end  : {stats['opens']} (excluded from stats)")
    print(f"Trade list        : {args.out}")
    print("=====================================================")


if __name__ == "__main__":
    main()