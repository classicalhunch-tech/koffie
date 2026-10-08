"""Turn MetaTrader 5 candle data into the CSV files the Koffie backtest reads.
 
This module holds the PURE logic and does not import MetaTrader5. The thin script
`tools/export_mt5_xauusd.py` supplies the real MT5 calls through a `fetch` function, so the
logic here can be tested without a terminal. It contains no market data.
 
What the export does (and does not do)
--------------------------------------
- asks MT5 for the candles of one timeframe in windows of `window_days` days;
- each rate is (open_time_epoch, open, high, low, close); MT5 `time` is the candle OPEN time
  on the BROKER SERVER clock, and that clock is kept exactly as delivered (no timezone
  conversion), the same for H1, M15 and M5;
- drops any candle that is still forming: a candle is kept only if open + timeframe <= the
  server time read from the latest tick (`server_now`), so only completed candles remain;
- sorts by time and removes EXACT duplicate rows that come from overlapping windows; two
  different rows with the same time are an error, never silently resolved;
- writes `<out>/XAUUSD_<TF>.csv` (columns time,open,high,low,close) and a sidecar
  `<out>/XAUUSD_<TF>.meta.json` recording the symbol, timeframe, row count, first/last open
  time, time basis, MT5 server and requested range, which `koffie.backtest.verify` checks;
- never fills gaps and never invents a candle.
"""
from __future__ import annotations
 
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple
 
from koffie.strategy.models.candle import Timeframe
 
Rate = Tuple[int, float, float, float, float]            # (open_time_epoch, open, high, low, close)
Fetch = Callable[[str, Timeframe, datetime, datetime], Optional[Sequence[Rate]]]
 
TIMEFRAME_LABELS = {Timeframe.H1: "H1", Timeframe.M15: "M15", Timeframe.M5: "M5"}
TIME_BASIS = "MT5 broker server clock as delivered by MetaTrader 5; no timezone conversion"
 
 
class ExportError(RuntimeError):
    """The export cannot continue. The message says why."""
 
 
def epoch_to_text(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
 
 
def windows(start: datetime, end: datetime, window_days: int = 30) -> List[Tuple[datetime, datetime]]:
    """Consecutive (from, to) windows covering [start, end]."""
    if end <= start:
        raise ExportError("the end of the range must be after its start")
    step = timedelta(days=window_days)
    out, cursor = [], start
    while cursor < end:
        out.append((cursor, min(cursor + step, end)))
        cursor += step
    return out
 
 
def collect_rates(
    fetch: Fetch, symbol: str, timeframe: Timeframe, start: datetime, end: datetime, window_days: int = 30
) -> List[Rate]:
    """Ask for each window in turn. `None` from MT5 is an error; an empty window is fine (weekends)."""
    rates: List[Rate] = []
    for window_start, window_end in windows(start, end, window_days):
        got = fetch(symbol, timeframe, window_start, window_end)
        if got is None:
            raise ExportError(
                f"MetaTrader 5 returned an error for {symbol} {TIMEFRAME_LABELS[timeframe]} "
                f"{window_start:%Y-%m-%d} -> {window_end:%Y-%m-%d}"
            )
        rates.extend((int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in got)
    return rates
 
 
def finalize_rates(rates: Sequence[Rate], timeframe: Timeframe, server_now: int) -> Tuple[List[Rate], int]:
    """Sort, drop exact duplicates, drop unfinished candles. Returns (rows, dropped_unfinished)."""
    ordered = sorted(rates, key=lambda r: r[0])
    unique: List[Rate] = []
    for rate in ordered:
        if unique and unique[-1][0] == rate[0]:
            if unique[-1] != rate:
                raise ExportError(
                    f"two different candles share the open time {epoch_to_text(rate[0])}; "
                    "refusing to choose between them"
                )
            continue
        unique.append(rate)
    seconds = int(timeframe.duration.total_seconds())
    complete = [r for r in unique if r[0] + seconds <= server_now]
    return complete, len(unique) - len(complete)
 
 
def write_export(
    rows: Sequence[Rate],
    symbol: str,
    timeframe: Timeframe,
    out_dir: Path,
    mt5_server: str,
    requested_start: datetime,
    requested_end: datetime,
    dropped_unfinished: int = 0,
    exported_at: Optional[datetime] = None,
    file_prefix: str = "XAUUSD",
) -> Tuple[Path, Path]:
    """Write the CSV and its meta file; returns (csv_path, meta_path)."""
    if not rows:
        raise ExportError(
            f"MetaTrader 5 returned no completed {TIMEFRAME_LABELS[timeframe]} candles for {symbol} in the requested range"
        )
    label = TIMEFRAME_LABELS[timeframe]
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path = folder / f"{file_prefix}_{label}.csv"
    meta_path = csv_path.with_suffix(".meta.json")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "open", "high", "low", "close"])
        for epoch, o, h, l, c in rows:
            writer.writerow([epoch_to_text(epoch), repr(o), repr(h), repr(l), repr(c)])
    meta = {
        "symbol": symbol,
        "timeframe": label,
        "rows": len(rows),
        "first_open_time": epoch_to_text(rows[0][0]),
        "last_open_time": epoch_to_text(rows[-1][0]),
        "time_basis": TIME_BASIS,
        "mt5_server": mt5_server,
        "requested_start": requested_start.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_end": requested_end.strftime("%Y-%m-%d %H:%M:%S"),
        "unfinished_candles_dropped": dropped_unfinished,
        "exported_at_local": (exported_at or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return csv_path, meta_path
 
 
def export_timeframe(
    fetch: Fetch,
    symbol: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
    server_now: int,
    out_dir: Path,
    mt5_server: str,
    window_days: int = 30,
    exported_at: Optional[datetime] = None,
    file_prefix: str = "XAUUSD",
) -> Tuple[Path, Path, int, int]:
    """Fetch, clean and write one timeframe. Returns (csv, meta, rows, dropped_unfinished)."""
    rates = collect_rates(fetch, symbol, timeframe, start, end, window_days)
    rows, dropped = finalize_rates(rates, timeframe, server_now)
    csv_path, meta_path = write_export(
        rows, symbol, timeframe, out_dir, mt5_server, start, end, dropped, exported_at, file_prefix
    )
    return csv_path, meta_path, len(rows), dropped