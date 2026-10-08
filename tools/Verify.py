"""Verify a historical XAUUSD dataset BEFORE it is backtested.
 
It checks the three files (H1, M15, M5) and reports what they actually contain. It never
repairs, fills or changes data, and it contains no market data of its own.
 
Checks (any failure = errors, and the backtest refuses to run)
--------------------------------------------------------------
1. all three CSV files exist;
2. each file passes the strict loader in `koffie.backtest.data`: required columns, finite
   numbers, valid OHLC, open times on the timeframe grid, strictly chronological rows,
   NO duplicate timestamps within a timeframe;
3. XAUUSD: a CSV has no symbol column, so each file needs a small sidecar `<name>.meta.json`
   (written by `tools/export_mt5_xauusd.py`) naming the symbol. The symbol must contain
   "XAUUSD", must be the same for all three files, and must agree with the file's
   timeframe, row count, first and last open time;
4. all three files come from the same MT5 server.
Use `require_symbol_meta=False` only if you accept that nothing proves the data is XAUUSD.
 
Reported, not failed (information you need to judge the result)
---------------------------------------------------------------
- first and last candle of each timeframe and how many rows it has;
- warnings when the three series do not cover the same period;
- every gap longer than 3 days inside a series (normal weekends are shorter, so these are
  holidays or genuinely missing history).
"""
from __future__ import annotations
 
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple
 
from koffie.backtest.data import DataError, load_candles_csv
from koffie.strategy.models.candle import Candle, Timeframe
 
TIMEFRAME_LABELS = {Timeframe.H1: "H1", Timeframe.M15: "M15", Timeframe.M5: "M5"}
REQUIRED_SYMBOL_TEXT = "XAUUSD"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
 
 
def meta_path_for(csv_path: Path) -> Path:
    """data/XAUUSD_H1.csv -> data/XAUUSD_H1.meta.json"""
    return Path(csv_path).with_suffix(".meta.json")
 
 
@dataclass(frozen=True)
class Gap:
    after: datetime        # close time of the candle before the gap
    until: datetime        # open time of the candle after the gap
 
    @property
    def length(self) -> timedelta:
        return self.until - self.after
 
 
def find_gaps(candles: List[Candle], threshold: timedelta = timedelta(days=3)) -> Tuple[Gap, ...]:
    gaps = []
    for previous, current in zip(candles, candles[1:]):
        if current.open_time - previous.close_time > threshold:
            gaps.append(Gap(previous.close_time, current.open_time))
    return tuple(gaps)
 
 
@dataclass(frozen=True)
class SeriesInfo:
    timeframe: Timeframe
    path: Path
    rows: int
    first_open: datetime
    last_close: datetime
    gaps: Tuple[Gap, ...]
    meta: Optional[dict]
 
 
@dataclass
class Verification:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    series: Dict[Timeframe, SeriesInfo] = field(default_factory=dict)
    candles: Dict[Timeframe, List[Candle]] = field(default_factory=dict)
 
    @property
    def ok(self) -> bool:
        return not self.errors and len(self.candles) == 3
 
 
def _fmt(moment: datetime) -> str:
    return moment.strftime(TIME_FORMAT)
 
 
def _read_meta(csv_path: Path, label: str, require: bool, out: Verification) -> Optional[dict]:
    path = meta_path_for(csv_path)
    if not path.exists():
        if require:
            out.errors.append(
                f"{label}: {path.name} is missing, so nothing proves this file is XAUUSD. "
                "Export the data with tools/export_mt5_xauusd.py, or pass --skip-symbol-check "
                "if you accept that the symbol cannot be verified."
            )
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        out.errors.append(f"{label}: cannot read {path.name} ({exc})")
        return None
    if not isinstance(meta, dict):
        out.errors.append(f"{label}: {path.name} is not a JSON object")
        return None
    return meta
 
 
def _check_meta(meta: dict, timeframe: Timeframe, label: str, candles: List[Candle], out: Verification) -> None:
    symbol = str(meta.get("symbol", ""))
    if REQUIRED_SYMBOL_TEXT not in symbol.upper():
        out.errors.append(f"{label}: the symbol in the meta file is {symbol!r}, not XAUUSD")
    if str(meta.get("timeframe", "")).upper() != TIMEFRAME_LABELS[timeframe]:
        out.errors.append(f"{label}: the meta file says timeframe {meta.get('timeframe')!r}, expected {TIMEFRAME_LABELS[timeframe]}")
    if meta.get("rows") != len(candles):
        out.errors.append(f"{label}: the meta file says {meta.get('rows')} rows but the CSV has {len(candles)}")
    if meta.get("first_open_time") != _fmt(candles[0].open_time):
        out.errors.append(f"{label}: the meta first_open_time {meta.get('first_open_time')!r} does not match the CSV ({_fmt(candles[0].open_time)})")
    if meta.get("last_open_time") != _fmt(candles[-1].open_time):
        out.errors.append(f"{label}: the meta last_open_time {meta.get('last_open_time')!r} does not match the CSV ({_fmt(candles[-1].open_time)})")
 
 
def verify_dataset(
    paths: Mapping[Timeframe, Path],
    require_symbol_meta: bool = True,
    delimiter: str = ",",
    encoding: str = "utf-8-sig",
    gap_threshold: timedelta = timedelta(days=3),
) -> Verification:
    """Verify the H1, M15 and M5 files. Returns a Verification; check `.ok`."""
    out = Verification()
    for timeframe in (Timeframe.H1, Timeframe.M15, Timeframe.M5):
        label = f"{TIMEFRAME_LABELS[timeframe]} ({Path(paths[timeframe]).name})"
        csv_path = Path(paths[timeframe])
        if not csv_path.exists():
            out.errors.append(f"{label}: the file does not exist ({csv_path})")
            continue
        try:
            candles = load_candles_csv(csv_path, timeframe, delimiter, encoding)
        except DataError as exc:
            out.errors.append(f"{label}: {exc}")
            continue
        meta = _read_meta(csv_path, label, require_symbol_meta, out)
        if meta is not None:
            _check_meta(meta, timeframe, label, candles, out)
        out.candles[timeframe] = candles
        out.series[timeframe] = SeriesInfo(
            timeframe=timeframe,
            path=csv_path,
            rows=len(candles),
            first_open=candles[0].open_time,
            last_close=candles[-1].close_time,
            gaps=find_gaps(candles, gap_threshold),
            meta=meta,
        )
 
    metas = [(tf, info.meta) for tf, info in out.series.items() if info.meta is not None]
    for field_name in ("symbol", "mt5_server"):
        values = {str(m.get(field_name)) for _, m in metas}
        if len(values) > 1:
            out.errors.append(f"the three meta files disagree on {field_name}: {sorted(values)}")
 
    if len(out.series) == 3:
        firsts = {tf: info.first_open for tf, info in out.series.items()}
        lasts = {tf: info.last_close for tf, info in out.series.items()}
        if max(firsts.values()) - min(firsts.values()) > timedelta(days=1):
            out.warnings.append(
                "the three series do not start at the same time ("
                + ", ".join(f"{TIMEFRAME_LABELS[tf]} {_fmt(v)}" for tf, v in firsts.items())
                + "); trades can only start once H1 history exists"
            )
        if max(lasts.values()) - min(lasts.values()) > timedelta(days=1):
            out.warnings.append(
                "the three series do not end at the same time ("
                + ", ".join(f"{TIMEFRAME_LABELS[tf]} {_fmt(v)}" for tf, v in lasts.items())
                + "); the tail of the longer series is not covered by the others"
            )
    for info in out.series.values():
        if info.gaps:
            out.warnings.append(
                f"{TIMEFRAME_LABELS[info.timeframe]}: {len(info.gaps)} gap(s) longer than "
                f"{gap_threshold.days} days (missing history or holidays); first ones: "
                + "; ".join(f"{_fmt(g.after)} -> {_fmt(g.until)}" for g in info.gaps[:5])
            )
    return out
 
 
def format_verification(result: Verification) -> str:
    lines = ["DATA VERIFICATION", f"Result: {'PASSED' if result.ok else 'FAILED'}"]
    for timeframe in (Timeframe.H1, Timeframe.M15, Timeframe.M5):
        info = result.series.get(timeframe)
        name = TIMEFRAME_LABELS[timeframe]
        if info is None:
            lines.append(f"  {name:<4}: not loaded")
            continue
        symbol = info.meta.get("symbol") if info.meta else "unverified (no meta file)"
        server = info.meta.get("mt5_server") if info.meta else "unknown"
        lines.append(
            f"  {name:<4}: {info.rows} candles, {_fmt(info.first_open)} -> {_fmt(info.last_close)}, "
            f"symbol {symbol}, server {server}"
        )
        lines.append(f"        file {info.path}")
    for error in result.errors:
        lines.append(f"ERROR: {error}")
    for warning in result.warnings:
        lines.append(f"WARNING: {warning}")
    if result.ok:
        lines.append(
            "Checked: files exist, columns and numbers valid, open times on the timeframe grid, "
            "chronological, no duplicate timestamps, symbol and counts match the meta files."
        )
    return "\n".join(lines)