"""Export real XAUUSD H1, M15 and M5 candles from your connected MetaTrader 5 terminal.
 
Read-only: it only asks MT5 for historical candles; it places no orders.
 
Run from C:\\Users\\FELIX\\trading_bot with the MT5 desktop app OPEN and logged in:
 
    .\\.venv\\Scripts\\python.exe tools\\export_mt5_xauusd.py --start 2024-01-01
 
It writes data/XAUUSD_H1.csv, data/XAUUSD_M15.csv, data/XAUUSD_M5.csv and a matching
.meta.json beside each, then verifies them. Nothing is invented: if your broker has less
history than you asked for, the files simply start later, and the report says so.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
 
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
 
try:
    import MetaTrader5 as mt5
except ImportError:
    sys.exit("The MetaTrader5 package is missing. Run:  .\\.venv\\Scripts\\python.exe -m pip install MetaTrader5")
 
from koffie.backtest.mt5_export import TIMEFRAME_LABELS, ExportError, epoch_to_text, export_timeframe
from koffie.backtest.verify import format_verification, verify_dataset
from koffie.strategy.models.candle import Timeframe
 
MT5_CODES = {
    Timeframe.H1: mt5.TIMEFRAME_H1,
    Timeframe.M15: mt5.TIMEFRAME_M15,
    Timeframe.M5: mt5.TIMEFRAME_M5,
}
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Export XAUUSD H1/M15/M5 candles from MetaTrader 5 to CSV.")
    parser.add_argument("--start", default="2024-01-01", help="first day to request, YYYY-MM-DD (default 2024-01-01)")
    parser.add_argument("--end", default=None, help="last day to request, YYYY-MM-DD (default: now on the broker clock)")
    parser.add_argument("--out", default=str(ROOT / "data"), help="output folder (default: data)")
    parser.add_argument("--symbol", default=None, help="exact MT5 symbol name if auto-detection fails (e.g. XAUUSDm)")
    return parser.parse_args()
 
 
def find_symbol(requested):
    if requested:
        if mt5.symbol_info(requested) is None:
            sys.exit(f"MT5 has no symbol called {requested!r}.")
        return requested
    if mt5.symbol_info("XAUUSD") is not None:
        return "XAUUSD"
    matches = [s.name for s in (mt5.symbols_get() or ()) if "XAUUSD" in s.name.upper()]
    if len(matches) == 1:
        return matches[0]
    sys.exit(
        "Could not pick the gold symbol automatically. Symbols containing XAUUSD: "
        f"{matches or 'none'}. Re-run with --symbol <exact name>."
    )
 
 
def main():
    args = parse_args()
    if not mt5.initialize():
        sys.exit(f"Could not attach to MetaTrader 5: {mt5.last_error()}\nOpen the MT5 app and log in first (run mt5_check.py).")
    try:
        terminal, account = mt5.terminal_info(), mt5.account_info()
        if not (terminal and terminal.connected) or account is None:
            sys.exit("MT5 is open but not connected to a trading account. Log in first.")
        symbol = find_symbol(args.symbol)
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None or tick.time == 0:
            sys.exit(f"No price received yet for {symbol}; open its chart in MT5 and try again.")
        server_now = int(tick.time)
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc) if args.end
               else datetime.fromtimestamp(server_now, timezone.utc))
 
        def fetch(sym, timeframe, window_start, window_end):
            code = MT5_CODES[timeframe]
            rates = mt5.copy_rates_range(sym, code, window_start, window_end)
            if rates is None:
                return None
            return [(int(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])) for r in rates]
 
        print(f"Symbol {symbol} | account server {account.server} | broker clock now {epoch_to_text(server_now)}")
        for timeframe in (Timeframe.H1, Timeframe.M15, Timeframe.M5):
            mt5.copy_rates_from_pos(symbol, MT5_CODES[timeframe], 0, 5)      # asks MT5 to load this timeframe's history
            csv_path, _, rows, dropped = export_timeframe(
                fetch, symbol, timeframe, start, end, server_now, Path(args.out), account.server
            )
            print(f"  {TIMEFRAME_LABELS[timeframe]:<4}: {rows} completed candles -> {csv_path}"
                  + (f"  (dropped {dropped} unfinished)" if dropped else ""))
    except ExportError as exc:
        sys.exit(f"EXPORT ERROR: {exc}\nIf MT5 returned nothing, open an XAUUSD chart in MT5, press Home to load older history, "
                 "and raise Tools > Options > Charts > Max bars in chart.")
    finally:
        mt5.shutdown()
 
    out = Path(args.out)
    result = verify_dataset({tf: out / f"XAUUSD_{TIMEFRAME_LABELS[tf]}.csv" for tf in MT5_CODES})
    print()
    print(format_verification(result))
    sys.exit(0 if result.ok else 1)
 
 
if __name__ == "__main__":
    main()