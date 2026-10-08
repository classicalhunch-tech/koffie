"""
Runs tools/Backtest_Strategy1.py unchanged, but after N seconds prints exactly
which line of code it is stuck on and then stops it.

Usage (from C:\\Users\\FELIX\\trading_bot):
    .\\.venv\\Scripts\\python.exe tools\\run_with_watchdog.py 120

120 = seconds to wait before dumping the location (change as you like).
"""
import faulthandler
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "tools" / "Backtest_Strategy1.py"

seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 120

# Do not forward the "seconds" number to the backtest's own argument parser.
# Anything typed after the seconds value is passed on to the backtest.
sys.argv = [str(TARGET)] + sys.argv[2:]

# Make sure the project root is importable (same as running from the repo root).
sys.path.insert(0, str(ROOT))

if not TARGET.exists():
    sys.exit(f"Cannot find {TARGET}")

print(f"[watchdog] running {TARGET.name}; will dump its location after {seconds}s")

# After `seconds`, print the call stack of every thread, then exit the process.
faulthandler.dump_traceback_later(seconds, repeat=False, exit=True, file=sys.stderr)

start = time.time()
try:
    runpy.run_path(str(TARGET), run_name="__main__")
except KeyboardInterrupt:
    print(f"\n[watchdog] interrupted after {time.time() - start:.1f}s")
    raise
finally:
    faulthandler.cancel_dump_traceback_later()

print(f"[watchdog] finished normally in {time.time() - start:.1f}s")