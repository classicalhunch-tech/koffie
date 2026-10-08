"""Tests for koffie/backtest/verify.py and koffie/backtest/mt5_export.py.
 
The MetaTrader 5 terminal is replaced by a plain function that returns hand-made rates, because
the real terminal cannot run inside the test suite. Every price in this file is synthetic and
exists only to exercise the file and verification logic; none of it is market data.
"""
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
 
import pytest
 
from koffie.backtest.mt5_export import (
    ExportError,
    collect_rates,
    epoch_to_text,
    export_timeframe,
    finalize_rates,
    windows,
    write_export,
)
from koffie.backtest.verify import find_gaps, format_verification, meta_path_for, verify_dataset
from koffie.strategy.models.candle import Candle, Timeframe
 
H1, M15, M5 = Timeframe.H1, Timeframe.M15, Timeframe.M5
START = datetime(2024, 1, 2, 0, 0, tzinfo=timezone.utc)
EPOCH0 = int(START.timestamp())
 
 
def epoch(minutes):
    return EPOCH0 + minutes * 60
 
 
def rates_for(tf, count, start_minute=0):
    step = int(tf.duration.total_seconds() // 60)
    return [(epoch(start_minute + i * step), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(count)]
 
 
def fake_fetch(table):
    """table: {Timeframe: [rates]} -> a fetch(symbol, timeframe, from, to) like MT5's copy_rates_range."""
    calls = []
 
    def fetch(symbol, timeframe, window_start, window_end):
        calls.append((symbol, timeframe, window_start, window_end))
        lo, hi = int(window_start.timestamp()), int(window_end.timestamp())
        return [r for r in table[timeframe] if lo <= r[0] <= hi]
 
    fetch.calls = calls
    return fetch
 
 
def export_all(folder, counts=(30, 120, 360), symbol="XAUUSD", server="Test-Server", now_minutes=100000):
    table = {tf: rates_for(tf, n) for tf, n in zip((H1, M15, M5), counts)}
    fetch = fake_fetch(table)
    for tf in (H1, M15, M5):
        export_timeframe(fetch, symbol, tf, START, START + timedelta(days=40), epoch(now_minutes), Path(folder), server)
    return {tf: Path(folder) / f"XAUUSD_{label}.csv" for tf, label in ((H1, "H1"), (M15, "M15"), (M5, "M5"))}
 
 
# =============================================================== windows and collection
def test_windows_cover_the_range_without_gaps():
    ws = windows(START, START + timedelta(days=70), 30)
    assert ws[0][0] == START and ws[-1][1] == START + timedelta(days=70)
    assert all(a[1] == b[0] for a, b in zip(ws, ws[1:]))
    assert len(ws) == 3
    with pytest.raises(ExportError):
        windows(START, START)
 
 
def test_collect_rates_asks_for_every_window_and_keeps_empty_ones():
    fetch = fake_fetch({H1: rates_for(H1, 3)})
    rates = collect_rates(fetch, "XAUUSD", H1, START, START + timedelta(days=65), 30)
    assert len(fetch.calls) == 3 and len(rates) == 3
 
 
def test_an_mt5_error_is_an_export_error_not_an_empty_result():
    with pytest.raises(ExportError):
        collect_rates(lambda *a: None, "XAUUSD", H1, START, START + timedelta(days=5))
 
 
# =============================================================== cleaning
def test_finalize_sorts_and_removes_exact_duplicates():
    rates = rates_for(M5, 4)
    rows, dropped = finalize_rates([rates[2], rates[0], rates[1], rates[1], rates[3]], M5, epoch(10_000))
    assert [r[0] for r in rows] == [r[0] for r in rates] and dropped == 0
 
 
def test_two_different_candles_with_one_open_time_are_an_error():
    a = (epoch(0), 100.0, 101.0, 99.0, 100.5)
    b = (epoch(0), 100.0, 102.0, 99.0, 100.5)
    with pytest.raises(ExportError):
        finalize_rates([a, b], M5, epoch(10_000))
 
 
def test_unfinished_candles_are_dropped_and_finished_ones_kept_exactly_at_the_boundary():
    rates = rates_for(M5, 3)                                  # opens at minutes 0, 5, 10
    rows, dropped = finalize_rates(rates, M5, epoch(15))      # the 10:00 candle closes at minute 15 -> complete
    assert len(rows) == 3 and dropped == 0
    rows, dropped = finalize_rates(rates, M5, epoch(14))      # the last candle is still forming
    assert len(rows) == 2 and dropped == 1
    h1 = rates_for(H1, 2)
    rows, dropped = finalize_rates(h1, H1, epoch(90))         # second H1 candle (60..120) unfinished
    assert len(rows) == 1 and dropped == 1
 
 
# =============================================================== writing the files
def test_the_exported_files_have_the_documented_format():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        lines = paths[M5].read_text(encoding="utf-8").splitlines()
        assert lines[0] == "time,open,high,low,close"
        assert lines[1] == "2024-01-02 00:00:00,100.0,101.0,99.0,100.5"
        assert len(lines) == 1 + 360
        meta = json.loads(meta_path_for(paths[M5]).read_text(encoding="utf-8"))
        assert meta["symbol"] == "XAUUSD" and meta["timeframe"] == "M5" and meta["rows"] == 360
        assert meta["first_open_time"] == "2024-01-02 00:00:00" and meta["mt5_server"] == "Test-Server"
        assert "no timezone conversion" in meta["time_basis"]
 
 
def test_an_empty_export_is_refused_instead_of_writing_empty_files():
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(ExportError):
            write_export([], "XAUUSD", M5, Path(folder), "S", START, START + timedelta(days=1))
        assert list(Path(folder).iterdir()) == []
 
 
def test_epoch_text_is_the_broker_clock_without_conversion():
    assert epoch_to_text(EPOCH0) == "2024-01-02 00:00:00"
    assert epoch_to_text(EPOCH0 + 3661) == "2024-01-02 01:01:01"
 
 
# =============================================================== verification: the passing path
def test_a_clean_export_passes_verification_and_is_reported():
    with tempfile.TemporaryDirectory() as folder:
        result = verify_dataset(export_all(folder))
        assert result.ok and result.errors == []
        assert set(result.candles) == {H1, M15, M5}
        assert [len(result.candles[tf]) for tf in (H1, M15, M5)] == [30, 120, 360]
        text = format_verification(result)
        assert "Result: PASSED" in text and "symbol XAUUSD" in text and "server Test-Server" in text
        assert "no duplicate timestamps" in text
 
 
def test_a_broker_symbol_with_a_suffix_is_accepted_as_long_as_it_contains_xauusd():
    with tempfile.TemporaryDirectory() as folder:
        assert verify_dataset(export_all(folder, symbol="XAUUSDm")).ok
 
 
# =============================================================== verification: failures
def test_a_missing_file_fails():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        paths[M15].unlink()
        result = verify_dataset(paths)
        assert not result.ok and any("does not exist" in e for e in result.errors)
 
 
def test_a_missing_meta_file_fails_unless_the_symbol_check_is_skipped():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        meta_path_for(paths[H1]).unlink()
        strict = verify_dataset(paths)
        assert not strict.ok and any("nothing proves this file is XAUUSD" in e for e in strict.errors)
        assert verify_dataset(paths, require_symbol_meta=False).ok
 
 
def test_a_non_xauusd_symbol_fails():
    with tempfile.TemporaryDirectory() as folder:
        result = verify_dataset(export_all(folder, symbol="EURUSD"))
        assert not result.ok and any("not XAUUSD" in e for e in result.errors)
 
 
def test_files_exported_from_different_symbols_or_servers_fail():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        meta = meta_path_for(paths[M5])
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["mt5_server"] = "Other-Server"
        meta.write_text(json.dumps(data), encoding="utf-8")
        result = verify_dataset(paths)
        assert not result.ok and any("disagree on mt5_server" in e for e in result.errors)
        data["mt5_server"] = "Test-Server"
        data["symbol"] = "XAUUSDm"
        meta.write_text(json.dumps(data), encoding="utf-8")
        assert any("disagree on symbol" in e for e in verify_dataset(paths).errors)
 
 
def test_a_meta_file_that_does_not_match_the_csv_fails():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        meta = meta_path_for(paths[M5])
        data = json.loads(meta.read_text(encoding="utf-8"))
        data["rows"] = 5
        data["timeframe"] = "M15"
        data["first_open_time"] = "2020-01-01 00:00:00"
        meta.write_text(json.dumps(data), encoding="utf-8")
        errors = " | ".join(verify_dataset(paths).errors)
        assert "5 rows" in errors and "expected M5" in errors and "first_open_time" in errors
 
 
def test_an_unreadable_meta_file_fails():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        meta_path_for(paths[H1]).write_text("{not json", encoding="utf-8")
        assert any("cannot read" in e for e in verify_dataset(paths).errors)
        meta_path_for(paths[H1]).write_text("[1, 2]", encoding="utf-8")
        assert any("not a JSON object" in e for e in verify_dataset(paths).errors)
 
 
def test_duplicate_unsorted_or_off_grid_timestamps_fail_verification():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        lines = paths[M5].read_text(encoding="utf-8").splitlines()
        paths[M5].write_text("\n".join(lines + [lines[-1]]) + "\n", encoding="utf-8")        # duplicate last row
        result = verify_dataset(paths, require_symbol_meta=False)
        assert not result.ok and any("repeats" in e or "chronological" in e for e in result.errors)
        paths[M5].write_text("\n".join([lines[0], lines[2], lines[1]] + lines[3:]) + "\n", encoding="utf-8")   # swapped rows
        assert not verify_dataset(paths, require_symbol_meta=False).ok
        paths[M5].write_text("\n".join([lines[0]] + [l.replace("00:00:00", "00:03:00", 1) for l in lines[1:4]]) + "\n", encoding="utf-8")
        assert any("grid" in e for e in verify_dataset(paths, require_symbol_meta=False).errors)
 
 
def test_a_candle_file_with_the_wrong_timeframe_fails():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        swapped = dict(paths)
        swapped[M5], swapped[M15] = paths[M15], paths[M5]
        assert not verify_dataset(swapped, require_symbol_meta=False).ok
 
 
# =============================================================== verification: information
def test_series_that_cover_different_periods_produce_warnings_not_errors():
    with tempfile.TemporaryDirectory() as folder:
        table = {H1: rates_for(H1, 30), M15: rates_for(M15, 120, start_minute=5 * 24 * 60), M5: rates_for(M5, 360)}
        fetch = fake_fetch(table)
        for tf in (H1, M15, M5):
            export_timeframe(fetch, "XAUUSD", tf, START, START + timedelta(days=40), epoch(10 ** 6), Path(folder), "S")
        paths = {tf: Path(folder) / f"XAUUSD_{l}.csv" for tf, l in ((H1, "H1"), (M15, "M15"), (M5, "M5"))}
        result = verify_dataset(paths)
        assert result.ok
        assert any("do not start at the same time" in w for w in result.warnings)
        assert any("do not end at the same time" in w for w in result.warnings)
 
 
def test_gaps_longer_than_three_days_are_reported_and_weekends_are_not():
    def c(day, hour):
        return Candle(H1, datetime(2024, 1, day, hour), 1.0, 2.0, 0.5, 1.5)
    weekend = [c(5, 21), c(7, 22)]                      # Fri 21:00 -> Sun 22:00: about 48 hours
    assert find_gaps(weekend) == ()
    long_gap = [c(2, 0), c(9, 0)]
    (gap,) = find_gaps(long_gap)
    assert gap.after == datetime(2024, 1, 2, 1) and gap.until == datetime(2024, 1, 9, 0)
    assert gap.length > timedelta(days=3)
    with tempfile.TemporaryDirectory() as folder:
        table = {H1: rates_for(H1, 5) + rates_for(H1, 5, start_minute=10 * 24 * 60), M15: rates_for(M15, 20), M5: rates_for(M5, 60)}
        fetch = fake_fetch(table)
        for tf in (H1, M15, M5):
            export_timeframe(fetch, "XAUUSD", tf, START, START + timedelta(days=40), epoch(10 ** 6), Path(folder), "S")
        paths = {tf: Path(folder) / f"XAUUSD_{l}.csv" for tf, l in ((H1, "H1"), (M15, "M15"), (M5, "M5"))}
        result = verify_dataset(paths)
        assert result.ok and any("H1: 1 gap(s) longer than 3 days" in w for w in result.warnings)
        assert "WARNING" in format_verification(result)
 
 
def test_a_failed_verification_is_reported_with_every_problem_and_loads_no_candles_for_the_bad_file():
    with tempfile.TemporaryDirectory() as folder:
        paths = export_all(folder)
        paths[M15].write_text("time,open,high,low,close\n2024-01-02 00:00:00,1,2,0.5,x\n", encoding="utf-8")
        result = verify_dataset(paths)
        text = format_verification(result)
        assert not result.ok and "Result: FAILED" in text and "ERROR" in text and M15 not in result.candles