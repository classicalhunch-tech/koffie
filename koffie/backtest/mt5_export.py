from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from koffie.strategy.models.candle import Timeframe


TIMEFRAME_LABELS = {
    Timeframe.H1: "H1",
    Timeframe.M15: "M15",
    Timeframe.M5: "M5",
}


class ExportError(Exception):
    pass


def epoch_to_text(epoch: int) -> str:
    return datetime.fromtimestamp(
        int(epoch), timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")


def export_timeframe(
    fetch,
    symbol,
    timeframe,
    start,
    end,
    server_now,
    output_dir,
    account_server,
):
    label = TIMEFRAME_LABELS[timeframe]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = fetch(symbol, timeframe, start, end)

    if rows is None:
        raise ExportError(
            f"MT5 returned no data for {symbol} {label}."
        )

    rows = sorted(rows, key=lambda row: int(row[0]))

    cleaned = []

    for row in rows:
        timestamp, open_, high, low, close = row

        timestamp = int(timestamp)
        open_ = float(open_)
        high = float(high)
        low = float(low)
        close = float(close)

        if high < max(open_, close, low):
            raise ExportError(
                f"Invalid OHLC data at {epoch_to_text(timestamp)}."
            )

        if low > min(open_, close, high):
            raise ExportError(
                f"Invalid OHLC data at {epoch_to_text(timestamp)}."
            )

        cleaned.append(
            (timestamp, open_, high, low, close)
        )

    unique = []
    seen = set()

    for row in cleaned:
        if row[0] in seen:
            raise ExportError(
                f"Duplicate candle timestamp: {epoch_to_text(row[0])}"
            )
        seen.add(row[0])
        unique.append(row)

    completed = [
        row for row in unique
        if row[0] < server_now
    ]

    dropped = len(unique) - len(completed)

    if not completed:
        raise ExportError(
            f"No completed candles found for {symbol} {label}."
        )

    csv_path = output_dir / f"XAUUSD_{label}.csv"
    meta_path = output_dir / f"XAUUSD_{label}.meta.json"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            ["time", "open", "high", "low", "close"]
        )

        for timestamp, open_, high, low, close in completed:
            writer.writerow(
                [
                    epoch_to_text(timestamp),
                    open_,
                    high,
                    low,
                    close,
                ]
            )

    metadata = {
        "symbol": symbol,
        "timeframe": label,
        "account_server": account_server,
        "rows": len(completed),
        "dropped_unfinished": dropped,
        "first_candle_utc": epoch_to_text(
            completed[0][0]
        ),
        "last_candle_utc": epoch_to_text(
            completed[-1][0]
        ),
    }

    meta_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return (
        csv_path,
        meta_path,
        len(completed),
        dropped,
    )