from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class VerificationResult:
    ok: bool
    path: Path
    rows: int
    first_time: str | None
    last_time: str | None
    errors: list[str]


@dataclass
class DatasetVerification:
    ok: bool
    results: dict


def _parse_time(value: str) -> int:
    value = value.strip()

    if value.endswith(" UTC"):
        value = value[:-4]

    dt = datetime.strptime(
        value,
        "%Y-%m-%d %H:%M:%S",
    )

    return int(
        dt.replace(tzinfo=timezone.utc).timestamp()
    )


def verify_file(path: Path) -> VerificationResult:
    errors = []
    timestamps = []

    if not path.exists():
        return VerificationResult(
            False,
            path,
            0,
            None,
            None,
            ["file does not exist"],
        )

    required = [
        "time",
        "open",
        "high",
        "low",
        "close",
    ]

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames != required:
                errors.append(
                    f"Expected columns {required}, "
                    f"got {reader.fieldnames}"
                )

                return VerificationResult(
                    False,
                    path,
                    0,
                    None,
                    None,
                    errors,
                )

            previous = None

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                try:
                    timestamp = _parse_time(
                        row["time"]
                    )

                    open_ = float(row["open"])
                    high = float(row["high"])
                    low = float(row["low"])
                    close = float(row["close"])

                except Exception as exc:
                    errors.append(
                        f"Line {line_number}: {exc}"
                    )
                    continue

                if previous is not None:

                    if timestamp == previous:
                        errors.append(
                            f"Line {line_number}: "
                            "duplicate timestamp"
                        )

                    if timestamp < previous:
                        errors.append(
                            f"Line {line_number}: "
                            "timestamps are not chronological"
                        )

                if high < max(
                    open_,
                    close,
                    low,
                ):
                    errors.append(
                        f"Line {line_number}: "
                        "invalid high"
                    )

                if low > min(
                    open_,
                    close,
                    high,
                ):
                    errors.append(
                        f"Line {line_number}: "
                        "invalid low"
                    )

                timestamps.append(timestamp)
                previous = timestamp

    except Exception as exc:
        errors.append(str(exc))

    first_time = None
    last_time = None

    if timestamps:
        first_time = datetime.fromtimestamp(
            timestamps[0],
            timezone.utc,
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        last_time = datetime.fromtimestamp(
            timestamps[-1],
            timezone.utc,
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    if not timestamps:
        errors.append(
            "No candle rows found."
        )

    return VerificationResult(
        ok=not errors,
        path=path,
        rows=len(timestamps),
        first_time=first_time,
        last_time=last_time,
        errors=errors,
    )


def verify_dataset(paths):
    results = {}

    for timeframe, path in paths.items():
        results[timeframe] = verify_file(
            Path(path)
        )

    return DatasetVerification(
        ok=all(
            result.ok
            for result in results.values()
        ),
        results=results,
    )


def format_verification(result):
    lines = [
        "",
        "Historical data verification",
        "============================",
        f"Overall: "
        f"{'PASS' if result.ok else 'FAIL'}",
    ]

    for timeframe, verification in (
        result.results.items()
    ):
        label = getattr(
            timeframe,
            "value",
            str(timeframe),
        )

        lines.extend(
            [
                "",
                f"{label}:",
                f"  File: {verification.path}",
                f"  Rows: {verification.rows}",
                f"  First: {verification.first_time}",
                f"  Last: {verification.last_time}",
                f"  Status: "
                f"{'PASS' if verification.ok else 'FAIL'}",
            ]
        )

        for error in verification.errors:
            lines.append(
                f"  ERROR: {error}"
            )

    return "\n".join(lines)