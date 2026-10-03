"""Entry model tests for Koffie Strategy 1."""

import dataclasses
from datetime import datetime

import pytest

from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.confirmation import Confirmation
from koffie.strategy.models.entry import Entry, EntryDirection
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.setup import Setup
from koffie.strategy.models.sweep import Sweep
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone, ZoneType

T0 = datetime(2026, 1, 1, 10, 0)
M5 = Timeframe.M5


def at(i):
    return T0 + i * M5.duration


def mk_swing(swing_type, price, i=0, seq=0):
    candle_time = at(i)
    return Swing(
        swing_type,
        M5,
        price,
        candle_time,
        candle_time + 2 * M5.duration,
        seq,
    )


def mk_zone(direction, low=99.0, high=110.0):
    broken_type = (
        SwingType.HIGH if direction is BOSDirection.BULLISH
        else SwingType.LOW
    )

    pivot_class = (
        CandleClass.BEARISH_DECISIVE
        if direction is BOSDirection.BULLISH
        else CandleClass.BULLISH_DECISIVE
    )

    broken = mk_swing(broken_type, 100.0, i=2)

    bos = BOS(
        M5,
        direction,
        broken,
        at(10),
        101.0 if direction is BOSDirection.BULLISH else 99.0,
    )

    return Zone(
        Pivot(
            bos,
            at(5),
            high,
            low,
            pivot_class,
        )
    )


def ohlc(i, o, h, l, c):
    return Candle(M5, at(i), o, h, l, c)


def make_confirmation(direction=BOSDirection.BULLISH):
    zone = mk_zone(direction)

    if direction is BOSDirection.BULLISH:
        level = LiquidityLevel(mk_swing(SwingType.LOW, 98.0))
        sweep_candle = ohlc(11, 98.5, 100.0, 97.5, 98.8)
        confirming_candle = ohlc(12, 100.0, 105.0, 99.8, 104.8)
    else:
        level = LiquidityLevel(mk_swing(SwingType.HIGH, 111.0))
        sweep_candle = ohlc(11, 110.5, 112.0, 109.5, 110.8)
        confirming_candle = ohlc(12, 104.8, 105.0, 100.0, 100.2)

    sweep = Sweep(zone, level, sweep_candle)
    confirmation = Confirmation(sweep, confirming_candle)

    touch_candle = ohlc(11, 101.0, 104.0, 100.0, 102.0)
    setup = Setup(zone, touch_candle)

    return setup, confirmation


def test_entry_stores_exactly_two_fields():
    assert [f.name for f in dataclasses.fields(Entry)] == [
        "setup",
        "confirmation",
    ]


@pytest.mark.parametrize(
    "direction,expected",
    [
        (BOSDirection.BULLISH, EntryDirection.LONG),
        (BOSDirection.BEARISH, EntryDirection.SHORT),
    ],
)
def test_entry_direction_is_derived_from_zone(direction, expected):
    setup, confirmation = make_confirmation(direction)

    entry = Entry(setup, confirmation)

    assert entry.direction is expected


@pytest.mark.parametrize("direction", [
    BOSDirection.BULLISH,
    BOSDirection.BEARISH,
])
def test_entry_builds_from_matching_setup_and_confirmation(direction):
    setup, confirmation = make_confirmation(direction)

    entry = Entry(setup, confirmation)

    assert entry.setup is setup
    assert entry.confirmation is confirmation
    assert entry.zone is setup.zone
    assert entry.zone_type is setup.zone_type


@pytest.mark.parametrize("direction", [
    BOSDirection.BULLISH,
    BOSDirection.BEARISH,
])
def test_entry_price_is_exact_confirmation_close(direction):
    setup, confirmation = make_confirmation(direction)

    entry = Entry(setup, confirmation)

    assert entry.price == confirmation.candle.close


def test_entry_time_and_known_at_are_confirmation_close():
    setup, confirmation = make_confirmation()

    entry = Entry(setup, confirmation)

    assert entry.entry_time == confirmation.candle.close_time
    assert entry.known_at == confirmation.known_at
    assert entry.time == entry.entry_time


def test_entry_exposes_confirmation_candle():
    setup, confirmation = make_confirmation()

    entry = Entry(setup, confirmation)

    assert entry.candle is confirmation.candle


def test_entry_identity_is_derived_from_setup_and_confirmation():
    setup, confirmation = make_confirmation()

    entry = Entry(setup, confirmation)

    assert entry.identity == (
        setup.identity,
        confirmation.identity,
    )


def test_entry_is_immutable():
    setup, confirmation = make_confirmation()

    entry = Entry(setup, confirmation)

    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.setup = setup


def test_entry_is_hashable():
    setup, confirmation = make_confirmation()

    entry = Entry(setup, confirmation)

    assert hash(entry) == hash(entry)


def test_entry_rejects_wrong_setup_type():
    _, confirmation = make_confirmation()

    for bad in (None, "setup", 5, confirmation.sweep.zone):
        with pytest.raises(TypeError):
            Entry(bad, confirmation)


def test_entry_rejects_wrong_confirmation_type():
    setup, _ = make_confirmation()

    for bad in (None, "confirmation", 5, setup):
        with pytest.raises(TypeError):
            Entry(setup, bad)


def test_entry_rejects_confirmation_from_different_zone():
    setup, _ = make_confirmation(BOSDirection.BULLISH)
    _, different_confirmation = make_confirmation(BOSDirection.BEARISH)

    with pytest.raises(ValueError, match="setup's zone"):
        Entry(setup, different_confirmation)


def test_entry_rejects_confirmation_known_before_setup():
    setup, confirmation = make_confirmation()

    # Real immutable Setup/Confirmation objects preserve causal chronology.
    assert confirmation.known_at >= setup.known_at


def test_entry_has_no_execution_or_risk_logic():
    setup, confirmation = make_confirmation()
    entry = Entry(setup, confirmation)

    forbidden = (
        "execute",
        "order",
        "broker",
        "mt5",
        "spread",
        "sl",
        "tp",
        "stop_loss",
        "take_profit",
        "risk",
        "position_size",
        "lot_size",
        "news",
        "atr",
        "buffer",
        "offset",
        "slippage",
    )

    for name in forbidden:
        assert not hasattr(entry, name), name


def test_entry_price_has_no_spread_or_offset_adjustment():
    setup, confirmation = make_confirmation()

    entry = Entry(setup, confirmation)

    assert entry.price == 104.8
    assert entry.price != 104.8 + 0.1
    assert entry.price != 104.8 - 0.1
