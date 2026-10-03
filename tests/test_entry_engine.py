"""EntryEngine tests for Koffie Strategy 1."""
from datetime import datetime

import pytest

from koffie.strategy.engines.entry_engine import EntryEngine
from koffie.strategy.engines.setup_engine import SetupEngine
from koffie.strategy.models.bos import BOS, BOSDirection
from koffie.strategy.models.candle import Candle, CandleClass, Timeframe
from koffie.strategy.models.confirmation import Confirmation
from koffie.strategy.models.entry import Entry
from koffie.strategy.models.liquidity import LiquidityLevel
from koffie.strategy.models.pivot import Pivot
from koffie.strategy.models.setup import Setup
from koffie.strategy.models.sweep import Sweep
from koffie.strategy.models.swing import Swing, SwingType
from koffie.strategy.models.zone import Zone

T0 = datetime(2026, 1, 1, 10, 0)
M5 = Timeframe.M5


def at(i):
    return T0 + i * M5.duration


def make_setup_and_confirmation():
    broken = Swing(
        SwingType.HIGH,
        M5,
        100.0,
        at(2),
        at(4),
        0,
    )

    bos = BOS(
        M5,
        BOSDirection.BULLISH,
        broken,
        at(10),
        101.0,
    )

    zone = Zone(
        Pivot(
            bos,
            at(5),
            110.0,
            99.0,
            CandleClass.BEARISH_DECISIVE,
        )
    )

    touch = Candle(
        M5,
        at(11),
        101.0,
        104.0,
        100.0,
        102.0,
    )

    setup = Setup(zone, touch)

    liquidity = LiquidityLevel(
        Swing(
            SwingType.LOW,
            M5,
            98.0,
            at(0),
            at(2),
            0,
        )
    )

    sweep_candle = Candle(
        M5,
        at(11),
        98.5,
        100.0,
        97.5,
        98.8,
    )

    sweep = Sweep(zone, liquidity, sweep_candle)

    confirmation_candle = Candle(
        M5,
        at(12),
        100.0,
        105.0,
        99.8,
        104.8,
    )

    confirmation = Confirmation(sweep, confirmation_candle)

    return setup, confirmation


class StubSetupEngine(SetupEngine):
    """Controlled SetupEngine surface used only to test EntryEngine."""

    def __init__(self, setup, confirmation, can_create=True):
        self._setup = setup
        self._confirmation = confirmation
        self._can_create = can_create
        self.recorded = []

    def can_create_trade(self, setup):
        assert setup is self._setup
        return self._can_create

    def confirmation_for(self, setup):
        assert setup is self._setup
        return self._confirmation

    def record_trade_created(self, setup, now):
        assert setup is self._setup
        self.recorded.append((setup, now))


def test_entry_engine_requires_setup_engine():
    with pytest.raises(TypeError):
        EntryEngine(None)

    with pytest.raises(TypeError):
        EntryEngine("setup engine")


def test_create_entry_returns_entry_with_exact_confirmation_close():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    entry = engine.create_entry(setup, confirmation.known_at)

    assert isinstance(entry, Entry)
    assert entry.price == confirmation.candle.close
    assert entry.price == 104.8
    assert entry.time == confirmation.candle.close_time


def test_create_entry_records_trade_created_after_entry_is_constructed():
    setup, confirmation = make_setup_and_confirmation()
    stub = StubSetupEngine(setup, confirmation)
    engine = EntryEngine(stub)

    entry = engine.create_entry(setup, confirmation.known_at)

    assert stub.recorded == [(setup, confirmation.known_at)]
    assert engine.entries == (entry,)


def test_create_entry_is_idempotent_for_same_setup():
    setup, confirmation = make_setup_and_confirmation()
    stub = StubSetupEngine(setup, confirmation)
    engine = EntryEngine(stub)

    first = engine.create_entry(setup, confirmation.known_at)
    second = engine.create_entry(setup, confirmation.known_at)

    assert first is second
    assert engine.entries == (first,)
    assert len(stub.recorded) == 1


def test_entry_for_returns_existing_entry():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    assert engine.entry_for(setup) is None

    entry = engine.create_entry(setup, confirmation.known_at)

    assert engine.entry_for(setup) is entry


def test_entry_for_rejects_wrong_type():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    for bad in (None, "setup", 5, confirmation):
        with pytest.raises(TypeError):
            engine.entry_for(bad)


def test_create_entry_rejects_unconfirmed_or_unavailable_trade():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(
        StubSetupEngine(
            setup,
            confirmation,
            can_create=False,
        )
    )

    with pytest.raises(ValueError, match="confirmed"):
        engine.create_entry(setup, confirmation.known_at)


def test_create_entry_rejects_before_confirmation_close():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    before_close = confirmation.known_at.replace(
        minute=confirmation.known_at.minute
    )

    # Move one second before the actual close.
    from datetime import timedelta
    before_close = confirmation.known_at - timedelta(seconds=1)

    with pytest.raises(ValueError, match="closed"):
        engine.create_entry(setup, before_close)

    assert engine.entries == ()


def test_process_creates_entries_only_for_confirmed_setups_known_at_now():
    setup, confirmation = make_setup_and_confirmation()

    class ProcessStub(StubSetupEngine):
        def __init__(self):
            super().__init__(setup, confirmation)

        def confirmed_setups_known_at(self, moment):
            if moment >= confirmation.known_at:
                return (setup,)
            return ()

    stub = ProcessStub()
    engine = EntryEngine(stub)

    assert engine.process(confirmation.known_at - __import__("datetime").timedelta(seconds=1)) == ()
    assert engine.entries == ()

    created = engine.process(confirmation.known_at)

    assert len(created) == 1
    assert created[0].price == confirmation.candle.close
    assert engine.entries == created


def test_process_does_not_duplicate_existing_entry():
    setup, confirmation = make_setup_and_confirmation()

    class ProcessStub(StubSetupEngine):
        def __init__(self):
            super().__init__(setup, confirmation)

        def confirmed_setups_known_at(self, moment):
            return (setup,) if moment >= confirmation.known_at else ()

    stub = ProcessStub()
    engine = EntryEngine(stub)

    first = engine.process(confirmation.known_at)
    second = engine.process(confirmation.known_at)

    assert len(first) == 1
    assert second == ()
    assert engine.entries == first


def test_process_rejects_wrong_now_type():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    for bad in (None, "now", 5, confirmation.known_at.date()):
        with pytest.raises(TypeError):
            engine.process(bad)


def test_create_entry_rejects_wrong_setup_type():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    for bad in (None, "setup", 5, confirmation):
        with pytest.raises(TypeError):
            engine.create_entry(bad, confirmation.known_at)


def test_create_entry_rejects_wrong_now_type():
    setup, confirmation = make_setup_and_confirmation()
    engine = EntryEngine(StubSetupEngine(setup, confirmation))

    for bad in (None, "now", 5, confirmation.known_at.date()):
        with pytest.raises(TypeError):
            engine.create_entry(setup, bad)
