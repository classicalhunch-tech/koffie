from datetime import datetime, timedelta

import pytest

from koffie.strategy.engines.entry_engine import EntryEngine
from koffie.strategy.engines.sl_tp_engine import SLTPEngine
from koffie.strategy.models.entry import Entry, EntryDirection


BASE_TIME = datetime(2026, 1, 1, 12, 0, 0)


class StubEntry(Entry):
    """Minimal Entry test double for SLTPEngine tests."""

    def __init__(
        self,
        *,
        direction: EntryDirection,
        entry_price: float,
        sweep_low: float,
        sweep_high: float,
        known_at: datetime = BASE_TIME,
        identity_value=None,
    ):
        self._direction = direction
        self._entry_price = entry_price
        self._known_at = known_at
        self._identity = (
            identity_value
            if identity_value is not None
            else object()
        )

        class Candle:
            pass

        candle = Candle()
        candle.low = sweep_low
        candle.high = sweep_high

        sweep = type("Sweep", (), {})()
        sweep.candle = candle

        confirmation = type("Confirmation", (), {})()
        confirmation.sweep = sweep
        confirmation.known_at = known_at

        self._confirmation = confirmation

    @property
    def direction(self):
        return self._direction

    @property
    def price(self):
        return self._entry_price

    @property
    def confirmation(self):
        return self._confirmation

    @property
    def known_at(self):
        return self._known_at

    @property
    def identity(self):
        return self._identity


class StubEntryEngine(EntryEngine):
    """EntryEngine-compatible source for deterministic engine tests."""

    def __init__(self, entries):
        self._entries = tuple(entries)

    @property
    def entries(self):
        return self._entries


def make_buy_entry(
    *,
    entry_price=110.0,
    sweep_low=105.0,
    sweep_high=115.0,
    known_at=BASE_TIME,
    identity_value=None,
):
    return StubEntry(
        direction=EntryDirection.LONG,
        entry_price=entry_price,
        sweep_low=sweep_low,
        sweep_high=sweep_high,
        known_at=known_at,
        identity_value=identity_value,
    )


def make_sell_entry(
    *,
    entry_price=90.0,
    sweep_low=85.0,
    sweep_high=95.0,
    known_at=BASE_TIME,
    identity_value=None,
):
    return StubEntry(
        direction=EntryDirection.SHORT,
        entry_price=entry_price,
        sweep_low=sweep_low,
        sweep_high=sweep_high,
        known_at=known_at,
        identity_value=identity_value,
    )


def test_constructor_accepts_no_entry_engine():
    engine = SLTPEngine()

    assert engine.entry_engine is None
    assert engine.sl_tps == ()


def test_constructor_rejects_wrong_entry_engine_type():
    with pytest.raises(TypeError):
        SLTPEngine(object())


def test_create_sl_tp_rejects_wrong_entry_type():
    engine = SLTPEngine()

    with pytest.raises(TypeError):
        engine.create_sl_tp(object(), BASE_TIME)


def test_create_sl_tp_rejects_wrong_now_type():
    engine = SLTPEngine()
    entry = make_buy_entry()

    with pytest.raises(TypeError):
        engine.create_sl_tp(entry, "not a datetime")


def test_create_sl_tp_rejects_entry_not_known_yet():
    engine = SLTPEngine()
    entry = make_buy_entry(
        known_at=BASE_TIME + timedelta(minutes=5)
    )

    with pytest.raises(ValueError, match="not known"):
        engine.create_sl_tp(entry, BASE_TIME)

    assert engine.sl_tps == ()


def test_create_buy_sl_tp():
    engine = SLTPEngine()
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )

    sl_tp = engine.create_sl_tp(entry, BASE_TIME)

    assert sl_tp.entry_price == 110.0
    assert sl_tp.stop_loss == 105.0
    assert sl_tp.take_profit == 120.0
    assert sl_tp.rr_ratio == 2.0


def test_create_sell_sl_tp():
    engine = SLTPEngine()
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )

    sl_tp = engine.create_sl_tp(entry, BASE_TIME)

    assert sl_tp.entry_price == 90.0
    assert sl_tp.stop_loss == 95.0
    assert sl_tp.take_profit == 80.0
    assert sl_tp.rr_ratio == 2.0


def test_create_sl_tp_is_idempotent():
    engine = SLTPEngine()
    entry = make_buy_entry()

    first = engine.create_sl_tp(entry, BASE_TIME)
    second = engine.create_sl_tp(entry, BASE_TIME + timedelta(minutes=10))

    assert second is first
    assert engine.sl_tps == (first,)


def test_sl_tp_for_returns_created_record():
    engine = SLTPEngine()
    entry = make_buy_entry()

    sl_tp = engine.create_sl_tp(entry, BASE_TIME)

    assert engine.sl_tp_for(entry) is sl_tp


def test_sl_tp_for_returns_none_when_missing():
    engine = SLTPEngine()
    entry = make_buy_entry()

    assert engine.sl_tp_for(entry) is None


def test_sl_tp_for_rejects_wrong_entry_type():
    engine = SLTPEngine()

    with pytest.raises(TypeError):
        engine.sl_tp_for(object())


def test_sl_tps_are_append_only():
    engine = SLTPEngine()

    entry_a = make_buy_entry(identity_value="A")
    entry_b = make_sell_entry(identity_value="B")

    sl_tp_a = engine.create_sl_tp(entry_a, BASE_TIME)
    sl_tp_b = engine.create_sl_tp(entry_b, BASE_TIME)

    assert engine.sl_tps == (sl_tp_a, sl_tp_b)


def test_sl_tps_returns_tuple():
    engine = SLTPEngine()
    entry = make_buy_entry()

    engine.create_sl_tp(entry, BASE_TIME)

    assert isinstance(engine.sl_tps, tuple)


def test_sl_tps_known_at_rejects_wrong_time_type():
    engine = SLTPEngine()

    with pytest.raises(TypeError):
        engine.sl_tps_known_at("not a datetime")


def test_sl_tps_known_at_returns_known_records():
    engine = SLTPEngine()

    early = make_buy_entry(
        known_at=BASE_TIME,
        identity_value="early",
    )
    late = make_sell_entry(
        known_at=BASE_TIME + timedelta(minutes=10),
        identity_value="late",
    )

    early_sl_tp = engine.create_sl_tp(early, BASE_TIME)
    late_sl_tp = engine.create_sl_tp(
        late,
        BASE_TIME + timedelta(minutes=10),
    )

    assert engine.sl_tps_known_at(BASE_TIME) == (early_sl_tp,)
    assert engine.sl_tps_known_at(
        BASE_TIME + timedelta(minutes=10)
    ) == (early_sl_tp, late_sl_tp)


def test_process_requires_entry_engine():
    engine = SLTPEngine()

    with pytest.raises(TypeError, match="EntryEngine"):
        engine.process(BASE_TIME)


def test_process_rejects_wrong_time_type():
    entry_engine = StubEntryEngine([])
    engine = SLTPEngine(entry_engine)

    with pytest.raises(TypeError):
        engine.process("not a datetime")


def test_process_creates_only_entries_known_at_now():
    early = make_buy_entry(
        known_at=BASE_TIME,
        identity_value="early",
    )
    late = make_sell_entry(
        known_at=BASE_TIME + timedelta(minutes=10),
        identity_value="late",
    )

    entry_engine = StubEntryEngine([early, late])
    engine = SLTPEngine(entry_engine)

    created = engine.process(BASE_TIME)

    assert len(created) == 1
    assert created[0].entry is early


def test_process_creates_later_entry_when_it_becomes_known():
    early = make_buy_entry(
        known_at=BASE_TIME,
        identity_value="early",
    )
    late = make_sell_entry(
        known_at=BASE_TIME + timedelta(minutes=10),
        identity_value="late",
    )

    entry_engine = StubEntryEngine([early, late])
    engine = SLTPEngine(entry_engine)

    first = engine.process(BASE_TIME)
    second = engine.process(BASE_TIME + timedelta(minutes=10))

    assert len(first) == 1
    assert len(second) == 1
    assert second[0].entry is late
    assert len(engine.sl_tps) == 2


def test_process_does_not_duplicate_existing_sl_tp():
    entry = make_buy_entry(identity_value="same")
    entry_engine = StubEntryEngine([entry])
    engine = SLTPEngine(entry_engine)

    first = engine.process(BASE_TIME)
    second = engine.process(BASE_TIME + timedelta(minutes=10))

    assert len(first) == 1
    assert second == ()
    assert len(engine.sl_tps) == 1


def test_process_preserves_entry_order():
    first_entry = make_buy_entry(identity_value="first")
    second_entry = make_sell_entry(identity_value="second")
    third_entry = make_buy_entry(identity_value="third")

    entry_engine = StubEntryEngine(
        [first_entry, second_entry, third_entry]
    )
    engine = SLTPEngine(entry_engine)

    created = engine.process(BASE_TIME)

    assert [sl_tp.entry for sl_tp in created] == [
        first_entry,
        second_entry,
        third_entry,
    ]


def test_process_does_not_create_future_entry():
    future_entry = make_buy_entry(
        known_at=BASE_TIME + timedelta(hours=1),
        identity_value="future",
    )

    entry_engine = StubEntryEngine([future_entry])
    engine = SLTPEngine(entry_engine)

    created = engine.process(BASE_TIME)

    assert created == ()
    assert engine.sl_tps == ()


def test_invalid_geometry_does_not_modify_engine():
    invalid_entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=112.0,
        identity_value="invalid",
    )

    engine = SLTPEngine()

    with pytest.raises(ValueError):
        engine.create_sl_tp(invalid_entry, BASE_TIME)

    assert engine.sl_tps == ()
    assert engine.sl_tp_for(invalid_entry) is None


def test_engine_stores_same_sl_tp_instance_for_identity():
    identity = "same-entry"

    entry = make_buy_entry(identity_value=identity)
    engine = SLTPEngine()

    first = engine.create_sl_tp(entry, BASE_TIME)

    equivalent_entry = make_buy_entry(
        identity_value=identity,
        entry_price=110.0,
        sweep_low=105.0,
    )

    second = engine.create_sl_tp(
        equivalent_entry,
        BASE_TIME + timedelta(minutes=1),
    )

    assert second is first
    assert len(engine.sl_tps) == 1


def test_engine_does_not_mutate_entry():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
        identity_value="immutable-check",
    )

    original_price = entry.price
    original_known_at = entry.known_at

    engine = SLTPEngine()
    engine.create_sl_tp(entry, BASE_TIME)

    assert entry.price == original_price
    assert entry.known_at == original_known_at


def test_engine_does_not_read_or_store_zone_data():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
        identity_value="zone-independent",
    )

    engine = SLTPEngine()
    sl_tp = engine.create_sl_tp(entry, BASE_TIME)

    assert sl_tp.stop_loss == 95.0
    assert sl_tp.take_profit == 80.0