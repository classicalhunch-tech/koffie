from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from koffie.strategy.models.entry import Entry, EntryDirection
from koffie.strategy.models.sl_tp import SLTP


BASE_TIME = datetime(2026, 1, 1, 12, 0, 0)


class StubEntry(Entry):
    """Minimal Entry test double exposing exactly what SLTP consumes."""

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

        candle = SimpleNamespace(
            low=sweep_low,
            high=sweep_high,
        )

        sweep = SimpleNamespace(candle=candle)
        confirmation = SimpleNamespace(
            sweep=sweep,
            known_at=known_at,
        )

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


def test_requires_entry():
    with pytest.raises(TypeError):
        SLTP(None)


def test_buy_direction_is_long():
    entry = make_buy_entry()

    sl_tp = SLTP(entry)

    assert sl_tp.direction is EntryDirection.LONG


def test_sell_direction_is_short():
    entry = make_sell_entry()

    sl_tp = SLTP(entry)

    assert sl_tp.direction is EntryDirection.SHORT


def test_buy_entry_price_is_exact_entry_price():
    entry = make_buy_entry(entry_price=110.25)

    sl_tp = SLTP(entry)

    assert sl_tp.entry_price == 110.25


def test_sell_entry_price_is_exact_entry_price():
    entry = make_sell_entry(entry_price=91.75)

    sl_tp = SLTP(entry)

    assert sl_tp.entry_price == 91.75


def test_buy_stop_loss_is_exact_sweep_candle_low():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=104.25,
        sweep_high=116.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.stop_loss == 104.25


def test_sell_stop_loss_is_exact_sweep_candle_high():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_low=84.0,
        sweep_high=96.75,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.stop_loss == 96.75


def test_buy_risk_is_entry_minus_stop():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.risk == 5.0


def test_sell_risk_is_stop_minus_entry():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.risk == 5.0


def test_buy_take_profit_is_exactly_two_risk():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.take_profit == 120.0


def test_sell_take_profit_is_exactly_two_risk():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.take_profit == 80.0


def test_buy_reward_is_two_risk():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.reward == 10.0


def test_sell_reward_is_two_risk():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.reward == 10.0


def test_buy_rr_is_exactly_two():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.rr_ratio == 2.0


def test_sell_rr_is_exactly_two():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.rr_ratio == 2.0


def test_buy_sl_must_be_strictly_below_entry():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=110.0,
    )

    with pytest.raises(ValueError, match="BUY stop loss"):
        SLTP(entry)


def test_buy_sl_above_entry_is_rejected():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=112.0,
    )

    with pytest.raises(ValueError, match="BUY stop loss"):
        SLTP(entry)


def test_sell_sl_must_be_strictly_above_entry():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=90.0,
    )

    with pytest.raises(ValueError, match="SELL stop loss"):
        SLTP(entry)


def test_sell_sl_below_entry_is_rejected():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=88.0,
    )

    with pytest.raises(ValueError, match="SELL stop loss"):
        SLTP(entry)


def test_buy_uses_exact_low_without_buffer():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.stop_loss == 105.0
    assert sl_tp.stop_loss != 105.1
    assert sl_tp.stop_loss != 104.9


def test_sell_uses_exact_high_without_buffer():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.stop_loss == 95.0
    assert sl_tp.stop_loss != 95.1
    assert sl_tp.stop_loss != 94.9


def test_buy_tp_has_no_zone_target_dependency():
    entry = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
        sweep_high=150.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.take_profit == 120.0


def test_sell_tp_has_no_zone_target_dependency():
    entry = make_sell_entry(
        entry_price=90.0,
        sweep_low=40.0,
        sweep_high=95.0,
    )

    sl_tp = SLTP(entry)

    assert sl_tp.take_profit == 80.0


def test_sl_tp_is_immutable():
    entry = make_buy_entry()

    sl_tp = SLTP(entry)

    with pytest.raises(Exception):
        sl_tp.entry = entry


def test_known_at_matches_entry_known_at():
    known_at = BASE_TIME + timedelta(minutes=5)
    entry = make_buy_entry(known_at=known_at)

    sl_tp = SLTP(entry)

    assert sl_tp.known_at == known_at


def test_is_known_at_requires_datetime():
    entry = make_buy_entry()
    sl_tp = SLTP(entry)

    with pytest.raises(TypeError):
        sl_tp.is_known_at("not a datetime")


def test_is_known_at_before_known_time_is_false():
    known_at = BASE_TIME + timedelta(minutes=5)
    entry = make_buy_entry(known_at=known_at)

    sl_tp = SLTP(entry)

    assert sl_tp.is_known_at(BASE_TIME) is False


def test_is_known_at_at_known_time_is_true():
    known_at = BASE_TIME + timedelta(minutes=5)
    entry = make_buy_entry(known_at=known_at)

    sl_tp = SLTP(entry)

    assert sl_tp.is_known_at(known_at) is True


def test_is_known_at_after_known_time_is_true():
    known_at = BASE_TIME + timedelta(minutes=5)
    entry = make_buy_entry(known_at=known_at)

    sl_tp = SLTP(entry)

    assert sl_tp.is_known_at(known_at + timedelta(minutes=1)) is True


def test_identity_is_entry_identity():
    identity = ("entry", 123)
    entry = make_buy_entry(identity_value=identity)

    sl_tp = SLTP(entry)

    assert sl_tp.identity == identity


def test_buy_risk_changes_only_when_sweep_low_changes():
    entry_a = make_buy_entry(
        entry_price=110.0,
        sweep_low=105.0,
    )
    entry_b = make_buy_entry(
        entry_price=110.0,
        sweep_low=103.0,
    )

    sl_tp_a = SLTP(entry_a)
    sl_tp_b = SLTP(entry_b)

    assert sl_tp_a.risk == 5.0
    assert sl_tp_b.risk == 7.0


def test_sell_risk_changes_only_when_sweep_high_changes():
    entry_a = make_sell_entry(
        entry_price=90.0,
        sweep_high=95.0,
    )
    entry_b = make_sell_entry(
        entry_price=90.0,
        sweep_high=97.0,
    )

    sl_tp_a = SLTP(entry_a)
    sl_tp_b = SLTP(entry_b)

    assert sl_tp_a.risk == 5.0
    assert sl_tp_b.risk == 7.0