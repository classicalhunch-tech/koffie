"""Entry model for Koffie Strategy 1.

An Entry is the strategy-level trade entry event produced from a confirmed
Setup and its causal Confirmation.

Locked rule
-----------
Entry price is exactly the CLOSE of the qualifying completed M5 confirmation
candle.

No spread adjustment, offset, buffer, ATR adjustment, slippage, SL/TP, risk,
position sizing, or execution logic belongs in this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from koffie.strategy.models.confirmation import Confirmation
from koffie.strategy.models.setup import Setup
from koffie.strategy.models.zone import ZoneType


class EntryDirection(Enum):
    """Direction of the strategy-level Entry."""

    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True)
class Entry:
    """Immutable strategy-level entry event."""

    setup: Setup
    confirmation: Confirmation

    def __post_init__(self) -> None:
        if not isinstance(self.setup, Setup):
            raise TypeError("setup must be a Setup")

        if not isinstance(self.confirmation, Confirmation):
            raise TypeError("confirmation must be a Confirmation")

        if self.confirmation.sweep.zone != self.setup.zone:
            raise ValueError(
                "confirmation must belong to the setup's zone"
            )

        if self.confirmation.known_at < self.setup.known_at:
            raise ValueError(
                "confirmation cannot be known before the setup exists"
            )

    @property
    def price(self) -> float:
        """Exact close price of the confirmation candle."""
        return self.confirmation.candle.close

    @property
    def candle(self):
        """The qualifying completed M5 confirmation candle."""
        return self.confirmation.candle

    @property
    def direction(self) -> EntryDirection:
        """DEMAND produces LONG; SUPPLY produces SHORT."""
        if self.setup.zone_type is ZoneType.DEMAND:
            return EntryDirection.LONG
        if self.setup.zone_type is ZoneType.SUPPLY:
            return EntryDirection.SHORT

        raise ValueError(
            f"unsupported zone type for Entry: {self.setup.zone_type!r}"
        )

    @property
    def entry_time(self) -> datetime:
        """The confirmation candle close time."""
        return self.confirmation.candle.close_time

    @property
    def known_at(self) -> datetime:
        """The time at which the Entry became causally knowable."""
        return self.confirmation.known_at

    @property
    def time(self) -> datetime:
        """Compatibility alias for the Entry time."""
        return self.entry_time

    @property
    def zone(self):
        """The originating Setup zone."""
        return self.setup.zone

    @property
    def zone_type(self):
        """The originating Setup zone type."""
        return self.setup.zone_type

    @property
    def identity(self):
        """Identity derived from the causal Setup and Confirmation."""
        return self.setup.identity, self.confirmation.identity
