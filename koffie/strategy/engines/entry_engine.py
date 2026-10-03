"""EntryEngine for Koffie Strategy 1.

Consumes confirmed Setups and creates immutable strategy-level Entry events.

Entry price is exactly the close of the qualifying completed confirmation
candle. MT5 execution, SL/TP, risk, position sizing, spread, and slippage are
outside this component.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from koffie.strategy.engines.setup_engine import SetupEngine
from koffie.strategy.models.entry import Entry
from koffie.strategy.models.setup import Setup


class EntryEngine:
    """Creates strategy-level Entries from confirmed Setups."""

    def __init__(self, setup_engine: SetupEngine) -> None:
        if not isinstance(setup_engine, SetupEngine):
            raise TypeError("setup_engine must be a SetupEngine")

        self._setup_engine = setup_engine
        self._entries: List[Entry] = []
        self._by_setup: Dict[object, Entry] = {}

    @property
    def entries(self) -> Tuple[Entry, ...]:
        """All Entries in creation order."""
        return tuple(self._entries)

    @property
    def setup_engine(self) -> SetupEngine:
        """The SetupEngine supplying confirmed Setups."""
        return self._setup_engine

    def entry_for(self, setup: Setup) -> Optional[Entry]:
        """Return the Entry for a Setup, if one exists."""
        if not isinstance(setup, Setup):
            raise TypeError("setup must be a Setup")

        return self._by_setup.get(setup.identity)

    def create_entry(self, setup: Setup, now: datetime) -> Entry:
        """Create one Entry for a confirmed Setup.

        The Entry price is exactly confirmation.candle.close.

        Repeating the same causal Setup returns the existing Entry without
        mutating engine state.
        """
        if not isinstance(setup, Setup):
            raise TypeError("setup must be a Setup")

        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")

        existing = self._by_setup.get(setup.identity)
        if existing is not None:
            return existing

        if not self._setup_engine.can_create_trade(setup):
            raise ValueError(
                "an Entry can only be created from a confirmed, "
                "unconsumed Setup"
            )

        confirmation = self._setup_engine.confirmation_for(setup)

        if confirmation is None:
            raise ValueError(
                "the confirmed Setup has no Confirmation"
            )

        if not confirmation.is_known_at(now):
            raise ValueError(
                "the Entry cannot be created before the confirmation "
                "candle has closed"
            )

        entry = Entry(
            setup=setup,
            confirmation=confirmation,
        )

        self._setup_engine.record_trade_created(setup, now)

        self._entries.append(entry)
        self._by_setup[setup.identity] = entry

        return entry

    def process(self, now: datetime) -> Tuple[Entry, ...]:
        """Create Entries for confirmed Setups known at ``now``.

        Only Entries created during this call are returned.
        """
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")

        created: List[Entry] = []

        for setup in self._setup_engine.confirmed_setups_known_at(now):
            if self.entry_for(setup) is not None:
                continue

            created.append(self.create_entry(setup, now))

        return tuple(created)
