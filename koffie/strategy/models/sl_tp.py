"""SL/TP model for Koffie Strategy 1.

An SLTP is the strategy-level stop-loss / take-profit pair produced from a
single Entry.

Locked rules
------------
BUY / DEMAND (Entry.direction is LONG):
    SL = exact LOW of the sweep candle:  entry.confirmation.sweep.candle.low
    R  = entry.price - SL
    TP = entry.price + 2 * R

SELL / SUPPLY (Entry.direction is SHORT):
    SL = exact HIGH of the sweep candle: entry.confirmation.sweep.candle.high
    R  = SL - entry.price
    TP = entry.price - 2 * R

No buffer, no offset, no ATR, no spread, no sweep-range fraction, no
opposite-zone target, no target-zone search, no fallback. TP is always 1:2.

The Entry price is exactly the close of the confirmation candle (Entry.price).

Geometry is validated: if the SL would be on the wrong side of the Entry
(BUY: SL >= entry.price; SELL: SL <= entry.price), the SLTP is rejected
rather than silently corrected.

This module is DATA ONLY. It contains no risk, position-sizing, execution,
MT5, news, spread, or broker logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from koffie.strategy.models.entry import Entry, EntryDirection


@dataclass(frozen=True)
class SLTP:
    """Immutable stop-loss / take-profit pair derived entirely from one Entry."""

    entry: Entry

    def __post_init__(self) -> None:
        if not isinstance(self.entry, Entry):
            raise TypeError("entry must be an Entry")
        if self.direction is EntryDirection.LONG:
            if not self.entry.price > self.stop_loss:
                raise ValueError(
                    "invalid geometry: BUY stop loss must be strictly below the entry price"
                )
        else:
            if not self.entry.price < self.stop_loss:
                raise ValueError(
                    "invalid geometry: SELL stop loss must be strictly above the entry price"
                )

    # -- derivation ----------------------------------------------------------
    @property
    def direction(self) -> EntryDirection:
        """LONG for a DEMAND zone, SHORT for a SUPPLY zone."""
        return self.entry.direction

    @property
    def entry_price(self) -> float:
        """Exact close of the confirmation candle."""
        return self.entry.price

    @property
    def sweep_candle(self):
        """The actual candle stored on the Sweep that the confirmation confirmed."""
        return self.entry.confirmation.sweep.candle

    @property
    def stop_loss(self) -> float:
        """BUY: sweep candle low. SELL: sweep candle high."""
        if self.direction is EntryDirection.LONG:
            return self.sweep_candle.low
        return self.sweep_candle.high

    @property
    def risk(self) -> float:
        """Absolute distance from entry to stop loss (always > 0 by construction)."""
        if self.direction is EntryDirection.LONG:
            return self.entry_price - self.stop_loss
        return self.stop_loss - self.entry_price

    @property
    def take_profit(self) -> float:
        """Entry shifted by exactly 2R in the trade direction."""
        if self.direction is EntryDirection.LONG:
            return self.entry_price + 2.0 * self.risk
        return self.entry_price - 2.0 * self.risk

    @property
    def reward(self) -> float:
        """Absolute distance from entry to take profit (always 2R)."""
        if self.direction is EntryDirection.LONG:
            return self.take_profit - self.entry_price
        return self.entry_price - self.take_profit

    @property
    def rr_ratio(self) -> float:
        """Reward divided by risk. Always exactly 2.0 by construction."""
        return self.reward / self.risk

    # -- timing / identity ---------------------------------------------------
    @property
    def known_at(self) -> datetime:
        """The SLTP is known exactly when its Entry is known."""
        return self.entry.known_at

    def is_known_at(self, moment: datetime) -> bool:
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")
        return moment >= self.known_at

    @property
    def identity(self):
        """Identity of the SLTP is the identity of its Entry (at most one SLTP per Entry)."""
        return self.entry.identity