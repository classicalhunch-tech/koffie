"""SL/TP model for Koffie Strategy 1.

An SLTP is the strategy-level stop-loss / take-profit pair produced from a
single Entry.

Locked rules
------------
BUY / DEMAND (Entry.direction is LONG):
    Base SL = exact LOW of the sweep candle:  entry.confirmation.sweep.candle.low
    With ATR widening: SL = sweep.low - (stop_atr_mult * ATR at entry time)
    R  = entry.price - SL
    TP = entry.price + 2 * R

SELL / SUPPLY (Entry.direction is SHORT):
    Base SL = exact HIGH of the sweep candle: entry.confirmation.sweep.candle.high
    With ATR widening: SL = sweep.high + (stop_atr_mult * ATR at entry time)
    R  = SL - entry.price
    TP = entry.price - 2 * R

ATR widening
------------
If stop_atr_mult > 0:
    Widens the stop beyond the zone edge by stop_atr_mult × ATR(14) known at entry.
    - LONG: stop moves lower (wider risk)
    - SHORT: stop moves higher (wider risk)

No buffer, no offset, no spread, no sweep-range fraction, no opposite-zone target,
no target-zone search, no fallback. TP is always 1:2 (after stop widening if applied).

The Entry price is exactly the close of the confirmation candle (Entry.price).

Geometry is validated: if the SL would be on the wrong side of the Entry
(BUY: SL >= entry.price; SELL: SL <= entry.price), the SLTP is rejected
rather than silently corrected.

This module is DATA ONLY. It contains no risk, position-sizing, execution,
MT5, news, or broker logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from koffie.strategy.models.entry import Entry, EntryDirection


@dataclass(frozen=True)
class SLTP:
    """Immutable stop-loss / take-profit pair derived entirely from one Entry.
    
    Attributes:
        entry: The Entry this SLTP is derived from
        atr_value: Optional ATR(14) at entry time for stop widening
        stop_atr_mult: Multiplier for ATR-based stop widening (0.0 = no widening)
    """

    entry: Entry
    atr_value: Optional[float] = None
    stop_atr_mult: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.entry, Entry):
            raise TypeError("entry must be an Entry")
        
        if self.stop_atr_mult < 0:
            raise ValueError("stop_atr_mult must be >= 0")
        
        if self.stop_atr_mult > 0 and self.atr_value is None:
            raise ValueError(
                "atr_value is required when stop_atr_mult > 0"
            )
        
        if self.stop_atr_mult > 0 and self.atr_value <= 0:
            raise ValueError(
                "atr_value must be > 0 for stop widening"
            )
        
        # Validate geometry AFTER computing stop with ATR widening
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
    def base_stop_loss(self) -> float:
        """Stop loss before ATR widening.
        
        BUY: sweep candle low. SELL: sweep candle high.
        """
        if self.direction is EntryDirection.LONG:
            return self.sweep_candle.low
        return self.sweep_candle.high

    @property
    def stop_loss(self) -> float:
        """Stop loss after optional ATR widening.
        
        If stop_atr_mult > 0:
            LONG: stop = base_stop - (stop_atr_mult * ATR)
            SHORT: stop = base_stop + (stop_atr_mult * ATR)
        Otherwise:
            stop = base_stop_loss
        """
        base = self.base_stop_loss
        
        if self.stop_atr_mult <= 0 or self.atr_value is None:
            return base
        
        atr_adj = self.stop_atr_mult * float(self.atr_value)
        
        if self.direction is EntryDirection.LONG:
            # Widen downward for longs
            return base - atr_adj
        else:
            # Widen upward for shorts
            return base + atr_adj

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
