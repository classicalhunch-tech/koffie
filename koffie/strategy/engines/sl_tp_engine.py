"""SLTPEngine for Koffie Strategy 1.

Produces immutable SLTP records from Entry objects. Reads only. Never consumes
a zone, never records a trade, never touches Risk, Validation or Execution.

Locked behaviour
----------------
- One SLTP per Entry, keyed by Entry.identity. Supplying the same Entry again
  returns the stored SLTP and changes nothing.
- An Entry with invalid geometry (BUY stop >= entry price; SELL stop <= entry
  price) is rejected with ValueError, not silently corrected.
- SL/TP never reads zones, liquidity levels, pivots, protected swings, or
  opposite zones. The only references are the Entry, its Confirmation, its
  Sweep, and the Sweep's candle.
- Optional ATR-based stop widening: if stop_atr_mult > 0, widen stops beyond
  the zone edge by stop_atr_mult × ATR(14) known at entry time.

If an EntryEngine is injected, process(now) creates SLTPs for all Entries
known at now. Otherwise create_sl_tp(entry, now) is the only entry point.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from koffie.strategy.engines.entry_engine import EntryEngine
from koffie.strategy.models.entry import Entry
from koffie.strategy.models.sl_tp import SLTP


class SLTPEngine:
    def __init__(
        self,
        entry_engine: Optional[EntryEngine] = None,
        atr_series: Optional[pd.Series] = None,
        stop_atr_mult: float = 0.0,
    ) -> None:
        """Initialize SLTPEngine with optional ATR stop-widening.
        
        Args:
            entry_engine: Optional EntryEngine to auto-create SLTPs
            atr_series: Optional pd.Series indexed by timestamp with ATR(14) values
            stop_atr_mult: Multiplier for ATR-based stop widening (0.0 = no widening)
                          Requires atr_series if > 0
        """
        if entry_engine is not None and not isinstance(entry_engine, EntryEngine):
            raise TypeError("entry_engine must be an EntryEngine or None")
        
        if stop_atr_mult < 0:
            raise ValueError("stop_atr_mult must be >= 0")
        
        if stop_atr_mult > 0 and atr_series is None:
            raise ValueError(
                "atr_series is required when stop_atr_mult > 0"
            )
        
        if atr_series is not None and not isinstance(atr_series, pd.Series):
            raise TypeError("atr_series must be a pd.Series or None")

        self._entry_engine = entry_engine
        self._atr_series = atr_series
        self._stop_atr_mult = stop_atr_mult
        self._sl_tps: List[SLTP] = []
        self._by_entry: Dict[object, SLTP] = {}

    # ------------------------------------------------------------- read-only state

    @property
    def entry_engine(self) -> Optional[EntryEngine]:
        return self._entry_engine
    
    @property
    def atr_series(self) -> Optional[pd.Series]:
        return self._atr_series
    
    @property
    def stop_atr_mult(self) -> float:
        return self._stop_atr_mult

    @property
    def sl_tps(self) -> Tuple[SLTP, ...]:
        """Every SLTP created, in acceptance order. Never removed."""
        return tuple(self._sl_tps)

    def sl_tp_for(self, entry: Entry) -> Optional[SLTP]:
        """The SLTP recorded for this Entry, or None."""
        if not isinstance(entry, Entry):
            raise TypeError("entry must be an Entry")
        return self._by_entry.get(entry.identity)

    def sl_tps_known_at(self, moment: datetime) -> Tuple[SLTP, ...]:
        """SLTPs whose Entry is known by moment, in acceptance order."""
        if not isinstance(moment, datetime):
            raise TypeError("moment must be a datetime")

        return tuple(
            sl_tp
            for sl_tp in self._sl_tps
            if sl_tp.is_known_at(moment)
        )

    # --------------------------------------------------------------------- input
    
    def _get_atr_at_time(self, timestamp: datetime) -> Optional[float]:
        """Get causal ATR value (known at this timestamp's open).
        
        Returns None if atr_series is not available or no value exists.
        """
        if self._atr_series is None:
            return None
        
        try:
            # Use asof() to get the most recent known value up to this timestamp
            atr_val = self._atr_series.asof(timestamp)
            if pd.isna(atr_val):
                return None
            return float(atr_val)
        except (KeyError, IndexError, TypeError):
            return None

    def create_sl_tp(self, entry: Entry, now: datetime) -> SLTP:
        """Create or return the SLTP for one Entry.

        Idempotent per Entry.identity. Raises TypeError for wrong argument
        types and ValueError when the Entry is not yet known or has invalid
        SL geometry. Nothing changes when an exception is raised.
        """
        if not isinstance(entry, Entry):
            raise TypeError("entry must be an Entry")

        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")

        if now < entry.known_at:
            raise ValueError("the Entry is not known yet at `now`")

        existing = self._by_entry.get(entry.identity)
        if existing is not None:
            return existing

        # Get ATR value if stop widening is enabled
        atr_val = None
        if self._stop_atr_mult > 0:
            atr_val = self._get_atr_at_time(entry.known_at)
            if atr_val is None:
                raise ValueError(
                    f"ATR value not available at entry time {entry.known_at} "
                    f"(required for stop_atr_mult={self._stop_atr_mult})"
                )

        sl_tp = SLTP(
            entry=entry,
            atr_value=atr_val,
            stop_atr_mult=self._stop_atr_mult,
        )

        # Commit only after construction succeeded.
        self._sl_tps.append(sl_tp)
        self._by_entry[entry.identity] = sl_tp

        return sl_tp

    def process(self, now: datetime) -> Tuple[SLTP, ...]:
        """Create SLTPs for all Entries known at now.

        Requires an injected EntryEngine.

        Returns only the SLTPs created during this call.
        """
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")

        if self._entry_engine is None:
            raise TypeError("EntryEngine is required for process()")

        created: List[SLTP] = []

        for entry in self._entry_engine.entries:
            if entry.known_at <= now and self.sl_tp_for(entry) is None:
                created.append(self.create_sl_tp(entry, now))

        return tuple(created)
