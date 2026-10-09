"""Strategy1: the coordinator that composes the existing Koffie Strategy 1 engines.
 
It contains NO trading rule of its own. Every rule stays in the engine that owns it.
It only routes completed candles to the right engines, in a causal order, and decides
whether a confirmed Setup may become a trade.
 
    H1  candle -> StructureProcessor(H1)    -> directional permission
    M15 candle -> StructureProcessor(M15)   -> BOS
                -> PivotEngine(M15)         -> PivotOutcome
                -> ZoneEngine(M15)          -> Zone (the Supply/Demand zone)
    M5  candle -> SwingEngine(M5)           -> swings
                -> LiquidityEngine(M5)      -> required liquidity (only READ downstream)
                -> SetupEngine              -> touch -> Sweep -> Confirmation -> Setup
                   (its own private SweepEngine / ConfirmationEngine do the sweep and
                    the post-sweep confirmation)
                -> EntryEngine              -> Entry (price = close of the confirming candle)
                -> SLTPEngine               -> SL (sweep candle low/high + optional ATR widening) and TP (exactly 2R)
 
Directional permission (H1 only)
--------------------------------
LONG_ONLY permits DEMAND zones (BUY), SHORT_ONLY permits SUPPLY zones (SELL),
NO_TRADE_PERMITTED permits nothing. M15 and M5 never override it. The permission is
read at the moment a Setup is confirmed, and only a Setup confirmed on THIS candle can
become a trade. A Setup confirmed while the permission did not allow it is reported as
skipped and is never traded later, so a later permission change cannot create a stale
entry. The engines keep running regardless of permission; permission only decides
whether a confirmed Setup is acted on.
 
Causality (no lookahead)
------------------------
Candles must be supplied merged in time order, by (close time, timeframe), and when
several close at the same instant the order is H1, then M15, then M5 (a candle may use
everything that had already closed when it closed, never anything later). A candle that
goes backwards in that order, overlaps the previous candle of its timeframe, or is not
closed at `now` is rejected BEFORE any engine is touched, so a rejected call changes
nothing. An M5 candle is only evaluated against zones that already existed when it
opened (`zone.created_at <= candle.open_time`); required liquidity is selected as of
the candle's open time by the SweepEngine, so a level confirmed by a candle can never be
swept by that same candle.
 
Trade creation
--------------
The SL/TP geometry is checked on the immutable Entry/SLTP value objects BEFORE the
EntryEngine is asked to create the Entry, because creating an Entry consumes its zone.
If the geometry is invalid (BUY stop not below entry, SELL stop not above entry) no
Entry is created, the zone is not consumed and the Setup is reported as skipped. When
several zones are confirmed by the same candle they are tried in the existing
`tie_break_order`; zone consumption and the one-trade-per-causal-event rule are
enforced by SetupEngine.can_create_trade, not here.
 
ATR Stop Widening (optional)
----------------------------
When stop_atr_mult > 0, stops are widened beyond the zone edge by
stop_atr_mult × ATR(14) known at entry time:
  - LONG: stop moves lower (wider risk)
  - SHORT: stop moves higher (wider risk)
This accounts for volatility and is optional (default: 0, no widening).

Not in this module: H4, spread, order flow, volume, risk, position sizing, news,
execution, flip zones, Strategy 2, or any rule that an engine does not already own.
Call `process_candle(candle, now)` with `now = candle.close_time` when replaying history.
"""
from __future__ import annotations
 
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import pandas as pd
 
from koffie.strategy.atr import known_atr
from koffie.strategy.engines.entry_engine import EntryEngine
from koffie.strategy.engines.liquidity_engine import LiquidityEngine
from koffie.strategy.engines.pivot_engine import PivotEngine
from koffie.strategy.engines.setup_engine import SetupEngine
from koffie.strategy.engines.sl_tp_engine import SLTPEngine
from koffie.strategy.engines.structure_processor import StructureProcessor
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.engines.zone_engine import ZoneEngine
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.entry import Entry, EntryDirection
from koffie.strategy.models.setup import Setup, SetupStatus, is_touch, tie_break_order
from koffie.strategy.models.sl_tp import SLTP
from koffie.strategy.models.structure import DirectionalPermission
from koffie.strategy.models.zone import Zone, ZoneType
 
# The H1 permission that allows each zone type (DEMAND = BUY, SUPPLY = SELL).
PERMISSION_FOR_ZONE_TYPE = {
    ZoneType.DEMAND: DirectionalPermission.LONG_ONLY,
    ZoneType.SUPPLY: DirectionalPermission.SHORT_ONLY,
}
 
# Same-instant feeding order: a candle may use higher-timeframe information that closed at the same time.
_FEED_RANK = {Timeframe.H1: 0, Timeframe.M15: 1, Timeframe.M5: 2}
 
 
def permission_allows(permission: DirectionalPermission, zone_type: ZoneType) -> bool:
    """True when the H1 directional permission allows trading a zone of this type."""
    if not isinstance(permission, DirectionalPermission):
        raise TypeError("permission must be a DirectionalPermission")
    if not isinstance(zone_type, ZoneType):
        raise TypeError("zone_type must be a ZoneType")
    return PERMISSION_FOR_ZONE_TYPE[zone_type] is permission
 
 
class SkipReason(Enum):
    DIRECTION_NOT_PERMITTED = "DIRECTION_NOT_PERMITTED"   # H1 permission did not allow this zone type
    NOT_TRADEABLE = "NOT_TRADEABLE"                        # zone already consumed, or this causal event already traded
    INVALID_GEOMETRY = "INVALID_GEOMETRY"                  # SL on the wrong side of the entry; no entry created
 
 
@dataclass(frozen=True)
class SkippedSetup:
    """A Setup confirmed on this candle that did NOT become a trade, and why."""
 
    setup: Setup
    reason: SkipReason
 
    def __post_init__(self) -> None:
        if not isinstance(self.setup, Setup):
            raise ValueError("setup must be a Setup")
        if not isinstance(self.reason, SkipReason):
            raise ValueError("reason must be a SkipReason")
 
 
@dataclass(frozen=True)
class Strategy1Trade:
    """The final Strategy 1 trade result: one Entry and its SL/TP."""
 
    entry: Entry
    sl_tp: SLTP
 
    def __post_init__(self) -> None:
        if not isinstance(self.entry, Entry):
            raise ValueError("entry must be an Entry")
        if not isinstance(self.sl_tp, SLTP):
            raise ValueError("sl_tp must be an SLTP")
        if self.sl_tp.entry != self.entry:
            raise ValueError("the SLTP must belong to the Entry")
 
    @property
    def setup(self) -> Setup:
        return self.entry.setup
 
    @property
    def zone(self) -> Zone:
        return self.entry.zone
 
    @property
    def direction(self) -> EntryDirection:
        return self.entry.direction
 
    @property
    def entry_price(self) -> float:
        return self.entry.price
 
    @property
    def stop_loss(self) -> float:
        return self.sl_tp.stop_loss
 
    @property
    def take_profit(self) -> float:
        return self.sl_tp.take_profit
 
    @property
    def known_at(self) -> datetime:
        return self.entry.known_at
 
 
@dataclass(frozen=True)
class Strategy1Result:
    """What ONE candle produced. Fields that do not apply to the candle's timeframe are empty.
 
    candle         the candle that was processed
    permission     the H1 directional permission after this candle
    new_zones      M15 zones created by this candle (M15 candles only)
    setup_changes  Setups this M5 candle started or whose status it changed
    trades         trades created by this M5 candle
    skipped        Setups confirmed by this M5 candle that did not become a trade
    """
 
    candle: Candle
    permission: DirectionalPermission
    new_zones: Tuple[Zone, ...] = ()
    setup_changes: Tuple[Setup, ...] = ()
    trades: Tuple[Strategy1Trade, ...] = ()
    skipped: Tuple[SkippedSetup, ...] = ()
 
 
class Strategy1:
    def __init__(
        self,
        historical_data: Optional[pd.DataFrame] = None,
        stop_atr_mult: float = 0.0,
    ) -> None:
        """Initialize Strategy1 with optional ATR stop-widening.
        
        Args:
            historical_data: Optional DataFrame with OHLC data (indexed by timestamp)
                           Required if stop_atr_mult > 0
            stop_atr_mult: Multiplier for ATR-based stop widening (0.0 = no widening)
        """
        if stop_atr_mult < 0:
            raise ValueError("stop_atr_mult must be >= 0")
        
        if stop_atr_mult > 0 and historical_data is None:
            raise ValueError(
                "historical_data (DataFrame with OHLC) is required when stop_atr_mult > 0"
            )
        
        # Compute causal ATR if stop widening is enabled
        atr_series = None
        if stop_atr_mult > 0 and historical_data is not None:
            atr_series = known_atr(historical_data, period=14)
        
        self._h1 = StructureProcessor(Timeframe.H1)
        self._m15 = StructureProcessor(Timeframe.M15)
        self._pivots = PivotEngine(Timeframe.M15)
        self._zones = ZoneEngine(Timeframe.M15)
        self._m5_swings = SwingEngine(Timeframe.M5)
        self._liquidity = LiquidityEngine(Timeframe.M5)
        self._setups = SetupEngine(self._liquidity)
        self._entries = EntryEngine(self._setups)
        self._sl_tps = SLTPEngine(
            self._entries,
            atr_series=atr_series,
            stop_atr_mult=stop_atr_mult,
        )
        self._trades: List[Strategy1Trade] = []
        self._last_key: Optional[Tuple[datetime, int]] = None
        self._last_candle: Dict[Timeframe, Candle] = {}
        # Speed-only caches (no trading rule): zones seen so far as (created_at, zone), their
        # positions, zones whose Setup sequence is still running, and zones consumed by a trade.
        self._zone_cache: List[Tuple[datetime, Zone, bool, float, float]] = []
        self._zone_index: Dict[object, int] = {}
        self._active_idx: set = set()
        self._sweep_wait_idx: set = set()          # active Setups still waiting for their Sweep
        self._consumed_idx: set = set()
 
    # ------------------------------------------------------------- read-only state
    @property
    def permission(self) -> DirectionalPermission:
        """The H1 directional permission right now."""
        return self._h1.structure.permission
 
    @property
    def zones(self) -> Tuple[Zone, ...]:
        """Every M15 Supply/Demand zone created so far, oldest first."""
        return self._zones.zones
 
    @property
    def trades(self) -> Tuple[Strategy1Trade, ...]:
        """Every trade created, in creation order. Never removed."""
        return tuple(self._trades)
 
    @property
    def h1(self) -> StructureProcessor:
        return self._h1
 
    @property
    def m15(self) -> StructureProcessor:
        return self._m15
 
    @property
    def zone_engine(self) -> ZoneEngine:
        return self._zones
 
    @property
    def liquidity_engine(self) -> LiquidityEngine:
        return self._liquidity
 
    @property
    def setup_engine(self) -> SetupEngine:
        return self._setups
 
    @property
    def entry_engine(self) -> EntryEngine:
        return self._entries
 
    @property
    def sl_tp_engine(self) -> SLTPEngine:
        return self._sl_tps
 
    # --------------------------------------------------------------------- input
    def process_candle(self, candle: Candle, now: datetime) -> Strategy1Result:
        """Run ONE completed candle (H1, M15 or M5) through the Strategy 1 chain.
 
        Raises TypeError for wrong argument types (or mixed naive/aware datetimes) and
        ValueError for an unsupported timeframe, a candle not closed at `now`, a candle
        out of (close time, timeframe) order, or one that overlaps the previous candle of
        its timeframe. These checks run before any engine is touched, so a rejected call
        changes nothing.
        """
        self._validate(candle, now)
        if candle.timeframe is Timeframe.H1:
            result = self._on_h1(candle, now)
        elif candle.timeframe is Timeframe.M15:
            result = self._on_m15(candle, now)
        else:
            result = self._on_m5(candle, now)
        self._last_key = (candle.close_time, _FEED_RANK[candle.timeframe])
        self._last_candle[candle.timeframe] = candle
        return result
 
    def _validate(self, candle: Candle, now: datetime) -> None:
        if not isinstance(candle, Candle):
            raise TypeError("candle must be a Candle")
        if not isinstance(now, datetime):
            raise TypeError("now must be a datetime")
        if candle.timeframe not in _FEED_RANK:
            raise ValueError(f"Strategy 1 does not use {candle.timeframe.value} candles")
        if not candle.is_closed_at(now):
            raise ValueError("candle is not closed yet; only completed candles may be processed")
        key = (candle.close_time, _FEED_RANK[candle.timeframe])
        if self._last_key is not None and key < self._last_key:
            raise ValueError(
                "candles must be supplied in (close time, timeframe) order: "
                "H1 before M15 before M5 when they close at the same instant"
            )
        previous = self._last_candle.get(candle.timeframe)
        if previous is not None and candle.open_time < previous.close_time:
            raise ValueError(
                f"{candle.timeframe.value} candles must not overlap or repeat "
                "(candle opens before the previous candle closed)"
            )
 
    # ------------------------------------------------------------------ per timeframe
    def _on_h1(self, candle: Candle, now: datetime) -> Strategy1Result:
        self._h1.process_candle(candle, now)
        return Strategy1Result(candle=candle, permission=self.permission)
 
    def _on_m15(self, candle: Candle, now: datetime) -> Strategy1Result:
        processed = self._m15.process_candle(candle, now)
        outcome = self._pivots.process_candle(candle, now, processed.bos)
        new_zones: Tuple[Zone, ...] = ()
        if outcome is not None:
            zone = self._zones.process_outcome(outcome, now)
            if zone is not None:
                new_zones = (zone,)
        return Strategy1Result(candle=candle, permission=self.permission, new_zones=new_zones)
 
    def _sync_zone_cache(self) -> None:
        zones = self._zones.zones            # append-only, oldest first
        for zone in zones[len(self._zone_cache):]:
            self._zone_index[zone.identity] = len(self._zone_cache)
            self._zone_cache.append(
                (zone.created_at, zone, zone.zone_type is ZoneType.DEMAND, zone.low, zone.high)
            )
 
    def _on_m5(self, candle: Candle, now: datetime) -> Strategy1Result:
        # 1. liquidity: swings confirmed by this candle become known at its close, so the
        #    SweepEngine (which selects liquidity as of the candle OPEN) cannot use them yet.
        for swing in self._m5_swings.process_candle(candle, now):
            self._liquidity.process_swing(swing, now)
 
        # 2. setups: only zones that already existed when the candle opened.
        #    SPEED: a zone can only start a Setup when the candle touches it, and can only
        #    advance a Setup that is already active; consumed zones can never start one.
        #    Every other zone would return None and change nothing, so it is skipped.
        self._sync_zone_cache()
        changes: List[Setup] = []
        active = self._active_idx
        consumed = self._consumed_idx
        open_time = candle.open_time
        sweep_wait = self._sweep_wait_idx
        c_low = candle.low
        c_high = candle.high
        for idx, (created_at, zone, is_demand, z_low, z_high) in enumerate(self._zone_cache):
            if open_time < created_at:
                continue
            if idx in consumed:
                continue
            if idx in sweep_wait:
                # A Setup still waiting for its Sweep can only sweep a level strictly beyond the
                # zone edge (DEMAND: below zone.low, SUPPLY: above zone.high). A candle that does
                # not go beyond that edge cannot sweep anything, so the engine would return None.
                if is_demand:
                    if not c_low < z_low:
                        continue
                elif not c_high > z_high:
                    continue
            elif idx not in active and not is_touch(zone, candle):
                continue
            setup = self._setups.process_candle(zone, candle, now)
            if setup is not None:
                changes.append(setup)
                status = self._setups.status_for(setup)
                if status.is_active:
                    active.add(idx)
                else:
                    active.discard(idx)
                if status is SetupStatus.WAITING_FOR_SWEEP:
                    sweep_wait.add(idx)
                else:
                    sweep_wait.discard(idx)
 
        # 3. only Setups confirmed by THIS candle can become a trade.
        confirmed_now = [
            s for s in changes
            if self._setups.status_for(s) is SetupStatus.CONFIRMED
            and self._setups.confirmation_for(s).candle == candle
        ]
        permission = self.permission
        trades: List[Strategy1Trade] = []
        skipped: List[SkippedSetup] = []
        for setup in tie_break_order(confirmed_now):
            if not permission_allows(permission, setup.zone_type):
                skipped.append(SkippedSetup(setup, SkipReason.DIRECTION_NOT_PERMITTED))
                continue
            if not self._setups.can_create_trade(setup):
                skipped.append(SkippedSetup(setup, SkipReason.NOT_TRADEABLE))
                continue
            candidate = Entry(setup, self._setups.confirmation_for(setup))
            try:
                SLTP(candidate)                       # pure geometry check; creating the Entry would consume the zone
            except ValueError:
                skipped.append(SkippedSetup(setup, SkipReason.INVALID_GEOMETRY))
                continue
            entry = self._entries.create_entry(setup, now)
            trade = Strategy1Trade(entry, self._sl_tps.create_sl_tp(entry, now))
            self._trades.append(trade)
            trades.append(trade)
            consumed_idx = self._zone_index.get(setup.zone.identity)
            if consumed_idx is not None:
                self._consumed_idx.add(consumed_idx)
 
        return Strategy1Result(
            candle=candle,
            permission=permission,
            setup_changes=tuple(changes),
            trades=tuple(trades),
            skipped=tuple(skipped),
        )
