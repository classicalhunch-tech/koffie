"""StructureProcessor: the caller that runs ONE closed candle through the engines.

There was no production caller before this module: SwingEngine, StructureEngine,
BOSEngine and CHOCHEngine each work on their own and were only ever wired
together by the test harnesses. This is the smallest piece that does that wiring
for ONE timeframe. It contains no rules of its own; every rule stays in the
engine that owns it.

Per closed candle, in this order:

    1. SwingEngine.process_candle      -> newly confirmed swings
    2. StructureEngine.process_swings  -> updated StructureSnapshot
    3. CHOCHEngine.process_candle      -> CHOCH event or None (detection only)
    4. BOSEngine.process_candle        -> BOS event or None (same snapshot)
    5. if a CHOCH was returned:
           structure.enter_revaluating(candle.close_time)

Step 5 is the ONLY place the structure moves to REVALUATING. CHOCHEngine never
does it. A CHOCH does not itself establish a new trend: the state stays
REVALUATING until later confirmed swings form a directional pair, which
StructureEngine decides on its own.

BOS and CHOCH are separate events and are returned separately, never merged.
The "CHOCH wins over BOS" rule (Option A) stays inside BOSEngine: BOS receives
the same snapshot CHOCH saw, so a candle that qualifies for both yields the
CHOCH and no BOS. This class does not decide it.

Only closed candles are accepted (each engine enforces it), so the transition is
made only after the breaking candle has closed, and `at` is its close time.

Failure behaviour: every input check happens in SwingEngine.process_candle,
before anything is changed, so a rejected candle leaves all four engines exactly
as they were.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

from koffie.strategy.engines.bos_engine import BOSEngine
from koffie.strategy.engines.choch_engine import CHOCHEngine
from koffie.strategy.engines.structure_engine import StructureEngine
from koffie.strategy.engines.swing_engine import SwingEngine
from koffie.strategy.models.bos import BOS
from koffie.strategy.models.candle import Candle, Timeframe
from koffie.strategy.models.choch import CHOCH
from koffie.strategy.models.structure import StructureSnapshot, StructureTransition
from koffie.strategy.models.swing import Swing


@dataclass(frozen=True)
class CandleResult:
    """What one closed candle produced. Every field is separate on purpose.

    swings                newly confirmed swings (zero, one or two)
    structure_transitions state changes made by those swings (directional pairs)
    choch                 the CHOCH event, or None
    bos                   the BOS event, or None (never set together with choch)
    revaluating           the CHOCH transition into REVALUATING, or None
    snapshot              the structure after everything above
    """

    swings: Tuple[Swing, ...]
    structure_transitions: Tuple[StructureTransition, ...]
    choch: Optional[CHOCH]
    bos: Optional[BOS]
    revaluating: Optional[StructureTransition]
    snapshot: StructureSnapshot


class StructureProcessor:
    def __init__(self, timeframe: Timeframe) -> None:
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe")
        self._timeframe = timeframe
        self._swing = SwingEngine(timeframe)
        self._structure = StructureEngine(timeframe)
        self._choch = CHOCHEngine(timeframe)
        self._bos = BOSEngine(timeframe)

    # ------------------------------------------------------------------
    # Read-only access to the engines it drives
    # ------------------------------------------------------------------
    @property
    def timeframe(self) -> Timeframe:
        return self._timeframe

    @property
    def swing_engine(self) -> SwingEngine:
        return self._swing

    @property
    def structure(self) -> StructureEngine:
        return self._structure

    @property
    def choch_engine(self) -> CHOCHEngine:
        return self._choch

    @property
    def bos_engine(self) -> BOSEngine:
        return self._bos

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def process_candle(self, candle: Candle, now: datetime) -> CandleResult:
        """Run one CLOSED candle through the whole flow and return what it produced.

        Raises whatever the engines raise (TypeError for wrong argument types
        or mixed naive/aware datetimes, ValueError for a wrong timeframe, an
        unfinished, repeated or overlapping candle). Nothing is changed when
        an exception is raised.
        """
        swings = self._swing.process_candle(candle, now)            # validates first
        structure_transitions = self._structure.process_swings(swings)
        snapshot = self._structure.snapshot

        choch = self._choch.process_candle(candle, now, snapshot)   # detection only
        bos = self._bos.process_candle(candle, now, snapshot)       # same snapshot

        revaluating = None
        if choch is not None:
            revaluating = self._structure.enter_revaluating(candle.close_time)

        return CandleResult(
            swings=tuple(swings),
            structure_transitions=tuple(structure_transitions),
            choch=choch,
            bos=bos,
            revaluating=revaluating,
            snapshot=self._structure.snapshot,
        )