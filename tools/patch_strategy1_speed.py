"""Applies the speed fix to koffie/strategy/strategy_1.py (makes strategy_1.py.bak first)."""
import shutil
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "koffie" / "strategy" / "strategy_1.py"
text = PATH.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in text else "\n"


def fix(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\n", nl)


REPLACEMENTS = [
    # 1. import is_touch
    (
        "from koffie.strategy.models.setup import Setup, SetupStatus, tie_break_order",
        "from koffie.strategy.models.setup import Setup, SetupStatus, is_touch, tie_break_order",
    ),
    # 2. new fields
    (
        "        self._last_candle: Dict[Timeframe, Candle] = {}",
        "        self._last_candle: Dict[Timeframe, Candle] = {}\n"
        "        # Speed-only caches (no trading rule): zones seen so far as (created_at, zone), their\n"
        "        # positions, zones whose Setup sequence is still running, and zones consumed by a trade.\n"
        "        self._zone_cache: List[Tuple[datetime, Zone]] = []\n"
        "        self._zone_index: Dict[object, int] = {}\n"
        "        self._active_idx: set = set()\n"
        "        self._consumed_idx: set = set()",
    ),
    # 3. cache sync helper placed right before _on_m5
    (
        "    def _on_m5(self, candle: Candle, now: datetime) -> Strategy1Result:",
        "    def _sync_zone_cache(self) -> None:\n"
        "        zones = self._zones.zones            # append-only, oldest first\n"
        "        for zone in zones[len(self._zone_cache):]:\n"
        "            self._zone_index[zone.identity] = len(self._zone_cache)\n"
        "            self._zone_cache.append((zone.created_at, zone))\n"
        "\n"
        "    def _on_m5(self, candle: Candle, now: datetime) -> Strategy1Result:",
    ),
    # 4. the slow loop
    (
        "        # 2. setups: only zones that already existed when the candle opened.\n"
        "        changes: List[Setup] = []\n"
        "        for zone in self._zones.zones:\n"
        "            if candle.open_time < zone.created_at:\n"
        "                continue\n"
        "            setup = self._setups.process_candle(zone, candle, now)\n"
        "            if setup is not None:\n"
        "                changes.append(setup)",
        "        # 2. setups: only zones that already existed when the candle opened.\n"
        "        #    SPEED: a zone can only start a Setup when the candle touches it, and can only\n"
        "        #    advance a Setup that is already active; consumed zones can never start one.\n"
        "        #    Every other zone would return None and change nothing, so it is skipped.\n"
        "        self._sync_zone_cache()\n"
        "        changes: List[Setup] = []\n"
        "        active = self._active_idx\n"
        "        consumed = self._consumed_idx\n"
        "        open_time = candle.open_time\n"
        "        for idx, (created_at, zone) in enumerate(self._zone_cache):\n"
        "            if open_time < created_at:\n"
        "                continue\n"
        "            if idx in consumed:\n"
        "                continue\n"
        "            if idx not in active and not is_touch(zone, candle):\n"
        "                continue\n"
        "            setup = self._setups.process_candle(zone, candle, now)\n"
        "            if setup is not None:\n"
        "                changes.append(setup)\n"
        "                if self._setups.status_for(setup).is_active:\n"
        "                    active.add(idx)\n"
        "                else:\n"
        "                    active.discard(idx)",
    ),
    # 5. mark the zone consumed when a trade is created
    (
        "            self._trades.append(trade)\n"
        "            trades.append(trade)",
        "            self._trades.append(trade)\n"
        "            trades.append(trade)\n"
        "            consumed_idx = self._zone_index.get(setup.zone.identity)\n"
        "            if consumed_idx is not None:\n"
        "                self._consumed_idx.add(consumed_idx)",
    ),
]

for old, new in REPLACEMENTS:
    count = text.count(fix(old))
    if count != 1:
        sys.exit(f"STOPPED: expected 1 match, found {count} for:\n{old[:120]}\nNothing was changed.")

shutil.copyfile(PATH, str(PATH) + ".bak")
for old, new in REPLACEMENTS:
    text = text.replace(fix(old), fix(new))
PATH.write_bytes(text.encode("utf-8"))
print("Patched. Backup saved as strategy_1.py.bak")