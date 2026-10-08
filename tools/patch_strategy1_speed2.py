"""Second speed fix for koffie/strategy/strategy_1.py (makes strategy_1.py.bak2 first).
Must be run AFTER patch_strategy1_speed.py."""
import shutil
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "koffie" / "strategy" / "strategy_1.py"
text = PATH.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in text else "\n"


def fix(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\n", nl)


REPLACEMENTS = [
    (
        "        self._zone_cache: List[Tuple[datetime, Zone]] = []",
        "        self._zone_cache: List[Tuple[datetime, Zone, bool, float, float]] = []",
    ),
    (
        "        self._active_idx: set = set()",
        "        self._active_idx: set = set()\n"
        "        self._sweep_wait_idx: set = set()          # active Setups still waiting for their Sweep",
    ),
    (
        "            self._zone_cache.append((zone.created_at, zone))",
        "            self._zone_cache.append(\n"
        "                (zone.created_at, zone, zone.zone_type is ZoneType.DEMAND, zone.low, zone.high)\n"
        "            )",
    ),
    (
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
        "        sweep_wait = self._sweep_wait_idx\n"
        "        c_low = candle.low\n"
        "        c_high = candle.high\n"
        "        for idx, (created_at, zone, is_demand, z_low, z_high) in enumerate(self._zone_cache):\n"
        "            if open_time < created_at:\n"
        "                continue\n"
        "            if idx in consumed:\n"
        "                continue\n"
        "            if idx in sweep_wait:\n"
        "                # A Setup still waiting for its Sweep can only sweep a level strictly beyond the\n"
        "                # zone edge (DEMAND: below zone.low, SUPPLY: above zone.high). A candle that does\n"
        "                # not go beyond that edge cannot sweep anything, so the engine would return None.\n"
        "                if is_demand:\n"
        "                    if not c_low < z_low:\n"
        "                        continue\n"
        "                elif not c_high > z_high:\n"
        "                    continue\n"
        "            elif idx not in active and not is_touch(zone, candle):\n"
        "                continue\n"
        "            setup = self._setups.process_candle(zone, candle, now)\n"
        "            if setup is not None:\n"
        "                changes.append(setup)\n"
        "                status = self._setups.status_for(setup)\n"
        "                if status.is_active:\n"
        "                    active.add(idx)\n"
        "                else:\n"
        "                    active.discard(idx)\n"
        "                if status is SetupStatus.WAITING_FOR_SWEEP:\n"
        "                    sweep_wait.add(idx)\n"
        "                else:\n"
        "                    sweep_wait.discard(idx)",
    ),
]

for old, new in REPLACEMENTS:
    count = text.count(fix(old))
    if count != 1:
        sys.exit(f"STOPPED: expected 1 match, found {count} for:\n{old[:120]}\nNothing was changed.")

shutil.copyfile(PATH, str(PATH) + ".bak2")
for old, new in REPLACEMENTS:
    text = text.replace(fix(old), fix(new))
PATH.write_bytes(text.encode("utf-8"))
print("Patched. Backup saved as strategy_1.py.bak2")