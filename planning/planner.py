import os
from typing import Dict, List

import pandas as pd


# =============================
# STEP4: 出力（制御区分別 & 室内機別）
# =============================
class Planner:
    def __init__(self, store_name: str, master: dict):
        self.store_name = store_name
        self.master = master

    @staticmethod
    def _mode_text(n: int) -> str:
        return {0: "COOL", 1: "DEHUM", 2: "FAN", 3: "HEAT"}.get(n, "FAN")

    def export(self, schedule: Dict[str, Dict[pd.Timestamp, dict]], out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        # 日付
        any_ts = None
        for z in schedule:
            if schedule[z]:
                any_ts = list(schedule[z].keys())[0]
                break
        if any_ts is None:
            from datetime import datetime

            any_ts = pd.Timestamp(pd.Timestamp.now().date())
        date_str = pd.Timestamp(any_ts).strftime("%Y%m%d")

        # 制御区分別
        rows = []
        for t in sorted({ts for z in schedule for ts in schedule[z].keys()}):
            rec = {"Date Time": t.strftime("%Y/%m/%d %H:%M")}
            for z, zs in schedule.items():
                s = zs.get(t, {})
                rec[f"{z}_OnOFF"] = "ON" if s else "OFF"
                rec[f"{z}_Mode"] = self._mode_text(s.get("mode", 2)) if s else "FAN"
                rec[f"{z}_SetTemp"] = s.get("set_temp", 25) if s else 25
                rec[f"{z}_FanSpeed"] = s.get("fan", 1) if s else 1
            rows.append(rec)
        ctrl_df = pd.DataFrame(rows)
        ctrl_path = os.path.join(out_dir, f"control_type_schedule_{date_str}.csv")
        ctrl_df.to_csv(ctrl_path, index=False, encoding="utf-8-sig")

        # 室内機別
        unit_rows = []
        zones = self.master.get("zones", {})
        zone_to_units: Dict[str, List[str]] = {}
        for z, zinfo in zones.items():
            units = []
            for _, ou in zinfo.get("outdoor_units", {}).items():
                units.extend(ou.get("indoor_units", []))
            zone_to_units[z] = list(dict.fromkeys(units))
        for t in sorted({ts for z in schedule for ts in schedule[z].keys()}):
            rec = {"Date Time": t.strftime("%Y/%m/%d %H:%M")}
            for z, units in zone_to_units.items():
                s = schedule.get(z, {}).get(t, {})
                for u in units:
                    rec[f"{u}_OnOFF"] = "ON" if s else "OFF"
                    rec[f"{u}_Mode"] = self._mode_text(s.get("mode", 2)) if s else "FAN"
                    rec[f"{u}_SetTemp"] = s.get("set_temp", 25) if s else 25
                    rec[f"{u}_FanSpeed"] = s.get("fan", 1) if s else 1
            unit_rows.append(rec)
        unit_df = pd.DataFrame(unit_rows)
        unit_path = os.path.join(out_dir, f"unit_schedule_{date_str}.csv")
        unit_df.to_csv(unit_path, index=False, encoding="utf-8-sig")

        print(f"[Planner] control schedule: {ctrl_path}")
        print(f"[Planner] unit schedule: {unit_path}")
