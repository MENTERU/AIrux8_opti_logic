from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from processing.utilities.category_mapping_loader import (
    get_category_mapping,
    get_inverse_category_mapping,
    normalize_candidate_values,
)
from training.model_builder import EnvPowerModels

MODE_MAPPING = get_category_mapping("A/C Mode")
MODE_INVERSE_MAPPING = get_inverse_category_mapping("A/C Mode")
FALLBACK_MODE_LABEL = (
    "FAN" if "FAN" in MODE_MAPPING else next(iter(MODE_MAPPING.keys()))
)
FALLBACK_MODE_CODE = MODE_MAPPING[FALLBACK_MODE_LABEL]

FAN_SPEED_MAPPING = get_category_mapping("A/C Fan Speed")
FALLBACK_FAN_LABEL = (
    "Low" if "Low" in FAN_SPEED_MAPPING else next(iter(FAN_SPEED_MAPPING.keys()))
)
FALLBACK_FAN_CODE = FAN_SPEED_MAPPING[FALLBACK_FAN_LABEL]


# =============================
# STEP3: 最適化（制御区分毎）
# =============================
class Optimizer:
    def __init__(self, master: dict, models: Dict[str, EnvPowerModels]):
        self.m = master
        self.models = models

    @staticmethod
    def _mode_name(n: int) -> str:
        return MODE_INVERSE_MAPPING.get(n, FALLBACK_MODE_LABEL)

    def _eval_score(
        self,
        comfort_weight: float,
        power_weight: float,
        predicted_temp: float,
        comfort_min: float,
        comfort_max: float,
        predicted_power: float,
    ) -> float:
        # 快適性ペナルティを単純化
        if comfort_min <= predicted_temp <= comfort_max:
            penalty = 0.0
        elif predicted_temp < comfort_min:
            penalty = (comfort_min - predicted_temp) * 100
        else:
            penalty = (predicted_temp - comfort_max) * 100
        return comfort_weight * penalty + power_weight * predicted_power

    def _gen_candidates(self, z: str) -> Tuple[List[int], List[int], List[int]]:
        ctrl = self.m.get("zones", {}).get(z, {})
        sp_min = int(ctrl.get("setpoint_min", 22))
        sp_max = int(ctrl.get("setpoint_max", 28))
        setpoints = list(range(sp_min, sp_max + 1))
        mode_candidates = ctrl.get("mode_candidates")
        if mode_candidates is not None and not isinstance(
            mode_candidates, (list, tuple, set)
        ):
            mode_candidates = [mode_candidates]
        modes = normalize_candidate_values(
            "A/C Mode", mode_candidates, ("COOL", "HEAT", "FAN")
        )

        fan_candidates = ctrl.get("fan_candidates")
        if fan_candidates is not None and not isinstance(
            fan_candidates, (list, tuple, set)
        ):
            fan_candidates = [fan_candidates]
        fans = normalize_candidate_values(
            "A/C Fan Speed", fan_candidates, ("Low", "Medium", "High")
        )

        return setpoints, modes, fans

    def optimize_day(
        self,
        date_range: pd.DatetimeIndex,
        weather_df: pd.DataFrame,
        preference: str = "balanced",
    ) -> Dict[str, Dict[pd.Timestamp, dict]]:
        """preference: 'comfort'|'energy'|'balanced' で重み変更"""
        print(f"[Optimizer] Starting optimization for {len(date_range)} hours")
        print(f"[Optimizer] Date range: {date_range[0]} to {date_range[-1]}")
        print(f"[Optimizer] Preference: {preference}")

        if preference == "comfort":
            comfort_w, power_w = 0.7, 0.3
        elif preference == "energy":
            comfort_w, power_w = 0.2, 0.8
        else:
            comfort_w, power_w = 0.5, 0.5

        print(f"[Optimizer] Weights - Comfort: {comfort_w}, Power: {power_w}")
        print(f"[Optimizer] Available models: {list(self.models.keys())}")

        results: Dict[str, Dict[pd.Timestamp, dict]] = {}
        for z, pack in self.models.items():
            print(f"\n[Optimizer] Processing zone: {z}")

            ctrl = self.m.get("zones", {}).get(z, {})
            start_h = int(str(ctrl.get("start_time", "07:00")).split(":")[0])
            end_h = int(str(ctrl.get("end_time", "20:00")).split(":")[0])
            comfort_min = float(ctrl.get("comfort_min", 22))
            comfort_max = float(ctrl.get("comfort_max", 24))

            print(f"[Optimizer] Zone {z} settings:")
            print(f"  - Business hours: {start_h}:00 to {end_h}:00")
            print(f"  - Comfort range: {comfort_min}°C to {comfort_max}°C")

            unit_count = 0
            for _, ou in ctrl.get("outdoor_units", {}).items():
                unit_count += len(ou.get("indoor_units", []))
            unit_count = max(unit_count, 1)
            print(f"  - Indoor units: {unit_count}")

            sp_list, mode_list, fan_list = self._gen_candidates(z)
            print(f"  - Temperature candidates: {sp_list}")
            print(f"  - Mode candidates: {mode_list}")
            print(f"  - Fan candidates: {fan_list}")
            print(
                f"  - Total combinations per hour: {len(sp_list) * len(mode_list) * len(fan_list)}"
            )

            z_schedule: Dict[pd.Timestamp, dict] = {}
            last_temp = float(ctrl.get("target_room_temp", 25.0))
            print(f"  - Initial temperature: {last_temp}°C")

            for i, t in enumerate(date_range):
                if i % 6 == 0:  # Log every 6 hours
                    print(
                        f"[Optimizer] Zone {z} - Processing hour {i+1}/{len(date_range)}: {t}"
                    )

                is_biz = start_h <= t.hour <= end_h
                wrow = weather_df[weather_df["datetime"] == t]
                if wrow.empty:
                    ot, oh = 25.0, 60.0
                    if i % 6 == 0:
                        print(
                            f"    - No weather data for {t}, using defaults: {ot}°C, {oh}%"
                        )
                else:
                    wr = wrow.iloc[0]
                    ot = float(wr.get("Outdoor Temp.", wr.get("temperature C", 25.0)))
                    oh = float(wr.get("Outdoor Humidity", wr.get("humidity", 60.0)))
                    if i % 6 == 0:
                        print(f"    - Weather: {ot}°C, {oh}% humidity")

                best = None
                best_score = np.inf
                combinations_tested = 0
                for sp in sp_list:
                    for md in mode_list:
                        for fs in fan_list:
                            combinations_tested += 1

                            feats = pd.DataFrame(
                                [
                                    {
                                        "A/C Set Temperature": sp,
                                        "Indoor Temp. Lag1": last_temp,
                                        "A/C ON/OFF": 1 if is_biz else 0,
                                        "A/C Mode": md,
                                        "A/C Fan Speed": fs,
                                        "Outdoor Temp.": ot,
                                        "Outdoor Humidity": oh,
                                    }
                                ]
                            )[pack.feature_cols]

                            temp_pred = float(pack.temp_model.predict(feats)[0])
                            power_pred = (
                                float(pack.power_model.predict(feats)[0]) * unit_count
                            )  # 室内機数で補正

                            score = self._eval_score(
                                comfort_w,
                                power_w,
                                temp_pred,
                                comfort_min,
                                comfort_max,
                                power_pred,
                            )

                            if score < best_score:
                                best_score = score
                                best = {
                                    "set_temp": sp,
                                    "mode": md,
                                    "fan": fs,
                                    "pred_temp": temp_pred,
                                    "pred_power": power_pred,
                                    "score": score,
                                }

                if i % 6 == 0:  # Log every 6 hours
                    print(f"    - Tested {combinations_tested} combinations")
                    print(
                        f"    - Best: Temp={best['set_temp']}°C, Mode={best['mode']}, Fan={best['fan']}"
                    )
                    print(
                        f"    - Predicted: {best['pred_temp']:.1f}°C, {best['pred_power']:.0f}W, Score={best['score']:.1f}"
                    )
            z_schedule[t] = (
                best
                if best is not None
                else {
                    "set_temp": 25,
                    "mode": FALLBACK_MODE_CODE,
                    "fan": FALLBACK_FAN_CODE,
                    "pred_temp": last_temp,
                    "pred_power": 0.0,
                    "score": 9e9,
                }
            )
            last_temp = z_schedule[t]["pred_temp"]

            results[z] = z_schedule
            print(f"[Optimizer] Zone {z} completed - {len(z_schedule)} hours scheduled")

        # self._save_results_to_csv(results) # デバッグ用

        print(f"\n[Optimizer] Optimization completed for {len(results)} zone\n")
        return results

    def _save_results_to_csv(self, results: Dict[str, Dict[pd.Timestamp, dict]]):
        """
        Convert optimization results to DataFrame and save as CSV for inspection
        最適化結果をDataFrameに変換し、確認用にCSVとして保存する
        """
        rows = []

        for zone_name, zone_schedule in results.items():
            for timestamp, settings in zone_schedule.items():
                row = {
                    "zone": zone_name,
                    "datetime": timestamp,
                    "set_temp": settings.get("set_temp", 25),
                    "mode": settings.get("mode", FALLBACK_MODE_CODE),
                    "fan": settings.get("fan", FALLBACK_FAN_CODE),
                    "pred_temp": settings.get("pred_temp", 25.0),
                    "pred_power": settings.get("pred_power", 0.0),
                    "score": settings.get("score", 0.0),
                }
                rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Sort by zone and datetime
        df = df.sort_values(["zone", "datetime"])

        # Save to CSV
        output_path = "optimization_results_inspection.csv"
        df.to_csv(output_path, index=False)
        print(f"[Optimizer] Results saved to: {output_path}")
        print(f"[Optimizer] DataFrame shape: {df.shape}")
        print(f"[Optimizer] Sample data:")
        print(df.head(10))
