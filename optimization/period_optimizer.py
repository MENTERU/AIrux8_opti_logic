# -*- coding: utf-8 -*-
"""
期間最適化システム
================
計算期間全体で評価する最適化システム
- 電力: 合計
- 室温: 平均（執務時間内のみ）
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from training.model_builder import EnvPowerModels


def optimize_zone_period(
    zone_name: str,
    zone_data: dict,
    models: EnvPowerModels,
    date_range: pd.DatetimeIndex,
    weather_df: pd.DataFrame,
    comfort_w: float,
    power_w: float,
) -> Tuple[str, Dict[pd.Timestamp, dict]]:
    """
    単一ゾーンの期間最適化を実行（並列処理用）

    Args:
        zone_name: ゾーン名
        zone_data: ゾーンの設定データ
        models: 予測モデル
        date_range: 最適化対象の時間範囲
        weather_df: 天候データ
        comfort_w: 快適性重み
        power_w: 電力重み

    Returns:
        (ゾーン名, 最適化結果)
    """
    print(f"[PeriodOptimizer] Starting period optimization for zone: {zone_name}")

    # ゾーン設定の取得
    start_h = int(str(zone_data.get("start_time", "07:00")).split(":")[0])
    end_h = int(str(zone_data.get("end_time", "20:00")).split(":")[0])
    comfort_min = float(zone_data.get("comfort_min", 22))
    comfort_max = float(zone_data.get("comfort_max", 24))

    # 室内機数の計算
    unit_count = 0
    for _, ou in zone_data.get("outdoor_units", {}).items():
        unit_count += len(ou.get("indoor_units", []))
    unit_count = max(unit_count, 1)

    # 候補の生成
    sp_min = int(zone_data.get("setpoint_min", 22))
    sp_max = int(zone_data.get("setpoint_max", 28))
    sp_list = list(range(sp_min, sp_max + 1))
    mode_list = zone_data.get("mode_candidates", [0, 1, 2])

    fan_candidates = zone_data.get("fan_candidates", [1, 2, 3])
    fan_mapping = {"Auto": 0, "Low": 1, "Medium": 2, "High": 3, "Top": 4}
    fan_list = []
    for fan in fan_candidates:
        if isinstance(fan, str):
            fan_list.append(fan_mapping.get(fan, 1))
        else:
            fan_list.append(int(fan))

    print(
        f"[PeriodOptimizer] Zone {zone_name}: {len(sp_list)}×{len(mode_list)}×{len(fan_list)} = {len(sp_list) * len(mode_list) * len(fan_list)} combinations"
    )

    # 天候データの準備
    weather_dict = {}
    for _, row in weather_df.iterrows():
        weather_dict[row["datetime"]] = {
            "outdoor_temp": row["Outdoor Temp."],
            "outdoor_humidity": row["Outdoor Humidity"],
            "solar_radiation": row.get("Solar Radiation", 0),
        }

    # 期間最適化の実行
    best_schedule = None
    best_score = np.inf

    # 全組み合わせを評価
    total_combinations = len(sp_list) * len(mode_list) * len(fan_list)
    print(
        f"[PeriodOptimizer] Zone {zone_name}: Evaluating {total_combinations} combinations for entire period"
    )

    for sp in sp_list:
        for md in mode_list:
            for fs in fan_list:
                # 期間全体のスケジュールを生成
                schedule = {}
                pred_temps = []
                pred_powers = []
                last_temp = float(zone_data.get("target_room_temp", 25.0))

                for timestamp in date_range:
                    is_biz = start_h <= timestamp.hour <= end_h

                    # 天候データの取得
                    weather = weather_dict.get(
                        timestamp,
                        {
                            "outdoor_temp": 25.0,
                            "outdoor_humidity": 60.0,
                            "solar_radiation": 0,
                        },
                    )

                    # 特徴量の作成
                    features = pd.DataFrame(
                        [
                            {
                                "A/C Set Temperature": sp,
                                "Indoor Temp. Lag1": last_temp,
                                "A/C ON/OFF": 1 if is_biz else 0,
                                "A/C Mode": md,
                                "A/C Fan Speed": fs,
                                "Outdoor Temp.": weather["outdoor_temp"],
                                "Outdoor Humidity": weather["outdoor_humidity"],
                                "Solar Radiation": weather["solar_radiation"],
                            }
                        ]
                    )[models.feature_cols]

                    # 予測
                    temp_pred = float(models.temp_model.predict(features)[0])
                    power_pred = (
                        float(models.power_model.predict(features)[0]) * unit_count
                    )

                    # スケジュールに追加
                    schedule[timestamp] = {
                        "set_temp": sp,
                        "mode": md,
                        "fan": fs,
                        "pred_temp": temp_pred,
                        "pred_power": power_pred,
                    }

                    pred_temps.append(temp_pred)
                    pred_powers.append(power_pred)
                    last_temp = temp_pred

                # 期間全体での評価
                # 1. 電力合計
                total_power = sum(pred_powers)

                # 2. 室温平均（執務時間内のみ）
                business_hours_temps = []
                for i, timestamp in enumerate(date_range):
                    if start_h <= timestamp.hour <= end_h:
                        business_hours_temps.append(pred_temps[i])

                avg_temp = (
                    np.mean(business_hours_temps)
                    if business_hours_temps
                    else np.mean(pred_temps)
                )

                # 3. 快適性ペナルティ（執務時間内のみ）
                comfort_penalty = 0
                for temp in business_hours_temps:
                    if temp < comfort_min:
                        comfort_penalty += (comfort_min - temp) * 100
                    elif temp > comfort_max:
                        comfort_penalty += (temp - comfort_max) * 100

                # 4. 総合スコア
                period_score = comfort_w * comfort_penalty + power_w * total_power

                # 最適解の更新
                if period_score < best_score:
                    best_score = period_score
                    best_schedule = schedule

                    print(
                        f"[PeriodOptimizer] Zone {zone_name}: New best score = {period_score:.1f} "
                        f"(Power: {total_power:.0f}W, Avg Temp: {avg_temp:.1f}°C, Penalty: {comfort_penalty:.1f})"
                    )

    print(
        f"[PeriodOptimizer] Zone {zone_name} completed - Best score: {best_score:.1f}"
    )
    return zone_name, best_schedule


class PeriodOptimizer:
    """期間最適化クラス"""

    def __init__(
        self, master: dict, models: Dict[str, EnvPowerModels], max_workers: int = None
    ):
        self.master = master
        self.models = models
        self.max_workers = max_workers or min(len(models), mp.cpu_count())

    def optimize_period(
        self,
        date_range: pd.DatetimeIndex,
        weather_df: pd.DataFrame,
        preference: str = "balanced",
    ) -> Dict[str, Dict[pd.Timestamp, dict]]:
        """期間最適化を実行"""
        print(
            f"[PeriodOptimizer] Starting period optimization for {len(date_range)} hours"
        )
        print(f"[PeriodOptimizer] Date range: {date_range[0]} to {date_range[-1]}")
        print(f"[PeriodOptimizer] Preference: {preference}")
        print(f"[PeriodOptimizer] Max workers: {self.max_workers}")

        # 重みの設定
        if preference == "comfort":
            comfort_w, power_w = 0.7, 0.3
        elif preference == "energy":
            comfort_w, power_w = 0.2, 0.8
        else:
            comfort_w, power_w = 0.5, 0.5

        print(f"[PeriodOptimizer] Weights - Comfort: {comfort_w}, Power: {power_w}")
        print(f"[PeriodOptimizer] Available zones: {list(self.models.keys())}")

        # 並列処理の準備
        zone_tasks = []
        for zone_name, models in self.models.items():
            zone_data = self.master.get("zones", {}).get(zone_name, {})
            zone_tasks.append(
                (
                    zone_name,
                    zone_data,
                    models,
                    date_range,
                    weather_df,
                    comfort_w,
                    power_w,
                )
            )

        # 並列実行
        results = {}
        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # タスクの投入
            future_to_zone = {
                executor.submit(optimize_zone_period, *task): task[0]
                for task in zone_tasks
            }

            # 結果の収集
            completed_count = 0
            for future in as_completed(future_to_zone):
                zone_name = future_to_zone[future]
                try:
                    zone_name, zone_schedule = future.result()
                    results[zone_name] = zone_schedule
                    completed_count += 1
                    print(
                        f"[PeriodOptimizer] Completed {completed_count}/{len(zone_tasks)} zones: {zone_name}"
                    )
                except Exception as exc:
                    print(
                        f"[PeriodOptimizer] Zone {zone_name} generated an exception: {exc}"
                    )

        end_time = time.perf_counter()
        print(
            f"[PeriodOptimizer] Period optimization completed in {end_time - start_time:.2f} seconds"
        )
        print(f"[PeriodOptimizer] Optimized {len(results)} zones")

        return results
