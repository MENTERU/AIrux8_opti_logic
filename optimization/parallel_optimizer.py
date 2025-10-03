# -*- coding: utf-8 -*-
"""
並列処理版エアコン最適化システム
================================
各ゾーンの最適化を並列実行して処理時間を短縮
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from optimization.feature_builder import OptimizationFeatureBuilder
from processing.utilities.category_mapping_loader import (
    get_category_mapping,
    normalize_candidate_values,
)
from training.model_builder import EnvPowerModels

MODE_MAPPING = get_category_mapping("A/C Mode")
FALLBACK_MODE_CODE = (
    MODE_MAPPING["FAN"] if "FAN" in MODE_MAPPING else next(iter(MODE_MAPPING.values()))
)

FAN_SPEED_MAPPING = get_category_mapping("A/C Fan Speed")
FALLBACK_FAN_CODE = (
    FAN_SPEED_MAPPING["Low"]
    if "Low" in FAN_SPEED_MAPPING
    else next(iter(FAN_SPEED_MAPPING.values()))
)


def optimize_single_zone(
    zone_name: str,
    zone_data: dict,
    models: EnvPowerModels,
    date_range: pd.DatetimeIndex,
    weather_df: pd.DataFrame,
    comfort_w: float,
    power_w: float,
) -> Tuple[str, Dict[pd.Timestamp, dict]]:
    """
    単一ゾーンの最適化を実行（並列処理用）

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
    print(f"[Parallel] Starting optimization for zone: {zone_name}")

    # Initialize feature builder
    feature_builder = OptimizationFeatureBuilder()

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
    mode_candidates = zone_data.get("mode_candidates")
    if mode_candidates is not None and not isinstance(
        mode_candidates, (list, tuple, set)
    ):
        mode_candidates = [mode_candidates]
    mode_list = normalize_candidate_values(
        "A/C Mode", mode_candidates, ("COOL", "HEAT", "FAN")
    )

    fan_candidates = zone_data.get("fan_candidates")
    if fan_candidates is not None and not isinstance(
        fan_candidates, (list, tuple, set)
    ):
        fan_candidates = [fan_candidates]
    fan_list = normalize_candidate_values(
        "A/C Fan Speed", fan_candidates, ("Low", "Medium", "High")
    )

    print(
        f"[Parallel] Zone {zone_name}: {len(sp_list)}×{len(mode_list)}×{len(fan_list)} = {len(sp_list) * len(mode_list) * len(fan_list)} combinations"
    )

    # スコア評価関数
    def eval_score(predicted_temp: float, predicted_power: float) -> float:
        if comfort_min <= predicted_temp <= comfort_max:
            penalty = 0.0
        elif predicted_temp < comfort_min:
            penalty = (comfort_min - predicted_temp) * 100
        else:
            penalty = (predicted_temp - comfort_max) * 100
        return comfort_w * penalty + power_w * predicted_power

    # 最適化実行
    z_schedule: Dict[pd.Timestamp, dict] = {}
    last_temp = float(zone_data.get("target_room_temp", 25.0))

    for i, t in enumerate(date_range):
        is_biz = start_h <= t.hour <= end_h

        # 天候データの取得
        wrow = weather_df[weather_df["datetime"] == t]
        if wrow.empty:
            ot, oh = 25.0, 60.0
        else:
            wr = wrow.iloc[0]
            ot = float(wr.get("Outdoor Temp.", wr.get("temperature C", 25.0)))
            oh = float(wr.get("Outdoor Humidity", wr.get("humidity", 60.0)))

        best = None
        best_score = np.inf

        # 全組み合わせの評価
        for sp in sp_list:
            for md in mode_list:
                for fs in fan_list:
                    # 特徴量の作成
                    base_features = {
                        "A/C Set Temperature": sp,
                        "Indoor Temp. Lag1": last_temp,
                        "A/C ON/OFF": 1 if is_biz else 0,
                        "A/C Mode": md,
                        "A/C Fan Speed": fs,
                        "Outdoor Temp.": ot,
                        "Outdoor Humidity": oh,
                        "Solar Radiation": 0.0,  # Default value
                    }

                    # Build complete feature set using feature builder
                    features_df = feature_builder.build_features(
                        base_features=base_features,
                        timestamp=t,
                        zone_name=zone_name,
                        weather_history=None,  # Could be enhanced with actual history
                        power_history=None,  # Could be enhanced with actual history
                    )

                    # Select only the features the model expects
                    feats = features_df[models.feature_cols]

                    # 予測
                    temp_pred = float(models.temp_model.predict(feats)[0])
                    power_pred = (
                        float(models.power_model.predict(feats)[0]) * unit_count
                    )

                    # スコア計算
                    score = eval_score(temp_pred, power_pred)

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

        # 結果の保存
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

    print(f"[Parallel] Zone {zone_name} completed - {len(z_schedule)} hours scheduled")
    return zone_name, z_schedule


class ParallelOptimizer:
    """並列処理版最適化クラス"""

    def __init__(
        self, master: dict, models: Dict[str, EnvPowerModels], max_workers: int = None
    ):
        self.master = master
        self.models = models
        self.max_workers = max_workers or min(len(models), mp.cpu_count())

    def optimize_day(
        self,
        date_range: pd.DatetimeIndex,
        weather_df: pd.DataFrame,
        preference: str = "balanced",
    ) -> Dict[str, Dict[pd.Timestamp, dict]]:
        """並列処理で最適化を実行"""
        print(
            f"[ParallelOptimizer] Starting parallel optimization for {len(date_range)} hours"
        )
        print(f"[ParallelOptimizer] Date range: {date_range[0]} to {date_range[-1]}")
        print(f"[ParallelOptimizer] Preference: {preference}")
        print(f"[ParallelOptimizer] Max workers: {self.max_workers}")

        # 重みの設定
        if preference == "comfort":
            comfort_w, power_w = 0.7, 0.3
        elif preference == "energy":
            comfort_w, power_w = 0.2, 0.8
        else:
            comfort_w, power_w = 0.5, 0.5

        print(f"[ParallelOptimizer] Weights - Comfort: {comfort_w}, Power: {power_w}")
        print(f"[ParallelOptimizer] Available zones: {list(self.models.keys())}")

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
                executor.submit(optimize_single_zone, *task): task[0]
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
                        f"[ParallelOptimizer] Completed {completed_count}/{len(zone_tasks)} zones: {zone_name}"
                    )
                except Exception as exc:
                    print(
                        f"[ParallelOptimizer] Zone {zone_name} generated an exception: {exc}"
                    )

        end_time = time.perf_counter()
        print(
            f"[ParallelOptimizer] Parallel optimization completed in {end_time - start_time:.2f} seconds"
        )
        print(f"[ParallelOptimizer] Optimized {len(results)} zones")

        return results
