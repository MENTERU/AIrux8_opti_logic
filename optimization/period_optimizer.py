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
    時刻別の意思決定を許容しつつ、期間全体スコア（電力合計＋快適性ペナルティ）を最小化。
    シンプルなビームサーチ（貪欲拡張＋幅優先N件保持）で探索量を制御。

    戻り値: (ゾーン名, スケジュール辞書)
    """
    print(f"[PeriodOptimizer] Starting period optimization for zone: {zone_name}")

    # パラメータ
    beam_width = 5  # 時刻ごとに保持する候補数（高速化のため削減）

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
        f"[PeriodOptimizer] Zone {zone_name}: Beam search with width={beam_width}, "
        f"candidates={len(sp_list)}×{len(mode_list)}×{len(fan_list)}"
    )

    # 天候データの準備
    weather_dict = {}
    for _, row in weather_df.iterrows():
        weather_dict[row["datetime"]] = {
            "outdoor_temp": (
                row["Outdoor Temp."]
                if "Outdoor Temp." in row
                else row.get("Outdoor Temp.", np.nan)
            ),
            "outdoor_humidity": (
                row["Outdoor Humidity"]
                if "Outdoor Humidity" in row
                else row.get("Outdoor Humidity", np.nan)
            ),
            "solar_radiation": row.get("Solar Radiation", 0),
        }

    # 初期ビーム（直前温度=目標温度、累積スコア=0）
    initial_temp = float(zone_data.get("target_room_temp", 25.0))
    BeamState = dict  # 型エイリアス
    beam: List[BeamState] = [
        {
            "last_temp": initial_temp,
            "total_power": 0.0,
            "comfort_penalty": 0.0,
            "schedule": {},
        }
    ]

    # 時間特徴の事前計算
    time_features = {
        ts: {
            "DayOfWeek": int(ts.dayofweek),
            "Hour": int(ts.hour),
            "Month": int(ts.month),
            "IsWeekend": 1 if int(ts.dayofweek) in (5, 6) else 0,
            "IsHoliday": 0,  # 期間最適化中は休日情報未連携のため0で初期化
        }
        for ts in date_range
    }

    # 時系列に沿って拡張
    for timestamp in date_range:
        is_biz = start_h <= timestamp.hour <= end_h

        expanded: List[BeamState] = []
        for state in beam:
            last_temp = float(state["last_temp"])
            total_power = float(state["total_power"])
            comfort_penalty = float(state["comfort_penalty"])
            schedule = state["schedule"]

            weather = weather_dict.get(
                timestamp,
                {
                    "outdoor_temp": 25.0,
                    "outdoor_humidity": 60.0,
                    "solar_radiation": 0,
                },
            )

            for sp in sp_list:
                for md in mode_list:
                    for fs in fan_list:
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
                                    "DayOfWeek": time_features[timestamp]["DayOfWeek"],
                                    "Hour": time_features[timestamp]["Hour"],
                                    "Month": time_features[timestamp]["Month"],
                                    "IsWeekend": time_features[timestamp]["IsWeekend"],
                                    "IsHoliday": time_features[timestamp]["IsHoliday"],
                                }
                            ]
                        )[models.feature_cols]

                        # 電力予測の条件を学習時と統一（ON/OFF状態に基づく）
                        power_prediction_onoff = 1 if is_biz else 0

                        # 予測（マルチアウトプットモデルが利用可能な場合は使用）
                        if models.multi_output_model is not None:
                            multi_pred = models.multi_output_model.predict(features)
                            temp_pred = float(multi_pred[0][0])
                            power_pred = float(multi_pred[0][1]) * unit_count
                        else:
                            temp_pred = float(models.temp_model.predict(features)[0])
                            # 電力予測：ON/OFF状態に基づいて調整
                            base_power_pred = float(
                                models.power_model.predict(features)[0]
                            )
                            # OFF状態の場合は電力予測を0に近づける
                            if power_prediction_onoff == 0:
                                power_pred = base_power_pred * 0.1  # OFF時は10%に減衰
                            else:
                                power_pred = base_power_pred * unit_count

                        # ペナルティ更新（執務時間内のみ）
                        new_penalty = comfort_penalty
                        if is_biz:
                            if temp_pred < comfort_min:
                                new_penalty += (comfort_min - temp_pred) * 100
                            elif temp_pred > comfort_max:
                                new_penalty += (temp_pred - comfort_max) * 100

                        # 新状態の作成
                        new_schedule = dict(schedule)
                        new_schedule[timestamp] = {
                            "set_temp": sp,
                            "mode": md,
                            "fan": fs,
                            "pred_temp": temp_pred,
                            "pred_power": power_pred,
                        }

                        new_state = {
                            "last_temp": temp_pred,
                            "total_power": total_power + power_pred,
                            "comfort_penalty": new_penalty,
                            "schedule": new_schedule,
                        }

                        expanded.append(new_state)

        # ビーム幅で剪定（現在までの累積スコアで評価）
        expanded.sort(
            key=lambda s: comfort_w * s["comfort_penalty"] + power_w * s["total_power"]
        )
        beam = expanded[:beam_width]

    # 最良解の選択
    best_state = min(
        beam,
        key=lambda s: comfort_w * s["comfort_penalty"] + power_w * s["total_power"],
    )

    best_score = (
        comfort_w * best_state["comfort_penalty"] + power_w * best_state["total_power"]
    )
    print(
        f"[PeriodOptimizer] Zone {zone_name} completed - Best score: {best_score:.1f}"
    )
    return zone_name, best_state["schedule"]


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
                        f"[PeriodOptimizer] Completed {completed_count}/"
                        f"{len(zone_tasks)} zones: {zone_name}"
                    )
                except Exception as exc:
                    print(
                        f"[PeriodOptimizer] Zone {zone_name} generated an exception: {exc}"
                    )

        end_time = time.perf_counter()
        print(
            f"[PeriodOptimizer] Period optimization completed in "
            f"{end_time - start_time:.2f} seconds"
        )
        print(f"[PeriodOptimizer] Optimized {len(results)} zones")

        return results
