# -*- coding: utf-8 -*-
"""
エアコン最適化システム（STEP1〜STEP4一貫版 / 再考リファクタ）
============================================================
要点（ご要望を反映）
- IF（インタフェース）と前処理、天気予報取得、マスタ情報の活用を最優先で尊重
- 1時間粒度をデフォルトに、予測粒度は可変
- マスタJSONから制御エリア（zones）、室外機/室内機、負荷率(load_share)、目標温度や範囲を取得
- 実績（空調・電力・天候）を整備して、制御エリア単位へ集約
  - 空調: 設定条件は最頻値、室内温度は平均
  - 電力: 室外機の消費電力×負荷率の合計
  - 天候: エリア間で共通
- 予測モデル: 制御エリア別に環境（室温/湿度）と電力を学習
- 最適化: 制御区分毎、パターン探索→環境は平均、電力は合計（電力は室内機数で補正）
- 出力: 制御区分別 & 室内機別（制御区分の運転条件を室内機に展開）

ディレクトリ前提（config.utils.get_data_path を使用）
- raw_data_path/<store> に ac-control-*.csv, ac-power-meter-*.csv
- processed_data_path/<store> に前処理済みCSVを出力
- models_path/<store> に学習済みモデルを保存
- master/MASTER_<store>.json にマスタ
- planning/<store> に制御スケジュールCSV
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from optimization.optimizer import Optimizer
from optimization.parallel_optimizer import ParallelOptimizer
from optimization.period_optimizer import PeriodOptimizer
from planning.planner import Planner
from processing.aggregator import AreaAggregator
from processing.preprocessor import DataPreprocessor
from processing.utilities.master_loader import MasterLoader
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher
from training.model_builder import ModelBuilder

warnings.filterwarnings("ignore")


# =============================
# 統合ランナー
# =============================
class AirconOptimizer:
    def __init__(self, store_name: str, enable_preprocessing: bool = True):
        self.store_name = store_name
        self.enable_preprocessing = enable_preprocessing
        self.master = MasterLoader(store_name).load()
        from config.utils import get_data_path

        self.proc_dir = os.path.join(get_data_path("processed_data_path"), store_name)
        self.plan_dir = os.path.join(get_data_path("output_data_path"), store_name)
        os.makedirs(self.plan_dir, exist_ok=True)

    def _load_processed(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        ac_p = os.path.join(
            self.proc_dir, f"ac_control_processed_{self.store_name}.csv"
        )
        pm_p = os.path.join(
            self.proc_dir, f"power_meter_processed_{self.store_name}.csv"
        )
        ac = pd.read_csv(ac_p) if os.path.exists(ac_p) else None
        pm = pd.read_csv(pm_p) if os.path.exists(pm_p) else None
        return ac, pm

    def run(
        self,
        weather_api_key: Optional[str] = None,
        coordinates: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "1H",
        preference: str = "balanced",
        temperature_std_multiplier: float = 5.0,
        power_std_multiplier: float = 5.0,
    ):
        if self.master is None:
            print("[Run] マスタ未読込")
            return None

        # 処理時間計測開始
        total_start_time = time.perf_counter()
        processing_times = {}

        # 座標情報をマスタから取得（デフォルト値も設定）
        if coordinates is None:
            coordinates = self.master.get("store_info", {}).get(
                "coordinates", "35.681236%2C139.767125"
            )
            print(f"[Run] Using coordinates from master: {coordinates}")
        else:
            print(f"[Run] Using provided coordinates: {coordinates}")

        # STEP1: 前処理
        if self.enable_preprocessing:
            preprocessing_start_time = time.perf_counter()
            print("[Run] Starting preprocessing...")
            print(f"[Run] Temperature std multiplier: {temperature_std_multiplier}")
            print(f"[Run] Power std multiplier: {power_std_multiplier}")
            preprocessor = DataPreprocessor(self.store_name)
            print("[Run] DataPreprocessor created, loading raw data...")
            ac_raw_data, pm_raw_data = preprocessor.load_raw()
            print("[Run] Raw data loaded, preprocessing AC data...")
            ac_processed_data = preprocessor.preprocess_ac(
                ac_raw_data, temperature_std_multiplier
            )
            print("[Run] AC data preprocessed, preprocessing PM data...")
            pm_processed_data = preprocessor.preprocess_pm(
                pm_raw_data, power_std_multiplier
            )
            print("[Run] PM data preprocessed, saving...")
            preprocessor.save(ac_processed_data, pm_processed_data)
            preprocessing_end_time = time.perf_counter()
            processing_times["前処理"] = (
                preprocessing_end_time - preprocessing_start_time
            )
            print(
                f"[Run] Preprocessing completed - 処理時間: {processing_times['前処理']:.2f}秒"
            )
        else:
            print("[Run] Loading processed data...")
            ac_processed_data, pm_processed_data = self._load_processed()
            print("[Run] Processed data loaded")
        # 天候データ取得（実績期間＋最適化期間をカバー）
        weather_start_time = time.perf_counter()

        # 実績期間の推定（前処理済みデータから）
        actual_start_dt = None
        actual_end_dt = None
        try:
            if ac_processed_data is not None and not ac_processed_data.empty:
                ac_dt = pd.to_datetime(ac_processed_data.get("datetime"))
                actual_start_dt = ac_dt.min() if actual_start_dt is None else min(actual_start_dt, ac_dt.min())
                actual_end_dt = ac_dt.max() if actual_end_dt is None else max(actual_end_dt, ac_dt.max())
            if pm_processed_data is not None and not pm_processed_data.empty:
                pm_dt = pd.to_datetime(pm_processed_data.get("datetime"))
                actual_start_dt = pm_dt.min() if actual_start_dt is None else min(actual_start_dt, pm_dt.min())
                actual_end_dt = pm_dt.max() if actual_end_dt is None else max(actual_end_dt, pm_dt.max())
        except Exception:
            # 無視（実績期間が取れない場合は最適化期間のみ）
            pass

        # 最適化期間の既定
        if start_date is None or end_date is None:
            today = pd.Timestamp.today().normalize()
            start_date = today.strftime("%Y-%m-%d")
            end_date = (today + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

        # 実績期間と最適化期間を統合した取得レンジ
        combined_start_dt = pd.to_datetime(start_date)
        combined_end_dt = pd.to_datetime(end_date)
        if actual_start_dt is not None:
            combined_start_dt = min(combined_start_dt, actual_start_dt.normalize())
        if actual_end_dt is not None:
            combined_end_dt = max(combined_end_dt, actual_end_dt.normalize())
        combined_start_date = combined_start_dt.strftime("%Y-%m-%d")
        combined_end_date = combined_end_dt.strftime("%Y-%m-%d")

        print(f"[Run] Weather API Key provided: {weather_api_key is not None}")
        if weather_api_key:
            print(f"[Run] Weather API Key: {weather_api_key[:10]}...")
        else:
            print("[Run] No Weather API Key provided")

        print(f"[Run] Date range (optimization): {start_date} to {end_date}")
        print(
            f"[Run] Date range (weather fetch combined): {combined_start_date} to {combined_end_date}"
        )
        print(f"[Run] Coordinates: {coordinates}")

        weather_df = None
        if weather_api_key:
            print("[Run] Attempting to fetch weather data from API...")
            try:
                # 月次チャンクで分割取得（APIコスト回避）
                cur = combined_start_dt
                chunks = []
                while cur <= combined_end_dt:
                    chunk_start = cur
                    # 月末まで or combined_end_dt まで
                    next_month = (chunk_start.replace(day=1) + pd.offsets.MonthEnd(1)).to_pydatetime()
                    chunk_end = pd.Timestamp(min(next_month, combined_end_dt))
                    s = chunk_start.strftime("%Y-%m-%d")
                    e = chunk_end.strftime("%Y-%m-%d")
                    print(f"[Weather] Fetching chunk: {s} -> {e}")
                    try:
                        chunk_df = VisualCrossingWeatherAPIDataFetcher(
                            coordinates=coordinates,
                            start_date=s,
                            end_date=e,
                            unit="metric",
                            api_key=weather_api_key,
                        ).fetch()
                        if chunk_df is not None and not chunk_df.empty:
                            chunks.append(chunk_df)
                    except Exception as ce:
                        print(f"[Weather] Chunk fetch failed: {ce}")
                    # 次の開始は翌月1日
                    cur = (chunk_end + pd.Timedelta(days=1)).normalize()
                if chunks:
                    weather_df = pd.concat(chunks, axis=0, ignore_index=True)
                    # 重複除去
                    if "datetime" in weather_df.columns:
                        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"]).dt.floor("H")
                        weather_df = weather_df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
                print(f"[Run] Weather API result: {weather_df is not None}")
                if weather_df is not None:
                    print(f"[Run] Weather data shape: {weather_df.shape}")
                    print(f"[Run] Weather data columns: {list(weather_df.columns)}")
            except Exception as e:
                print(f"[Run] Weather API exception: {e}")
                weather_df = None

        weather_end_time = time.perf_counter()
        processing_times["天候データ取得"] = weather_end_time - weather_start_time
        print(
            f"[Run] Weather data processing completed - 処理時間: {processing_times['天候データ取得']:.2f}秒"
        )

        # 制御エリア集約
        aggregation_start_time = time.perf_counter()
        print("[Run] Starting area aggregation...")
        aggregator = AreaAggregator(self.master)
        area_df = aggregator.build(
            ac_processed_data, pm_processed_data, weather_df, freq=freq
        )
        print(
            f"[Run] Area aggregation completed. Shape: {area_df.shape if area_df is not None else 'None'}"
        )

        if area_df is not None and not area_df.empty:
            print(f"[Run] Area data columns: {list(area_df.columns)}")
            print(
                f"[Run] Zones found: {area_df['zone'].unique() if 'zone' in area_df.columns else 'No zone column'}"
            )
            print(
                f"[Run] Adjusted power data: {area_df['adjusted_power'].notna().sum() if 'adjusted_power' in area_df.columns else 'No adjusted_power column'}"
            )

        area_out = os.path.join(
            self.proc_dir, f"features_processed_{self.store_name}.csv"
        )
        os.makedirs(self.proc_dir, exist_ok=True)
        area_df.to_csv(area_out, index=False, encoding="utf-8-sig")
        print(f"[Run] Area data saved to: {area_out}")
        
        # 天気予報データの出力（最適化期間のみ）
        if weather_df is not None:
            forecast_df = weather_df[
                (weather_df["datetime"] >= pd.to_datetime(start_date))
                & (weather_df["datetime"] <= pd.to_datetime(end_date))
            ].copy()
            forecast_path = os.path.join(
                get_data_path("output_data_path"), store_name, "weather_forecast.csv"
            )
            os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
            forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
            print(f"[Run] Weather forecast saved to: {forecast_path}")

        aggregation_end_time = time.perf_counter()
        processing_times["エリア集約"] = aggregation_end_time - aggregation_start_time
        print(
            f"[Run] Area aggregation completed - 処理時間: {processing_times['エリア集約']:.2f}秒"
        )

        # STEP2: 予測モデル
        model_training_start_time = time.perf_counter()
        print("[Run] Starting model training...")
        builder = ModelBuilder(self.store_name)
        models = builder.train_by_zone(area_df, self.master)
        model_training_end_time = time.perf_counter()
        processing_times["モデル学習"] = (
            model_training_end_time - model_training_start_time
        )
        print(
            f"[Run] Model training completed. Models created: {len(models)} - 処理時間: {processing_times['モデル学習']:.2f}秒"
        )
        if not models:
            print("[Run] モデル作成不可（データ不足）")
            return None

        # STEP3: 最適化（並列処理版）
        optimization_start_time = time.perf_counter()
        date_range = pd.date_range(
            start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=freq
        )
        date_range = date_range[(date_range.hour >= 0) & (date_range.hour <= 23)]

        # 期間最適化版を使用（電力合計、室温平均で評価）
        opt = PeriodOptimizer(self.master, models, max_workers=6)  # 6ゾーン分
        schedule = opt.optimize_period(date_range, weather_df, preference=preference)
        optimization_end_time = time.perf_counter()
        processing_times["最適化"] = optimization_end_time - optimization_start_time
        print(
            f"[Run] Optimization completed - 処理時間: {processing_times['最適化']:.2f}秒"
        )

        # STEP4: 出力
        output_start_time = time.perf_counter()
        Planner(self.store_name, self.master).export(schedule, self.plan_dir)
        output_end_time = time.perf_counter()
        processing_times["計画出力"] = output_end_time - output_start_time
        print(
            f"[Run] Planning output completed - 処理時間: {processing_times['計画出力']:.2f}秒"
        )

        # 総処理時間の表示
        total_end_time = time.perf_counter()
        processing_times["総処理時間"] = total_end_time - total_start_time

        print(f"\n{'='*60}")
        print("📊 処理時間サマリー")
        print(f"{'='*60}")
        for process_name, duration in processing_times.items():
            if process_name != "総処理時間":
                percentage = (duration / processing_times["総処理時間"]) * 100
                print(f"{process_name:12}: {duration:6.2f}秒 ({percentage:5.1f}%)")
        print(f"{'='*60}")
        print(f"{'総処理時間':12}: {processing_times['総処理時間']:6.2f}秒 (100.0%)")
        print(f"{'='*60}")

        return schedule


if __name__ == "__main__":
    # 例:
    # AirconOptimizer("STORE_A").run(weather_api_key="YOUR_API_KEY")
    pass
