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
from processing.utilities.helper_functions import analyze_feature_correlations
from processing.utilities.master_loader import MasterLoader
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher
from training.data_processor import DataProcessor
from training.model_builder import ModelBuilder

warnings.filterwarnings("ignore")


# =============================
# 統合ランナー
# =============================
class AirconOptimizer:
    def __init__(
        self,
        store_name: str,
        enable_preprocessing: bool = True,
        skip_aggregation: bool = False,
    ):
        self.store_name = store_name
        self.enable_preprocessing = enable_preprocessing
        self.skip_aggregation = skip_aggregation
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
        weather_p = os.path.join(
            self.proc_dir, f"weather_processed_{self.store_name}.csv"
        )
        ac = pd.read_csv(ac_p) if os.path.exists(ac_p) else None
        pm = pd.read_csv(pm_p) if os.path.exists(pm_p) else None
        weather = pd.read_csv(weather_p) if os.path.exists(weather_p) else None
        return ac, pm, weather

    def _get_weather_forecast_path(self, start_date: str, end_date: str) -> str:
        """
        Generate weather forecast file path with date range in filename

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Full path to the weather forecast file
        """
        # Format dates for filename (remove dashes)
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        filename = f"weather_forecast_{start_clean}_{end_clean}.csv"
        return os.path.join(self.plan_dir, filename)

    def _load_weather_forecast(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Load weather forecast from cached file if it exists

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Weather DataFrame if file exists, None otherwise
        """
        forecast_path = self._get_weather_forecast_path(start_date, end_date)

        if os.path.exists(forecast_path):
            print(f"[Run] Loading cached weather forecast: {forecast_path}")
            try:
                weather_df = pd.read_csv(forecast_path)

                # Convert datetime column to datetime type if it exists
                if "datetime" in weather_df.columns:
                    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
                    print(f"[Run] Converted datetime column to datetime type")

                print(f"[Run] Cached weather data loaded. Shape: {weather_df.shape}")
                return weather_df
            except Exception as e:
                print(f"[Run] Error loading cached weather data: {e}")
                return None
        else:
            print(f"[Run] No cached weather forecast found: {forecast_path}")
            return None

    def _save_weather_forecast(
        self, weather_df: pd.DataFrame, start_date: str, end_date: str
    ) -> None:
        """
        Save weather forecast to cached file with date range in filename

        Args:
            weather_df: Weather DataFrame to save
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        forecast_path = self._get_weather_forecast_path(start_date, end_date)

        try:
            os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
            weather_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
            print(f"[Run] Weather forecast cached to: {forecast_path}")
        except Exception as e:
            print(f"[Run] Error saving weather forecast: {e}")

    def _load_features_directly(self) -> Optional[pd.DataFrame]:
        """
        Load features directly from the processed CSV file, skipping aggregation
        """
        features_path = os.path.join(
            self.proc_dir, f"features_processed_{self.store_name}.csv"
        )

        if os.path.exists(features_path):
            print(f"[Run] Loading features directly from: {features_path}")
            try:
                area_df = pd.read_csv(features_path)
                print(f"[Run] Features loaded successfully. Shape: {area_df.shape}")

                if "zone" in area_df.columns:
                    zones = area_df["zone"].unique()
                    print(f"[Run] Zones found: {zones}")
                else:
                    print("[Run] Warning: No 'zone' column found in features data")

                return area_df
            except Exception as e:
                print(f"[Run] Error loading features: {e}")
                return None
        else:
            print(f"[Run] Features file not found: {features_path}")
            return None

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
            print("[Run] PM data preprocessed, checking for cached weather data...")

            # Check if weather_processed file exists
            weather_file = os.path.join(
                self.proc_dir, f"weather_processed_{self.store_name}.csv"
            )
            if os.path.exists(weather_file):
                print(f"[Run] Found cached weather data: {weather_file}")
                historical_weather_data = pd.read_csv(weather_file)
                # Convert datetime column if it exists
                if "datetime" in historical_weather_data.columns:
                    historical_weather_data["datetime"] = pd.to_datetime(
                        historical_weather_data["datetime"]
                    )
                print(
                    f"[Run] Loaded cached weather data: {len(historical_weather_data)} records"
                )
            else:
                print("[Run] No cached weather data found, fetching from API...")
                historical_weather_data = preprocessor._fetch_historical_weather(
                    ac_processed_data, pm_processed_data, weather_api_key, coordinates
                )
                print("[Run] Historical weather data fetched from API, saving...")
            print("[Run] Saving processed data...")
            preprocessor.save(
                ac_processed_data, pm_processed_data, historical_weather_data
            )
            preprocessing_end_time = time.perf_counter()
            processing_times["前処理"] = (
                preprocessing_end_time - preprocessing_start_time
            )
            print(
                f"[Run] Preprocessing completed - 処理時間: {processing_times['前処理']:.2f}秒"
            )
        else:
            print("[Run] Loading processed data...")
            ac_processed_data, pm_processed_data, historical_weather_data = (
                self._load_processed()
            )
            print("[Run] Processed data loaded")
        # 天候データ取得（実績期間＋最適化期間をカバー）
        weather_start_time = time.perf_counter()

        # 実績期間の推定（前処理済みデータから）
        actual_start_dt = None
        actual_end_dt = None
        try:
            if ac_processed_data is not None and not ac_processed_data.empty:
                ac_dt = pd.to_datetime(ac_processed_data.get("datetime"))
                actual_start_dt = (
                    ac_dt.min()
                    if actual_start_dt is None
                    else min(actual_start_dt, ac_dt.min())
                )
                actual_end_dt = (
                    ac_dt.max()
                    if actual_end_dt is None
                    else max(actual_end_dt, ac_dt.max())
                )
            if pm_processed_data is not None and not pm_processed_data.empty:
                pm_dt = pd.to_datetime(pm_processed_data.get("datetime"))
                actual_start_dt = (
                    pm_dt.min()
                    if actual_start_dt is None
                    else min(actual_start_dt, pm_dt.min())
                )
                actual_end_dt = (
                    pm_dt.max()
                    if actual_end_dt is None
                    else max(actual_end_dt, pm_dt.max())
                )
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

        # ----------------------------

        # 天候データ取得 (Cached or API)
        # ----------------------------
        weather_df = None

        # First, try to load from cached file
        weather_df = self._load_weather_forecast(start_date, end_date)

        # If no cached data found, fetch from API
        if weather_df is None and weather_api_key:
            print("[Run] No cached weather data found. Fetching from API...")
            try:
                weather_df = VisualCrossingWeatherAPIDataFetcher(
                    coordinates=coordinates,
                    start_date=start_date,
                    end_date=end_date,
                    unit="metric",
                    api_key=weather_api_key,
                ).fetch()
                print(f"[Run] Weather API result: {weather_df is not None}")
                if weather_df is not None:
                    print(f"[Run] Weather data shape: {weather_df.shape}")
                    print(f"[Run] Weather data columns: {list(weather_df.columns)}")

                    # Save the fetched data to cache
                    self._save_weather_forecast(weather_df, start_date, end_date)
            except Exception as e:
                print(f"[Run] Weather API exception: {e}")
                weather_df = None
        elif weather_df is not None:
            print("[Run] Using cached weather data - no API call needed")
        else:
            print("[Run] No weather API key provided and no cached data found")

        # 天候データの統合（履歴 + 未来）
        combined_weather_df = None
        if historical_weather_data is not None and not historical_weather_data.empty:
            print("[Run] Combining historical and future weather data...")
            if weather_df is not None and not weather_df.empty:
                # 重複を避けるため、未来の天候データから履歴期間を除外
                historical_max_date = pd.to_datetime(
                    historical_weather_data["datetime"]
                ).max()
                weather_df_filtered = weather_df[
                    pd.to_datetime(weather_df["datetime"]) > historical_max_date
                ]
                if not weather_df_filtered.empty:
                    combined_weather_df = pd.concat(
                        [historical_weather_data, weather_df_filtered],
                        ignore_index=True,
                    )
                    print(
                        f"[Run] Combined weather data: {len(historical_weather_data)} historical + {len(weather_df_filtered)} future records"
                    )
                else:
                    combined_weather_df = historical_weather_data
                    print(
                        "[Run] Using only historical weather data (no future data needed)"
                    )
            else:
                combined_weather_df = historical_weather_data
                print("[Run] Using only historical weather data")
        else:
            combined_weather_df = weather_df
            print("[Run] Using only future weather data")

        weather_end_time = time.perf_counter()
        processing_times["天候データ取得"] = weather_end_time - weather_start_time
        print(
            f"[Run] Weather data processing completed - 処理時間: {processing_times['天候データ取得']:.2f}秒"
        )

        # 制御エリア集約
        aggregation_start_time = time.perf_counter()

        if self.skip_aggregation:
            print("[Run] Skipping aggregation, loading features directly...")
            area_df = self._load_features_directly()
        else:
            print("[Run] Starting area aggregation...")
            aggregator = AreaAggregator(self.master)
            area_df = aggregator.build(
                ac_processed_data, pm_processed_data, combined_weather_df, freq=freq
            )
            area_out = os.path.join(
                self.proc_dir, f"features_processed_{self.store_name}.csv"
            )
            os.makedirs(self.proc_dir, exist_ok=True)
            area_df.to_csv(area_out, index=False, encoding="utf-8-sig")
            print(f"[Run] Area data saved to: {area_out}")
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

        # 特徴量の確認と相関の簡易レポート
        if area_df is not None:
            # Use helper function for correlation analysis
            correlation_results = analyze_feature_correlations(area_df)

            # Process features for model training
            print("[Run] Processing features for model training...")
            data_processor = DataProcessor()
            area_df = data_processor.process_features(area_df)

            # Print feature summary
            data_processor.print_feature_summary(area_df)

        # 天気予報データの出力（最適化期間のみ）
        if weather_df is not None:
            forecast_df = weather_df[
                (weather_df["datetime"] >= pd.to_datetime(start_date))
                & (weather_df["datetime"] <= pd.to_datetime(end_date))
            ].copy()

            # Save forecast with date-based filename (already cached above, but save filtered version too)
            forecast_path = self._get_weather_forecast_path(start_date, end_date)
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
        # schedule = None
        return schedule


if __name__ == "__main__":
    # 例:
    # AirconOptimizer("STORE_A").run(weather_api_key="YOUR_API_KEY")
    pass
