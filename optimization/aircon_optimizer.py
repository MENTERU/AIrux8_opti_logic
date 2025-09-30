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
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from optimization.optimizer import Optimizer
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
        self.plan_dir = os.path.join(
            get_data_path("output_data_path", use_remote_paths=True), store_name
        )
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
        coordinates: str = "35.681236%2C139.767125",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "1H",
        preference: str = "balanced",
    ):
        if self.master is None:
            print("[Run] マスタ未読込")
            return None
        # STEP1: 前処理
        if self.enable_preprocessing:
            print("[Run] Starting preprocessing...")
            prep = DataPreprocessor(self.store_name)
            print("[Run] DataPreprocessor created, loading raw data...")
            ac_raw, pm_raw = prep.load_raw()
            print("[Run] Raw data loaded, preprocessing AC data...")
            ac = prep.preprocess_ac(ac_raw, 5.0)
            print("[Run] AC data preprocessed, preprocessing PM data...")
            pm = prep.preprocess_pm(pm_raw, 5.0)
            print("[Run] PM data preprocessed, saving...")
            prep.save(ac, pm)
            print("[Run] Preprocessing completed")
        else:
            print("[Run] Loading processed data...")
            ac, pm = self._load_processed()
            print("[Run] Processed data loaded")
        # 天候
        if start_date is None or end_date is None:
            # 今日〜明日
            today = pd.Timestamp.today().normalize()
            start_date = today.strftime("%Y-%m-%d")
            end_date = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"[Run] Weather API Key provided: {weather_api_key is not None}")
        if weather_api_key:
            print(f"[Run] Weather API Key: {weather_api_key[:10]}...")
        else:
            print("[Run] No Weather API Key provided")

        print(f"[Run] Date range: {start_date} to {end_date}")
        print(f"[Run] Coordinates: {coordinates}")

        weather_df = None
        if weather_api_key:
            print("[Run] Attempting to fetch weather data from API...")
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
            except Exception as e:
                print(f"[Run] Weather API exception: {e}")
                weather_df = None

        if weather_df is None:
            print("[Run] Using synthetic weather data (fallback)")
            # フォールバック（正弦で擬似生成）
            rng = pd.date_range(start=start_date, end=end_date, freq=freq)
            weather_df = pd.DataFrame(
                {
                    "datetime": rng,
                    "Outdoor Temp.": 23
                    + 5 * np.sin(np.linspace(0, 2 * np.pi, len(rng))),
                    "Outdoor Humidity": 60
                    + 15 * np.sin(np.linspace(0, 2 * np.pi, len(rng))),
                }
            )
            print(f"[Run] Generated synthetic weather: {weather_df.shape}")

        # 制御エリア集約
        print("[Run] Starting area aggregation...")
        aggregator = AreaAggregator(self.master)
        area_df = aggregator.build(ac, pm, weather_df, freq=freq)
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

        # STEP2: 予測モデル
        print("[Run] Starting model training...")
        builder = ModelBuilder(self.store_name)
        models = builder.train_by_zone(area_df, self.master)
        print(f"[Run] Model training completed. Models created: {len(models)}")
        if not models:
            print("[Run] モデル作成不可（データ不足）")
            return None

        # STEP3: 最適化
        date_range = pd.date_range(
            start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=freq
        )
        date_range = date_range[(date_range.hour >= 0) & (date_range.hour <= 23)]
        opt = Optimizer(self.master, models)
        schedule = opt.optimize_day(date_range, weather_df, preference=preference)

        # STEP4: 出力
        Planner(self.store_name, self.master).export(schedule, self.plan_dir)
        return schedule


if __name__ == "__main__":
    # 例:
    # AirconOptimizer("STORE_A").run(weather_api_key="YOUR_API_KEY")
    pass
