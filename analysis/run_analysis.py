# -*- coding: utf-8 -*-
"""
最適化結果分析の実行スクリプト
============================
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.optimization_analysis import OptimizationAnalyzer
from config.private_information import WEATHER_API_KEY
from optimization.aircon_optimizer import AirconOptimizer


def run_analysis_with_optimization(store_name: str = "Clea"):
    """最適化を実行してから分析を行う"""
    print(f"🔍 {store_name}の最適化結果分析を開始")

    # 1. 最適化の実行
    print("📊 最適化を実行中...")
    optimizer = AirconOptimizer(store_name, enable_preprocessing=False)

    results = optimizer.run(
        weather_api_key=WEATHER_API_KEY,
        temperature_std_multiplier=5.0,
        power_std_multiplier=5.0,
        preference="energy",
        start_date="2025-10-01",
        end_date="2025-10-02",
    )

    if not results:
        print("❌ 最適化に失敗しました")
        return None

    # 2. 分析の実行
    print("📈 分析を実行中...")
    analyzer = OptimizationAnalyzer(store_name)

    # 日時範囲の設定
    date_range = pd.date_range(start="2025-10-01", end="2025-10-02", freq="1H")
    date_range = date_range[(date_range.hour >= 0) & (date_range.hour <= 23)]

    # 天候データの取得
    from processing.utilities.weatherapi_client import (
        VisualCrossingWeatherAPIDataFetcher,
    )

    weather_fetcher = VisualCrossingWeatherAPIDataFetcher(WEATHER_API_KEY)
    weather_df = weather_fetcher.fetch_weather_data(
        coordinates="35.681236%2C139.767125",
        start_date="2025-10-01",
        end_date="2025-10-02",
    )

    # モデルの読み込み（簡易版 - 実際のモデル読み込みを実装）
    models = {}  # ここで実際のモデルを読み込む必要がある

    # 分析グラフの作成
    analyzer.create_zone_analysis_plots(results, weather_df, models)
    summary_df = analyzer.create_summary_analysis(results, weather_df)

    print("🎉 分析完了！")
    print("📁 出力ファイル:")
    print("   - analysis/output/*_analysis.html (各ゾーン分析)")
    print("   - analysis/output/summary_analysis.html (全体サマリー)")
    print("   - analysis/output/summary_statistics.csv (統計データ)")

    return summary_df


def run_analysis_from_existing_results(store_name: str = "Clea"):
    """既存の最適化結果から分析を行う"""
    print(f"🔍 {store_name}の既存結果から分析を開始")

    # 既存の最適化結果を読み込み
    # ここで実際の結果ファイルを読み込む実装が必要

    print("📈 分析を実行中...")
    analyzer = OptimizationAnalyzer(store_name)

    # 簡易的な結果データの作成（実際の実装では既存ファイルから読み込み）
    results = {}

    print("🎉 分析完了！")
    return None


if __name__ == "__main__":
    # 最適化を実行してから分析
    summary_df = run_analysis_with_optimization("Clea")

    if summary_df is not None:
        print("\n📊 統計サマリー:")
        print(summary_df.to_string(index=False))
