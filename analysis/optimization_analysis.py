# -*- coding: utf-8 -*-
"""
最適化結果の分析・可視化ツール
================================
制御エリアごとの最適化根拠をグラフで表示
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimization.parallel_optimizer import ParallelOptimizer
from processing.utilities.master_loader import MasterLoader
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher


class OptimizationAnalyzer:
    """最適化結果の分析・可視化クラス"""

    def __init__(self, store_name: str):
        self.store_name = store_name
        self.master = MasterLoader(store_name).load()

    def load_optimization_results(
        self,
        date_range: pd.DatetimeIndex,
        weather_df: pd.DataFrame,
        models: dict,
        preference: str = "energy",
    ) -> dict:
        """最適化結果を取得"""
        print(f"[Analysis] Loading optimization results for {self.store_name}")

        # 並列最適化を実行
        optimizer = ParallelOptimizer(self.master, models, max_workers=6)
        results = optimizer.optimize_day(date_range, weather_df, preference=preference)

        return results

    def create_zone_analysis_plots(
        self,
        results: dict,
        weather_df: pd.DataFrame,
        models: dict,
        output_dir: str = "analysis/output",
    ):
        """各ゾーンの分析グラフを作成"""
        os.makedirs(output_dir, exist_ok=True)

        # 天候データの準備
        weather_dict = {}
        for _, row in weather_df.iterrows():
            weather_dict[row["datetime"]] = {
                "outdoor_temp": row["Outdoor Temp."],
                "outdoor_humidity": row["Outdoor Humidity"],
            }

        for zone_name, zone_schedule in results.items():
            print(f"[Analysis] Creating analysis for zone: {zone_name}")

            # データの準備
            timestamps = []
            set_temps = []
            modes = []
            fan_speeds = []
            pred_temps = []
            pred_powers = []
            scores = []
            outdoor_temps = []
            outdoor_humidities = []

            for timestamp, settings in zone_schedule.items():
                timestamps.append(timestamp)
                set_temps.append(settings["set_temp"])
                modes.append(settings["mode"])
                fan_speeds.append(settings["fan"])
                pred_temps.append(settings["pred_temp"])
                pred_powers.append(settings["pred_power"])
                scores.append(settings["score"])

                # 天候データの取得
                weather = weather_dict.get(
                    timestamp, {"outdoor_temp": 25.0, "outdoor_humidity": 60.0}
                )
                outdoor_temps.append(weather["outdoor_temp"])
                outdoor_humidities.append(weather["outdoor_humidity"])

            # ゾーン設定の取得
            zone_config = self.master.get("zones", {}).get(zone_name, {})
            comfort_min = float(zone_config.get("comfort_min", 22))
            comfort_max = float(zone_config.get("comfort_max", 24))
            target_temp = float(zone_config.get("target_room_temp", 23))

            # サブプロットの作成
            fig = make_subplots(
                rows=4,
                cols=2,
                subplot_titles=[
                    f"{zone_name} - 温度設定と予測室温",
                    f"{zone_name} - 電力消費予測",
                    f"{zone_name} - 運転モード",
                    f"{zone_name} - ファン速度",
                    f"{zone_name} - 外気温度",
                    f"{zone_name} - 外気湿度",
                    f"{zone_name} - 快適性評価",
                    f"{zone_name} - 最適化スコア",
                ],
                specs=[
                    [{"secondary_y": True}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # 1. 温度設定と予測室温
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=set_temps,
                    name="設定温度",
                    line=dict(color="blue", width=2),
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pred_temps,
                    name="予測室温",
                    line=dict(color="red", width=2),
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )
            # 快適性範囲の表示
            fig.add_hline(
                y=comfort_min,
                line_dash="dash",
                line_color="green",
                annotation_text=f"快適下限({comfort_min}°C)",
                row=1,
                col=1,
            )
            fig.add_hline(
                y=comfort_max,
                line_dash="dash",
                line_color="green",
                annotation_text=f"快適上限({comfort_max}°C)",
                row=1,
                col=1,
            )
            fig.add_hline(
                y=target_temp,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"目標温度({target_temp}°C)",
                row=1,
                col=1,
            )

            # 2. 電力消費予測
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=pred_powers,
                    name="予測電力",
                    line=dict(color="purple", width=2),
                    mode="lines+markers",
                    fill="tonexty",
                ),
                row=1,
                col=2,
            )

            # 3. 運転モード
            mode_names = {0: "COOL", 1: "DEHUM", 2: "FAN", 3: "HEAT"}
            mode_colors = {0: "blue", 1: "cyan", 2: "green", 3: "red"}
            for mode_val in set(modes):
                mode_timestamps = [
                    t for t, m in zip(timestamps, modes) if m == mode_val
                ]
                mode_y = [m for m in modes if m == mode_val]
                fig.add_trace(
                    go.Scatter(
                        x=mode_timestamps,
                        y=mode_y,
                        name=f"モード: {mode_names.get(mode_val, mode_val)}",
                        mode="markers",
                        marker=dict(color=mode_colors.get(mode_val, "gray"), size=8),
                    ),
                    row=2,
                    col=1,
                )

            # 4. ファン速度
            fan_names = {0: "Auto", 1: "Low", 2: "Medium", 3: "High", 4: "Top"}
            fan_colors = {
                0: "gray",
                1: "lightblue",
                2: "blue",
                3: "darkblue",
                4: "navy",
            }
            for fan_val in set(fan_speeds):
                fan_timestamps = [
                    t for t, f in zip(timestamps, fan_speeds) if f == fan_val
                ]
                fan_y = [f for f in fan_speeds if f == fan_val]
                fig.add_trace(
                    go.Scatter(
                        x=fan_timestamps,
                        y=fan_y,
                        name=f"ファン: {fan_names.get(fan_val, fan_val)}",
                        mode="markers",
                        marker=dict(color=fan_colors.get(fan_val, "gray"), size=8),
                    ),
                    row=2,
                    col=2,
                )

            # 5. 外気温度
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=outdoor_temps,
                    name="外気温度",
                    line=dict(color="orange", width=2),
                    mode="lines+markers",
                ),
                row=3,
                col=1,
            )

            # 6. 外気湿度
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=outdoor_humidities,
                    name="外気湿度",
                    line=dict(color="brown", width=2),
                    mode="lines+markers",
                ),
                row=3,
                col=2,
            )

            # 7. 快適性評価
            comfort_scores = []
            for pred_temp in pred_temps:
                if comfort_min <= pred_temp <= comfort_max:
                    comfort_scores.append(0)  # 快適範囲内
                elif pred_temp < comfort_min:
                    comfort_scores.append(
                        (comfort_min - pred_temp) * 100
                    )  # 寒すぎるペナルティ
                else:
                    comfort_scores.append(
                        (pred_temp - comfort_max) * 100
                    )  # 暑すぎるペナルティ

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=comfort_scores,
                    name="快適性ペナルティ",
                    line=dict(color="red", width=2),
                    mode="lines+markers",
                ),
                row=4,
                col=1,
            )

            # 8. 最適化スコア
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=scores,
                    name="最適化スコア",
                    line=dict(color="darkgreen", width=2),
                    mode="lines+markers",
                ),
                row=4,
                col=2,
            )

            # レイアウトの設定
            fig.update_layout(
                title=f"{zone_name} 最適化結果分析",
                height=1200,
                showlegend=True,
                template="plotly_white",
            )

            # 軸ラベルの設定
            fig.update_xaxes(title_text="時間", row=4, col=1)
            fig.update_xaxes(title_text="時間", row=4, col=2)
            fig.update_yaxes(title_text="温度 (°C)", row=1, col=1)
            fig.update_yaxes(title_text="電力 (W)", row=1, col=2)
            fig.update_yaxes(title_text="モード", row=2, col=1)
            fig.update_yaxes(title_text="ファン速度", row=2, col=2)
            fig.update_yaxes(title_text="外気温度 (°C)", row=3, col=1)
            fig.update_yaxes(title_text="外気湿度 (%)", row=3, col=2)
            fig.update_yaxes(title_text="ペナルティ", row=4, col=1)
            fig.update_yaxes(title_text="スコア", row=4, col=2)

            # ファイル保存
            output_file = os.path.join(output_dir, f"{zone_name}_analysis.html")
            fig.write_html(output_file)
            print(f"[Analysis] Saved analysis plot: {output_file}")

    def create_summary_analysis(
        self,
        results: dict,
        weather_df: pd.DataFrame,
        output_dir: str = "analysis/output",
    ):
        """全体サマリー分析を作成"""
        print("[Analysis] Creating summary analysis")

        # データの準備
        all_data = []
        for zone_name, zone_schedule in results.items():
            zone_config = self.master.get("zones", {}).get(zone_name, {})
            comfort_min = float(zone_config.get("comfort_min", 22))
            comfort_max = float(zone_config.get("comfort_max", 24))

            for timestamp, settings in zone_schedule.items():
                weather = weather_df[weather_df["datetime"] == timestamp]
                outdoor_temp = (
                    weather["Outdoor Temp."].iloc[0] if not weather.empty else 25.0
                )
                outdoor_humidity = (
                    weather["Outdoor Humidity"].iloc[0] if not weather.empty else 60.0
                )

                all_data.append(
                    {
                        "zone": zone_name,
                        "timestamp": timestamp,
                        "set_temp": settings["set_temp"],
                        "mode": settings["mode"],
                        "fan_speed": settings["fan"],
                        "pred_temp": settings["pred_temp"],
                        "pred_power": settings["pred_power"],
                        "score": settings["score"],
                        "outdoor_temp": outdoor_temp,
                        "outdoor_humidity": outdoor_humidity,
                        "comfort_min": comfort_min,
                        "comfort_max": comfort_max,
                        "in_comfort_zone": comfort_min
                        <= settings["pred_temp"]
                        <= comfort_max,
                    }
                )

        df = pd.DataFrame(all_data)

        # 全体サマリーグラフ
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "全ゾーン 電力消費比較",
                "全ゾーン 室温予測比較",
                "全ゾーン 設定温度比較",
                "快適性達成率",
            ],
        )

        # 1. 電力消費比較
        for zone in df["zone"].unique():
            zone_data = df[df["zone"] == zone]
            fig.add_trace(
                go.Scatter(
                    x=zone_data["timestamp"],
                    y=zone_data["pred_power"],
                    name=zone,
                    mode="lines+markers",
                ),
                row=1,
                col=1,
            )

        # 2. 室温予測比較
        for zone in df["zone"].unique():
            zone_data = df[df["zone"] == zone]
            fig.add_trace(
                go.Scatter(
                    x=zone_data["timestamp"],
                    y=zone_data["pred_temp"],
                    name=zone,
                    mode="lines+markers",
                ),
                row=1,
                col=2,
            )

        # 3. 設定温度比較
        for zone in df["zone"].unique():
            zone_data = df[df["zone"] == zone]
            fig.add_trace(
                go.Scatter(
                    x=zone_data["timestamp"],
                    y=zone_data["set_temp"],
                    name=zone,
                    mode="lines+markers",
                ),
                row=2,
                col=1,
            )

        # 4. 快適性達成率
        comfort_rate = df.groupby("zone")["in_comfort_zone"].mean() * 100
        fig.add_trace(
            go.Bar(x=comfort_rate.index, y=comfort_rate.values, name="快適性達成率"),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="最適化結果 全体サマリー", height=800, template="plotly_white"
        )

        # ファイル保存
        output_file = os.path.join(output_dir, "summary_analysis.html")
        fig.write_html(output_file)
        print(f"[Analysis] Saved summary plot: {output_file}")

        # 統計サマリーの出力
        summary_stats = {
            "zone": [],
            "avg_power": [],
            "avg_temp": [],
            "comfort_rate": [],
            "total_power": [],
        }

        for zone in df["zone"].unique():
            zone_data = df[df["zone"] == zone]
            summary_stats["zone"].append(zone)
            summary_stats["avg_power"].append(zone_data["pred_power"].mean())
            summary_stats["avg_temp"].append(zone_data["pred_temp"].mean())
            summary_stats["comfort_rate"].append(
                zone_data["in_comfort_zone"].mean() * 100
            )
            summary_stats["total_power"].append(zone_data["pred_power"].sum())

        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(output_dir, "summary_statistics.csv")
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        print(f"[Analysis] Saved summary statistics: {summary_file}")

        return summary_df


def main():
    """メイン実行関数"""
    store_name = "Clea"
    start_date = "2025-10-01"
    end_date = "2025-10-02"

    print(f"🔍 {store_name}の最適化結果分析を開始")

    # 分析器の初期化
    analyzer = OptimizationAnalyzer(store_name)

    # 日時範囲の設定
    date_range = pd.date_range(start=start_date, end=end_date, freq="1H")
    date_range = date_range[(date_range.hour >= 0) & (date_range.hour <= 23)]

    # 天候データの取得
    from config.private_information import WEATHER_API_KEY

    weather_fetcher = VisualCrossingWeatherAPIDataFetcher(WEATHER_API_KEY)
    weather_df = weather_fetcher.fetch_weather_data(
        coordinates="35.681236%2C139.767125", start_date=start_date, end_date=end_date
    )

    # モデルの読み込み（簡易版）
    # 実際の実装では、学習済みモデルを読み込む
    models = {}  # ここでモデルを読み込む

    # 最適化結果の取得
    results = analyzer.load_optimization_results(
        date_range, weather_df, models, preference="energy"
    )

    # 分析グラフの作成
    analyzer.create_zone_analysis_plots(results, weather_df, models)
    summary_df = analyzer.create_summary_analysis(results, weather_df)

    print("🎉 分析完了！")
    print("📁 出力ファイル:")
    print("   - analysis/output/*_analysis.html (各ゾーン分析)")
    print("   - analysis/output/summary_analysis.html (全体サマリー)")
    print("   - analysis/output/summary_statistics.csv (統計データ)")


if __name__ == "__main__":
    main()
