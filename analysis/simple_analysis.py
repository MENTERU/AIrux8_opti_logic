# -*- coding: utf-8 -*-
"""
簡易最適化結果分析ツール
======================
既存の最適化結果から分析グラフを作成
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_optimization_results(store_name: str = "Clea"):
    """最適化結果を読み込み"""
    # 制御スケジュールの読み込み
    control_file = (
        f"data/04_PlanningData/{store_name}/control_type_schedule_20251001.csv"
    )

    if not os.path.exists(control_file):
        print(f"❌ ファイルが見つかりません: {control_file}")
        return None

    df = pd.read_csv(control_file)
    df["Date Time"] = pd.to_datetime(df["Date Time"])

    print(f"✅ 最適化結果を読み込みました: {len(df)} 時間分")
    return df


def create_zone_analysis(store_name: str = "Clea"):
    """各ゾーンの分析グラフを作成"""
    print(f"🔍 {store_name}の最適化結果分析を開始")

    # データの読み込み
    df = load_optimization_results(store_name)
    if df is None:
        return

    # ゾーン一覧の取得
    zones = []
    for col in df.columns:
        if col != "Date Time" and col.endswith("_OnOFF"):
            zone_name = col.replace("_OnOFF", "")
            zones.append(zone_name)

    print(f"📊 分析対象ゾーン: {zones}")

    # 各ゾーンの分析
    for zone in zones:
        print(f"📈 {zone} の分析を作成中...")

        # ゾーンデータの抽出
        zone_data = {
            "timestamp": df["Date Time"],
            "onoff": df[f"{zone}_OnOFF"],
            "mode": df[f"{zone}_Mode"],
            "set_temp": df[f"{zone}_SetTemp"],
            "fan_speed": df[f"{zone}_FanSpeed"],
        }

        # サブプロットの作成
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                f"{zone} - 運転状態と設定温度",
                f"{zone} - 運転モード",
                f"{zone} - ファン速度",
                f"{zone} - 設定温度の分布",
                f"{zone} - 運転モードの分布",
                f"{zone} - ファン速度の分布",
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        # 1. 運転状態と設定温度
        fig.add_trace(
            go.Scatter(
                x=zone_data["timestamp"],
                y=zone_data["set_temp"],
                name="設定温度",
                line=dict(color="blue", width=2),
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=zone_data["timestamp"],
                y=zone_data["onoff"],
                name="運転状態",
                line=dict(color="red", width=1),
                mode="lines+markers",
                yaxis="y2",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # 2. 運転モード
        mode_mapping = {"COOL": 0, "DEHUM": 1, "FAN": 2, "HEAT": 3}
        mode_colors = {"COOL": "blue", "DEHUM": "cyan", "FAN": "green", "HEAT": "red"}

        for mode_name, mode_val in mode_mapping.items():
            mode_mask = zone_data["mode"] == mode_name
            if mode_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=zone_data["timestamp"][mode_mask],
                        y=[mode_val] * mode_mask.sum(),
                        name=f"モード: {mode_name}",
                        mode="markers",
                        marker=dict(color=mode_colors[mode_name], size=8),
                    ),
                    row=1,
                    col=2,
                )

        # 3. ファン速度
        fan_mapping = {"Auto": 0, "Low": 1, "Medium": 2, "High": 3, "Top": 4}
        fan_colors = {
            "Auto": "gray",
            "Low": "lightblue",
            "Medium": "blue",
            "High": "darkblue",
            "Top": "navy",
        }

        for fan_name, fan_val in fan_mapping.items():
            fan_mask = zone_data["fan_speed"] == fan_name
            if fan_mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=zone_data["timestamp"][fan_mask],
                        y=[fan_val] * fan_mask.sum(),
                        name=f"ファン: {fan_name}",
                        mode="markers",
                        marker=dict(color=fan_colors[fan_name], size=8),
                    ),
                    row=2,
                    col=1,
                )

        # 4. 設定温度の分布
        fig.add_trace(
            go.Histogram(
                x=zone_data["set_temp"],
                name="設定温度分布",
                nbinsx=10,
                marker_color="lightblue",
            ),
            row=2,
            col=2,
        )

        # 5. 運転モードの分布
        mode_counts = zone_data["mode"].value_counts()
        fig.add_trace(
            go.Bar(
                x=mode_counts.index,
                y=mode_counts.values,
                name="モード分布",
                marker_color="lightgreen",
            ),
            row=3,
            col=1,
        )

        # 6. ファン速度の分布
        fan_counts = zone_data["fan_speed"].value_counts()
        fig.add_trace(
            go.Bar(
                x=fan_counts.index,
                y=fan_counts.values,
                name="ファン分布",
                marker_color="lightcoral",
            ),
            row=3,
            col=2,
        )

        # レイアウトの設定
        fig.update_layout(
            title=f"{zone} 最適化結果分析",
            height=900,
            showlegend=True,
            template="plotly_white",
        )

        # 軸ラベルの設定
        fig.update_xaxes(title_text="時間", row=3, col=1)
        fig.update_xaxes(title_text="時間", row=3, col=2)
        fig.update_yaxes(title_text="設定温度 (°C)", row=1, col=1)
        fig.update_yaxes(title_text="運転状態", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="モード", row=1, col=2)
        fig.update_yaxes(title_text="ファン速度", row=2, col=1)
        fig.update_yaxes(title_text="頻度", row=2, col=2)
        fig.update_yaxes(title_text="頻度", row=3, col=1)
        fig.update_yaxes(title_text="頻度", row=3, col=2)

        # ファイル保存
        output_file = f"analysis/output/{zone}_analysis.html"
        fig.write_html(output_file)
        print(f"✅ 分析グラフを保存: {output_file}")


def create_summary_analysis(store_name: str = "Clea"):
    """全体サマリー分析を作成"""
    print("📊 全体サマリー分析を作成中...")

    # データの読み込み
    df = load_optimization_results(store_name)
    if df is None:
        return

    # ゾーン一覧の取得
    zones = []
    for col in df.columns:
        if col != "Date Time" and col.endswith("_OnOFF"):
            zone_name = col.replace("_OnOFF", "")
            zones.append(zone_name)

    # 全体サマリーグラフ
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "全ゾーン 設定温度比較",
            "全ゾーン 運転モード比較",
            "全ゾーン ファン速度比較",
            "ゾーン別 運転時間比較",
        ],
    )

    # 1. 設定温度比較
    for zone in zones:
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"],
                y=df[f"{zone}_SetTemp"],
                name=zone,
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

    # 2. 運転モード比較（時間別）
    for zone in zones:
        mode_mapping = {"COOL": 0, "DEHUM": 1, "FAN": 2, "HEAT": 3}
        mode_values = [mode_mapping.get(mode, -1) for mode in df[f"{zone}_Mode"]]
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"], y=mode_values, name=zone, mode="lines+markers"
            ),
            row=1,
            col=2,
        )

    # 3. ファン速度比較（時間別）
    for zone in zones:
        fan_mapping = {"Auto": 0, "Low": 1, "Medium": 2, "High": 3, "Top": 4}
        fan_values = [fan_mapping.get(fan, -1) for fan in df[f"{zone}_FanSpeed"]]
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"], y=fan_values, name=zone, mode="lines+markers"
            ),
            row=2,
            col=1,
        )

    # 4. 運転時間比較
    onoff_counts = {}
    for zone in zones:
        onoff_counts[zone] = df[f"{zone}_OnOFF"].sum()

    fig.add_trace(
        go.Bar(
            x=list(onoff_counts.keys()), y=list(onoff_counts.values()), name="運転時間"
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="最適化結果 全体サマリー", height=800, template="plotly_white"
    )

    # 軸ラベルの設定
    fig.update_xaxes(title_text="時間", row=1, col=1)
    fig.update_xaxes(title_text="時間", row=1, col=2)
    fig.update_xaxes(title_text="時間", row=2, col=1)
    fig.update_xaxes(title_text="ゾーン", row=2, col=2)
    fig.update_yaxes(title_text="設定温度 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="モード", row=1, col=2)
    fig.update_yaxes(title_text="ファン速度", row=2, col=1)
    fig.update_yaxes(title_text="運転時間 (時間)", row=2, col=2)

    # ファイル保存
    output_file = "analysis/output/summary_analysis.html"
    fig.write_html(output_file)
    print(f"✅ 全体サマリーを保存: {output_file}")

    # 統計データの出力
    stats_data = []
    for zone in zones:
        stats_data.append(
            {
                "zone": zone,
                "avg_set_temp": df[f"{zone}_SetTemp"].mean(),
                "min_set_temp": df[f"{zone}_SetTemp"].min(),
                "max_set_temp": df[f"{zone}_SetTemp"].max(),
                "most_common_mode": df[f"{zone}_Mode"].mode().iloc[0],
                "most_common_fan": df[f"{zone}_FanSpeed"].mode().iloc[0],
                "operation_hours": df[f"{zone}_OnOFF"].sum(),
            }
        )

    stats_df = pd.DataFrame(stats_data)
    stats_file = "analysis/output/summary_statistics.csv"
    stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
    print(f"✅ 統計データを保存: {stats_file}")

    return stats_df


def main():
    """メイン実行関数"""
    store_name = "Clea"

    print(f"🔍 {store_name}の最適化結果分析を開始")

    # 各ゾーンの分析
    create_zone_analysis(store_name)

    # 全体サマリー分析
    stats_df = create_summary_analysis(store_name)

    print("\n🎉 分析完了！")
    print("📁 出力ファイル:")
    print("   - analysis/output/*_analysis.html (各ゾーン分析)")
    print("   - analysis/output/summary_analysis.html (全体サマリー)")
    print("   - analysis/output/summary_statistics.csv (統計データ)")

    if stats_df is not None:
        print("\n📊 統計サマリー:")
        print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
