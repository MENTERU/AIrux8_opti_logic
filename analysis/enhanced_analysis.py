# -*- coding: utf-8 -*-
"""
拡張最適化結果分析ツール
======================
室内温度予測を含む詳細な分析グラフを作成
"""

import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import numpy as np
import pickle
from pathlib import Path


def load_models(store_name: str = "Clea"):
    """学習済みモデルを読み込み"""
    models_dir = f"data/03_Models/{store_name}"
    models = {}
    
    if not os.path.exists(models_dir):
        print(f"❌ モデルディレクトリが見つかりません: {models_dir}")
        return None
    
    # モデルファイルを検索
    for model_file in Path(models_dir).glob("*.pkl"):
        zone_name = model_file.stem.replace("_model", "")
        try:
            with open(model_file, 'rb') as f:
                models[zone_name] = pickle.load(f)
            print(f"✅ モデル読み込み完了: {zone_name}")
        except Exception as e:
            print(f"❌ モデル読み込みエラー {zone_name}: {e}")
    
    return models if models else None


def predict_room_temperature(zone_name: str, models: dict, control_data: dict, 
                           weather_data: dict, last_temp: float = 25.0) -> float:
    """室内温度を予測"""
    if zone_name not in models:
        return last_temp
    
    try:
        # 特徴量の作成
        features = pd.DataFrame([{
            "A/C Set Temperature": control_data['set_temp'],
            "Indoor Temp. Lag1": last_temp,
            "A/C ON/OFF": control_data['onoff'],
            "A/C Mode": control_data['mode'],
            "A/C Fan Speed": control_data['fan'],
            "Outdoor Temp.": weather_data['outdoor_temp'],
            "Outdoor Humidity": weather_data['outdoor_humidity'],
        }])
        
        # 温度予測
        temp_model = models[zone_name].temp_model
        predicted_temp = float(temp_model.predict(features[temp_model.feature_names_])[0])
        return predicted_temp
    except Exception as e:
        print(f"❌ 温度予測エラー {zone_name}: {e}")
        return last_temp


def predict_power_consumption(zone_name: str, models: dict, control_data: dict, 
                            weather_data: dict, unit_count: int = 1) -> float:
    """電力消費を予測"""
    if zone_name not in models:
        return 0.0
    
    try:
        # 特徴量の作成
        features = pd.DataFrame([{
            "A/C Set Temperature": control_data['set_temp'],
            "Indoor Temp. Lag1": control_data['last_temp'],
            "A/C ON/OFF": control_data['onoff'],
            "A/C Mode": control_data['mode'],
            "A/C Fan Speed": control_data['fan'],
            "Outdoor Temp.": weather_data['outdoor_temp'],
            "Outdoor Humidity": weather_data['outdoor_humidity'],
        }])
        
        # 電力予測
        power_model = models[zone_name].power_model
        predicted_power = float(power_model.predict(features[power_model.feature_names_])[0])
        return predicted_power * unit_count
    except Exception as e:
        print(f"❌ 電力予測エラー {zone_name}: {e}")
        return 0.0


def load_optimization_results(store_name: str = "Clea"):
    """最適化結果を読み込み"""
    control_file = f"data/04_PlanningData/{store_name}/control_type_schedule_20251001.csv"
    
    if not os.path.exists(control_file):
        print(f"❌ ファイルが見つかりません: {control_file}")
        return None
    
    df = pd.read_csv(control_file)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    
    print(f"✅ 最適化結果を読み込みました: {len(df)} 時間分")
    return df


def load_weather_data():
    """天候データを読み込み（簡易版）"""
    # 実際の実装では、天候APIから取得したデータを使用
    weather_data = {
        "2025-10-01 00:00:00": {"outdoor_temp": 23.5, "outdoor_humidity": 78.28},
        "2025-10-01 01:00:00": {"outdoor_temp": 23.2, "outdoor_humidity": 79.1},
        "2025-10-01 02:00:00": {"outdoor_temp": 22.8, "outdoor_humidity": 80.5},
        "2025-10-01 03:00:00": {"outdoor_temp": 22.5, "outdoor_humidity": 81.2},
        "2025-10-01 04:00:00": {"outdoor_temp": 22.1, "outdoor_humidity": 82.0},
        "2025-10-01 05:00:00": {"outdoor_temp": 21.8, "outdoor_humidity": 83.1},
        "2025-10-01 06:00:00": {"outdoor_temp": 21.6, "outdoor_humidity": 84.8},
        "2025-10-01 07:00:00": {"outdoor_temp": 21.9, "outdoor_humidity": 83.5},
        "2025-10-01 08:00:00": {"outdoor_temp": 22.3, "outdoor_humidity": 81.8},
        "2025-10-01 09:00:00": {"outdoor_temp": 22.8, "outdoor_humidity": 79.5},
        "2025-10-01 10:00:00": {"outdoor_temp": 23.2, "outdoor_humidity": 77.8},
        "2025-10-01 11:00:00": {"outdoor_temp": 23.8, "outdoor_humidity": 75.2},
        "2025-10-01 12:00:00": {"outdoor_temp": 20.4, "outdoor_humidity": 80.33},
        "2025-10-01 13:00:00": {"outdoor_temp": 20.1, "outdoor_humidity": 81.5},
        "2025-10-01 14:00:00": {"outdoor_temp": 19.8, "outdoor_humidity": 82.8},
        "2025-10-01 15:00:00": {"outdoor_temp": 19.5, "outdoor_humidity": 84.1},
        "2025-10-01 16:00:00": {"outdoor_temp": 19.2, "outdoor_humidity": 85.5},
        "2025-10-01 17:00:00": {"outdoor_temp": 19.0, "outdoor_humidity": 86.8},
        "2025-10-01 18:00:00": {"outdoor_temp": 20.4, "outdoor_humidity": 79.82},
        "2025-10-01 19:00:00": {"outdoor_temp": 20.6, "outdoor_humidity": 78.5},
        "2025-10-01 20:00:00": {"outdoor_temp": 20.8, "outdoor_humidity": 77.2},
        "2025-10-01 21:00:00": {"outdoor_temp": 21.0, "outdoor_humidity": 76.1},
        "2025-10-01 22:00:00": {"outdoor_temp": 21.2, "outdoor_humidity": 75.5},
        "2025-10-01 23:00:00": {"outdoor_temp": 21.4, "outdoor_humidity": 74.8},
        "2025-10-02 00:00:00": {"outdoor_temp": 20.6, "outdoor_humidity": 77.35},
    }
    return weather_data


def create_enhanced_zone_analysis(store_name: str = "Clea"):
    """室内温度予測を含む拡張ゾーン分析を作成"""
    print(f"🔍 {store_name}の拡張最適化結果分析を開始")
    
    # データの読み込み
    df = load_optimization_results(store_name)
    if df is None:
        return
    
    # モデルの読み込み
    models = load_models(store_name)
    if models is None:
        print("❌ モデルの読み込みに失敗しました")
        return
    
    # 天候データの読み込み
    weather_data = load_weather_data()
    
    # ゾーン一覧の取得
    zones = []
    for col in df.columns:
        if col != "Date Time" and col.endswith("_OnOFF"):
            zone_name = col.replace("_OnOFF", "")
            zones.append(zone_name)
    
    print(f"📊 分析対象ゾーン: {zones}")
    
    # 各ゾーンの分析
    for zone in zones:
        print(f"📈 {zone} の拡張分析を作成中...")
        
        # ゾーンデータの抽出
        zone_data = {
            "timestamp": df["Date Time"],
            "onoff": df[f"{zone}_OnOFF"],
            "mode": df[f"{zone}_Mode"],
            "set_temp": df[f"{zone}_SetTemp"],
            "fan_speed": df[f"{zone}_FanSpeed"],
        }
        
        # 室内温度と電力消費の予測
        pred_temps = []
        pred_powers = []
        last_temp = 25.0  # 初期温度
        
        for i, timestamp in enumerate(zone_data["timestamp"]):
            # 制御データ
            control_data = {
                "set_temp": zone_data["set_temp"].iloc[i],
                "onoff": 1 if zone_data["onoff"].iloc[i] == "ON" else 0,
                "mode": {"COOL": 0, "DEHUM": 1, "FAN": 2, "HEAT": 3}.get(zone_data["mode"].iloc[i], 2),
                "fan": {"Auto": 0, "Low": 1, "Medium": 2, "High": 3, "Top": 4}.get(zone_data["fan_speed"].iloc[i], 1),
                "last_temp": last_temp,
            }
            
            # 天候データ
            weather_key = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            weather = weather_data.get(weather_key, {"outdoor_temp": 25.0, "outdoor_humidity": 60.0})
            
            # 予測実行
            pred_temp = predict_room_temperature(zone, models, control_data, weather, last_temp)
            pred_power = predict_power_consumption(zone, models, control_data, weather, unit_count=1)
            
            pred_temps.append(pred_temp)
            pred_powers.append(pred_power)
            last_temp = pred_temp
        
        # サブプロットの作成
        fig = make_subplots(
            rows=4,
            cols=2,
            subplot_titles=[
                f"{zone} - 設定温度と予測室温",
                f"{zone} - 予測電力消費",
                f"{zone} - 運転モード",
                f"{zone} - ファン速度",
                f"{zone} - 外気温度",
                f"{zone} - 外気湿度",
                f"{zone} - 温度差（設定-予測）",
                f"{zone} - 電力消費の分布",
            ],
            specs=[
                [{"secondary_y": True}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )
        
        # 1. 設定温度と予測室温
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
                y=pred_temps,
                name="予測室温",
                line=dict(color="red", width=2),
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
        
        # 快適性範囲の表示
        fig.add_hline(y=22, line_dash="dash", line_color="green", 
                     annotation_text="快適下限(22°C)", row=1, col=1)
        fig.add_hline(y=24, line_dash="dash", line_color="green", 
                     annotation_text="快適上限(24°C)", row=1, col=1)
        
        # 2. 予測電力消費
        fig.add_trace(
            go.Scatter(
                x=zone_data["timestamp"],
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
                    row=2,
                    col=1,
                )
        
        # 4. ファン速度
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
                    col=2,
                )
        
        # 5. 外気温度
        outdoor_temps = [weather_data.get(t.strftime("%Y-%m-%d %H:%M:%S"), 
                                        {"outdoor_temp": 25.0})["outdoor_temp"] 
                        for t in zone_data["timestamp"]]
        fig.add_trace(
            go.Scatter(
                x=zone_data["timestamp"],
                y=outdoor_temps,
                name="外気温度",
                line=dict(color="orange", width=2),
                mode="lines+markers",
            ),
            row=3,
            col=1,
        )
        
        # 6. 外気湿度
        outdoor_humidities = [weather_data.get(t.strftime("%Y-%m-%d %H:%M:%S"), 
                                             {"outdoor_humidity": 60.0})["outdoor_humidity"] 
                             for t in zone_data["timestamp"]]
        fig.add_trace(
            go.Scatter(
                x=zone_data["timestamp"],
                y=outdoor_humidities,
                name="外気湿度",
                line=dict(color="brown", width=2),
                mode="lines+markers",
            ),
            row=3,
            col=2,
        )
        
        # 7. 温度差（設定-予測）
        temp_diff = [set_temp - pred_temp for set_temp, pred_temp in 
                    zip(zone_data["set_temp"], pred_temps)]
        fig.add_trace(
            go.Scatter(
                x=zone_data["timestamp"],
                y=temp_diff,
                name="温度差",
                line=dict(color="darkgreen", width=2),
                mode="lines+markers",
            ),
            row=4,
            col=1,
        )
        
        # 8. 電力消費の分布
        fig.add_trace(
            go.Histogram(
                x=pred_powers,
                name="電力分布",
                nbinsx=10,
                marker_color="lightcoral",
            ),
            row=4,
            col=2,
        )
        
        # レイアウトの設定
        fig.update_layout(
            title=f"{zone} 拡張最適化結果分析（室内温度予測含む）",
            height=1200,
            showlegend=True,
            template="plotly_white",
        )
        
        # 軸ラベルの設定
        fig.update_xaxes(title_text="時間", row=4, col=1)
        fig.update_xaxes(title_text="電力 (W)", row=4, col=2)
        fig.update_yaxes(title_text="温度 (°C)", row=1, col=1)
        fig.update_yaxes(title_text="電力 (W)", row=1, col=2)
        fig.update_yaxes(title_text="モード", row=2, col=1)
        fig.update_yaxes(title_text="ファン速度", row=2, col=2)
        fig.update_yaxes(title_text="外気温度 (°C)", row=3, col=1)
        fig.update_yaxes(title_text="外気湿度 (%)", row=3, col=2)
        fig.update_yaxes(title_text="温度差 (°C)", row=4, col=1)
        fig.update_yaxes(title_text="頻度", row=4, col=2)
        
        # ファイル保存
        output_file = f"analysis/output/{zone}_enhanced_analysis.html"
        fig.write_html(output_file)
        print(f"✅ 拡張分析グラフを保存: {output_file}")


def create_enhanced_summary_analysis(store_name: str = "Clea"):
    """拡張全体サマリー分析を作成"""
    print("📊 拡張全体サマリー分析を作成中...")
    
    # データの読み込み
    df = load_optimization_results(store_name)
    if df is None:
        return
    
    # モデルの読み込み
    models = load_models(store_name)
    if models is None:
        print("❌ モデルの読み込みに失敗しました")
        return
    
    # 天候データの読み込み
    weather_data = load_weather_data()
    
    # ゾーン一覧の取得
    zones = []
    for col in df.columns:
        if col != "Date Time" and col.endswith("_OnOFF"):
            zone_name = col.replace("_OnOFF", "")
            zones.append(zone_name)
    
    # 全ゾーンの予測データを準備
    all_zone_data = {}
    
    for zone in zones:
        zone_data = {
            "timestamp": df["Date Time"],
            "onoff": df[f"{zone}_OnOFF"],
            "mode": df[f"{zone}_Mode"],
            "set_temp": df[f"{zone}_SetTemp"],
            "fan_speed": df[f"{zone}_FanSpeed"],
        }
        
        # 予測実行
        pred_temps = []
        pred_powers = []
        last_temp = 25.0
        
        for i, timestamp in enumerate(zone_data["timestamp"]):
            control_data = {
                "set_temp": zone_data["set_temp"].iloc[i],
                "onoff": 1 if zone_data["onoff"].iloc[i] == "ON" else 0,
                "mode": {"COOL": 0, "DEHUM": 1, "FAN": 2, "HEAT": 3}.get(zone_data["mode"].iloc[i], 2),
                "fan": {"Auto": 0, "Low": 1, "Medium": 2, "High": 3, "Top": 4}.get(zone_data["fan_speed"].iloc[i], 1),
                "last_temp": last_temp,
            }
            
            weather_key = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            weather = weather_data.get(weather_key, {"outdoor_temp": 25.0, "outdoor_humidity": 60.0})
            
            pred_temp = predict_room_temperature(zone, models, control_data, weather, last_temp)
            pred_power = predict_power_consumption(zone, models, control_data, weather, unit_count=1)
            
            pred_temps.append(pred_temp)
            pred_powers.append(pred_power)
            last_temp = pred_temp
        
        all_zone_data[zone] = {
            "pred_temps": pred_temps,
            "pred_powers": pred_powers,
            "set_temps": zone_data["set_temp"].tolist(),
        }
    
    # 全体サマリーグラフ
    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[
            "全ゾーン 予測室温比較",
            "全ゾーン 予測電力消費比較",
            "全ゾーン 設定温度比較",
            "全ゾーン 温度差比較",
            "ゾーン別 平均電力消費",
            "ゾーン別 快適性達成率",
        ],
    )
    
    # 1. 予測室温比較
    for zone in zones:
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"],
                y=all_zone_data[zone]["pred_temps"],
                name=zone,
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )
    
    # 2. 予測電力消費比較
    for zone in zones:
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"],
                y=all_zone_data[zone]["pred_powers"],
                name=zone,
                mode="lines+markers",
            ),
            row=1,
            col=2,
        )
    
    # 3. 設定温度比較
    for zone in zones:
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"],
                y=all_zone_data[zone]["set_temps"],
                name=zone,
                mode="lines+markers",
            ),
            row=2,
            col=1,
        )
    
    # 4. 温度差比較
    for zone in zones:
        temp_diff = [set_temp - pred_temp for set_temp, pred_temp in 
                    zip(all_zone_data[zone]["set_temps"], all_zone_data[zone]["pred_temps"])]
        fig.add_trace(
            go.Scatter(
                x=df["Date Time"],
                y=temp_diff,
                name=zone,
                mode="lines+markers",
            ),
            row=2,
            col=2,
        )
    
    # 5. 平均電力消費
    avg_powers = [np.mean(all_zone_data[zone]["pred_powers"]) for zone in zones]
    fig.add_trace(
        go.Bar(x=zones, y=avg_powers, name="平均電力消費"),
        row=3,
        col=1,
    )
    
    # 6. 快適性達成率
    comfort_rates = []
    for zone in zones:
        pred_temps = all_zone_data[zone]["pred_temps"]
        in_comfort = sum(1 for temp in pred_temps if 22 <= temp <= 24)
        comfort_rate = (in_comfort / len(pred_temps)) * 100
        comfort_rates.append(comfort_rate)
    
    fig.add_trace(
        go.Bar(x=zones, y=comfort_rates, name="快適性達成率"),
        row=3,
        col=2,
    )
    
    fig.update_layout(
        title="拡張最適化結果 全体サマリー（室内温度予測含む）",
        height=1200,
        template="plotly_white",
    )
    
    # 軸ラベルの設定
    fig.update_xaxes(title_text="時間", row=1, col=1)
    fig.update_xaxes(title_text="時間", row=1, col=2)
    fig.update_xaxes(title_text="時間", row=2, col=1)
    fig.update_xaxes(title_text="時間", row=2, col=2)
    fig.update_xaxes(title_text="ゾーン", row=3, col=1)
    fig.update_xaxes(title_text="ゾーン", row=3, col=2)
    fig.update_yaxes(title_text="予測室温 (°C)", row=1, col=1)
    fig.update_yaxes(title_text="予測電力 (W)", row=1, col=2)
    fig.update_yaxes(title_text="設定温度 (°C)", row=2, col=1)
    fig.update_yaxes(title_text="温度差 (°C)", row=2, col=2)
    fig.update_yaxes(title_text="平均電力 (W)", row=3, col=1)
    fig.update_yaxes(title_text="快適性達成率 (%)", row=3, col=2)
    
    # ファイル保存
    output_file = "analysis/output/enhanced_summary_analysis.html"
    fig.write_html(output_file)
    print(f"✅ 拡張全体サマリーを保存: {output_file}")
    
    # 統計データの出力
    stats_data = []
    for zone in zones:
        pred_temps = all_zone_data[zone]["pred_temps"]
        pred_powers = all_zone_data[zone]["pred_powers"]
        set_temps = all_zone_data[zone]["set_temps"]
        
        in_comfort = sum(1 for temp in pred_temps if 22 <= temp <= 24)
        comfort_rate = (in_comfort / len(pred_temps)) * 100
        
        stats_data.append({
            "zone": zone,
            "avg_pred_temp": np.mean(pred_temps),
            "min_pred_temp": np.min(pred_temps),
            "max_pred_temp": np.max(pred_temps),
            "avg_set_temp": np.mean(set_temps),
            "avg_power": np.mean(pred_powers),
            "total_power": np.sum(pred_powers),
            "comfort_rate": comfort_rate,
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_file = "analysis/output/enhanced_summary_statistics.csv"
    stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
    print(f"✅ 拡張統計データを保存: {stats_file}")
    
    return stats_df


def main():
    """メイン実行関数"""
    store_name = "Clea"
    
    print(f"🔍 {store_name}の拡張最適化結果分析を開始")
    
    # 拡張ゾーン分析
    create_enhanced_zone_analysis(store_name)
    
    # 拡張全体サマリー分析
    stats_df = create_enhanced_summary_analysis(store_name)
    
    print("\n🎉 拡張分析完了！")
    print("📁 出力ファイル:")
    print("   - analysis/output/*_enhanced_analysis.html (各ゾーン拡張分析)")
    print("   - analysis/output/enhanced_summary_analysis.html (拡張全体サマリー)")
    print("   - analysis/output/enhanced_summary_statistics.csv (拡張統計データ)")
    
    if stats_df is not None:
        print("\n📊 拡張統計サマリー:")
        print(stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
