# -*- coding: utf-8 -*-
"""
ダッシュボード生成
 - 実績ダッシュボード（時別/日別）
 - 計画妥当性ダッシュボード（実績との比較、計画開始を明示）
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _load_actual(store_name: str) -> Optional[pd.DataFrame]:
    path = f"data/02_PreprocessedData/{store_name}/features_processed_{store_name}.csv"
    if not os.path.exists(path):
        print(f"❌ 実績データが見つかりません: {path}")
        return None
    df = pd.read_csv(path)
    # Datetime 列の互換対応
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "Datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["Datetime"])  # 統一
    else:
        print("❌ 実績データに Datetime/datetime 列がありません")
        return None
    return df


def _load_plan(store_name: str) -> Optional[pd.DataFrame]:
    # 直近のコントロールスケジュールを推定（命名規則yyyyMMdd）
    plan_dir = f"data/04_PlanningData/{store_name}"
    if not os.path.isdir(plan_dir):
        print(f"❌ 計画ディレクトリが見つかりません: {plan_dir}")
        return None
    files = sorted(
        [
            f
            for f in os.listdir(plan_dir)
            if f.startswith("control_type_schedule_") and f.endswith(".csv")
        ]
    )
    if not files:
        print("❌ コントロールスケジュールが見つかりません")
        return None
    latest = files[-1]
    path = os.path.join(plan_dir, latest)
    df = pd.read_csv(path)
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    return df


def _load_weather_forecast(store_name: str) -> Optional[pd.DataFrame]:
    """天気予報データを読み込み"""
    # 保存先は PlanningData に統一
    path = f"data/04_PlanningData/{store_name}/weather_forecast.csv"
    if not os.path.exists(path):
        print(f"❌ 天気予報データが見つかりません: {path}")
        return None
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        print("❌ 天気予報データに datetime 列がありません")
        return None
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def create_historical_dashboard(store_name: str = "Clea", freq: str = "H") -> None:
    """実績ダッシュボード（時別/日別）を出力"""
    df = _load_actual(store_name)
    if df is None or df.empty:
        return
    if freq.upper() not in ("H", "D"):
        freq = "H"

    # 集約（A/C関連は最頻値、温度/電力は平均）
    group_cols = [
        c
        for c in [
            "A/C Set Temperature",
            "Indoor Temp.",
            "adjusted_power",
            "A/C ON/OFF",
            "A/C Mode",
            "A/C Fan Speed",
            "Outdoor Temp.",
            "Outdoor Humidity",
            "Solar Radiation",
        ]
        if c in df.columns
    ]
    agg_dict = {col: "mean" for col in group_cols}
    for cat_col in ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed", "A/C Set Temperature"]:
        if cat_col in agg_dict:
            agg_dict[cat_col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan

    agg_df = (
        df.set_index("datetime")[group_cols + ["zone"]]
        .groupby([pd.Grouper(freq=freq), "zone"])
        .agg(agg_dict)
        .reset_index()
    )

    zones = agg_df["zone"].dropna().unique().tolist()
    os.makedirs("analysis/output", exist_ok=True)

    for z in zones:
        sub = agg_df[agg_df["zone"] == z]
        fig = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[
                f"{z} 空調設定（設定温度/ON）",
                f"{z} 室温",
                f"{z} 電力",
                f"{z} 外気温",
                f"{z} 外気湿度",
                f"{z} モード/ファン（実績）",
            ],
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": True}],
                [{"secondary_y": False}],
            ],
        )
        if "A/C Set Temperature" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"], y=sub["A/C Set Temperature"], name="設定温度"
                ),
                row=1,
                col=1,
            )
        if "Indoor Temp." in sub.columns:
            fig.add_trace(
                go.Scatter(x=sub["datetime"], y=sub["Indoor Temp."], name="室温"),
                row=2,
                col=1,
            )
        if "adjusted_power" in sub.columns:
            fig.add_trace(
                go.Scatter(x=sub["datetime"], y=sub["adjusted_power"], name="電力"),
                row=3,
                col=1,
            )
        if "Outdoor Temp." in sub.columns:
            fig.add_trace(
                go.Scatter(x=sub["datetime"], y=sub["Outdoor Temp."], name="外気温"),
                row=4,
                col=1,
            )
        if "Outdoor Humidity" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"], y=sub["Outdoor Humidity"], name="外気湿度"
                ),
                row=5,
                col=1,
            )
        if "Solar Radiation" in sub.columns:
            fig.add_trace(
                go.Scatter(x=sub["datetime"], y=sub["Solar Radiation"], name="日射量"),
                row=5,
                col=1,
            )

        # 実績 モード/ファン
        if "A/C Mode" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"],
                    y=sub["A/C Mode"],
                    name="モード(実績)",
                    mode="markers",
                ),
                row=6,
                col=1,
            )
        if "A/C Fan Speed" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub["datetime"],
                    y=sub["A/C Fan Speed"],
                    name="ファン(実績)",
                    mode="markers",
                ),
                row=6,
                col=1,
            )

        # ON/OFF（実績）を1〜5段目に副軸で重ねる
        if "A/C ON/OFF" in sub.columns:
            on_series = sub["A/C ON/OFF"].fillna(0).clip(lower=0)
            for r in range(1, 6):
                fig.add_trace(
                    go.Scatter(
                        x=sub["datetime"],
                        y=on_series,
                        name="ON(実績)",
                        line=dict(color="rgba(255,0,0,0.35)", width=1),
                        mode="lines",
                        showlegend=(r == 1),
                    ),
                    row=r,
                    col=1,
                    secondary_y=True,
                )
                fig.update_yaxes(
                    range=[0, 1], title_text="ON", row=r, col=1, secondary_y=True
                )

        fig.update_layout(
            title=f"{store_name} 実績ダッシュボード（{z}、freq={freq})",
            height=1200,
            template="plotly_white",
        )
        # 出力先: store/zone 階層
        out_dir = os.path.join("analysis/output", store_name, z)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"historical_{freq}.html")
        fig.write_html(out)
        print(f"✅ 実績ダッシュボード出力: {out}")


def create_plan_validation_dashboard(
    store_name: str = "Clea", lookback_days: int = 7
) -> None:
    """計画妥当性ダッシュボードを出力（直近実績との比較、計画開始を明示）"""
    actual = _load_actual(store_name)
    plan = _load_plan(store_name)
    weather_forecast = _load_weather_forecast(store_name)
    if actual is None or plan is None:
        return

    # 直近実績（lookback_days）
    plan_start = plan["Date Time"].min()
    actual_win = actual[
        (actual["datetime"] >= plan_start - pd.Timedelta(days=lookback_days))
        & (actual["datetime"] <= plan["Date Time"].max())
    ].copy()

    zones = actual_win["zone"].dropna().unique().tolist()
    base_out = os.path.join("analysis/output", store_name)
    os.makedirs(base_out, exist_ok=True)

    for z in zones:
        sub_a = actual_win[actual_win["zone"] == z]
        # 計画の列名を解決
        cols = {
            "on": f"{z}_OnOFF",
            "mode": f"{z}_Mode",
            "set": f"{z}_SetTemp",
            "fan": f"{z}_FanSpeed",
            "pt": f"{z}_PredTemp",
            "pp": f"{z}_PredPower",
        }
        # 計画サブ
        sub_p = plan[
            ["Date Time"] + [c for c in cols.values() if c in plan.columns]
        ].copy()

        fig = make_subplots(
            rows=6,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=[
                f"{z} 室温・設定温度（実績/計画）",
                f"{z} 電力（実績/計画）",
                f"{z} 運転モード・ファン（実績/計画）",
                f"{z} 運転状態（実績/計画）",
                f"{z} 外気温（実績/予報）",
                f"{z} 外気湿度・日射量（実績/予報）",
            ],
            specs=[
                [{"secondary_y": False}],  # 室温・設定温度のみ
                [{"secondary_y": False}],  # 電力のみ
                [{"secondary_y": False}],  # モード・ファンのみ
                [{"secondary_y": False}],  # 運転状態のみ
                [{"secondary_y": False}],  # 外気温のみ
                [{"secondary_y": True}],  # 外気湿度・日射量（副軸）
            ],
        )

        # 室温・設定温度（同じ項目は同じ色、線種で区別）
        if not sub_a.empty and "Indoor Temp." in sub_a.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["Indoor Temp."],
                    name="室温",
                    line=dict(color="blue", width=2),
                    mode="lines",
                ),
                row=1,
                col=1,
            )
        if cols["pt"] in sub_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_p["Date Time"],
                    y=sub_p[cols["pt"]],
                    name="室温(計画)",
                    line=dict(color="blue", width=2, dash="dash"),
                    mode="lines",
                ),
                row=1,
                col=1,
            )
        if not sub_a.empty and "A/C Set Temperature" in sub_a.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["A/C Set Temperature"],
                    name="設定温度",
                    line=dict(color="red", width=2),
                    mode="lines",
                ),
                row=1,
                col=1,
            )
        if cols["set"] in sub_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_p["Date Time"],
                    y=sub_p[cols["set"]],
                    name="設定温度(計画)",
                    line=dict(color="red", width=2, dash="dash"),
                    mode="lines",
                ),
                row=1,
                col=1,
            )

        # 電力（同じ項目は同じ色、線種で区別）
        if not sub_a.empty and "adjusted_power" in sub_a.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["adjusted_power"],
                    name="電力",
                    line=dict(color="green", width=2),
                    mode="lines",
                ),
                row=2,
                col=1,
            )
        if cols["pp"] in sub_p.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_p["Date Time"],
                    y=sub_p[cols["pp"]],
                    name="電力(計画)",
                    line=dict(color="green", width=2, dash="dash"),
                    mode="lines",
                ),
                row=2,
                col=1,
            )

        # 運転モード・ファン（実績/計画）
        if not sub_a.empty and "A/C Mode" in sub_a.columns:
            # モード数値を文字列に変換
            mode_mapping = {0: "COOL", 1: "DEHUM", 2: "FAN", 3: "HEAT"}
            mode_labels = sub_a["A/C Mode"].map(mode_mapping).fillna("UNKNOWN")

            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["A/C Mode"],
                    name="モード(実績)",
                    mode="markers",
                    marker=dict(color="orange", size=6, symbol="circle"),
                    text=mode_labels,
                    hovertemplate="%{text}<br>時刻: %{x}<br>モード: %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )
        if cols["mode"] in sub_p.columns:
            # 計画のモードも文字列に変換
            plan_mode_mapping = {"COOL": 0, "DEHUM": 1, "FAN": 2, "HEAT": 3}
            plan_mode_numeric = sub_p[cols["mode"]].map(plan_mode_mapping).fillna(2)
            plan_mode_labels = sub_p[cols["mode"]]

            fig.add_trace(
                go.Scatter(
                    x=sub_p["Date Time"],
                    y=plan_mode_numeric,
                    name="モード(計画)",
                    mode="markers",
                    marker=dict(color="orange", size=6, symbol="diamond"),
                    text=plan_mode_labels,
                    hovertemplate="%{text}<br>時刻: %{x}<br>モード: %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )
        if not sub_a.empty and "A/C Fan Speed" in sub_a.columns:
            # ファン速度数値を文字列に変換
            fan_mapping = {0: "Auto", 1: "Low", 2: "Medium", 3: "High", 4: "Top"}
            fan_labels = sub_a["A/C Fan Speed"].map(fan_mapping).fillna("UNKNOWN")

            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["A/C Fan Speed"],
                    name="ファン(実績)",
                    mode="markers",
                    marker=dict(color="purple", size=6, symbol="square"),
                    text=fan_labels,
                    hovertemplate="%{text}<br>時刻: %{x}<br>ファン: %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )
        if cols["fan"] in sub_p.columns:
            # 計画のファン速度も数値に変換
            plan_fan_numeric = sub_p[cols["fan"]]
            plan_fan_mapping = {0: "Auto", 1: "Low", 2: "Medium", 3: "High", 4: "Top"}
            plan_fan_labels = plan_fan_numeric.map(plan_fan_mapping).fillna("UNKNOWN")

            fig.add_trace(
                go.Scatter(
                    x=sub_p["Date Time"],
                    y=plan_fan_numeric,
                    name="ファン(計画)",
                    mode="markers",
                    marker=dict(color="purple", size=6, symbol="star"),
                    text=plan_fan_labels,
                    hovertemplate="%{text}<br>時刻: %{x}<br>ファン: %{y}<extra></extra>",
                ),
                row=3,
                col=1,
            )

        # 運転状態（実績/計画）
        if not sub_a.empty and "A/C ON/OFF" in sub_a.columns:
            # ON/OFF状態を文字列に変換
            onoff_labels = (
                sub_a["A/C ON/OFF"].map({0: "OFF", 1: "ON"}).fillna("UNKNOWN")
            )

            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["A/C ON/OFF"],
                    name="ON/OFF(実績)",
                    mode="lines",
                    line=dict(color="black", width=2),
                    text=onoff_labels,
                    hovertemplate="%{text}<br>時刻: %{x}<br>状態: %{y}<extra></extra>",
                ),
                row=4,
                col=1,
            )
        if cols["on"] in sub_p.columns:
            # 計画のON/OFF状態を数値に変換
            plan_onoff_numeric = (sub_p[cols["on"]] == "ON").astype(int)
            plan_onoff_labels = sub_p[cols["on"]]

            fig.add_trace(
                go.Scatter(
                    x=sub_p["Date Time"],
                    y=plan_onoff_numeric,
                    name="ON/OFF(計画)",
                    mode="lines",
                    line=dict(color="black", width=2, dash="dash"),
                    text=plan_onoff_labels,
                    hovertemplate="%{text}<br>時刻: %{x}<br>状態: %{y}<extra></extra>",
                ),
                row=4,
                col=1,
            )

        # 外気温（実績/予報）
        if not sub_a.empty and "Outdoor Temp." in sub_a.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["Outdoor Temp."],
                    name="外気温(実績)",
                    line=dict(color="orange", width=2),
                    mode="lines",
                ),
                row=5,
                col=1,
            )
        if (
            weather_forecast is not None
            and not weather_forecast.empty
            and "Outdoor Temp." in weather_forecast.columns
        ):
            fig.add_trace(
                go.Scatter(
                    x=weather_forecast["datetime"],
                    y=weather_forecast["Outdoor Temp."],
                    name="外気温(予報)",
                    line=dict(color="orange", width=2, dash="dash"),
                    mode="lines",
                ),
                row=5,
                col=1,
            )

        # 外気湿度・日射量（実績/予報）
        if not sub_a.empty and "Outdoor Humidity" in sub_a.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["Outdoor Humidity"],
                    name="外気湿度(実績)",
                    line=dict(color="lightblue", width=2),
                    mode="lines",
                ),
                row=6,
                col=1,
            )
        if (
            weather_forecast is not None
            and not weather_forecast.empty
            and "Outdoor Humidity" in weather_forecast.columns
        ):
            fig.add_trace(
                go.Scatter(
                    x=weather_forecast["datetime"],
                    y=weather_forecast["Outdoor Humidity"],
                    name="外気湿度(予報)",
                    line=dict(color="lightblue", width=2, dash="dash"),
                    mode="lines",
                ),
                row=6,
                col=1,
            )
        if not sub_a.empty and "Solar Radiation" in sub_a.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub_a["datetime"],
                    y=sub_a["Solar Radiation"],
                    name="日射量(実績)",
                    line=dict(color="yellow", width=2),
                    mode="lines",
                    yaxis="y12",
                ),
                row=6,
                col=1,
                secondary_y=True,
            )
        if (
            weather_forecast is not None
            and not weather_forecast.empty
            and "Solar Radiation" in weather_forecast.columns
        ):
            fig.add_trace(
                go.Scatter(
                    x=weather_forecast["datetime"],
                    y=weather_forecast["Solar Radiation"],
                    name="日射量(予報)",
                    line=dict(color="yellow", width=2, dash="dash"),
                    mode="lines",
                    yaxis="y12",
                ),
                row=6,
                col=1,
                secondary_y=True,
            )

        # 計画開始の縦線とシェーディング
        fig.add_vline(x=plan_start, line_width=2, line_dash="dash", line_color="red")
        fig.add_vrect(
            x0=plan_start,
            x1=sub_p["Date Time"].max(),
            fillcolor="rgba(255,0,0,0.05)",
            line_width=0,
        )

        # 副軸の設定
        fig.update_yaxes(title_text="湿度 (%)", row=6, col=1)
        fig.update_yaxes(title_text="日射量 (W/m²)", row=6, col=1, secondary_y=True)

        fig.update_layout(
            title=f"{store_name} 計画妥当性ダッシュボード（{z}）",
            height=1500,
            template="plotly_white",
        )
        out_dir = os.path.join("analysis/output", store_name, z)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "plan_validation.html")
        fig.write_html(out)
        print(f"✅ 計画妥当性ダッシュボード出力: {out}")
