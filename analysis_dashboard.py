#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HVAC分析ダッシュボード
制御エリア・室外機・室内機単位での運転条件と電力・環境の関係を分析
"""

import json
import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots

from processing.utilities.category_mapping_loader import get_inverse_category_mapping

MODE_CODE_TO_LABEL = get_inverse_category_mapping("A/C Mode")
FAN_CODE_TO_LABEL = get_inverse_category_mapping("A/C Fan Speed")

# ページ設定
st.set_page_config(
    page_title="HVAC分析ダッシュボード",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# カスタムCSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 2rem;
        color: #A23B72;
        margin: 1rem 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2E86AB;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stPlotlyChart {
        width: 100% !important;
    }
    .plot-container {
        margin: 2rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# デフォルトカラーパレット
DEFAULT_COLORS = [
    "#2E86AB",  # 青
    "#A23B72",  # 紫
    "#F18F01",  # オレンジ
    "#C73E1D",  # 赤
    "#4CAF50",  # 緑
    "#9C27B0",  # 紫
    "#FF9800",  # オレンジ
    "#2196F3",  # 青
]


@st.cache_data
def load_data():
    """データ読み込み（キャッシュ付き）"""
    try:
        # 実績データ
        features_df = pd.read_csv(
            "data/02_PreprocessedData/Clea/features_processed_Clea.csv"
        )
        features_df["Datetime"] = pd.to_datetime(features_df["Datetime"])

        # マスタデータ
        with open("data/01_MasterData/MASTER_Clea.json", "r", encoding="utf-8") as f:
            master_data = json.load(f)

        # 計画データ（最新）
        plan_files = [
            f
            for f in os.listdir("data/04_PlanningData/Clea/")
            if f.startswith("control_type_schedule_") and f.endswith(".csv")
        ]
        if plan_files:
            latest_plan = sorted(plan_files)[-1]
            plan_df = pd.read_csv(f"data/04_PlanningData/Clea/{latest_plan}")
            plan_df["Date Time"] = pd.to_datetime(plan_df["Date Time"])
        else:
            plan_df = None

        return features_df, master_data, plan_df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None, None


def resample_data(df, freq):
    """データの時間粒度を変更"""
    if freq == "時別":
        return df
    elif freq == "日別":
        # 日別に集約（平均値）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_dict = {col: "mean" for col in numeric_cols}
        agg_dict["Datetime"] = "first"  # 最初の時刻を保持

        # カテゴリカル変数は最頻値
        categorical_cols = ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed", "zone"]
        for col in categorical_cols:
            if col in df.columns:
                agg_dict[col] = lambda x: (
                    x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
                )

        return df.groupby(df["Datetime"].dt.date).agg(agg_dict).reset_index()
    else:
        return df


def create_ac_status_analysis(df, zone, start_date, end_date, freq):
    """A/C状態分析プロット作成（縦並び）"""
    zone_data = df[df["zone"] == zone].copy()
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    zone_data = zone_data[
        (zone_data["Datetime"] >= start_datetime)
        & (zone_data["Datetime"] <= end_datetime)
    ]

    if zone_data.empty:
        return None

    # データの時間粒度を変更
    zone_data = resample_data(zone_data, freq)

    # サブプロット作成（縦並び）
    fig = make_subplots(
        rows=5,
        cols=1,
        subplot_titles=[
            f"{zone} - A/C ON/OFF状態",
            f"{zone} - 設定温度",
            f"{zone} - 運転モード",
            f"{zone} - 風量設定",
            f"{zone} - 室温・電力",
        ],
        vertical_spacing=0.08,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],  # Row 5 needs secondary_y for power
        ],
    )

    # 1. A/C ON/OFF状態
    onoff_numeric = zone_data["A/C ON/OFF"].fillna(0).astype(int)
    fig.add_trace(
        go.Scatter(
            x=zone_data["Datetime"],
            y=onoff_numeric,
            name="ON/OFF",
            line=dict(color=DEFAULT_COLORS[0], width=3),
            mode="lines+markers",
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )

    # 2. 設定温度
    fig.add_trace(
        go.Scatter(
            x=zone_data["Datetime"],
            y=zone_data["A/C Set Temperature"],
            name="設定温度",
            line=dict(color=DEFAULT_COLORS[1], width=3),
            mode="lines+markers",
            marker=dict(size=6),
        ),
        row=2,
        col=1,
    )

    # 3. 運転モード
    mode_numeric = zone_data["A/C Mode"].fillna(0).astype(int)
    fig.add_trace(
        go.Scatter(
            x=zone_data["Datetime"],
            y=mode_numeric,
            name="運転モード",
            line=dict(color=DEFAULT_COLORS[2], width=3),
            mode="lines+markers",
            marker=dict(size=6),
        ),
        row=3,
        col=1,
    )

    # 4. 風量設定
    fan_numeric = zone_data["A/C Fan Speed"].fillna(0).astype(int)
    fig.add_trace(
        go.Scatter(
            x=zone_data["Datetime"],
            y=fan_numeric,
            name="風量設定",
            line=dict(color=DEFAULT_COLORS[3], width=3),
            mode="lines+markers",
            marker=dict(size=6),
        ),
        row=4,
        col=1,
    )

    # 5. 室温・電力（2軸）
    fig.add_trace(
        go.Scatter(
            x=zone_data["Datetime"],
            y=zone_data["Indoor Temp."],
            name="室温",
            line=dict(color=DEFAULT_COLORS[4], width=3),
            mode="lines+markers",
            marker=dict(size=6),
        ),
        row=5,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=zone_data["Datetime"],
            y=zone_data["adjusted_power"] / 1000,
            name="電力(kW)",
            line=dict(color=DEFAULT_COLORS[5], width=3),
            mode="lines+markers",
            marker=dict(size=6),
        ),
        row=5,
        col=1,
        secondary_y=True,
    )

    # レイアウト設定
    fig.update_layout(
        height=1500,
        showlegend=True,
        title=f"{zone} - A/C状態分析 ({freq})",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # Y軸ラベル
    fig.update_yaxes(title_text="ON/OFF", row=1, col=1)
    fig.update_yaxes(title_text="設定温度(°C)", row=2, col=1)
    fig.update_yaxes(title_text="運転モード", row=3, col=1)
    fig.update_yaxes(title_text="風量設定", row=4, col=1)
    fig.update_yaxes(title_text="室温(°C)", row=5, col=1)
    fig.update_yaxes(title_text="電力(kW)", row=5, col=1, secondary_y=True)

    return fig


def create_scatter_analysis(df, zone, freq):
    """散布図分析（相関係数の代わり）"""
    zone_data = df[df["zone"] == zone].copy()

    if zone_data.empty:
        return None

    # データの時間粒度を変更
    zone_data = resample_data(zone_data, freq)

    # サブプロット作成（2x2）
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "電力 vs 設定温度",
            "電力 vs 室温",
            "室温 vs 設定温度",
            "電力 vs 運転モード",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # 1. 電力 vs 設定温度
    fig.add_trace(
        go.Scatter(
            x=zone_data["A/C Set Temperature"],
            y=zone_data["adjusted_power"] / 1000,
            mode="markers",
            name="電力 vs 設定温度",
            opacity=0.7,
            marker=dict(color=DEFAULT_COLORS[0], size=8),
            text=zone_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M"),
            hovertemplate="設定温度: %{x}°C<br>電力: %{y}kW<br>時刻: %{text}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. 電力 vs 室温
    fig.add_trace(
        go.Scatter(
            x=zone_data["Indoor Temp."],
            y=zone_data["adjusted_power"] / 1000,
            mode="markers",
            name="電力 vs 室温",
            opacity=0.7,
            marker=dict(color=DEFAULT_COLORS[1], size=8),
            text=zone_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M"),
            hovertemplate="室温: %{x}°C<br>電力: %{y}kW<br>時刻: %{text}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. 室温 vs 設定温度
    fig.add_trace(
        go.Scatter(
            x=zone_data["A/C Set Temperature"],
            y=zone_data["Indoor Temp."],
            mode="markers",
            name="室温 vs 設定温度",
            opacity=0.7,
            marker=dict(color=DEFAULT_COLORS[2], size=8),
            text=zone_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M"),
            hovertemplate="設定温度: %{x}°C<br>室温: %{y}°C<br>時刻: %{text}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 4. 電力 vs 運転モード
    mode_numeric = zone_data["A/C Mode"].fillna(0).astype(int)
    fig.add_trace(
        go.Scatter(
            x=mode_numeric,
            y=zone_data["adjusted_power"] / 1000,
            mode="markers",
            name="電力 vs 運転モード",
            opacity=0.7,
            marker=dict(color=DEFAULT_COLORS[3], size=8),
            text=zone_data["Datetime"].dt.strftime("%Y-%m-%d %H:%M"),
            hovertemplate="運転モード: %{x}<br>電力: %{y}kW<br>時刻: %{text}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # レイアウト設定
    fig.update_layout(
        height=800,
        showlegend=False,
        title=f"{zone} - 散布図分析 ({freq})",
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # 軸ラベル
    fig.update_xaxes(title_text="設定温度(°C)", row=1, col=1)
    fig.update_yaxes(title_text="電力(kW)", row=1, col=1)
    fig.update_xaxes(title_text="室温(°C)", row=1, col=2)
    fig.update_yaxes(title_text="電力(kW)", row=1, col=2)
    fig.update_xaxes(title_text="設定温度(°C)", row=2, col=1)
    fig.update_yaxes(title_text="室温(°C)", row=2, col=1)
    fig.update_xaxes(title_text="運転モード", row=2, col=2)
    fig.update_yaxes(title_text="電力(kW)", row=2, col=2)

    return fig


def create_mode_analysis(df, zone, freq):
    """モード別分析"""
    zone_data = df[df["zone"] == zone].copy()

    # データの時間粒度を変更
    zone_data = resample_data(zone_data, freq)

    # モード名マッピング
    zone_data["Mode_Name"] = (
        zone_data["A/C Mode"].map(MODE_CODE_TO_LABEL).fillna("UNKNOWN")
    )

    # モード別統計
    mode_stats = (
        zone_data.groupby("Mode_Name")
        .agg(
            {
                "adjusted_power": ["count", "mean", "std"],
                "Indoor Temp.": ["mean", "std"],
                "A/C Set Temperature": ["mean", "std"],
            }
        )
        .round(2)
    )

    # モード別電力消費
    fig1 = px.box(
        zone_data,
        x="Mode_Name",
        y="adjusted_power",
        title=f"{zone} - モード別電力消費 ({freq})",
        labels={"adjusted_power": "電力(W)", "Mode_Name": "運転モード"},
        color="Mode_Name",
        color_discrete_sequence=DEFAULT_COLORS,
    )

    # モード別室温
    fig2 = px.box(
        zone_data,
        x="Mode_Name",
        y="Indoor Temp.",
        title=f"{zone} - モード別室温 ({freq})",
        labels={"Indoor Temp.": "室温(°C)", "Mode_Name": "運転モード"},
        color="Mode_Name",
        color_discrete_sequence=DEFAULT_COLORS,
    )

    # レイアウト設定
    fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=500)
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=500)

    return fig1, fig2, mode_stats


def main():
    """メイン処理"""
    st.markdown(
        '<h1 class="main-header">🌡️ HVAC分析ダッシュボード</h1>', unsafe_allow_html=True
    )

    # データ読み込み
    with st.spinner("データ読み込み中..."):
        features_df, master_data, plan_df = load_data()

    if features_df is None:
        st.error("データの読み込みに失敗しました")
        return

    # サイドバー設定
    st.sidebar.header("🔧 分析設定")

    # ゾーン選択
    zones = features_df["zone"].unique()
    selected_zone = st.sidebar.selectbox("制御エリア選択", zones)

    # 時間粒度選択
    freq = st.sidebar.selectbox("時間粒度", ["時別", "日別"])

    # 期間選択
    min_date = features_df["Datetime"].min().date()
    max_date = features_df["Datetime"].max().date()

    start_date = st.sidebar.date_input("開始日", min_date)
    end_date = st.sidebar.date_input("終了日", max_date)

    # 分析タイプ選択
    analysis_type = st.sidebar.selectbox(
        "分析タイプ",
        ["A/C状態分析", "散布図分析", "モード別分析", "全体概要"],
    )

    # メインコンテンツ
    if analysis_type == "A/C状態分析":
        st.markdown('<h2 class="sub-header">A/C状態分析</h2>', unsafe_allow_html=True)

        fig = create_ac_status_analysis(
            features_df, selected_zone, start_date, end_date, freq
        )
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("選択した期間にデータがありません")

    elif analysis_type == "散布図分析":
        st.markdown('<h2 class="sub-header">散布図分析</h2>', unsafe_allow_html=True)

        fig = create_scatter_analysis(features_df, selected_zone, freq)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("データがありません")

    elif analysis_type == "モード別分析":
        st.markdown('<h2 class="sub-header">モード別分析</h2>', unsafe_allow_html=True)

        fig1, fig2, mode_stats = create_mode_analysis(features_df, selected_zone, freq)

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("📋 モード別統計")
        st.dataframe(mode_stats)

    elif analysis_type == "全体概要":
        st.markdown('<h2 class="sub-header">全体概要</h2>', unsafe_allow_html=True)

        # 基本統計
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("総レコード数", f"{len(features_df):,}")
        with col2:
            st.metric("制御エリア数", len(zones))
        with col3:
            st.metric("データ期間", f"{(max_date - min_date).days}日")
        with col4:
            avg_power = features_df["adjusted_power"].mean() / 1000
            st.metric("平均電力", f"{avg_power:.1f}kW")

        # ゾーン別統計
        st.subheader("🏢 ゾーン別統計")
        zone_stats = (
            features_df.groupby("zone")
            .agg(
                {
                    "adjusted_power": ["count", "mean", "std"],
                    "Indoor Temp.": ["mean", "std"],
                    "A/C ON/OFF": "mean",
                }
            )
            .round(2)
        )

        st.dataframe(zone_stats)

        # 電力分布
        fig = px.histogram(
            features_df,
            x="adjusted_power",
            title="電力消費分布",
            labels={"adjusted_power": "電力(W)", "count": "頻度"},
            color_discrete_sequence=DEFAULT_COLORS,
        )
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=500)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
