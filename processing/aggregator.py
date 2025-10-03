from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from processing.utilities.category_mapping_loader import (
    get_default_category_value,
    map_category_series,
)


# =============================
# STEP1: 集約（制御エリア単位テーブル）
# =============================
class AreaAggregator:
    """制御エリア単位に、空調・電力・天候を1時間単位で統合"""

    def __init__(self, master_info: dict):
        self.m = master_info

    @staticmethod
    def _most_frequent(s: pd.Series):
        return s.mode().iloc[0] if not s.mode().empty else np.nan

    def build(
        self,
        ac: Optional[pd.DataFrame],
        pm: Optional[pd.DataFrame],
        weather: Optional[pd.DataFrame],
        freq: str = "1H",
        apply_zone_mapping: bool = True,
    ) -> pd.DataFrame:
        if self.m is None or "zones" not in self.m:
            raise ValueError("マスタに zones がありません")
        zones = self.m["zones"]

        # 天候（共通）
        weather = weather.copy() if weather is not None else pd.DataFrame()
        if not weather.empty:
            # 天気データの列名を統一（datetime -> Datetime）
            if "datetime" in weather.columns:
                weather["Datetime"] = pd.to_datetime(weather["datetime"]).dt.floor(freq)
            elif "Datetime" in weather.columns:
                weather["Datetime"] = pd.to_datetime(weather["Datetime"]).dt.floor(freq)
            else:
                print(
                    f"⚠️ 天気データにDatetime列が見つかりません。利用可能な列: {list(weather.columns)}"
                )
                return pd.DataFrame()
            wcols = [
                c
                for c in [
                    "Outdoor Temp.",
                    "Outdoor Humidity",
                    "Solar Radiation",
                    "temperature C",
                    "humidity",
                ]
                if c in weather.columns
            ]
            weather = (
                weather[["Datetime"] + wcols]
                .groupby("Datetime")
                .agg("mean")
                .reset_index()
            )
            # 列名統一
            if (
                "temperature C" in weather.columns
                and "Outdoor Temp." not in weather.columns
            ):
                weather.rename(columns={"temperature C": "Outdoor Temp."}, inplace=True)
            if (
                "humidity" in weather.columns
                and "Outdoor Humidity" not in weather.columns
            ):
                weather.rename(columns={"humidity": "Outdoor Humidity"}, inplace=True)

        # 制御エリアごとにテーブル構築
        area_rows = []
        for zone_name, zinfo in zones.items():
            # 室内機一覧
            indoor_units: List[str] = []
            # 室外機: {id: {load_share: x}}
            outdoor_units: Dict[str, dict] = zinfo.get("outdoor_units", {})
            for _, ou in outdoor_units.items():
                indoor_units.extend(ou.get("indoor_units", []))
            indoor_units = list(dict.fromkeys(indoor_units))  # unique & keep order

            # 空調（室内機）: 1時間ごと 最頻値/平均
            if ac is not None and not ac.empty and indoor_units:
                ac_sub = ac[ac["A/C Name"].isin(indoor_units)].copy()
                if not ac_sub.empty:
                    # エリア別カテゴリカル変数マッピングを適用
                    if apply_zone_mapping:
                        ac_sub = self._apply_zone_categorical_mapping(ac_sub, zone_name)

                    ac_sub["Datetime"] = pd.to_datetime(ac_sub["Datetime"]).dt.floor(
                        freq
                    )
                    g = (
                        ac_sub.groupby("Datetime")
                        .agg(
                            {
                                "A/C Set Temperature": AreaAggregator._most_frequent,
                                "Indoor Temp.": "mean",  # 学習は平均室温
                                "A/C ON/OFF": AreaAggregator._most_frequent,
                                "A/C Mode": AreaAggregator._most_frequent,
                                "A/C Fan Speed": AreaAggregator._most_frequent,
                            }
                        )
                        .reset_index()
                    )

                    # Create A/C Status column based on ON/OFF and Mode
                    # Status mapping: OFF=0, COOL=1, HEAT=2, FAN=3
                    if "A/C ON/OFF" in g.columns and "A/C Mode" in g.columns:
                        g["A/C Status"] = 0  # Default to OFF
                        # If AC is ON, use the mode value
                        on_mask = g["A/C ON/OFF"] == 1
                        g.loc[on_mask, "A/C Status"] = g.loc[on_mask, "A/C Mode"]
                        # Convert to integer type (not float)
                        g["A/C Status"] = g["A/C Status"].fillna(0).astype(int)
                        print(
                            f"[AreaAggregator] Zone {zone_name}: Created A/C Status column"
                        )
                else:
                    g = pd.DataFrame(
                        columns=[
                            "Datetime",
                            "A/C Set Temperature",
                            "Indoor Temp.",
                            "A/C ON/OFF",
                            "A/C Mode",
                            "A/C Fan Speed",
                            "A/C Status",
                        ]
                    )
            else:
                g = pd.DataFrame(
                    columns=[
                        "Datetime",
                        "A/C Set Temperature",
                        "Indoor Temp.",
                        "A/C ON/OFF",
                        "A/C Mode",
                        "A/C Fan Speed",
                        "A/C Status",
                    ]
                )

            # 電力（室外機×負荷率の合計）
            p_list = []
            if pm is not None and not pm.empty and outdoor_units:
                print(
                    f"[AreaAggregator] Zone {zone_name}: Processing {len(outdoor_units)} outdoor units"
                )

                for ou_id, ou in outdoor_units.items():
                    share = float(ou.get("load_share", 1.0))

                    # Try exact match first
                    sub = pm[pm["Mesh ID"] == ou_id].copy()

                    # If no exact match, try extracting the base number (e.g., "49-1" -> 49)
                    if sub.empty and "-" in str(ou_id):
                        base_id = int(str(ou_id).split("-")[0])
                        sub = pm[pm["Mesh ID"] == base_id].copy()

                    if sub.empty:
                        continue

                    print(
                        f"[AreaAggregator] Found {len(sub)} records for Mesh ID: {ou_id}"
                    )

                    # Total_kWh列の存在確認
                    if "Total_kWh" not in sub.columns:
                        print(
                            f"⚠️ Total_kWh列が存在しません。利用可能な列: {list(sub.columns)}"
                        )
                        if "Phase A" in sub.columns:
                            print(f"  Phase A列を使用します")
                            sub["Total_kWh"] = sub["Phase A"]
                        else:
                            print(f"  ❌ 電力データが見つかりません")
                            continue

                    sub["Datetime"] = pd.to_datetime(sub["Datetime"]).dt.floor(freq)
                    sub = sub.groupby("Datetime")["Total_kWh"].sum().reset_index()
                    sub["adjusted_power"] = sub["Total_kWh"] * share

                    print(
                        f"  Total_kWh統計: 平均={sub['Total_kWh'].mean():.2f}, 最大={sub['Total_kWh'].max():.2f}"
                    )
                    print(
                        f"  adjusted_power統計: 平均={sub['adjusted_power'].mean():.2f}, 最大={sub['adjusted_power'].max():.2f}"
                    )

                    p_list.append(sub[["Datetime", "adjusted_power"]])
            if p_list:
                p = (
                    pd.concat(p_list, ignore_index=True)
                    .groupby("Datetime")["adjusted_power"]
                    .sum()
                    .reset_index()
                )

                print(f"[AreaAggregator] 電力データ統合結果:")
                print(f"  統合前レコード数: {len(p_list)}")
                print(f"  統合後レコード数: {len(p)}")
                print(f"  adjusted_power欠損値: {p['adjusted_power'].isnull().sum()}件")
                print(
                    f"  adjusted_power統計: 平均={p['adjusted_power'].mean():.2f}, 最大={p['adjusted_power'].max():.2f}"
                )
            else:
                p = pd.DataFrame(columns=["Datetime", "adjusted_power"])
                print(f"[AreaAggregator] 電力データがありません")

            # マージ
            df = g.merge(p, on="Datetime", how="outer")

            print(f"[AreaAggregator] マージ後:")
            print(f"  dfレコード数: {len(df)}")
            print(f"  adjusted_power欠損値: {df['adjusted_power'].isnull().sum()}件")
            if not weather.empty:
                df = df.merge(weather, on="Datetime", how="left")

            # adjusted_powerの欠損値分析
            missing_power = df["adjusted_power"].isnull().sum()
            if missing_power > 0:
                print(f"⚠️ adjusted_powerに欠損値が{missing_power}件あります")

                # 欠損値の原因分析
                missing_df = df[df["adjusted_power"].isnull()].copy()
                print(f"  欠損値の詳細分析:")
                print(f"    欠損レコード数: {len(missing_df)}")

                # 時間範囲の確認
                if not missing_df.empty:
                    print(
                        f"    欠損期間: {missing_df['Datetime'].min()} ～ {missing_df['Datetime'].max()}"
                    )

                    # 電力データが存在するかチェック
                    has_power_data = (
                        "adjusted_power" in df.columns
                        and not df["adjusted_power"].isnull().all()
                    )
                    if has_power_data:
                        non_missing_count = df["adjusted_power"].notnull().sum()
                        print(f"    電力データ存在: {non_missing_count}件")
                        print(
                            f"    電力データ欠損率: {missing_power / len(df) * 100:.1f}%"
                        )
                    else:
                        print(f"    ❌ 電力データが全く存在しません")

                    # 空調データとの比較
                    if "Indoor Temp." in df.columns:
                        temp_missing = df["Indoor Temp."].isnull().sum()
                        print(f"    室温データ欠損: {temp_missing}件")
                        if temp_missing == 0:
                            print(f"    ⚠️ 室温データは存在するが電力データが欠損")
                        else:
                            print(f"    ⚠️ 室温データも欠損している可能性")

                # 電力データの統合前後の状況確認
                if p_list:
                    print(f"  電力データ統合前の状況:")
                    print(f"    統合前レコード数: {len(p_list)}")
                    print(f"    統合後レコード数: {len(p)}")
                    print(f"    統合後欠損値: {p['adjusted_power'].isnull().sum()}件")
                else:
                    print(f"    ❌ 電力データが統合されていません（p_listが空）")

                # マージの状況確認
                print(f"  マージ状況:")
                print(f"    空調データレコード数: {len(g)}")
                print(f"    電力データレコード数: {len(p)}")
                print(f"    マージ後レコード数: {len(df)}")

                # 時間範囲の重複確認
                if not g.empty and not p.empty:
                    g_time_range = (g["Datetime"].min(), g["Datetime"].max())
                    p_time_range = (p["Datetime"].min(), p["Datetime"].max())
                    print(
                        f"    空調データ時間範囲: {g_time_range[0]} ～ {g_time_range[1]}"
                    )
                    print(
                        f"    電力データ時間範囲: {p_time_range[0]} ～ {p_time_range[1]}"
                    )

                    # 時間範囲の重複チェック
                    overlap_start = max(g_time_range[0], p_time_range[0])
                    overlap_end = min(g_time_range[1], p_time_range[1])
                    if overlap_start <= overlap_end:
                        print(
                            f"    ✅ 時間範囲に重複があります: {overlap_start} ～ {overlap_end}"
                        )
                    else:
                        print(f"    ❌ 時間範囲に重複がありません")
            else:
                print(f"✅ adjusted_powerに欠損値はありません")
                print(
                    f"  adjusted_power統計: 平均={df['adjusted_power'].mean():.2f}, 最大={df['adjusted_power'].max():.2f}"
                )

            df["zone"] = zone_name
            df.sort_values("Datetime", inplace=True)
            area_rows.append(df)

        area_df = (
            pd.concat(area_rows, ignore_index=True) if area_rows else pd.DataFrame()
        )
        # ラグ（前時刻室温）
        if not area_df.empty:
            # 時間特徴量の付与（曜日・時刻・月・週末）
            area_df["Datetime"] = pd.to_datetime(area_df["Datetime"])  # 安全化
            area_df["Date"] = area_df["Datetime"].dt.date
            area_df["DayOfWeek"] = area_df["Datetime"].dt.dayofweek.astype(int)
            area_df["Hour"] = area_df["Datetime"].dt.hour.astype(int)
            area_df["Month"] = area_df["Datetime"].dt.month.astype(int)
            area_df["IsWeekend"] = area_df["DayOfWeek"].isin([5, 6]).astype(int)
            # 祝日フラグ（jpholidayが利用可能なら使用、なければ0）
            try:
                import jpholiday  # type: ignore

                area_df["IsHoliday"] = (
                    area_df["Datetime"]
                    .dt.date.map(lambda d: 1 if jpholiday.is_holiday(d) else 0)
                    .astype(int)
                )
            except Exception:
                area_df["IsHoliday"] = 0
            area_df["Indoor Temp. Lag1"] = (
                area_df.sort_values(["zone", "Datetime"])
                .groupby("zone")["Indoor Temp."]
                .shift(1)
            )
            area_df["Indoor Temp. Lag1"].fillna(area_df["Indoor Temp."], inplace=True)

            # 温度を小数点第1位に丸める
            if "Indoor Temp." in area_df.columns:
                area_df["Indoor Temp."] = area_df["Indoor Temp."].round(1)
            if "Indoor Temp. Lag1" in area_df.columns:
                area_df["Indoor Temp. Lag1"] = area_df["Indoor Temp. Lag1"].round(1)
            if "Outdoor Temp." in area_df.columns:
                area_df["Outdoor Temp."] = area_df["Outdoor Temp."].round(1)

            # 列の並び順を調整（Datetime, Dateを最初に配置）
            cols = list(area_df.columns)
            if "Datetime" in cols:
                cols.remove("Datetime")
            if "Date" in cols:
                cols.remove("Date")

            # Datetime, Dateを最初に配置
            area_df = area_df[["Datetime", "Date"] + cols]

        return area_df

    def _apply_zone_categorical_mapping(
        self, dataframe: pd.DataFrame, zone_name: str
    ) -> pd.DataFrame:
        """エリア別のカテゴリカル変数マッピングを適用"""
        import json
        import os
        from datetime import datetime

        # ログファイルの準備
        log_dir = f"logs/preprocessing/{self.m.get('store_name', 'unknown')}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"zone_mapping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # エリア別のマッピングログ
        zone_mapping_log = {
            "store_name": self.m.get("store_name", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "zones": {},
        }

        print(f"\n[AreaAggregator] エリア '{zone_name}' のカテゴリカル変数処理開始")
        zone_log = {
            "zone_name": zone_name,
            "total_records": len(dataframe),
            "categorical_mappings": {},
        }

        # 各カテゴリカル変数を処理
        for column in ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed"]:
            if column in dataframe.columns:
                print(f"[AreaAggregator] {zone_name} - {column} 処理中...")

                # エリア固有の値の分析
                unique_values = dataframe[column].value_counts()
                print(
                    f"[AreaAggregator] {zone_name} - {column} ユニーク値: {unique_values.to_dict()}"
                )

                original_series = dataframe[column]
                mapped_series, applied_mapping, unmapped_values = map_category_series(
                    original_series, column
                )
                dataframe[column] = mapped_series

                zone_log_entry = {
                    "original_values": unique_values.to_dict(),
                    "mapping": applied_mapping,
                    "mapped_count": len(applied_mapping),
                    "unmapped_count": int(sum(unmapped_values.values())),
                }
                if unmapped_values:
                    zone_log_entry["unmapped_values"] = unmapped_values
                zone_log["categorical_mappings"][column] = zone_log_entry

                if unmapped_values:
                    print(
                        f"[AreaAggregator] {zone_name} - {column} マッピングされなかった値: {unmapped_values}"
                    )
                    unmapped_mask = mapped_series.isna() & original_series.notna()
                    default_value = get_default_category_value(column)
                    if default_value is not None:
                        dataframe.loc[unmapped_mask, column] = default_value
                        zone_log_entry["default_value"] = default_value
                        print(
                            f"[AreaAggregator] {zone_name} - {column} デフォルト値({default_value})で置換: {int(unmapped_mask.sum())}件"
                        )

                dataframe[column] = dataframe[column].astype(pd.Int64Dtype())

        zone_mapping_log["zones"][zone_name] = zone_log

        # ログファイルに保存（安全な書き込み）
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(zone_mapping_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ ログファイル保存エラー: {e}")
            # バックアップファイルに保存
            backup_file = log_file.replace(".json", "_backup.json")
            try:
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(zone_mapping_log, f, ensure_ascii=False, indent=2)
                print(f"📋 バックアップファイルに保存: {backup_file}")
            except Exception as backup_e:
                print(f"❌ バックアップ保存も失敗: {backup_e}")

        print(f"\n[AreaAggregator] エリア別マッピングログ保存: {log_file}")

        return dataframe
