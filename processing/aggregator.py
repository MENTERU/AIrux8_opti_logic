from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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
    ) -> pd.DataFrame:
        if self.m is None or "zones" not in self.m:
            raise ValueError("マスタに zones がありません")
        zones = self.m["zones"]

        # 天候（共通）
        weather = weather.copy() if weather is not None else pd.DataFrame()
        if not weather.empty:
            weather["datetime"] = pd.to_datetime(weather["datetime"]).dt.floor(freq)
            wcols = [
                c
                for c in [
                    "Outdoor Temp.",
                    "Outdoor Humidity",
                    "temperature C",
                    "humidity",
                ]
                if c in weather.columns
            ]
            weather = (
                weather[["datetime"] + wcols]
                .groupby("datetime")
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
                    ac_sub["datetime"] = pd.to_datetime(ac_sub["datetime"]).dt.floor(
                        freq
                    )
                    g = (
                        ac_sub.groupby("datetime")
                        .agg(
                            {
                                "A/C Set Temperature": "mean",  # 参考用
                                "Indoor Temp.": "mean",  # 平均室温
                                "A/C ON/OFF": AreaAggregator._most_frequent,
                                "A/C Mode": AreaAggregator._most_frequent,
                                "A/C Fan Speed": AreaAggregator._most_frequent,
                            }
                        )
                        .reset_index()
                    )
                else:
                    g = pd.DataFrame(
                        columns=[
                            "datetime",
                            "A/C Set Temperature",
                            "Indoor Temp.",
                            "A/C ON/OFF",
                            "A/C Mode",
                            "A/C Fan Speed",
                        ]
                    )
            else:
                g = pd.DataFrame(
                    columns=[
                        "datetime",
                        "A/C Set Temperature",
                        "Indoor Temp.",
                        "A/C ON/OFF",
                        "A/C Mode",
                        "A/C Fan Speed",
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
                    sub["datetime"] = pd.to_datetime(sub["datetime"]).dt.floor(freq)
                    sub = sub.groupby("datetime")["Phase A"].sum().reset_index()
                    sub["adjusted_power"] = sub["Phase A"] * share
                    p_list.append(sub[["datetime", "adjusted_power"]])
            if p_list:
                p = (
                    pd.concat(p_list, ignore_index=True)
                    .groupby("datetime")["adjusted_power"]
                    .sum()
                    .reset_index()
                )
            else:
                p = pd.DataFrame(columns=["datetime", "adjusted_power"])

            # マージ
            df = g.merge(p, on="datetime", how="outer")
            if not weather.empty:
                df = df.merge(weather, on="datetime", how="left")
            df["zone"] = zone_name
            df.sort_values("datetime", inplace=True)
            area_rows.append(df)

        area_df = (
            pd.concat(area_rows, ignore_index=True) if area_rows else pd.DataFrame()
        )
        # ラグ（前時刻室温）
        if not area_df.empty:
            area_df["Indoor Temp. Lag1"] = (
                area_df.sort_values(["zone", "datetime"])
                .groupby("zone")["Indoor Temp."]
                .shift(1)
            )
            area_df["Indoor Temp. Lag1"].fillna(area_df["Indoor Temp."], inplace=True)
        return area_df
