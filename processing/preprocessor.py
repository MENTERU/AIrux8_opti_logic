import glob
import os
from typing import List, Optional, Tuple

import pandas as pd


# =============================
# STEP1: 前処理
# =============================
class DataPreprocessor:
    def __init__(self, store_name: str):
        self.store_name = store_name
        from config.utils import get_data_path

        self.data_dir = os.path.join(get_data_path("raw_data_path"), store_name)
        self.output_dir = os.path.join(get_data_path("processed_data_path"), store_name)
        os.makedirs(self.output_dir, exist_ok=True)

    # 共通
    @staticmethod
    def _unify_datetime(
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        cols = [c for c in df.columns if "datetime" in c.lower() or "日時" in c]
        if not cols:
            return None, None
        col = cols[0]
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None).dt.floor("T")
        return df, col

    @staticmethod
    def _rm_dup(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
        dev_col = (
            "A/C Name"
            if "A/C Name" in df.columns
            else ("Mesh ID" if "Mesh ID" in df.columns else None)
        )
        return df.drop_duplicates(subset=[dt_col, dev_col]) if dev_col else df

    @staticmethod
    def _rm_outliers(
        df: pd.DataFrame, columns: List[str], standard_deviation_multiplier: float = 3.0
    ) -> pd.DataFrame:
        for column in columns:
            if column in df.columns:
                mean_value, standard_deviation = df[column].mean(), df[column].std()
                df = df[
                    (df[column] - mean_value).abs()
                    <= standard_deviation_multiplier * standard_deviation
                ]
        return df

    def load_raw(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        print(f"[DataPreprocessor] Loading raw data from: {self.data_dir}")
        print(f"[DataPreprocessor] Directory exists: {os.path.exists(self.data_dir)}")

        ac_files = glob.glob(f"{self.data_dir}/**/ac-control-*.csv", recursive=True)
        pm_files = glob.glob(f"{self.data_dir}/**/ac-power-meter-*.csv", recursive=True)

        print(f"[DataPreprocessor] Found {len(ac_files)} AC control files")
        print(f"[DataPreprocessor] Found {len(pm_files)} power meter files")

        if ac_files:
            print(
                f"[DataPreprocessor] AC files: {ac_files[:1]}..."
            )  # Show first 3 files
        if pm_files:
            print(
                f"[DataPreprocessor] PM files: {pm_files[:1]}..."
            )  # Show first 3 files

        ac = (
            pd.concat([pd.read_csv(f) for f in ac_files], ignore_index=True)
            if ac_files
            else None
        )
        pm = (
            pd.concat([pd.read_csv(f) for f in pm_files], ignore_index=True)
            if pm_files
            else None
        )

        print(
            f"[DataPreprocessor] AC data shape: {ac.shape if ac is not None else 'None'}"
        )
        print(
            f"[DataPreprocessor] PM data shape: {pm.shape if pm is not None else 'None'}"
        )

        return ac, pm

    def _apply_categorical_mapping(
        self, dataframe: pd.DataFrame, column: str, mapping_dict: dict
    ) -> pd.DataFrame:
        """共通のカテゴリカル変数マッピングを適用"""
        # マッピング前の値の確認
        original_values = dataframe[column].value_counts()
        print(
            f"[DataPreprocessor] {column} マッピング前の値: "
            f"{original_values.head().to_dict()}"
        )

        # マッピング実行
        dataframe[column] = dataframe[column].map(mapping_dict)

        # マッピングされなかった値の確認
        unmapped_mask = dataframe[column].isnull()
        if unmapped_mask.any():
            unmapped_values = dataframe.loc[unmapped_mask, column].value_counts()
            print(
                f"[DataPreprocessor] {column} マッピングされなかった値: "
                f"{unmapped_values.to_dict()}"
            )

            # デフォルト値の設定
            if column == "A/C ON/OFF":
                default_value = 0  # OFFをデフォルト
            elif column == "A/C Mode":
                default_value = 2  # FANをデフォルト
            elif column == "A/C Fan Speed":
                default_value = 1  # Lowをデフォルト
            else:
                default_value = 0

            dataframe[column] = dataframe[column].fillna(default_value)
            print(
                f"[DataPreprocessor] {column} デフォルト値({default_value})で置換: "
                f"{unmapped_mask.sum()}件"
            )
        else:
            print(f"[DataPreprocessor] {column} 全ての値が正常にマッピングされました")

        return dataframe

    def _apply_zone_specific_mapping(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """エリア別のカテゴリカル変数マッピングを適用"""
        import json
        import os
        from datetime import datetime

        # ログファイルの準備
        log_dir = f"logs/preprocessing/{self.store_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"zone_mapping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # エリア別のマッピングログ
        zone_mapping_log = {
            "store_name": self.store_name,
            "timestamp": datetime.now().isoformat(),
            "zones": {},
        }

        # エリア別に処理
        if "A/C Name" in dataframe.columns:
            # A/C Nameからエリアを推定（命名規則に基づく）
            dataframe["zone"] = dataframe["A/C Name"].str.extract(
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)"
            )[0]
            dataframe["zone"] = dataframe["zone"].fillna("Unknown")
        else:
            dataframe["zone"] = "Unknown"

        for zone in dataframe["zone"].unique():
            if zone == "Unknown":
                continue

            print(f"\n[DataPreprocessor] エリア '{zone}' のカテゴリカル変数処理開始")
            zone_data = dataframe[dataframe["zone"] == zone].copy()
            zone_log = {
                "zone_name": zone,
                "total_records": len(zone_data),
                "categorical_mappings": {},
            }

            # 各カテゴリカル変数を処理
            for column in ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed"]:
                if column in zone_data.columns:
                    print(f"[DataPreprocessor] {zone} - {column} 処理中...")

                    # エリア固有の値の分析
                    unique_values = zone_data[column].value_counts()
                    print(
                        f"[DataPreprocessor] {zone} - {column} ユニーク値: {unique_values.to_dict()}"
                    )

                    # 自動マッピング生成
                    mapping = self._generate_zone_mapping(zone, column, unique_values)
                    zone_log["categorical_mappings"][column] = {
                        "original_values": unique_values.to_dict(),
                        "mapping": mapping,
                        "mapped_count": len(unique_values),
                        "unmapped_count": 0,
                    }

                    # マッピング適用
                    zone_data[column] = zone_data[column].map(mapping)

                    # マッピングされなかった値の処理
                    unmapped_mask = zone_data[column].isnull()
                    if unmapped_mask.any():
                        unmapped_values = zone_data.loc[
                            unmapped_mask, column
                        ].value_counts()
                        print(
                            f"[DataPreprocessor] {zone} - {column} マッピングされなかった値: {unmapped_values.to_dict()}"
                        )

                        # デフォルト値設定
                        default_value = self._get_default_value(column)
                        zone_data[column] = zone_data[column].fillna(default_value)

                        zone_log["categorical_mappings"][column][
                            "unmapped_values"
                        ] = unmapped_values.to_dict()
                        zone_log["categorical_mappings"][column][
                            "unmapped_count"
                        ] = unmapped_mask.sum()
                        zone_log["categorical_mappings"][column][
                            "default_value"
                        ] = default_value

                        print(
                            f"[DataPreprocessor] {zone} - {column} デフォルト値({default_value})で置換: {unmapped_mask.sum()}件"
                        )

                    # 元のデータフレームを更新
                    dataframe.loc[dataframe["zone"] == zone, column] = zone_data[column]

            zone_mapping_log["zones"][zone] = zone_log

        # ログファイルに保存
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(zone_mapping_log, f, ensure_ascii=False, indent=2)

        print(f"\n[DataPreprocessor] エリア別マッピングログ保存: {log_file}")

        # エリア列を削除（元のデータ構造を維持）
        dataframe = dataframe.drop("zone", axis=1)

        return dataframe

    def _generate_zone_mapping(
        self, zone: str, column: str, unique_values: pd.Series
    ) -> dict:
        """エリア固有のマッピングを自動生成"""
        mapping = {}

        for value in unique_values.index:
            if pd.isna(value):
                continue

            # 値の正規化
            normalized_value = str(value).strip().upper()

            if column == "A/C ON/OFF":
                if any(
                    keyword in normalized_value
                    for keyword in ["ON", "1", "TRUE", "有効"]
                ):
                    mapping[value] = 1
                elif any(
                    keyword in normalized_value
                    for keyword in ["OFF", "0", "FALSE", "無効"]
                ):
                    mapping[value] = 0
                else:
                    # デフォルトはOFF
                    mapping[value] = 0

            elif column == "A/C Mode":
                if any(
                    keyword in normalized_value
                    for keyword in ["COOL", "冷房", "COOLING"]
                ):
                    mapping[value] = 0
                elif any(
                    keyword in normalized_value
                    for keyword in ["DEHUM", "除湿", "DEHUMIDIFY"]
                ):
                    mapping[value] = 1
                elif any(
                    keyword in normalized_value
                    for keyword in ["FAN", "送風", "FAN_ONLY"]
                ):
                    mapping[value] = 2
                elif any(
                    keyword in normalized_value
                    for keyword in ["HEAT", "暖房", "HEATING"]
                ):
                    mapping[value] = 3
                else:
                    # デフォルトはFAN
                    mapping[value] = 2

            elif column == "A/C Fan Speed":
                if any(
                    keyword in normalized_value
                    for keyword in ["AUTO", "自動", "AUTOMATIC"]
                ):
                    mapping[value] = 0
                elif any(keyword in normalized_value for keyword in ["LOW", "低", "1"]):
                    mapping[value] = 1
                elif any(
                    keyword in normalized_value
                    for keyword in ["MEDIUM", "中", "2", "MID"]
                ):
                    mapping[value] = 2
                elif any(
                    keyword in normalized_value for keyword in ["HIGH", "高", "3"]
                ):
                    mapping[value] = 3
                elif any(
                    keyword in normalized_value
                    for keyword in ["TOP", "最高", "4", "MAX"]
                ):
                    mapping[value] = 4
                else:
                    # デフォルトはLow
                    mapping[value] = 1

        return mapping

    def _get_default_value(self, column: str) -> int:
        """カラム別のデフォルト値を取得"""
        if column == "A/C ON/OFF":
            return 0  # OFF
        elif column == "A/C Mode":
            return 2  # FAN
        elif column == "A/C Fan Speed":
            return 1  # Low
        else:
            return 0

    def preprocess_ac(
        self,
        dataframe: Optional[pd.DataFrame],
        standard_deviation_multiplier: float = 5.0,
        category_mapping: Optional[dict] = None,
        zone_specific_mapping: bool = True,
    ) -> Optional[pd.DataFrame]:
        if dataframe is None or dataframe.empty:
            return None
        dataframe, datetime_column = self._unify_datetime(dataframe)
        if dataframe is None:
            return None
        dataframe = self._rm_dup(dataframe, datetime_column)
        dataframe = self._rm_outliers(
            dataframe,
            ["Indoor Temp.", "Outdoor Temp."],
            standard_deviation_multiplier,
        )
        for column in ["Indoor Temp.", "Outdoor Temp."]:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].interpolate("linear")

        # カテゴリ変換は前処理段階では行わない
        # 後段階（集約時）でエリア別マッピングを実行
        print("[DataPreprocessor] カテゴリカル変数のマッピングは後段階で実行します")

        # 列名を統一（Datetime, Date）
        dataframe["Datetime"] = dataframe[datetime_column]
        dataframe["Date"] = dataframe[datetime_column].dt.date

        # 列の並び順を調整（Datetime, Dateを最初に配置）
        cols = list(dataframe.columns)
        if "Datetime" in cols:
            cols.remove("Datetime")
        if "Date" in cols:
            cols.remove("Date")

        # Datetime, Dateを最初に配置
        dataframe = dataframe[["Datetime", "Date"] + cols]

        return dataframe

    def preprocess_pm(
        self,
        dataframe: Optional[pd.DataFrame],
        standard_deviation_multiplier: float = 5.0,
    ) -> Optional[pd.DataFrame]:
        if dataframe is None or dataframe.empty:
            return None
        dataframe, datetime_column = self._unify_datetime(dataframe)
        if dataframe is None:
            return None
        dataframe = self._rm_dup(dataframe, datetime_column)
        # Total_kWh列を先に作成してから外れ値除去
        phase_columns = [col for col in dataframe.columns if col.startswith("Phase")]
        if phase_columns:
            dataframe["Total_kWh"] = dataframe[phase_columns].sum(axis=1)
        else:
            dataframe["Total_kWh"] = dataframe["Phase A"]

        dataframe = self._rm_outliers(
            dataframe, ["Total_kWh"], standard_deviation_multiplier
        )
        dataframe["Phase A"] = dataframe["Phase A"].fillna(0)
        dataframe["Total_kWh"] = dataframe["Total_kWh"].fillna(0)

        # 列名を統一（Datetime, Date）
        dataframe["Datetime"] = dataframe[datetime_column]
        dataframe["Date"] = dataframe[datetime_column].dt.date

        # 列の並び順を調整（Datetime, Dateを最初に配置）
        cols = list(dataframe.columns)
        if "Datetime" in cols:
            cols.remove("Datetime")
        if "Date" in cols:
            cols.remove("Date")

        # Datetime, Dateを最初に配置
        dataframe = dataframe[["Datetime", "Date"] + cols]

        return dataframe

    def save(
        self,
        ac_control_data: Optional[pd.DataFrame],
        power_meter_data: Optional[pd.DataFrame],
    ):
        if ac_control_data is not None:
            ac_control_data.to_csv(
                os.path.join(
                    self.output_dir, f"ac_control_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )
        if power_meter_data is not None:
            power_meter_data.to_csv(
                os.path.join(
                    self.output_dir, f"power_meter_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )
