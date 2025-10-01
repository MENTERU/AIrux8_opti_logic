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

    def preprocess_ac(
        self,
        dataframe: Optional[pd.DataFrame],
        standard_deviation_multiplier: float = 5.0,
        category_mapping: Optional[dict] = None,
    ) -> Optional[pd.DataFrame]:
        if dataframe is None or dataframe.empty:
            return None
        dataframe, datetime_column = self._unify_datetime(dataframe)
        if dataframe is None:
            return None
        dataframe = self._rm_dup(dataframe, datetime_column)
        dataframe = self._rm_outliers(
            dataframe,
            ["A/C Set Temperature", "Indoor Temp.", "Outdoor Temp."],
            standard_deviation_multiplier,
        )
        for column in ["A/C Set Temperature", "Indoor Temp.", "Outdoor Temp."]:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].interpolate("linear")

        # カテゴリ変換（設定から取得、デフォルト値も設定）
        if category_mapping is None:
            from config.utils import load_config

            config = load_config()
            category_mapping = config.get("preprocessing", {}).get(
                "category_mapping", {}
            )

        for column, mapping_dict in category_mapping.items():
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].map(mapping_dict).fillna(-1)
        dataframe["datetime"] = dataframe[datetime_column]
        dataframe["date"] = dataframe[datetime_column].dt.date
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
        dataframe = self._rm_outliers(
            dataframe, ["Phase A"], standard_deviation_multiplier
        )
        dataframe["Phase A"] = dataframe["Phase A"].fillna(0)
        dataframe["datetime"] = dataframe[datetime_column]
        dataframe["date"] = dataframe[datetime_column].dt.date
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
