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
    def _rm_outliers(df: pd.DataFrame, cols: List[str], k: float = 3.0) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                m, s = df[c].mean(), df[c].std()
                df = df[(df[c] - m).abs() <= k * s]
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
        self, df: Optional[pd.DataFrame], std_k: float = 3.0
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        df, dt = self._unify_datetime(df)
        if df is None:
            return None
        df = self._rm_dup(df, dt)
        df = self._rm_outliers(
            df, ["A/C Set Temperature", "Indoor Temp.", "Outdoor Temp."], std_k
        )
        for c in ["A/C Set Temperature", "Indoor Temp.", "Outdoor Temp."]:
            if c in df.columns:
                df[c] = df[c].interpolate("linear")
        # カテゴリ変換（固定ルール。必要ならマスタで上書き可）
        mapping = {
            "A/C ON/OFF": {"0": 0, "1": 1, 0: 0, 1: 1},
            "A/C Mode": {"COOL": 0, "DEHUM": 1, "FAN": 2, "HEAT": 3},
            "A/C Fan Speed": {"Auto": 0, "Low": 1, "Medium": 2, "High": 3, "Top": 4},
        }
        for c, m in mapping.items():
            if c in df.columns:
                df[c] = df[c].map(m).fillna(-1)
        df["datetime"] = df[dt]
        df["date"] = df[dt].dt.date
        return df

    def preprocess_pm(
        self, df: Optional[pd.DataFrame], std_k: float = 3.0
    ) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        df, dt = self._unify_datetime(df)
        if df is None:
            return None
        df = self._rm_dup(df, dt)
        df = self._rm_outliers(df, ["Phase A"], std_k)
        df["Phase A"] = df["Phase A"].fillna(0)
        df["datetime"] = df[dt]
        df["date"] = df[dt].dt.date
        return df

    def save(self, ac: Optional[pd.DataFrame], pm: Optional[pd.DataFrame]):
        if ac is not None:
            ac.to_csv(
                os.path.join(
                    self.output_dir, f"ac_control_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )
        if pm is not None:
            pm.to_csv(
                os.path.join(
                    self.output_dir, f"power_meter_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )
