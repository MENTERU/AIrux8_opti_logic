import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# =============================
# STEP2: 予測モデル（制御エリア別）
# =============================
@dataclass
class EnvPowerModels:
    temp_model: RandomForestRegressor
    hum_model: Optional[RandomForestRegressor]
    power_model: RandomForestRegressor
    feature_cols: List[str]


class ModelBuilder:
    def __init__(self, store_name: str):
        self.store_name = store_name

    @staticmethod
    def _split_xy(df: pd.DataFrame, target: str, feature_cols: List[str]):
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = df[target].astype(float)
        mask = y.notna()
        return X[mask], y[mask]

    def train_by_zone(
        self, area_df: pd.DataFrame, master: dict
    ) -> Dict[str, EnvPowerModels]:
        print(
            f"[ModelBuilder] Starting train_by_zone. Input shape: {area_df.shape if area_df is not None else 'None'}"
        )
        if area_df is None or area_df.empty:
            print("[ModelBuilder] Area data is None or empty")
            return {}
        models: Dict[str, EnvPowerModels] = {}
        feats = [
            "A/C Set Temperature",
            "Indoor Temp. Lag1",
            "A/C ON/OFF",
            "A/C Mode",
            "A/C Fan Speed",
            "Outdoor Temp.",
            "Outdoor Humidity",
            "Solar Radiation",  # 日射量を追加
        ]
        zones = sorted(area_df["zone"].dropna().unique().tolist())
        print(f"[ModelBuilder] Found zones: {zones}")
        for z in zones:
            sub = area_df[area_df["zone"] == z].copy()
            print(f"[ModelBuilder] Zone {z}: {len(sub)} records")
            if len(sub) < 20:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient data: {len(sub)} < 20)"
                )
                continue
            # 温度
            X_t, y_t = self._split_xy(sub, "Indoor Temp.", feats)
            Xtr, Xte, ytr, yte = train_test_split(
                X_t, y_t, test_size=0.2, random_state=42
            )
            temp_model = RandomForestRegressor(n_estimators=200, random_state=42)
            temp_model.fit(Xtr, ytr)
            # 湿度（実績が無い場合はNone）
            hum_model = None
            if "室内湿度" in sub.columns and sub["室内湿度"].notna().sum() > 10:
                X_h, y_h = self._split_xy(sub, "室内湿度", feats)
                Xtrh, Xteh, ytrh, yteh = train_test_split(
                    X_h, y_h, test_size=0.2, random_state=42
                )
                hum_model = RandomForestRegressor(n_estimators=200, random_state=42)
                hum_model.fit(Xtrh, ytrh)
            # 電力（adjusted_power）
            print(f"[ModelBuilder] Zone {z}: Checking adjusted_power column...")
            if "adjusted_power" not in sub.columns:
                print(
                    f"[ModelBuilder] Zone {z}: No adjusted_power column found. Available columns: {list(sub.columns)}"
                )
                continue
            power_count = sub["adjusted_power"].notna().sum()
            print(
                f"[ModelBuilder] Zone {z}: adjusted_power non-null values: {power_count}"
            )
            if power_count < 10:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient power data: {power_count} < 10)"
                )
                continue
            X_p, y_p = self._split_xy(sub, "adjusted_power", feats)
            Xtrp, Xtep, ytrp, ytep = train_test_split(
                X_p, y_p, test_size=0.2, random_state=42
            )
            power_model = RandomForestRegressor(n_estimators=200, random_state=42)
            power_model.fit(Xtrp, ytrp)
            # ざっくり評価
            y_hat_t = temp_model.predict(Xte)
            print(
                f"[Model] {z} Temp: MAE={mean_absolute_error(yte, y_hat_t):.2f} R2={r2_score(yte, y_hat_t):.3f}"
            )
            y_hat_p = power_model.predict(Xtep)
            print(
                f"[Model] {z} Power: MAE={mean_absolute_error(ytep, y_hat_p):.1f} R2={r2_score(ytep, y_hat_p):.3f}"
            )
            models[z] = EnvPowerModels(
                temp_model=temp_model,
                hum_model=hum_model,
                power_model=power_model,
                feature_cols=feats,
            )
        # 保存
        from config.utils import get_data_path

        mdir = os.path.join(get_data_path("models_path"), self.store_name)
        os.makedirs(mdir, exist_ok=True)
        for z, pack in models.items():
            joblib.dump(
                {
                    "temp_model": pack.temp_model,
                    "hum_model": pack.hum_model,
                    "power_model": pack.power_model,
                    "feature_cols": pack.feature_cols,
                },
                os.path.join(mdir, f"models_{z}.pkl"),
            )
        return models
