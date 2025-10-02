import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================
# STEP2: 予測モデル（制御エリア別）
# =============================
@dataclass
class EnvPowerModels:
    temp_model: Any
    hum_model: Optional[Any]
    power_model: Any
    multi_output_model: Optional[Any]  # マルチアウトプットモデル
    feature_cols: List[str]


class ModelBuilder:
    def __init__(self, store_name: str):
        self.store_name = store_name

    @staticmethod
    def _ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _shap_on_pipeline(
        pipeline: Any, X: pd.DataFrame, out_png: str, title: str
    ) -> None:
        try:
            if not isinstance(pipeline, Pipeline):
                return
            model = pipeline.named_steps.get("model")
            if model is None:
                return
            # サンプリング（重すぎ防止）
            Xs = X.sample(min(len(X), 2000), random_state=42) if len(X) > 0 else X
            if Xs is None or Xs.empty:
                return
            data_for_model = Xs
            scaler = pipeline.named_steps.get("scaler")
            if scaler is not None:
                try:
                    data_for_model = scaler.transform(Xs)
                except Exception:
                    pass
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(data_for_model)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                data_for_model,
                feature_names=Xs.columns,
                show=False,
                plot_type="bar",
            )
            plt.title(title)
            plt.tight_layout()
            plt.savefig(out_png)
            plt.close()
        except Exception as e:
            print(f"[ModelBuilder] SHAP plot failed: {e}")

    @staticmethod
    def _split_xy(df: pd.DataFrame, target: str, feature_cols: List[str]):
        """特徴量と単一ターゲットを返す。欠損を含む行は除外する。"""
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        y = df[target]
        combined = pd.concat([X, y], axis=1)
        combined = combined.dropna()
        X_clean = combined[feature_cols].astype(float)
        y_clean = combined[target].astype(float)
        return X_clean, y_clean

    @staticmethod
    def _split_multi(df: pd.DataFrame, targets: List[str], feature_cols: List[str]):
        """特徴量と複数ターゲットを返す。欠損を含む行は除外する。"""
        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        Y = df[targets]
        combined = pd.concat([X, Y], axis=1)
        combined = combined.dropna()
        X_clean = combined[feature_cols].astype(float)
        Y_clean = combined[targets].astype(float)
        return X_clean, Y_clean

    @staticmethod
    def _compute_additional_features(df: pd.DataFrame) -> pd.DataFrame:
        """追加特徴量を作成（ON連続長, Hour×DayOfWeek など）"""
        out = df.copy()
        # HourOfWeek = DayOfWeek*24 + Hour
        if all(c in out.columns for c in ["DayOfWeek", "Hour"]):
            out["HourOfWeek"] = out["DayOfWeek"] * 24 + out["Hour"]
        # ON/OFF連続長（A/C ON/OFFが存在する場合）
        if "A/C ON/OFF" in out.columns and "Datetime" in out.columns:
            out = out.sort_values("Datetime")
            onoff = out["A/C ON/OFF"].fillna(0).astype(int).values
            run = 0
            runs = []
            for v in onoff:
                if v > 0:
                    run += 1
                else:
                    run = 0
                runs.append(run)
            out["OnRunLength"] = runs
        return out

    def train_by_zone(
        self, area_df: pd.DataFrame, master: dict
    ) -> Dict[str, EnvPowerModels]:
        print(
            f"[ModelBuilder] Starting train_by_zone. "
            f"Input shape: {area_df.shape if area_df is not None else 'None'}"
        )
        if area_df is None or area_df.empty:
            print("[ModelBuilder] Area data is None or empty")
            return {}
        models: Dict[str, EnvPowerModels] = {}
        base_feats = [
            "A/C Set Temperature",
            "Indoor Temp. Lag1",
            "A/C ON/OFF",
            "A/C Mode",
            "A/C Fan Speed",
            "Outdoor Temp.",
            "Outdoor Humidity",
            "Solar Radiation",
            "DayOfWeek",
            "Hour",
            "Month",
            "IsWeekend",
            "IsHoliday",
            # 追加特徴量（存在すれば使う）
            "HourOfWeek",
            "OnRunLength",
        ]
        zones = sorted(area_df["zone"].dropna().unique().tolist())
        print(f"[ModelBuilder] Found zones: {zones}")
        for z in zones:
            sub = area_df[area_df["zone"] == z].copy()
            if "Datetime" in sub.columns:
                sub = self._compute_additional_features(sub)
            print(f"[ModelBuilder] Zone {z}: {len(sub)} records")
            if len(sub) < 20:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient data: {len(sub)} < 20)"
                )
                continue
            feats = [c for c in base_feats if c in sub.columns]
            if feats:
                nonnull_ratio = sub[feats].notna().mean()
                feats = [c for c in feats if nonnull_ratio.get(c, 0.0) >= 0.9]
            if len(feats) < 5:
                print(f"[ModelBuilder] Zone {z}: insufficient features {feats}")
            # 温度
            X_t, y_t = self._split_xy(sub, "Indoor Temp.", feats)
            if len(X_t) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient temp samples after dropping NaNs: {len(X_t)} < 5)"
                )
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X_t, y_t, test_size=0.2, random_state=42
            )
            temp_model = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=400,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            temp_model.fit(Xtr, ytr)
            # 湿度
            hum_model = None
            if "室内湿度" in sub.columns and sub["室内湿度"].notna().sum() > 10:
                X_h, y_h = self._split_xy(sub, "室内湿度", feats)
                Xtrh, Xteh, ytrh, yteh = train_test_split(
                    X_h, y_h, test_size=0.2, random_state=42
                )
                hum_model = Pipeline(
                    steps=[
                        ("scaler", MinMaxScaler()),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=400,
                                max_depth=6,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                hum_model.fit(Xtrh, ytrh)
            # 電力
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
            if len(X_p) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient power samples after dropping NaNs: {len(X_p)} < 5)"
                )
                continue
            sample_weight_series = None
            if "A/C ON/OFF" in sub.columns:
                try:
                    onoff_aligned = sub.loc[X_p.index, "A/C ON/OFF"].fillna(0)
                    sample_weight_series = pd.Series(
                        np.where(onoff_aligned > 0, 1.0, 0.2), index=X_p.index
                    )
                except Exception:
                    sample_weight_series = None
            if sample_weight_series is not None:
                Xtrp, Xtep, ytrp, ytep, wtrp, wtep = train_test_split(
                    X_p, y_p, sample_weight_series, test_size=0.2, random_state=42
                )
            else:
                Xtrp, Xtep, ytrp, ytep = train_test_split(
                    X_p, y_p, test_size=0.2, random_state=42
                )
                wtrp = None
            power_base = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=600,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.9,
                            random_state=42,
                            n_jobs=-1,
                            objective="reg:pseudohubererror",
                        ),
                    ),
                ]
            )
            power_model = PowerLogModel(power_base)
            if wtrp is not None:
                power_model.fit(Xtrp, ytrp, sample_weight=wtrp)
            else:
                power_model.fit(Xtrp, ytrp)
            # マルチ出力
            multi_output_model = None
            try:
                multi_targets = ["Indoor Temp.", "adjusted_power"]
                if all(t in sub.columns for t in multi_targets):
                    Xm, Ym = self._split_multi(sub, multi_targets, feats)
                    if len(Xm) >= 20:
                        Xm_tr, Xm_te, Ym_tr, Ym_te = train_test_split(
                            Xm, Ym, test_size=0.2, random_state=42
                        )
                        base_xgb = Pipeline(
                            steps=[
                                ("scaler", MinMaxScaler()),
                                (
                                    "model",
                                    XGBRegressor(
                                        n_estimators=500,
                                        max_depth=6,
                                        learning_rate=0.05,
                                        subsample=0.8,
                                        colsample_bytree=0.9,
                                        random_state=42,
                                        n_jobs=-1,
                                        objective="reg:pseudohubererror",
                                    ),
                                ),
                            ]
                        )
                        multi_output_model = MultiOutputRegressor(base_xgb)
                        multi_output_model.fit(Xm_tr, Ym_tr)
                        Ym_hat = pd.DataFrame(
                            multi_output_model.predict(Xm_te),
                            index=Ym_te.index,
                            columns=multi_targets,
                        )
                        temp_mae = mean_absolute_error(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        power_mae = mean_absolute_error(
                            Ym_te["adjusted_power"] / 1000.0,
                            Ym_hat["adjusted_power"] / 1000.0,
                        )
                        temp_mape = mean_absolute_percentage_error(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        nonzero_mask = (Ym_te["adjusted_power"] / 1000.0) != 0
                        power_mape = (
                            mean_absolute_percentage_error(
                                (Ym_te.loc[nonzero_mask, "adjusted_power"] / 1000.0),
                                (Ym_hat.loc[nonzero_mask, "adjusted_power"] / 1000.0),
                            )
                            if nonzero_mask.any()
                            else np.nan
                        )
                        temp_r2 = r2_score(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        power_r2 = r2_score(
                            Ym_te["adjusted_power"], Ym_hat["adjusted_power"]
                        )
                        print(
                            f"[Model] {z} Multi-Output (Temp/Power): Temp MAE={temp_mae:.2f} MAPE={temp_mape*100:.1f}% R2={temp_r2:.3f} | Power MAE(kW)={power_mae:.3f} MAPE={(power_mape*100 if not np.isnan(power_mape) else np.nan):.1f}% R2={power_r2:.3f}"
                        )
            except Exception as me:
                print(f"[ModelBuilder] Multi-output model error for zone {z}: {me}")
            # 評価
            y_hat_t = temp_model.predict(Xte)
            temp_mae = mean_absolute_error(yte, y_hat_t)
            temp_mape = mean_absolute_percentage_error(yte, y_hat_t)
            print(
                f"[Model] {z} Temp: MAE={temp_mae:.2f} MAPE={temp_mape*100:.1f}% R2={r2_score(yte, y_hat_t):.3f}"
            )
            y_hat_p = power_model.predict(Xtep)
            ytep_kw = ytep / 1000.0
            yhat_kw = pd.Series(y_hat_p, index=ytep.index) / 1000.0
            nonzero_mask_p = ytep_kw != 0
            power_mape = (
                mean_absolute_percentage_error(
                    ytep_kw[nonzero_mask_p], yhat_kw[nonzero_mask_p]
                )
                if nonzero_mask_p.any()
                else np.nan
            )
            print(
                f"[Model] {z} Power: MAE(kW)={mean_absolute_error(ytep_kw, yhat_kw):.3f} MAPE={(power_mape*100 if not np.isnan(power_mape) else np.nan):.1f}% R2={r2_score(ytep, y_hat_p):.3f}"
            )
            # SHAP
            out_dir = os.path.join("analysis", "output", self.store_name, z)
            ModelBuilder._ensure_dir(out_dir)
            try:
                self._shap_on_pipeline(
                    temp_model,
                    Xte,
                    os.path.join(out_dir, "shap_temp.png"),
                    f"SHAP - Temp ({z})",
                )
            except Exception as e:
                print(f"[ModelBuilder] SHAP temp failed ({z}): {e}")
            try:
                self._shap_on_pipeline(
                    (
                        power_model.pipeline
                        if hasattr(power_model, "pipeline")
                        else power_model
                    ),
                    Xtep,
                    os.path.join(out_dir, "shap_power.png"),
                    f"SHAP - Power ({z})",
                )
            except Exception as e:
                print(f"[ModelBuilder] SHAP power failed ({z}): {e}")
            models[z] = EnvPowerModels(
                temp_model=temp_model,
                hum_model=hum_model,
                power_model=power_model,
                multi_output_model=multi_output_model,
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
                    "multi_output_model": pack.multi_output_model,
                    "feature_cols": pack.feature_cols,
                },
                os.path.join(mdir, f"models_{z}.pkl"),
            )
        return models


class PowerLogModel:
    """目的変数log1pで学習し、予測をexpm1で返すラッパー。"""

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        y_log = np.log1p(y)
        if sample_weight is not None:
            # Pipeline内の最終推定器にsample_weightを渡す
            self.pipeline.fit(X, y_log, model__sample_weight=sample_weight)
        else:
            self.pipeline.fit(X, y_log)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_log_pred = self.pipeline.predict(X)
        # クリップでexpオーバーフローと負値を抑制
        y_log_pred = np.clip(y_log_pred, -20.0, 20.0)
        y_pred = np.expm1(y_log_pred)
        # 非負にクリップし、極端な外れを現実的上限で抑制（W単位）
        y_pred = np.clip(y_pred, 0.0, 2_000_000.0)
        # 数値化（inf/NaN除去）
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=2_000_000.0, neginf=0.0)
        return y_pred

    def train_by_zone(
        self, area_df: pd.DataFrame, master: dict
    ) -> Dict[str, EnvPowerModels]:
        print(
            f"[ModelBuilder] Starting train_by_zone. "
            f"Input shape: {area_df.shape if area_df is not None else 'None'}"
        )
        if area_df is None or area_df.empty:
            print("[ModelBuilder] Area data is None or empty")
            return {}
        models: Dict[str, EnvPowerModels] = {}
        base_feats = [
            "A/C Set Temperature",
            "Indoor Temp. Lag1",
            "A/C ON/OFF",
            "A/C Mode",
            "A/C Fan Speed",
            "Outdoor Temp.",
            "Outdoor Humidity",
            "Solar Radiation",  # 日射量
            # 時間特徴
            "DayOfWeek",
            "Hour",
            "Month",
            "IsWeekend",
            "IsHoliday",
        ]
        zones = sorted(area_df["zone"].dropna().unique().tolist())
        print(f"[ModelBuilder] Found zones: {zones}")
        for z in zones:
            sub = area_df[area_df["zone"] == z].copy()
            # 追加特徴量
            if "Datetime" in sub.columns:
                sub = self._compute_additional_features(sub)
            print(f"[ModelBuilder] Zone {z}: {len(sub)} records")
            if len(sub) < 20:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped "
                    f"(insufficient data: {len(sub)} < 20)"
                )
                continue
            # 利用可能な特徴量に自動絞り込み（欠損が多い列は除外）
            feats = [c for c in base_feats if c in sub.columns]
            if feats:
                nonnull_ratio = sub[feats].notna().mean()
                # 90%以上非欠損の列のみ使用
                feats = [c for c in feats if nonnull_ratio.get(c, 0.0) >= 0.9]
            if len(feats) < 5:
                print(f"[ModelBuilder] Zone {z}: insufficient features {feats}")
            # 温度
            X_t, y_t = self._split_xy(sub, "Indoor Temp.", feats)
            if len(X_t) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient temp samples after dropping NaNs: {len(X_t)} < 5)"
                )
                continue
            Xtr, Xte, ytr, yte = train_test_split(
                X_t, y_t, test_size=0.2, random_state=42
            )
            temp_model = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=400,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            random_state=42,
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            temp_model.fit(Xtr, ytr)
            # 湿度（実績が無い場合はNone）
            hum_model = None
            if "室内湿度" in sub.columns and sub["室内湿度"].notna().sum() > 10:
                X_h, y_h = self._split_xy(sub, "室内湿度", feats)
                Xtrh, Xteh, ytrh, yteh = train_test_split(
                    X_h, y_h, test_size=0.2, random_state=42
                )
                hum_model = Pipeline(
                    steps=[
                        ("scaler", MinMaxScaler()),
                        (
                            "model",
                            XGBRegressor(
                                n_estimators=400,
                                max_depth=6,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                random_state=42,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
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
                f"[ModelBuilder] Zone {z}: adjusted_power non-null values: "
                f"{power_count}"
            )
            if power_count < 10:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped "
                    f"(insufficient power data: {power_count} < 10)"
                )
                continue
            X_p, y_p = self._split_xy(sub, "adjusted_power", feats)
            if len(X_p) < 5:
                print(
                    f"[ModelBuilder] Zone {z}: Skipped (insufficient power samples after dropping NaNs: {len(X_p)} < 5)"
                )
                continue
            # サンプル重み: 非稼働(OFF)データの影響を抑える（ON=1.0, OFF=0.2）
            sample_weight_series = None
            if "A/C ON/OFF" in sub.columns:
                try:
                    onoff_aligned = sub.loc[X_p.index, "A/C ON/OFF"].fillna(0)
                    sample_weight_series = pd.Series(
                        np.where(onoff_aligned > 0, 1.0, 0.2), index=X_p.index
                    )
                except Exception:
                    sample_weight_series = None

            if sample_weight_series is not None:
                Xtrp, Xtep, ytrp, ytep, wtrp, wtep = train_test_split(
                    X_p, y_p, sample_weight_series, test_size=0.2, random_state=42
                )
            else:
                Xtrp, Xtep, ytrp, ytep = train_test_split(
                    X_p, y_p, test_size=0.2, random_state=42
                )
                wtrp = None

            power_base = Pipeline(
                steps=[
                    ("scaler", MinMaxScaler()),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=600,
                            max_depth=6,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.9,
                            random_state=42,
                            n_jobs=-1,
                            objective="reg:pseudohubererror",
                        ),
                    ),
                ]
            )
            # 目的変数のlog1p変換ラッパー
            power_model = PowerLogModel(power_base)
            if wtrp is not None:
                power_model.fit(Xtrp, ytrp, sample_weight=wtrp)
            else:
                power_model.fit(Xtrp, ytrp)

            # マルチアウトプットモデル（室温・電力を同時学習）
            multi_output_model = None
            try:
                multi_targets = ["Indoor Temp.", "adjusted_power"]
                # 利用可能チェック
                if all(t in sub.columns for t in multi_targets):
                    Xm, Ym = self._split_multi(sub, multi_targets, feats)
                    if len(Xm) >= 20:
                        Xm_tr, Xm_te, Ym_tr, Ym_te = train_test_split(
                            Xm, Ym, test_size=0.2, random_state=42
                        )
                        base_xgb = Pipeline(
                            steps=[
                                ("scaler", MinMaxScaler()),
                                (
                                    "model",
                                    XGBRegressor(
                                        n_estimators=500,
                                        max_depth=6,
                                        learning_rate=0.05,
                                        subsample=0.8,
                                        colsample_bytree=0.9,
                                        random_state=42,
                                        n_jobs=-1,
                                        objective="reg:pseudohubererror",
                                    ),
                                ),
                            ]
                        )
                        multi_output_model = MultiOutputRegressor(base_xgb)
                        multi_output_model.fit(Xm_tr, Ym_tr)
                        Ym_hat = pd.DataFrame(
                            multi_output_model.predict(Xm_te),
                            index=Ym_te.index,
                            columns=multi_targets,
                        )
                        # 評価（MAE / MAPE / R2）
                        temp_mae = mean_absolute_error(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        power_mae = mean_absolute_error(
                            Ym_te["adjusted_power"] / 1000.0,
                            Ym_hat["adjusted_power"] / 1000.0,
                        )

                        # MAPEは0除算回避（ゼロは除外）
                        temp_mape = mean_absolute_percentage_error(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        nonzero_mask = (Ym_te["adjusted_power"] / 1000.0) != 0
                        power_mape = (
                            mean_absolute_percentage_error(
                                (Ym_te.loc[nonzero_mask, "adjusted_power"] / 1000.0),
                                (Ym_hat.loc[nonzero_mask, "adjusted_power"] / 1000.0),
                            )
                            if nonzero_mask.any()
                            else np.nan
                        )

                        temp_r2 = r2_score(
                            Ym_te["Indoor Temp."], Ym_hat["Indoor Temp."]
                        )
                        power_r2 = r2_score(
                            Ym_te["adjusted_power"], Ym_hat["adjusted_power"]
                        )

                        print(
                            f"[Model] {z} Multi-Output (Temp/Power): "
                            f"Temp MAE={temp_mae:.2f} MAPE={temp_mape*100:.1f}% R2={temp_r2:.3f} | "
                            f"Power MAE(kW)={power_mae:.3f} MAPE={(power_mape*100 if not np.isnan(power_mape) else np.nan):.1f}% R2={power_r2:.3f}"
                        )
            except Exception as me:
                print(f"[ModelBuilder] Multi-output model error for zone {z}: {me}")

            # ざっくり評価
            y_hat_t = temp_model.predict(Xte)
            temp_mae = mean_absolute_error(yte, y_hat_t)
            temp_mape = mean_absolute_percentage_error(yte, y_hat_t)
            print(
                f"[Model] {z} Temp: MAE={temp_mae:.2f} MAPE={temp_mape*100:.1f}% R2={r2_score(yte, y_hat_t):.3f}"
            )

            y_hat_p = power_model.predict(Xtep)
            # kW換算で評価
            ytep_kw = ytep / 1000.0
            yhat_kw = pd.Series(y_hat_p, index=ytep.index) / 1000.0
            nonzero_mask_p = ytep_kw != 0
            power_mape = (
                mean_absolute_percentage_error(
                    ytep_kw[nonzero_mask_p], yhat_kw[nonzero_mask_p]
                )
                if nonzero_mask_p.any()
                else np.nan
            )
            print(
                f"[Model] {z} Power: MAE(kW)={mean_absolute_error(ytep_kw, yhat_kw):.3f} "
                f"MAPE={(power_mape*100 if not np.isnan(power_mape) else np.nan):.1f}% R2={r2_score(ytep, y_hat_p):.3f}"
            )

            # SHAP 可視化出力
            out_dir = os.path.join("analysis", "output", self.store_name, z)
            ModelBuilder._ensure_dir(out_dir)
            try:
                ModelBuilder._shap_on_pipeline(
                    temp_model,
                    Xte,
                    os.path.join(out_dir, "shap_temp.png"),
                    f"SHAP - Temp ({z})",
                )
            except Exception as e:
                print(f"[ModelBuilder] SHAP temp failed ({z}): {e}")
            try:
                ModelBuilder._shap_on_pipeline(
                    power_model,
                    Xtep,
                    os.path.join(out_dir, "shap_power.png"),
                    f"SHAP - Power ({z})",
                )
            except Exception as e:
                print(f"[ModelBuilder] SHAP power failed ({z}): {e}")
            models[z] = EnvPowerModels(
                temp_model=temp_model,
                hum_model=hum_model,
                power_model=power_model,
                multi_output_model=multi_output_model,
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
                    "multi_output_model": pack.multi_output_model,
                    "feature_cols": pack.feature_cols,
                },
                os.path.join(mdir, f"models_{z}.pkl"),
            )
        return models
