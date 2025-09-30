# =============================================================================
# エアコン最適化システム - 統合版
# =============================================================================

import glob
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.io as pio
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"


class VisualCrossingWeatherAPIDataFetcher:
    """
    Visual Crossing Weather APIを使用して天気予報データを取得するクラス
    API Website: https://www.visualcrossing.com/
    """

    def __init__(
        self,
        coordinates: str,
        start_date: str,
        end_date: str,
        unit: str,
        api_key: str,
        temperature_col_name: str,
        humidity_col_name: str,
    ):
        self.coordinates = coordinates
        self.start_date = start_date
        self.end_date = end_date
        self.unit = unit
        self.api_key = api_key
        self.temperature_col_name = temperature_col_name
        self.humidity_col_name = humidity_col_name
        self.forecast_list = []

    def fetch_forecast_data(self):
        """天気予報データを取得する"""
        try:
            url = (
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/"
                f"services/timeline/{self.coordinates}/{self.start_date}/{self.end_date}"
                f"?unitGroup={self.unit}&key={self.api_key}&include=hours"
            )
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            for day in data["days"]:
                for hour in day["hours"]:
                    self.forecast_list.append(
                        {
                            "datetime": f"{day['datetime']} {hour['datetime']}",
                            self.temperature_col_name: hour["temp"],
                            self.humidity_col_name: hour["humidity"],
                        }
                    )

            df_weather = pd.DataFrame(self.forecast_list)
            df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])
            return df_weather

        except Exception as e:
            print(f"天気データ取得エラー: {e}")
            return None


class DataPreprocessor:
    """データ前処理クラス"""

    def __init__(self, store_name):
        self.store_name = store_name
        from config.utils import get_data_path

        self.data_dir = os.path.join(get_data_path("raw_data_path"), store_name)
        self.output_dir = os.path.join(get_data_path("processed_data_path"), store_name)
        print(f"🔍 Data directory: {self.data_dir}")
        print(f"🔍 Output directory: {self.output_dir}")
        print(f"🔍 Data directory exists: {os.path.exists(self.data_dir)}")
        if os.path.exists(self.data_dir):
            print(f"🔍 Data directory contents: {os.listdir(self.data_dir)}")

    def load_raw_data(self):
        """生データの読み込み（サブフォルダ対応）"""
        print(f"📁 {self.store_name}の生データ読み込み中...")

        # AC制御データの読み込み（サブフォルダも含めて検索）
        ac_control_files = glob.glob(
            f"{self.data_dir}/**/ac-control-*.csv", recursive=True
        )
        ac_control_dfs = []

        for file in ac_control_files:
            try:
                df = pd.read_csv(file, encoding="utf-8")
                df["sourcefile"] = os.path.basename(file)
                ac_control_dfs.append(df)
                print(f"   ✅ AC制御: {os.path.basename(file)} ({len(df):,}件)")
            except Exception as e:
                print(f"   ❌ AC制御読み込みエラー {file}: {e}")

        # Phase Aメーターデータの読み込み（サブフォルダも含めて検索）
        power_meter_files = glob.glob(
            f"{self.data_dir}/**/ac-power-meter-*.csv", recursive=True
        )
        power_meter_dfs = []

        for file in power_meter_files:
            try:
                df = pd.read_csv(file, encoding="utf-8")
                df["sourcefile"] = os.path.basename(file)
                power_meter_dfs.append(df)
                print(
                    f"   ✅ Phase Aメーター: {os.path.basename(file)} ({len(df):,}件)"
                )
            except Exception as e:
                print(f"   ❌ Phase Aメーター読み込みエラー {file}: {e}")

        if not ac_control_dfs and not power_meter_dfs:
            print(f"❌ {self.store_name}のデータが見つかりません")
            return None, None

        ac_control_df = (
            pd.concat(ac_control_dfs, ignore_index=True) if ac_control_dfs else None
        )
        power_meter_df = (
            pd.concat(power_meter_dfs, ignore_index=True) if power_meter_dfs else None
        )

        return ac_control_df, power_meter_df

    def _preprocess_datetime_column(self, df):
        """日時列の統一処理（共通処理）"""
        datetime_cols = [
            col for col in df.columns if "datetime" in col.lower() or "日時" in col
        ]
        if not datetime_cols:
            print("❌ 日時列が見つかりません")
            return None, None

        datetime_col = datetime_cols[0]
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)
        df[datetime_col] = df[datetime_col].dt.tz_localize(None)
        df[datetime_col] = df[datetime_col].dt.floor("T")
        return df, datetime_col

    def _remove_outliers(self, df, numeric_cols, std_multiplier):
        """外れ値除去（共通処理）"""
        for col in numeric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                std_val = df[col].std()
                threshold = std_multiplier * std_val
                df = df[abs(df[col] - mean_val) <= threshold]
        return df

    def _remove_duplicates(self, df, datetime_col):
        """重複除去（共通処理）"""
        device_col = None
        if "A/C Name" in df.columns:
            device_col = "A/C Name"
        elif "Mesh ID" in df.columns:
            device_col = "Mesh ID"

        if device_col:
            df = df.drop_duplicates(subset=[datetime_col, device_col])
        return df

    def preprocess_ac_control_data(self, df, std_multiplier=3.0):
        """AC制御データの前処理"""
        if df is None or df.empty:
            return None

        print(f"🔧 AC制御データ前処理中... (外れ値係数: {std_multiplier})")

        # 日時列の統一
        df, datetime_col = self._preprocess_datetime_column(df)
        if df is None:
            return None

        # カラム名はそのまま使用（日本語リネームは不要）

        # 重複除去
        df = self._remove_duplicates(df, datetime_col)

        # 外れ値除去
        numeric_cols = ["A/C Set Temperature", "Indoor Temp.", "Outdoor Temp."]
        df = self._remove_outliers(df, numeric_cols, std_multiplier)

        # 欠損値処理（線形補間、湿度は除外）
        numeric_cols = ["A/C Set Temperature", "Indoor Temp.", "Outdoor Temp."]
        for col in numeric_cols:
            if col in df.columns:
                before_nan = df[col].isna().sum()
                df[col] = df[col].interpolate(method="linear")
                after_nan = df[col].isna().sum()
                print(f"   🔄 {col}欠損値補完: {before_nan:,} → {after_nan:,}件")
        # 湿度は補完しない（データが無くても補完は行わない）
        # カテゴリカル変数の処理（JSON定義を使用）
        categorical_cols = ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed"]
        for col in categorical_cols:
            if col in df.columns:
                # 変換前の値を確認（NaN値を除外）
                unique_values = df[col].dropna().unique()
                print(f"   📋 {col}変換前の値: {sorted(unique_values)}")
                # デフォルトの変換ルールを適用
                if col == "A/C ON/OFF":
                    mapping = {"0": 0, "1": 1}
                elif col == "A/C Mode":
                    mapping = {"COOL": 1, "FAN": 3, "HEAT": 2}
                elif col == "A/C Fan Speed":
                    mapping = {"Auto": 0, "High": 3, "Low": 1, "Medium": 2, "Top": 4}
                else:
                    mapping = {}
                df[col] = df[col].map(mapping)
                # 未定義の値がある場合は警告
                unmapped_values = df[col].isna().sum()
                if unmapped_values > 0:
                    print(f"   ⚠️ {col}で未定義の値が{unmapped_values}件あります")
                    # 未定義の値は-1で埋める
                    df[col] = df[col].fillna(-1)
                print(f"   🔄 {col}変換ルール: {mapping}")
                print(f"   ✅ {col}カテゴリ変換完了")

        # 日付列の追加
        df["date"] = df[datetime_col].dt.date
        df["datetime"] = df[datetime_col]

        # 不要な列の削除
        df = df.drop(columns=["sourcefile"], errors="ignore")

        print(f"   ✅ 前処理完了: {len(df):,}件")
        return df

    def preprocess_power_meter_data(self, df, std_multiplier=3.0):
        """Phase Aメーターデータの前処理"""
        if df is None or df.empty:
            return None

        print(f"🔧 Phase Aメーターデータ前処理中... (外れ値係数: {std_multiplier})")

        # 日時列の統一
        df, datetime_col = self._preprocess_datetime_column(df)
        if df is None:
            return None

        # カラム名はそのまま使用（日本語リネームは不要）

        # 重複除去
        df = self._remove_duplicates(df, datetime_col)

        # 外れ値除去
        numeric_cols = ["Phase A"]
        df = self._remove_outliers(df, numeric_cols, std_multiplier)

        # 欠損値処理
        df["Phase A"] = df["Phase A"].fillna(0)

        # 日付列の追加
        df["date"] = df[datetime_col].dt.date
        df["datetime"] = df[datetime_col]

        # 不要な列の削除
        df = df.drop(columns=["sourcefile"], errors="ignore")

        print(f"   ✅ 前処理完了: {len(df):,}件")
        return df

    def save_processed_data(self, ac_control_df, power_meter_df):
        """前処理済みデータの保存"""
        os.makedirs(self.output_dir, exist_ok=True)

        if ac_control_df is not None:
            ac_control_path = (
                f"{self.output_dir}/ac_control_processed_{self.store_name}.csv"
            )
            ac_control_df.to_csv(ac_control_path, index=False, encoding="utf-8-sig")
            print(f"💾 AC制御データ保存: {ac_control_path}")

        if power_meter_df is not None:
            power_meter_path = (
                f"{self.output_dir}/power_meter_processed_{self.store_name}.csv"
            )
            power_meter_df.to_csv(power_meter_path, index=False, encoding="utf-8-sig")
            print(f"💾 Phase Aメーターデータ保存: {power_meter_path}")


class FeatureEngineer:
    """特徴量エンジニアリングクラス"""

    def __init__(self, store_name):
        self.store_name = store_name
        self.master_info = None

    def load_master_data(self):
        """マスタデータの読み込み（JSON形式対応）"""
        print(f"📋 {self.store_name}のマスタデータ読み込み中...")
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = (
            os.path.dirname(current_dir)
            if current_dir.endswith("config")
            else current_dir
        )
        json_file = os.path.join(
            project_root, "master", f"MASTER_{self.store_name}.json"
        )
        excel_file = os.path.join(
            project_root, "master", f"MASTER_{self.store_name}.xlsx"
        )
        master_info = {}
        if os.path.exists(json_file):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                master_info.update(json_data)
                print(f"   ✅ JSONマスタデータ読み込み完了: {list(json_data.keys())}")
            except Exception as e:
                print(f"❌ JSONマスタデータ読み込みエラー: {e}")
        if os.path.exists(excel_file):
            try:
                excel_data = pd.read_excel(excel_file, sheet_name=None)
                master_info.update(excel_data)
                print(f"   ✅ Excelマスタデータ読み込み完了: {list(excel_data.keys())}")
            except Exception as e:
                print(f"❌ Excelマスタデータ読み込みエラー: {e}")
        if not master_info:
            print(f"❌ マスタファイルが見つかりません: {json_file} または {excel_file}")
            return None
        return master_info

    def create_features(self, ac_control_df, power_meter_df, master_info):
        """特徴量の作成"""
        print("🔧 特徴量作成中...")

        # 基本特徴量の作成
        features = []

        if ac_control_df is not None:
            # 時別集約（datetime列を使用）
            hourly_ac = (
                ac_control_df.groupby(["datetime", "A/C Name"])
                .agg(
                    {
                        "A/C Set Temperature": "mean",
                        "Indoor Temp.": "mean",
                        "Outdoor Temp.": "mean",
                        "Outdoor Humidity": "mean",
                        "A/C ON/OFF": "mean",
                        "A/C Mode": "mean",
                        "A/C Fan Speed": "mean",
                    }
                )
                .reset_index()
            )

            # ラグ特徴量（前の時刻のIndoor Temp.）
            hourly_ac = hourly_ac.sort_values(["A/C Name", "datetime"])
            hourly_ac["Indoor Temp. Lag1"] = hourly_ac.groupby("A/C Name")[
                "Indoor Temp."
            ].shift(1)
            hourly_ac["Indoor Temp. Lag1"] = hourly_ac["Indoor Temp. Lag1"].fillna(
                hourly_ac["Indoor Temp."]
            )

            features.append(hourly_ac)

        if power_meter_df is not None:
            # マスタデータからPM Mesh IDとPhase A予測区分の対応を取得
            if master_info and "zones" in master_info:
                # 新しいJSON形式のマスタデータに対応
                all_power_data = []
                zones = master_info["zones"]

                for zone_name, zone_info in zones.items():
                    if "outdoor_units" in zone_info:
                        outdoor_units = zone_info["outdoor_units"]
                        for unit_id, unit_info in outdoor_units.items():
                            # 該当するMesh IDのデータを抽出
                            category_data = power_meter_df[
                                power_meter_df["Mesh ID"] == unit_id
                            ].copy()
                            if not category_data.empty:
                                category_data["A/C Name"] = (
                                    zone_name  # 制御区分名をA/C Nameとして使用
                                )
                                all_power_data.append(category_data)

                if all_power_data:
                    power_meter_df_copy = pd.concat(all_power_data, ignore_index=True)
                else:
                    power_meter_df_copy = pd.DataFrame()

                if not power_meter_df_copy.empty:
                    # 時別集約（datetime列を使用）
                    hourly_power = (
                        power_meter_df_copy.groupby(["datetime", "A/C Name"])
                        .agg({"Phase A": "sum"})
                        .reset_index()
                    )
                    features.append(hourly_power)
                    print(f"   ✅ Phase Aデータ統合完了: {len(hourly_power)}件")
                    print(
                        f"   📋 Phase A予測区分: {sorted(hourly_power['A/C Name'].unique())}"
                    )
                else:
                    print("   ⚠️ Phase Aデータの統合に失敗しました")
            elif master_info and "control" in master_info:
                master_df = master_info["control"]
                # PM Mesh IDとPhase A予測区分の対応を作成（1つのPM Mesh IDに対して複数のPhase A予測区分）
                pm_to_power_categories_mapping = {}
                for _, row in master_df.iterrows():
                    pm_id = row.get("PM Mesh ID")
                    power_category = row.get("Phase A予測区分")
                    if (
                        pd.notna(pm_id)
                        and pd.notna(power_category)
                        and power_category != "---"
                    ):
                        if pm_id not in pm_to_power_categories_mapping:
                            pm_to_power_categories_mapping[pm_id] = []
                        pm_to_power_categories_mapping[pm_id].append(power_category)

                # 各PM Mesh IDのデータを複数のPhase A予測区分に分配
                all_power_data = []
                for pm_id, power_categories in pm_to_power_categories_mapping.items():
                    pm_data = power_meter_df[power_meter_df["A/C Name"] == pm_id].copy()
                    if not pm_data.empty:
                        for power_category in power_categories:
                            category_data = pm_data.copy()
                            category_data["A/C Name"] = power_category
                            all_power_data.append(category_data)

                if all_power_data:
                    power_meter_df_copy = pd.concat(all_power_data, ignore_index=True)
                else:
                    power_meter_df_copy = pd.DataFrame()

                if not power_meter_df_copy.empty:
                    # 時別集約（datetime列を使用）
                    hourly_power = (
                        power_meter_df_copy.groupby(["datetime", "A/C Name"])
                        .agg({"Phase A": "sum"})
                        .reset_index()
                    )
                    features.append(hourly_power)
                    print(f"   ✅ Phase Aデータ統合完了: {len(hourly_power)}件")
                    print(
                        f"   📋 Phase A予測区分: {sorted(hourly_power['A/C Name'].unique())}"
                    )
                else:
                    print("   ⚠️ Phase Aデータの統合に失敗しました")
            else:
                print(
                    "   ⚠️ マスタデータが見つからないため、Phase Aデータを統合できません"
                )

        if features:
            # 特徴量の結合
            feature_df = features[0]
            # A/C Name列のデータ型を統一
            feature_df["A/C Name"] = feature_df["A/C Name"].astype(str)

            for i in range(1, len(features)):
                features[i]["A/C Name"] = features[i]["A/C Name"].astype(str)
                feature_df = pd.merge(
                    feature_df, features[i], on=["datetime", "A/C Name"], how="outer"
                )

            # マスタ情報の結合
            if master_info and "environmental" in master_info:
                env_master = master_info["environmental"]
                # エリア区分の名称をIndoor Unit Nameとして使用
                if (
                    "PM Mesh ID" in env_master.columns
                    and "エリア区分" in env_master.columns
                ):
                    env_mapping = dict(
                        zip(env_master["PM Mesh ID"], env_master["エリア区分"])
                    )
                    feature_df["Indoor Unit Name"] = feature_df["A/C Name"].map(
                        env_mapping
                    )

            print(f"   ✅ 特徴量作成完了: {len(feature_df):,}件")
            return feature_df

        return None


class ModelTrainer:
    """モデル訓練クラス"""

    def __init__(self, store_name):
        self.store_name = store_name
        self.models = {}
        self.feature_importance = {}

    def _create_default_power_models(self, master_info):
        """デフォルトのPhase A予測モデルを作成（マスタデータの予測区分に基づく）"""
        print("🔧 デフォルトPhase A予測モデルを作成中...")

        # マスタデータから実際の予測区分を取得
        if master_info and "power" in master_info:
            power_master = master_info["power"]
            # Phase A予測区分の一意な値を取得
            power_categories = power_master["Phase A予測区分"].dropna().unique()
            print(f"   📋 マスタデータのPhase A予測区分: {list(power_categories)}")
        else:
            # マスタデータがない場合は制御区分から推測
            if master_info and "control" in master_info:
                control_master = master_info["control"]
                power_categories = control_master["制御区分"].dropna().unique()
                print(
                    f"   📋 制御区分からPhase A予測区分を推測: {list(power_categories)}"
                )
            else:
                # 最後の手段として固定カテゴリを使用
                power_categories = [
                    "エリア1",
                    "エリア2",
                    "エリア3",
                    "エリア4",
                    "会議室",
                    "休憩室",
                ]
                print(
                    f"   ⚠️ マスタデータが見つからないため、固定カテゴリを使用: {list(power_categories)}"
                )

        for category in power_categories:
            # デフォルトモデル情報を保存（関数は含めない）
            model_info = {
                "model": None,  # 実際のモデルは使用しない
                "feature_cols": [
                    "A/C Set Temperature",
                    "前時刻Indoor Temp.",
                    "A/C ON/OFF",
                    "A/C Mode",
                    "A/C Fan Speed",
                    "Outdoor Temp.",
                    "Outdoor Humidity",
                ],
                "unit_id": None,
                "r2": 0.5,
                "mae": 200,
                "is_default": True,  # デフォルトモデルであることを示すフラグ
            }

            # モデルディレクトリに保存
            from config.utils import get_data_path

            model_dir = os.path.join(get_data_path("models_path"), self.store_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_power_{category}.pkl")
            joblib.dump(model_info, model_path)
            print(f"💾 デフォルトPhase A予測モデル保存: {model_path}")

    def _create_default_environmental_models(self, master_info):
        """デフォルトの環境予測モデルを作成（マスタデータの予測区分に基づく）"""
        print("🔧 デフォルト環境予測モデルを作成中...")

        # マスタデータから実際の予測区分を取得
        if master_info and "environmental" in master_info:
            env_master = master_info["environmental"]
            # 環境予測区分の一意な値を取得
            env_categories = env_master["環境予測区分"].dropna().unique()
            print(f"   📋 マスタデータの環境予測区分: {list(env_categories)}")
        else:
            # マスタデータがない場合は制御区分から推測
            if master_info and "control" in master_info:
                control_master = master_info["control"]
                env_categories = control_master["制御区分"].dropna().unique()
                print(f"   📋 制御区分から環境予測区分を推測: {list(env_categories)}")
            else:
                # 最後の手段として固定カテゴリを使用
                env_categories = [
                    "エリア1",
                    "エリア2",
                    "エリア3",
                    "エリア4",
                    "会議室",
                    "休憩室",
                ]
                print(
                    f"   ⚠️ マスタデータが見つからないため、固定カテゴリを使用: {list(env_categories)}"
                )

        for category in env_categories:
            # デフォルトモデル情報を保存
            model_info = {
                "model": None,  # 実際のモデルは使用しない
                "feature_cols": [
                    "A/C Set Temperature",
                    "前時刻Indoor Temp.",
                    "A/C ON/OFF",
                    "A/C Mode",
                    "A/C Fan Speed",
                    "Outdoor Temp.",
                    "Outdoor Humidity",
                ],
                "unit_id": None,
                "r2": 0.5,
                "mae": 2.0,
                "is_default": True,  # デフォルトモデルであることを示すフラグ
            }

            # モデルディレクトリに保存
            from config.utils import get_data_path

            model_dir = os.path.join(get_data_path("models_path"), self.store_name)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_env_{category}.pkl")
            joblib.dump(model_info, model_path)
            print(f"💾 デフォルト環境予測モデル保存: {model_path}")

    def train_environmental_model(self, feature_df, master_info):
        """環境予測モデルの訓練（制御区分ごと）"""
        print("🌡️ 環境予測モデル訓練中...")

        if not master_info or "zones" not in master_info:
            print("❌ 制御区分のマスタデータが見つかりません")
            # デフォルトの環境予測モデルを作成
            self._create_default_environmental_models(master_info)
            return None

        zones = master_info["zones"]
        models = {}

        for zone_name, zone_info in zones.items():
            print(f"   🔍 {zone_name} の環境予測モデル学習中...")

            # 制御区分内の室内機を取得
            indoor_units = []
            for outdoor_unit_id, outdoor_info in zone_info.get(
                "outdoor_units", {}
            ).items():
                indoor_units.extend(outdoor_info.get("indoor_units", []))

            if not indoor_units:
                print(f"   ⚠️ {zone_name}: 室内機が見つかりません")
                continue

            # 該当する室内機のデータを取得
            env_data = feature_df[feature_df["A/C Name"].isin(indoor_units)].copy()

            if env_data.empty:
                print(f"   ⚠️ {zone_name}: データが見つかりません")
                continue

            print(f"   ✅ {zone_name}: {len(env_data)}件のデータを取得")

            # 制御区分内の室内機の平均温度・湿度を計算（時別集約）
            # 湿度データがない場合は50%で埋める
            if "室内湿度" in env_data.columns:
                grouped_data = (
                    env_data.groupby(["datetime"])
                    .agg(
                        {
                            "Indoor Temp.": "mean",
                            "室内湿度": "mean",
                            "A/C Set Temperature": "mean",
                            "前時刻Indoor Temp.": "mean",
                            "A/C ON/OFF": "mean",
                            "A/C Mode": "mean",
                            "A/C Fan Speed": "mean",
                            "Outdoor Temp.": "first",
                            "Outdoor Humidity": "first",
                        }
                    )
                    .reset_index()
                )
            else:
                # 室内湿度列がない場合は50%で埋める
                grouped_data = (
                    env_data.groupby(["datetime"])
                    .agg(
                        {
                            "Indoor Temp.": "mean",
                            "A/C Set Temperature": "mean",
                            "前時刻Indoor Temp.": "mean",
                            "A/C ON/OFF": "mean",
                            "A/C Mode": "mean",
                            "A/C Fan Speed": "mean",
                            "Outdoor Temp.": "first",
                            "Outdoor Humidity": "first",
                        }
                    )
                    .reset_index()
                )
                grouped_data["室内湿度"] = 50.0

            # 特徴量とターゲットの準備
            feature_cols = [
                "A/C Set Temperature",
                "前時刻Indoor Temp.",
                "A/C ON/OFF",
                "A/C Mode",
                "A/C Fan Speed",
                "Outdoor Temp.",
                "Outdoor Humidity",
            ]
            X = grouped_data[feature_cols].fillna(0)

            # 温度データの確認
            if "Indoor Temp." not in grouped_data.columns:
                print(f"   ⚠️ {zone_name} の温度データが見つかりません")
                continue

            y_temp = grouped_data["Indoor Temp."]
            y_humidity = grouped_data["室内湿度"]

            # NaN値を除去
            valid_mask = y_temp.notna()
            X = X[valid_mask]
            y_temp = y_temp[valid_mask]
            y_humidity = y_humidity[valid_mask]

            if len(X) < 10:
                print(f"   ⚠️ {zone_name} の学習データが不足しています ({len(X)}件)")
                continue

            # 訓練・テスト分割
            X_train, X_test, y_train_temp, y_test_temp = train_test_split(
                X, y_temp, test_size=0.2, random_state=42
            )
            _, _, y_train_humidity, y_test_humidity = train_test_split(
                X, y_humidity, test_size=0.2, random_state=42
            )

            # 温度予測モデル訓練
            temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
            temp_model.fit(X_train, y_train_temp)

            # 湿度予測モデル訓練
            humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
            humidity_model.fit(X_train, y_train_humidity)

            # 予測と評価
            y_pred_temp = temp_model.predict(X_test)
            y_pred_humidity = humidity_model.predict(X_test)

            temp_mae = mean_absolute_error(y_test_temp, y_pred_temp)
            temp_r2 = r2_score(y_test_temp, y_pred_temp)
            temp_mape = np.mean(np.abs((y_test_temp - y_pred_temp) / y_test_temp)) * 100

            humidity_mae = mean_absolute_error(y_test_humidity, y_pred_humidity)
            humidity_r2 = r2_score(y_test_humidity, y_pred_humidity)
            humidity_mape = (
                np.mean(np.abs((y_test_humidity - y_pred_humidity) / y_test_humidity))
                * 100
            )

            print(
                f"   ✅ {zone_name}: 温度 MAE={temp_mae:.2f}, "
                f"R²={temp_r2:.3f}, MAPE={temp_mape:.1f}%"
            )
            print(
                f"   ✅ {zone_name}: 湿度 MAE={humidity_mae:.2f}, "
                f"R²={humidity_r2:.3f}, MAPE={humidity_mape:.1f}%"
            )

            # モデル保存
            models[zone_name] = {
                "temp_model": temp_model,
                "humidity_model": humidity_model,
                "zone_name": zone_name,
                "feature_cols": feature_cols,
                "temp_mae": temp_mae,
                "temp_r2": temp_r2,
                "temp_mape": temp_mape,
                "humidity_mae": humidity_mae,
                "humidity_r2": humidity_r2,
                "humidity_mape": humidity_mape,
                "is_default": False,  # 実際の学習済みモデル
            }

        return models

    def train_power_model(self, feature_df, master_info):
        """Phase A予測モデルの訓練（MESH-ID組合せのみ対象）"""
        print("⚡ Phase A予測モデル訓練中...")

        if not master_info or "zones" not in master_info:
            print("❌ 制御区分のマスタデータが見つかりません")
            # デフォルトのPhase A予測モデルを作成
            self._create_default_power_models(master_info)
            return None

        zones = master_info["zones"]
        models = {}

        for zone_name, zone_info in zones.items():
            print(f"   🔍 {zone_name} のPhase A予測モデル学習中...")

            # 制御区分内の室外機を取得
            outdoor_units = zone_info.get("outdoor_units", {})
            if not outdoor_units:
                print(f"   ⚠️ {zone_name}: 室外機が見つかりません")
                continue

            # 室外機のPhase Aデータを取得し、負荷比率を乗じて合計
            zone_power_data = []
            for outdoor_unit_id, outdoor_info in outdoor_units.items():
                load_share = outdoor_info.get("load_share", 1.0)

                # 室外機のPhase Aデータを取得（{mesh}-{ID}形式で検索）
                outdoor_data = feature_df[
                    feature_df["A/C Name"] == outdoor_unit_id
                ].copy()

                if not outdoor_data.empty:
                    # 負荷比率を乗じたPhase A値を計算
                    outdoor_data["adjusted_power"] = (
                        outdoor_data["Phase A"] * load_share
                    )
                    zone_power_data.append(outdoor_data)
                    print(
                        f"   ✅ {outdoor_unit_id}: {len(outdoor_data)}件のPhase Aデータを取得"
                    )
                else:
                    print(f"   ⚠️ {outdoor_unit_id}: Phase Aデータが見つかりません")

            if not zone_power_data:
                print(f"   ⚠️ {zone_name}: Phase Aデータが見つかりません")
                continue

            # 制御区分内の室外機のPhase Aデータを統合
            combined_power_data = pd.concat(zone_power_data, ignore_index=True)

            # 時別にPhase A値を合計（時別集約）
            grouped_power_data = (
                combined_power_data.groupby(["datetime"])
                .agg(
                    {
                        "adjusted_power": "sum",
                        "A/C Set Temperature": "mean",
                        "前時刻Indoor Temp.": "mean",
                        "A/C ON/OFF": "mean",
                        "A/C Mode": "mean",
                        "A/C Fan Speed": "mean",
                        "Outdoor Temp.": "first",
                        "Outdoor Humidity": "first",
                    }
                )
                .reset_index()
            )

            # 特徴量とターゲットの準備
            feature_cols = [
                "A/C Set Temperature",
                "前時刻Indoor Temp.",
                "A/C ON/OFF",
                "A/C Mode",
                "A/C Fan Speed",
                "Outdoor Temp.",
                "Outdoor Humidity",
            ]
            X = grouped_power_data[feature_cols].fillna(0)
            y_power = grouped_power_data["adjusted_power"]

            # NaN値を除去
            valid_mask = y_power.notna()
            X = X[valid_mask]
            y_power = y_power[valid_mask]

            if len(X) < 10:
                print(f"   ⚠️ {zone_name} の学習データが不足しています ({len(X)}件)")
                continue

            # 訓練・テスト分割
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_power, test_size=0.2, random_state=42
            )

            # モデル訓練
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 予測と評価
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            print(f"   ✅ {zone_name}: MAE={mae:.2f}, R²={r2:.3f}, MAPE={mape:.1f}%")

            # モデル保存
            models[zone_name] = {
                "model": model,
                "zone_name": zone_name,
                "feature_cols": feature_cols,
                "mae": mae,
                "r2": r2,
                "mape": mape,
                "is_default": False,  # 実際の学習済みモデル
            }

        return models

    def save_models(self, env_models, power_models):
        """モデルの保存"""
        from config.utils import get_data_path

        model_dir = os.path.join(get_data_path("models_path"), self.store_name)
        os.makedirs(model_dir, exist_ok=True)

        if env_models:
            for category, model_info in env_models.items():
                model_path = os.path.join(
                    model_dir, f"model_environmental_{category}.pkl"
                )
                joblib.dump(model_info, model_path)
                print(f"💾 環境予測モデル保存: {model_path}")

        if power_models:
            for category, model_info in power_models.items():
                model_path = os.path.join(model_dir, f"model_power_{category}.pkl")
                joblib.dump(model_info, model_path)
                print(f"💾 Phase A予測モデル保存: {model_path}")


class Optimizer:
    """最適化クラス"""

    def __init__(self, store_name):
        self.store_name = store_name
        self.env_models = {}
        self.power_models = {}

    def load_models(self):
        """モデルの読み込み"""
        from config.utils import get_data_path

        model_dir = os.path.join(get_data_path("models_path"), self.store_name)

        # 環境予測モデルの読み込み
        env_model_files = glob.glob(
            os.path.join(model_dir, "model_environmental_*.pkl")
        )
        for file in env_model_files:
            category = (
                os.path.basename(file)
                .replace("model_environmental_", "")
                .replace(".pkl", "")
            )
            self.env_models[category] = joblib.load(file)

        # Phase A予測モデルの読み込み
        power_model_files = glob.glob(os.path.join(model_dir, "model_power_*.pkl"))
        for file in power_model_files:
            category = (
                os.path.basename(file).replace("model_power_", "").replace(".pkl", "")
            )
            self.power_models[category] = joblib.load(file)

        print(
            f"📦 モデル読み込み完了: 環境予測{len(self.env_models)}個, Phase A予測{len(self.power_models)}個"
        )

    def optimize_control_values(self, master_info, weather_data=None):
        """2日サイクルでの制御値最適化（時別予測、蓄熱効果考慮）"""
        print("🎯 2日サイクル制御値最適化中（時別予測）...")

        if not master_info or "zones" not in master_info:
            print("❌ 制御区分のマスタデータが見つかりません")
            return {}

        # 天気予報データの取得
        if weather_data is None:
            weather_data = self._get_weather_forecast()

        # 2日間の最適化実行
        optimization_results = self._optimize_2day_cycle(master_info, weather_data)

        return optimization_results

    def _get_weather_forecast(self):
        """天気予報データの取得"""
        print("🌤️ 天気予報データ取得中...")

        # Visual Crossing Weather APIを使用
        from datetime import datetime, timedelta

        # 2日間の期間を設定
        start_date = datetime.now()
        end_date = start_date + timedelta(days=2)

        # API設定（実際のAPIキーを使用）
        API_KEY = "5JYPWTTJGYN89DF4GJW7SY2PB"
        COORDINATES = "35.39291572%2C139.44288869"  # 東京都港区麻布台二丁目

        try:
            weather_forecast = VisualCrossingWeatherAPIDataFetcher(
                coordinates=COORDINATES,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                unit="metric",
                api_key=API_KEY,
                temperature_col_name="temperature C",
                humidity_col_name="humidity",
            )
            df_weather = weather_forecast.fetch_forecast_data()
            print(f"   ✅ 天気予報データ取得完了: {len(df_weather)}件")
            return df_weather
        except Exception as e:
            print(f"   ⚠️ 天気予報データ取得エラー: {e}")
            # デフォルトの天気データを生成
            return self._generate_default_weather_data(start_date, end_date)

    def _generate_default_weather_data(self, start_date, end_date):
        """デフォルトの天気データ生成"""
        import numpy as np
        import pandas as pd

        # 2日間の1時間毎のデータを生成
        date_range = pd.date_range(start=start_date, end=end_date, freq="H")

        weather_data = []
        for dt in date_range:
            # 時間帯に応じた温度変化（簡易モデル）
            hour = dt.hour
            if 6 <= hour <= 18:  # 日中
                temp = 25 + 5 * np.sin((hour - 6) * np.pi / 12)
            else:  # 夜間
                temp = 20 + 3 * np.sin((hour - 18) * np.pi / 12)

            weather_data.append(
                {
                    "datetime": dt,
                    "temperature": temp,
                    "humidity": 60 + 20 * np.sin(hour * np.pi / 12),
                }
            )

        return pd.DataFrame(weather_data)

    def _optimize_2day_cycle(self, master_info, weather_data):
        """2日サイクルでの最適化"""
        print("🔄 2日サイクル最適化実行中...")

        zones = master_info["zones"]
        optimization_results = {}

        # 各制御区分の最適化
        for zone_name, zone_info in zones.items():
            print(f"   🔍 {zone_name} の2日サイクル最適化中...")

            # 制御パラメータの取得
            control_params = self._get_control_parameters_from_zone(zone_info)

            # 2日間の最適化実行
            best_schedule = self._optimize_control_schedule(
                zone_name, control_params, weather_data
            )

            optimization_results[zone_name] = best_schedule
            print(f"   ✅ {zone_name}: 2日サイクル最適化完了")

        return optimization_results

    def _get_control_parameters_from_zone(self, zone_info):
        """制御区分情報から制御パラメータを取得"""
        return {
            "start_time": zone_info.get("start_time", "07:00"),
            "end_time": zone_info.get("end_time", "20:00"),
            "target_room_temp": zone_info.get("target_room_temp", 25.0),
            "setpoint_min": zone_info.get("setpoint_min", 22.0),
            "setpoint_max": zone_info.get("setpoint_max", 28.0),
            "fan_candidates": zone_info.get(
                "fan_candidates", ["Low", "Medium", "High"]
            ),
            "spot": zone_info.get("spot", False),
        }

    def _get_control_type(self, row):
        """制御区分の取得"""
        if "制御区分" in row:
            return row["制御区分"]
        elif "予測モデル(提案)" in row:
            return row["予測モデル(提案)"]
        return None

    def _get_control_parameters(self, row):
        """制御パラメータの取得"""
        return {
            "open_hour": self._get_business_hour(row, "始業時間"),
            "close_hour": self._get_business_hour(row, "就業時間"),
            "temp_min": row.get("A/C Set Temperature下限", 22),
            "temp_max": row.get("A/C Set Temperature上限", 28),
            "target_temp": row.get("目標Indoor Temp.", 25),
        }

    def _optimize_control_schedule(self, zone_name, control_params, weather_data):
        """制御スケジュールの最適化（時別予測を使用、0時から23時まで）"""
        # 1日分の1時間毎の最適化（時別予測を使用、0時から23時まで）
        schedule = {}

        # 初期室温設定
        current_temp = control_params["target_room_temp"]

        # 0時から23時までの時間を生成
        start_date = weather_data["datetime"].min().date()
        time_range = pd.date_range(start=start_date, periods=24, freq="H")

        for hour in range(24):
            current_datetime = time_range[hour]

            # 天気予報データから該当時刻のデータを取得
            weather_row = weather_data[weather_data["datetime"] == current_datetime]
            if weather_row.empty:
                # 該当時刻のデータがない場合は前後のデータから補間
                weather_row = weather_data.iloc[0] if not weather_data.empty else None
                if weather_row is None:
                    outdoor_temp = 25
                    outdoor_humidity = 60
                else:
                    outdoor_temp = weather_row.get(
                        "temperature", weather_row.get("temperature C", 25)
                    )
                    outdoor_humidity = weather_row.get("humidity", 60)
            else:
                weather_row = weather_row.iloc[0]
                outdoor_temp = weather_row.get(
                    "temperature", weather_row.get("temperature C", 25)
                )
                outdoor_humidity = weather_row.get("humidity", 60)

            # 営業時間のチェック
            start_hour = int(control_params["start_time"].split(":")[0])
            end_hour = int(control_params["end_time"].split(":")[0])
            is_business_hours = start_hour <= hour <= end_hour

            # 最適制御値の探索
            best_control = self._find_optimal_control(
                zone_name,
                control_params,
                current_temp,
                outdoor_temp,
                outdoor_humidity,
                is_business_hours,
            )

            # 室温予測（次の時刻の予測に使用）
            if self.env_models and zone_name in self.env_models:
                current_temp = self._predict_temperature(
                    zone_name,
                    best_control,
                    current_temp,
                    outdoor_temp,
                    outdoor_humidity,
                )

            schedule[current_datetime] = best_control

        return schedule

    def _find_optimal_control(
        self,
        control_type,
        control_params,
        current_temp,
        outdoor_temp,
        outdoor_humidity,
        is_business_hours,
    ):
        """最適制御値の探索（入れ子構造対応）"""
        best_score = float("inf")
        best_control = None

        # 制御値の候補を取得
        set_temp_candidates = list(
            range(
                int(control_params["setpoint_min"]),
                int(control_params["setpoint_max"]) + 1,
            )
        )
        mode_candidates = [0, 1, 2]  # 0:冷房, 1:除湿, 2:送風
        fan_speed_candidates = [1, 2, 3]  # A/C Fan Speed1-3

        print(
            f"   🔍 制御値候補: 温度={set_temp_candidates}, "
            f"A/C Mode={mode_candidates}, A/C Fan Speed={fan_speed_candidates}"
        )
        print(
            f"   🔍 営業時間: {control_params['start_time']}-{control_params['end_time']}"
        )

        for set_temp in set_temp_candidates:
            for mode in mode_candidates:
                for fan_speed in fan_speed_candidates:
                    # 制御値の設定
                    control_values = {
                        "A/C Set Temperature": set_temp,
                        "前時刻Indoor Temp.": current_temp,
                        "A/C ON/OFF": 1 if is_business_hours else 0,
                        "A/C Mode": mode,
                        "A/C Fan Speed": fan_speed,
                        "Outdoor Temp.": outdoor_temp,
                        "Outdoor Humidity": outdoor_humidity,
                    }

                    # 入れ子構造の最適化スコア計算
                    score = self._calculate_nested_optimization_score(
                        control_type, control_values, control_params, is_business_hours
                    )

                    # 最適解の更新
                    if score < best_score:
                        best_score = score
                        best_control = {
                            "set_temp": set_temp,
                            "mode": mode,
                            "fan_speed": fan_speed,
                            "score": score,
                        }
                        print(
                            f"   🎯 最適解更新: 温度={set_temp}°C, "
                            f"A/C Mode={mode}, A/C Fan Speed={fan_speed}, スコア={score:.2f}"
                        )

        return best_control or {
            "set_temp": 25,
            "mode": 0,
            "fan_speed": 1,
            "score": 1000,
        }

    def _get_control_range(self, control_type, control_param):
        """マスタ情報から制御範囲を取得"""
        if not hasattr(self, "master_info") or not self.master_info:
            # デフォルト値
            if control_param == "温度":
                return list(range(19, 28))
            elif control_param == "A/C Mode":
                return [0, 1, 2]
            elif control_param == "A/C Fan Speed":
                return [1, 2, 3]
            return []

        # マスタ情報から制御範囲を取得
        control_master = self.master_info.get("control")
        if control_master is None:
            return []

        # 制御区分に対応する制御範囲を取得
        control_data = control_master[control_master["制御区分"] == control_type]
        if control_data.empty:
            return []

        if control_param == "温度":
            temp_min = (
                control_data["温度下限"].iloc[0]
                if "温度下限" in control_data.columns
                else 19
            )
            temp_max = (
                control_data["温度上限"].iloc[0]
                if "温度上限" in control_data.columns
                else 27
            )
            return list(range(int(temp_min), int(temp_max) + 1))
        elif control_param == "A/C Mode":
            return [0, 1, 2]  # 冷房、暖房、送風
        elif control_param == "A/C Fan Speed":
            return [1, 2, 3]  # 弱、中、強

        return []

    def _get_comfort_range(self, control_type):
        """マスタ情報から快適性範囲を取得"""
        if not hasattr(self, "master_info") or not self.master_info:
            return {"min": 22, "max": 24}  # デフォルト値

        control_master = self.master_info.get("control")
        if control_master is None:
            return {"min": 22, "max": 24}

        control_data = control_master[control_master["制御区分"] == control_type]
        if control_data.empty:
            return {"min": 22, "max": 24}

        comfort_min = (
            control_data["快適性下限"].iloc[0]
            if "快適性下限" in control_data.columns
            else 22
        )
        comfort_max = (
            control_data["快適性上限"].iloc[0]
            if "快適性上限" in control_data.columns
            else 24
        )

        return {"min": comfort_min, "max": comfort_max}

    def _get_power_correction_factors(self, control_type):
        """Phase A予測区分ごとの室内機台数を取得"""
        if not hasattr(self, "master_info") or not self.master_info:
            return {}

        control_master = self.master_info.get("control")
        if control_master is None:
            return {}

        # 制御区分に対応するPhase A予測区分と室内機台数を取得
        control_data = control_master[control_master["制御区分"] == control_type]
        if control_data.empty:
            return {}

        correction_factors = {}
        for _, row in control_data.iterrows():
            power_category = row.get("Phase A予測区分", "")
            unit_count = row.get("室内機台数", 1)
            if power_category and power_category != "---":
                correction_factors[power_category] = unit_count

        return correction_factors

    def _calculate_nested_optimization_score(
        self, control_type, control_values, control_params, is_business_hours
    ):
        """入れ子構造の最適化スコア計算（Phase A補正と快適性ペナルティ対応）"""
        # 1. Phase A予測区分ごとのPhase A予測値の合計（室内機台数で補正）
        total_power = self._calculate_total_power_consumption_with_correction(
            control_type, control_values
        )

        # 2. 環境予測区分ごとの温度予測値の平均
        avg_predicted_temp = self._calculate_average_temperature_prediction(
            control_type, control_values
        )

        # 3. 快適性範囲の取得
        comfort_range = self._get_comfort_range(control_type)

        # 4. 快適性ペナルティの計算
        comfort_penalty = self._calculate_comfort_penalty(
            avg_predicted_temp, comfort_range
        )

        # 5. 営業時間外の場合はPhase A消費のみを考慮
        if not is_business_hours:
            return total_power

        # 6. 営業時間内の場合はPhase A消費と快適性ペナルティの重み付き合計
        power_weight = control_params.get("power_weight", 0.7)  # Phase A消費の重み
        comfort_weight = control_params.get("comfort_weight", 0.3)  # 快適性の重み

        score = power_weight * total_power + comfort_weight * comfort_penalty

        return score

    def _calculate_total_power_consumption_with_correction(
        self, control_type, control_values
    ):
        """制御区分に関連する全Phase A予測区分のPhase A消費合計を計算（室内機台数で補正）"""
        total_power = 0

        # 制御区分に関連するPhase A予測区分を特定
        related_power_categories = self._get_related_power_categories(control_type)

        # Phase A補正係数を取得
        correction_factors = self._get_power_correction_factors(control_type)

        for power_category in related_power_categories:
            if self.power_models and power_category in self.power_models:
                # 1台当たりのPhase A予測
                unit_power = self._predict_power_by_category(
                    power_category, control_values
                )

                # 室内機台数で補正
                unit_count = correction_factors.get(power_category, 1)
                corrected_power = unit_power * unit_count

                total_power += corrected_power
            else:
                # デフォルトPhase A予測
                unit_power = self._simple_power_predict(control_values)
                unit_count = correction_factors.get(power_category, 1)
                corrected_power = unit_power * unit_count
                total_power += corrected_power

        return total_power

    def _calculate_comfort_penalty(self, predicted_temp, comfort_range):
        """快適性ペナルティの計算"""
        comfort_min = comfort_range["min"]
        comfort_max = comfort_range["max"]

        if comfort_min <= predicted_temp <= comfort_max:
            # 快適性範囲内の場合はペナルティなし
            return 0
        elif predicted_temp < comfort_min:
            # 下限を下回る場合のペナルティ
            penalty = (comfort_min - predicted_temp) * 100  # 1°C = 100W相当
            return penalty
        else:
            # 上限を上回る場合のペナルティ
            penalty = (predicted_temp - comfort_max) * 100  # 1°C = 100W相当
            return penalty

    def _calculate_total_power_consumption(self, control_type, control_values):
        """制御区分に関連する全Phase A予測区分のPhase A消費合計を計算"""
        total_power = 0

        # 制御区分に関連するPhase A予測区分を特定
        related_power_categories = self._get_related_power_categories(control_type)

        for power_category in related_power_categories:
            if self.power_models and power_category in self.power_models:
                power = self._predict_power_by_category(power_category, control_values)
                total_power += power
            else:
                # デフォルトPhase A予測
                total_power += self._simple_power_predict(control_values)

        return total_power

    def _calculate_average_temperature_prediction(self, control_type, control_values):
        """制御区分に関連する全環境予測区分の温度予測平均を計算"""
        predicted_temps = []

        # 制御区分に関連する環境予測区分を特定
        related_env_categories = self._get_related_env_categories(control_type)

        for env_category in related_env_categories:
            if self.env_models and env_category in self.env_models:
                temp = self._predict_temperature_by_category(
                    env_category, control_values
                )
                predicted_temps.append(temp)
            else:
                # デフォルト温度予測
                predicted_temps.append(control_values["前時刻Indoor Temp."])

        return sum(predicted_temps) / len(predicted_temps) if predicted_temps else 22

    def _get_related_power_categories(self, control_type):
        """制御区分に関連するPhase A予測区分を取得"""
        if not hasattr(self, "master_info") or not self.master_info:
            return []

        # マスタデータから制御区分に対応するPhase A予測区分を取得
        power_master = self.master_info.get("control")
        if power_master is None:
            return []

        # 制御区分に対応するPhase A予測区分を取得
        power_categories = (
            power_master[power_master["制御区分"] == control_type]["Phase A予測区分"]
            .dropna()
            .unique()
        )
        return [str(cat) for cat in power_categories if pd.notna(cat) and cat != "---"]

    def _get_related_env_categories(self, control_type):
        """制御区分に関連する環境予測区分を取得"""
        # 制御区分と環境予測区分のマッピング
        mapping = {
            "エリア1": ["エリア1"],
            "エリア2": ["エリア2"],
            "エリア3": ["エリア3"],
            "エリア4": ["エリア4"],
            "会議室": ["会議室"],
            "休憩室": ["休憩室"],
        }

        return mapping.get(control_type, [control_type])

    def _predict_power_by_category(self, power_category, control_values):
        """特定のPhase A予測区分でのPhase A予測"""
        if not self.power_models or power_category not in self.power_models:
            return self._simple_power_predict(control_values)

        model_info = self.power_models[power_category]

        # デフォルトモデルの場合
        if model_info.get("is_default", False):
            return self._simple_power_predict(control_values)

        # 通常のモデルの場合
        X = pd.DataFrame([control_values])
        X = X[model_info["feature_cols"]]
        return model_info["model"].predict(X)[0]

    def _predict_temperature_by_category(self, env_category, control_values):
        """特定の環境予測区分での温度予測"""
        if not self.env_models or env_category not in self.env_models:
            return control_values["前時刻Indoor Temp."]

        model_info = self.env_models[env_category]

        # デフォルトモデルの場合
        if model_info.get("is_default", False):
            return control_values["前時刻Indoor Temp."]

        # 通常のモデルの場合
        X = pd.DataFrame([control_values])
        X = X[model_info["feature_cols"]]
        return model_info["temp_model"].predict(X)[0]

    def _predict_power(self, control_type, control_values):
        """Phase A予測"""
        if not self.power_models or control_type not in self.power_models:
            # デバッグ情報を追加
            print(f"   ⚠️ Phase A予測モデルが見つかりません: {control_type}")
            return 1000  # デフォルト値

        model_info = self.power_models[control_type]

        # デフォルトモデルの場合
        if model_info.get("is_default", False):
            predicted_power = self._simple_power_predict(control_values)
            print(
                f"   🔋 Phase A予測: {control_type} -> {predicted_power:.2f}W (デフォルトモデル)"
            )
            return predicted_power

        # 通常のモデルの場合
        X = pd.DataFrame([control_values])
        X = X[model_info["feature_cols"]]
        predicted_power = model_info["model"].predict(X)[0]
        print(f"   🔋 Phase A予測: {control_type} -> {predicted_power:.2f}W")
        return predicted_power

    def _simple_power_predict(self, control_values):
        """簡易的なPhase A予測（デフォルトモデル用）"""
        set_temp = control_values.get("A/C Set Temperature", 25)
        mode = control_values.get("A/C Mode", 0)
        fan_speed = control_values.get("A/C Fan Speed", 1)
        is_on = control_values.get("A/C ON/OFF", 1)

        if not is_on:
            return 0

        # A/C Set Temperatureが低いほどPhase A消費が大きい
        base_power = 1000 + (25 - set_temp) * 200

        # A/C Modeによる調整
        if mode == 0:  # 冷房
            power_multiplier = 1.0
        elif mode == 1:  # 除湿
            power_multiplier = 1.2
        else:  # 送風
            power_multiplier = 0.3

        # A/C Fan Speedによる調整
        fan_multiplier = 0.8 + (fan_speed - 1) * 0.2

        return base_power * power_multiplier * fan_multiplier

    def _get_mode_name(self, mode):
        """A/C Mode番号を文字列に変換"""
        mode_names = {
            0: "COOL",  # 冷房
            1: "DEHUM",  # 除湿
            2: "FAN",  # 送風
        }
        return mode_names.get(mode, "FAN")

    def _predict_temperature(
        self, control_type, control_values, current_temp, outdoor_temp, outdoor_humidity
    ):
        """室温予測"""
        if not self.env_models or control_type not in self.env_models:
            return current_temp  # 現在の温度を維持

        model_info = self.env_models[control_type]
        X = pd.DataFrame([control_values])

        # デバッグ情報を追加
        print(f"   🔍 利用可能な列: {list(X.columns)}")
        print(f"   🔍 必要な列: {model_info['feature_cols']}")

        # 必要な列が存在するかチェック
        missing_cols = [
            col for col in model_info["feature_cols"] if col not in X.columns
        ]
        if missing_cols:
            print(f"   ⚠️ 不足している列: {missing_cols}")
            # 不足している列をデフォルト値で埋める
            for col in missing_cols:
                X[col] = 0

        X = X[model_info["feature_cols"]]
        return model_info["temp_model"].predict(X)[0]

    def _get_business_hour(self, row, column_name):
        """営業時間を取得（文字列から時間を抽出）"""
        time_str = str(row.get(column_name, "09:00:00"))
        try:
            # "09:00:00" 形式から時間を抽出
            hour = int(time_str.split(":")[0])
            return hour
        except Exception:
            return 9  # デフォルト値

    def _is_control_target(self, row):
        """制御対象かどうかを判定"""
        # 制御区分列の確認
        control_type = None
        if "制御区分" in row:
            control_type = str(row.get("制御区分", "")).strip()
        elif "予測モデル(提案)" in row:
            control_type = str(row.get("予測モデル(提案)", "")).strip()
        else:
            return False

        unit_id_raw = row.get("PM Mesh ID", "")
        unit_id = str(unit_id_raw).strip() if unit_id_raw is not None else ""

        if not control_type or control_type in ["", "-", "ー", "nan"]:
            return False
        if not unit_id or unit_id in ["", "-", "ー", "nan"]:
            return False
        return True

    def generate_hourly_control_schedule(
        self, optimization_results, master_info, output_dir
    ):
        """時刻別制御スケジュール生成（0時から23時まで、対象日をファイル名に含める）"""
        print("📅 時刻別制御スケジュール生成中...")

        try:
            # 対象日を取得（最適化結果から）
            target_date = None
            for zone_name, zone_schedule in optimization_results.items():
                if zone_schedule and isinstance(zone_schedule, dict):
                    for datetime_key in zone_schedule.keys():
                        if hasattr(datetime_key, "date"):
                            target_date = datetime_key.date()
                            break
                    if target_date:
                        break

            if target_date is None:
                from datetime import datetime

                target_date = datetime.now().date()

            date_str = target_date.strftime("%Y%m%d")
            print(f"   📅 対象日: {target_date}")

            # 1日分の期間を設定（0時から23時まで）
            from datetime import datetime, timedelta

            start_date = datetime.combine(target_date, datetime.min.time())
            end_date = start_date + timedelta(hours=23)

            # 制御区分と室内機の対応を取得
            zones = master_info["zones"]
            control_units = {}
            unit_control_mapping = {}

            env_units = {}  # 制御区分 -> 室内機のマッピング

            for zone_name, zone_info in zones.items():
                # 制御区分内の室内機を取得
                indoor_units = []
                for outdoor_unit_id, outdoor_info in zone_info.get(
                    "outdoor_units", {}
                ).items():
                    indoor_units.extend(outdoor_info.get("indoor_units", []))

                if indoor_units:
                    env_units[zone_name] = []
                    for unit_name in indoor_units:
                        env_units[zone_name].append(
                            {"unit_id": unit_name, "unit_name": unit_name}
                        )
                        print(f"   📋 制御区分: {zone_name} -> 室内機: {unit_name}")

            # 制御区分の室内機を追加
            for zone_name, unit_list in env_units.items():
                for unit_info in unit_list:
                    unit_id = unit_info.get("unit_id")
                    unit_name = unit_info.get("unit_name")

                    print(
                        f"   🔍 制御区分処理: {zone_name} -> {unit_name} (ID: {unit_id})"
                    )

                    # 制御区分の室内機を個別に追加
                    unit_control_mapping[unit_name] = zone_name

                    if zone_name not in control_units:
                        control_units[zone_name] = []
                    control_units[zone_name].append(unit_name)

                    # 制御区分の最適化結果は制御区分の結果を使用
                    # 制御区分の最適化結果が存在する場合はそれを使用
                    if zone_name in optimization_results:
                        # 既存の最適化結果を使用
                        pass
                    else:
                        # 制御区分の最適化結果を環境予測区分にも適用
                        # 制御区分の結果を使用
                        if zone_name in optimization_results:
                            # 既存の最適化結果を使用
                            pass
                        else:
                            # デフォルト値を使用
                            optimization_results[zone_name] = {
                                "set_temp": 25.0,
                                "mode": "COOL",
                                "fan_speed": "Medium",
                            }
                    print(f"   ✅ 制御区分追加: {zone_name} -> {unit_name}")

            # 日時範囲の生成（1時間間隔）
            date_range = pd.date_range(start=start_date, end=end_date, freq="H")

            # 制御区分別ファイルの生成
            self._generate_control_type_schedule(
                optimization_results,
                control_units,
                date_range,
                output_dir,
                master_info,
                date_str,
            )

            # 室内機別ファイルの生成
            self._generate_unit_schedule(
                optimization_results,
                unit_control_mapping,
                date_range,
                output_dir,
                master_info,
                env_units,
                date_str,
            )

        except Exception as e:
            print(f"❌ 制御スケジュール生成エラー: {e}")
            import traceback

            traceback.print_exc()

    def _generate_control_type_schedule(
        self,
        optimization_results,
        control_units,
        date_range,
        output_dir,
        master_info,
        date_str,
    ):
        """制御区分別の制御スケジュール生成"""
        try:
            print("📋 制御区分別スケジュール生成中...")

            # マスタ情報から営業時間を取得
            control_master = master_info.get("control", pd.DataFrame())
            business_hours = {}

            for _, row in control_master.iterrows():
                control_type = row.get("制御区分", "")
                if control_type:
                    open_hour = self._get_business_hour(row, "始業時間")
                    close_hour = self._get_business_hour(row, "就業時間")
                    business_hours[control_type] = (open_hour, close_hour)

            schedule_data = []
            for dt in date_range:
                hour = dt.hour

                # 1つのレコードに全制御区分の情報を含める
                record = {"Date Time": dt.strftime("%Y/%m/%d %H:%M")}

                # 各制御区分の状態を追加
                for control_type, units in control_units.items():
                    target_result = optimization_results.get(control_type, {})

                    # 該当制御区分の営業時間を取得
                    open_hour, close_hour = business_hours.get(control_type, (9, 18))
                    is_business_hours = open_hour <= hour <= close_hour

                    # 2日サイクルの結果から該当時刻の制御値を取得
                    if isinstance(target_result, dict) and dt in target_result:
                        # 2日サイクルの結果の場合
                        control_data = target_result[dt]
                        print(
                            f"   🔍 {control_type} {dt}: 温度={control_data.get('set_temp', 25)}°C"
                        )
                    elif (
                        isinstance(target_result, dict) and "set_temp" in target_result
                    ):
                        # 単一の制御値の場合
                        control_data = target_result
                        print(
                            f"   🔍 {control_type}: 温度={control_data.get('set_temp', 25)}°C"
                        )
                    else:
                        # デフォルト値
                        control_data = {"set_temp": 25, "mode": 0, "fan_speed": 1}
                        print(f"   ⚠️ {control_type}: デフォルト値使用")

                    if is_business_hours and control_data:
                        # 営業時間内
                        record[f"{control_type}_OnOFF"] = "ON"
                        record[f"{control_type}_Mode"] = self._get_mode_name(
                            control_data.get("mode", 0)
                        )
                        record[f"{control_type}_SetTemp"] = control_data.get(
                            "set_temp", 25
                        )
                        record[f"{control_type}_FanSpeed"] = control_data.get(
                            "fan_speed", 1
                        )
                    else:
                        # 営業時間外
                        record[f"{control_type}_OnOFF"] = "OFF"
                        record[f"{control_type}_Mode"] = self._get_mode_name(
                            control_data.get("mode", 2)
                        )
                        record[f"{control_type}_SetTemp"] = control_data.get(
                            "set_temp", 25
                        )
                        record[f"{control_type}_FanSpeed"] = control_data.get(
                            "fan_speed", 1
                        )

                schedule_data.append(record)

            if schedule_data:
                schedule_df = pd.DataFrame(schedule_data)

                # 制御区分別ファイルの保存（対象日をファイル名に含める）
                control_schedule_path = os.path.join(
                    output_dir, f"control_type_schedule_{date_str}.csv"
                )
                schedule_df.to_csv(
                    control_schedule_path, index=False, encoding="utf-8-sig"
                )
                print("✅ 制御区分別スケジュール保存完了:")
                print(f"   📄 ファイル: {control_schedule_path}")
                print(f"   📊 レコード数: {len(schedule_df):,}件")

        except Exception as e:
            print(f"❌ 制御区分別スケジュール生成エラー: {e}")

    def _generate_unit_schedule(
        self,
        optimization_results,
        unit_control_mapping,
        date_range,
        output_dir,
        master_info,
        env_units=None,
        date_str=None,
    ):
        """室内機別の制御スケジュール生成"""
        try:
            print("📋 室内機別スケジュール生成中...")

            # マスタ情報から営業時間を取得
            control_master = master_info.get("control", pd.DataFrame())
            business_hours = {}

            for _, row in control_master.iterrows():
                control_type = row.get("制御区分", "")
                if control_type:
                    open_hour = self._get_business_hour(row, "始業時間")
                    close_hour = self._get_business_hour(row, "就業時間")
                    business_hours[control_type] = (open_hour, close_hour)

            schedule_data = []
            for dt in date_range:
                hour = dt.hour

                # 1つのレコードに全室内機の情報を含める
                record = {"Date Time": dt.strftime("%Y/%m/%d %H:%M")}

                # 各室内機の状態を追加
                for unit_id, control_type in unit_control_mapping.items():
                    target_result = optimization_results.get(control_type, {})

                    # 該当制御区分の営業時間を取得
                    open_hour, close_hour = business_hours.get(control_type, (9, 18))
                    is_business_hours = open_hour <= hour <= close_hour

                    # Indoor Unit Nameを取得（環境予測区分の場合は具体的な名前を使用）
                    unit_display_name = unit_id

                    # 環境予測区分の場合は常に出力
                    if is_business_hours and target_result:
                        # 営業時間内
                        record[f"{unit_display_name}_OnOFF"] = "ON"
                        record[f"{unit_display_name}_Mode"] = self._get_mode_name(
                            target_result.get("mode", 0)
                        )
                        record[f"{unit_display_name}_SetTemp"] = target_result.get(
                            "set_temp", 25
                        )
                        record[f"{unit_display_name}_FanSpeed"] = target_result.get(
                            "fan_speed", 1
                        )
                    else:
                        # 営業時間外
                        record[f"{unit_display_name}_OnOFF"] = "OFF"
                        record[f"{unit_display_name}_Mode"] = self._get_mode_name(
                            target_result.get("mode", 2) if target_result else 2
                        )
                        record[f"{unit_display_name}_SetTemp"] = (
                            target_result.get("set_temp", 25) if target_result else 25
                        )
                        record[f"{unit_display_name}_FanSpeed"] = (
                            target_result.get("fan_speed", 1) if target_result else 1
                        )

                schedule_data.append(record)

            if schedule_data:
                schedule_df = pd.DataFrame(schedule_data)

                # 室内機別ファイルの保存（対象日をファイル名に含める）
                unit_schedule_path = os.path.join(
                    output_dir, f"unit_schedule_{date_str}.csv"
                )
                schedule_df.to_csv(
                    unit_schedule_path, index=False, encoding="utf-8-sig"
                )
                print("✅ 室内機別スケジュール保存完了:")
                print(f"   📄 ファイル: {unit_schedule_path}")
                print(f"   📊 レコード数: {len(schedule_df):,}件")

        except Exception as e:
            print(f"❌ 室内機別スケジュール生成エラー: {e}")


class AirconOptimizer:
    """エアコン最適化システム統合クラス"""

    def __init__(self, store_name, enable_preprocessing=True):
        self.store_name = store_name
        self.enable_preprocessing = enable_preprocessing
        self.preprocessor = DataPreprocessor(store_name)
        self.feature_engineer = FeatureEngineer(store_name)
        self.model_trainer = ModelTrainer(store_name)
        self.optimizer = Optimizer(store_name)
        self.master_info = None

    def _load_existing_processed_data(self):
        """既存の前処理済みデータを読み込み"""
        print(f"📁 {self.store_name}の既存前処理済みデータ読み込み中...")

        ac_control_path = os.path.join(
            self.preprocessor.output_dir, f"ac_control_processed_{self.store_name}.csv"
        )
        power_meter_path = os.path.join(
            self.preprocessor.output_dir, f"power_meter_processed_{self.store_name}.csv"
        )

        ac_control_df = None
        power_meter_df = None

        if os.path.exists(ac_control_path):
            ac_control_df = pd.read_csv(ac_control_path)
            print(f"   ✅ AC制御データ読み込み: {len(ac_control_df):,}件")
        else:
            print(f"   ⚠️ AC制御データが見つかりません: {ac_control_path}")

        if os.path.exists(power_meter_path):
            power_meter_df = pd.read_csv(power_meter_path)
            print(f"   ✅ Phase Aメーターデータ読み込み: {len(power_meter_df):,}件")
        else:
            print(f"   ⚠️ Phase Aメーターデータが見つかりません: {power_meter_path}")

        return ac_control_df, power_meter_df

    def run_step_by_step(self, std_multiplier_temp=3.0, std_multiplier_power=3.0):
        """ステップ別実行（notebook用）"""
        results = {}

        print("=" * 60)
        print("🔧 STEP 1: データ前処理")
        print("=" * 60)

        if self.enable_preprocessing:
            ac_control_df, power_meter_df = self.preprocessor.load_raw_data()
            if ac_control_df is not None:
                ac_control_df = self.preprocessor.preprocess_ac_control_data(
                    ac_control_df, std_multiplier_temp
                )
            if power_meter_df is not None:
                power_meter_df = self.preprocessor.preprocess_power_meter_data(
                    power_meter_df, std_multiplier_power
                )
            self.preprocessor.save_processed_data(ac_control_df, power_meter_df)
        else:
            print(f"⏭️ {self.store_name}の前処理をスキップします")
            ac_control_df, power_meter_df = self._load_existing_processed_data()

        results["ac_control_df"] = ac_control_df
        results["power_meter_df"] = power_meter_df

        print("=" * 60)
        print("📋 STEP 2: マスタデータ読み込み")
        print("=" * 60)

        self.master_info = self.feature_engineer.load_master_data()
        if self.master_info is None:
            print("❌ マスタ情報の取得に失敗しました")
            return None

        results["master_info"] = self.master_info

        print("=" * 60)
        print("🔧 STEP 3: 特徴量作成")
        print("=" * 60)

        feature_df = self.feature_engineer.create_features(
            ac_control_df, power_meter_df, self.master_info
        )
        if feature_df is None:
            print("❌ 特徴量の作成に失敗しました")
            return None

        results["feature_df"] = feature_df

        print("=" * 60)
        print("🤖 STEP 4: モデル訓練")
        print("=" * 60)

        env_models = self.model_trainer.train_environmental_model(
            feature_df, self.master_info
        )
        power_models = self.model_trainer.train_power_model(
            feature_df, self.master_info
        )

        results["env_models"] = env_models
        results["power_models"] = power_models

        print("=" * 60)
        print("🎯 STEP 5: 最適化実行")
        print("=" * 60)

        optimization_results = self.optimizer.optimize_control_values(self.master_info)

        results["optimization_results"] = optimization_results

        print("=" * 60)
        print("📅 STEP 6: スケジュール生成")
        print("=" * 60)

        output_dir = f"planning/{self.store_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.optimizer.generate_hourly_control_schedule(
            optimization_results, self.master_info, output_dir
        )

        print("✅ 全ステップ完了")
        return results

    def run_full_pipeline(
        self,
        std_multiplier_temp=3.0,
        std_multiplier_power=3.0,
    ):
        """フルパイプラインの実行"""
        print(f"🚀 {self.store_name}の最適化パイプライン開始")

        # 1. データ前処理（前処理フラグが有効な場合のみ）
        if self.enable_preprocessing:
            ac_control_df, power_meter_df = self.preprocessor.load_raw_data()
            if ac_control_df is not None:
                ac_control_df = self.preprocessor.preprocess_ac_control_data(
                    ac_control_df, std_multiplier_temp
                )
            if power_meter_df is not None:
                power_meter_df = self.preprocessor.preprocess_power_meter_data(
                    power_meter_df, std_multiplier_power
                )
            self.preprocessor.save_processed_data(ac_control_df, power_meter_df)
        else:
            print(f"⏭️ {self.store_name}の前処理をスキップします")
            # 既存の前処理済みデータを読み込み
            ac_control_df, power_meter_df = self._load_existing_processed_data()

        # 2. マスタデータ読み込み
        self.master_info = self.feature_engineer.load_master_data()
        if self.master_info is None:
            print("❌ マスタ情報の取得に失敗しました")
            return None

        # 3. 特徴量作成
        feature_df = self.feature_engineer.create_features(
            ac_control_df, power_meter_df, self.master_info
        )
        if feature_df is None:
            print("❌ 特徴量の作成に失敗しました")
            return None

        # 統合された特徴量データを保存
        feature_path = (
            f"{self.preprocessor.output_dir}/features_processed_{self.store_name}.csv"
        )
        feature_df.to_csv(feature_path, index=False, encoding="utf-8-sig")
        print(f"💾 統合特徴量データ保存: {feature_path}")

        # 4. モデル訓練
        env_models = self.model_trainer.train_environmental_model(
            feature_df, self.master_info
        )
        power_models = self.model_trainer.train_power_model(
            feature_df, self.master_info
        )

        self.model_trainer.save_models(env_models, power_models)

        # 5. 最適化
        self.optimizer.env_models = env_models
        self.optimizer.power_models = power_models
        optimization_results = self.optimizer.optimize_control_values(self.master_info)

        # 6. 制御スケジュール生成
        output_dir = f"planning/{self.store_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.optimizer.generate_hourly_control_schedule(
            optimization_results, self.master_info, output_dir
        )

        print(f"✅ {self.store_name}の最適化パイプライン完了")
        return optimization_results


# 実行は run_optimization.py を使用してください
