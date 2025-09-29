# エアコン最適化システム

## 概要
エアコンの設定温度とモードを最適化し、電力消費を最小化するシステムです。営業時間内の室温制約を考慮した48時間スケジュールを生成します。

## ファイル構成

### メインファイル
- `aircon_optimizer.py` - メインクラス定義
- `example_usage.py` - 使用例
- `main.ipynb` - Jupyter Notebook版

### データディレクトリ
- `data/Clea/` - クレアデータ
- `data/IsetanMitsukoshi/` - 三越伊勢丹データ
- `data_processed/` - 前処理済みデータ
- `models/` - 学習済みモデル

## クラス構成

### 1. DataLoader
データ読み込みを担当
- `load_ac_control_data()` - AC Control データ読み込み
- `load_power_meter_data()` - Power Meter データ読み込み
- `load_all_data()` - 全データ読み込み

### 2. DataPreprocessor
データ前処理を担当
- `preprocess_data()` - データ前処理
- `save_processed_data()` - 前処理済みデータ保存

### 3. FeatureEngineer
特徴量生成を担当
- `load_master_data()` - マスターデータ読み込み
- `build_daily_features()` - 日次特徴量生成
- `build_hourly_features()` - 時間別特徴量生成

### 4. ModelTrainer
モデル学習を担当
- `train_store_model()` - ストア別モデル学習
- `load_best_model()` - 最良モデル読み込み

### 5. Predictor
予測実行を担当
- `predict_with_model()` - モデルで予測実行

### 6. Optimizer
最適化を担当
- `optimize_48h_schedule()` - 48時間最適化アルゴリズム

### 7. Visualizer
可視化を担当
- `create_optimization_plot()` - 最適化結果の可視化

### 8. AirconOptimizer
メインクラス
- `run_full_pipeline()` - フルパイプライン実行

## 使用方法

### 基本的な使用方法

```python
from aircon_optimizer import AirconOptimizer

# システム初期化
optimizer = AirconOptimizer()

# フルパイプライン実行
result = optimizer.run_full_pipeline(
    target_store="Clea",
    open_hour=10,
    close_hour=20,
    temp_min=22.0,
    temp_max=26.0,
    set_temp_min=22,
    set_temp_max=28,
    modes=["cool", "dry", "fan"],
    horizon_hours=48
)
```

### 設定項目

| パラメータ | 説明 | デフォルト値 |
|------------|------|-------------|
| `target_store` | 対象建物 | "Clea" |
| `open_hour` | 営業開始時間 | 10 |
| `close_hour` | 営業終了時間 | 20 |
| `temp_min` | 最低室温（℃） | 22.0 |
| `temp_max` | 最高室温（℃） | 26.0 |
| `set_temp_min` | 最低設定温度 | 22 |
| `set_temp_max` | 最高設定温度 | 28 |
| `modes` | 探索モード | ["cool", "dry", "fan"] |
| `horizon_hours` | 最適化期間（時間） | 48 |

### 実行例

```bash
# Pythonスクリプト実行
python example_usage.py

# Jupyter Notebook実行
jupyter notebook main.ipynb
```

## 出力結果

### CSVファイル
- `data_processed/{Store}/optimized_schedule_48h.csv` - 時刻別スケジュール
- `data_processed/{Store}/optimized_daily_48h.csv` - 日別合計

### 可視化
- 予測電力消費グラフ
- 予測室温グラフ
- 設定温度とモードグラフ
- 営業時間の背景色表示
- 室温制約線表示

## 特徴

1. **オブジェクト指向設計** - 各機能をクラスに分離
2. **設定項目の明確化** - パラメータを冒頭で設定
3. **エラーハンドリング** - 適切なエラーメッセージ
4. **進捗表示** - 処理状況をリアルタイム表示
5. **可視化** - 営業時間・制約線の表示
6. **柔軟性** - 設定項目の変更が容易

## 注意事項

- データファイルは `data/` ディレクトリに配置
- マスターデータは `master/` ディレクトリに配置
- 前処理済みデータは `data_processed/` に保存
- 学習済みモデルは `models/` に保存
