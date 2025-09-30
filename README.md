# エアコン最適化システム

## 概要

エアコンの設定温度とモードを最適化し、電力消費を最小化するシステムです。営業時間内の室温制約を考慮した 48 時間スケジュールを生成します。

## セットアップ

### 1. プライベート情報ファイルの作成

システムを使用する前に、`config/private_information.py` ファイルを作成し、以下の変数を設定してください：

```python
# config/private_information.py
# gmailのメールアドレス (Gdriveから取得する場合。現在停止中)
ACCESS_INFORMATION = "name@menteru.jp" または　ACCESS_INFORMATION = "name@gmail.com"
# visual crossing Weather API Key
WEATHER_API_KEY = "weather_api_key_here"
```

**注意**: このファイルは `.gitignore` に含まれているため、Git にコミットされません。各開発者が個別に作成する必要があります。

### 2. 必要な API キーの取得

- **Weather API Key**: Visual Crossing Weather API のキーを取得
  - サイト: https://www.visualcrossing.com/weather-api
  - 無料プランでも利用可能

### 3. データフォルダの準備

#### ローカルパスを使用する場合（推奨）

ローカルパスを使用する場合は、データフォルダをプロジェクトのルートディレクトリに配置してください。

**🗂️フォルダ構造:**

```
AIrux8_opti_logic/
├── aircon_optimizer.py
├── run_optimization.py
├── config/
├── processing/
├── machine_learning/
├── optimization/
├── planning/
├── service/
└── data/                    # ← このフォルダをダウンロードして配置
    ├── 00_InputData/        # 生データ
    ├── 01_MasterData/       # マスターデータ
    ├── 02_PreprocessedData/ # 前処理済みデータ（自動生成）
    ├── 03_Models/           # 学習済みモデル（自動生成）
    └── 04_PlanningData/     # 計画データ（自動生成）
```

**データフォルダの取得方法:**

1. Google Drive または共有ストレージから `data/` フォルダをダウンロード
2. プロジェクトのルートディレクトリ（`AIrux8_opti_logic/`）に配置
3. フォルダ構造が上記の通りになっていることを確認

### 4. パス設定の選択

システムは以下の 2 つのパス設定をサポートしています：

- **ローカルパス** (`local_paths`): プロジェクト内の `data/` フォルダを使用（デフォルト・推奨）
- **リモートパス** (`remote_paths`): Google Drive の共有フォルダを使用

```python
from config.utils import get_data_path

# ローカルパスを使用（デフォルト）
local_path = get_data_path("raw_data_path")

# リモートパスを使用
remote_path = get_data_path("raw_data_path", use_remote_paths=True)
```

### メインファイル

- `run_optimization.py` - 実行スクリプト
- `aircon_optimizer_simple.ipynb` - Jupyter Notebook 版

## クラス構成（プロセス別）

### 1. データ処理プロセス (`processing/`)

#### DataPreprocessor (`preprocessor.py`)

- `load_raw()` - 生データ読み込み
- `preprocess_ac()` - AC 制御データ前処理
- `preprocess_pm()` - 電力メーターデータ前処理
- `save()` - 前処理済みデータ保存

#### AreaAggregator (`aggregator.py`)

- `build()` - 制御エリア単位でのデータ集約
- `_most_frequent()` - 最頻値計算

#### MasterLoader (`utilities/master_loader.py`)

- `load()` - マスターデータ読み込み

#### VisualCrossingWeatherAPIDataFetcher (`utilities/weather_client.py`)

- `fetch()` - 天気データ取得

### 2. 機械学習プロセス (`machine_learning/`)

#### ModelBuilder (`model_builder.py`)

- `train_by_zone()` - 制御エリア別モデル学習
- `_split_xy()` - 特徴量・目的変数分割

### 3. 最適化プロセス (`optimization/`)

#### Optimizer (`optimizer.py`)

- `optimize_day()` - 1 日分の最適化実行
- `_eval_score()` - スコア評価
- `_gen_candidates()` - 候補生成

#### AirconOptimizer (`aircon_optimizer.py`)

- `run()` - フルパイプライン実行
- `_load_processed()` - 前処理済みデータ読み込み

### 4. 計画・出力プロセス (`planning/`)

#### Planner (`planner.py`)

- `export()` - スケジュール出力
- `_mode_text()` - モード名変換

## 使用方法

### 基本的な使用方法

```python
from optimization.aircon_optimizer import AirconOptimizer
from config.private_information import WEATHER_API_KEY

# システム初期化
optimizer = AirconOptimizer(
    store_name="Clea",
    enable_preprocessing=True
)

# フルパイプライン実行
result = optimizer.run(
    weather_api_key=WEATHER_API_KEY,
    coordinates="35.681236%2C139.767125",
    start_date="2024-01-01",
    end_date="2024-01-02",
    freq="1H",
    preference="balanced"  # "comfort", "energy", "balanced"
)
```

### 設定項目

| パラメータ             | 説明                 | デフォルト値             |
| ---------------------- | -------------------- | ------------------------ |
| `store_name`           | 対象ストア名         | "Clea"                   |
| `enable_preprocessing` | 前処理実行フラグ     | True                     |
| `weather_api_key`      | 天気 API キー        | None                     |
| `coordinates`          | 座標（緯度%2C 経度） | "35.681236%2C139.767125" |
| `start_date`           | 開始日               | 今日                     |
| `end_date`             | 終了日               | 明日                     |
| `freq`                 | 時間間隔             | "1H"                     |
| `preference`           | 最適化優先度         | "balanced"               |

### 実行例

```bash
# Pythonスクリプト実行
python run_optimization.py

# Jupyter Notebook実行
jupyter notebook aircon_optimizer_simple.ipynb
```

## 出力結果

### CSV ファイル

- `data/04_PlanningData/{Store}/control_type_schedule_YYYYMMDD.csv` - 制御区分別スケジュール
- `data/04_PlanningData/{Store}/unit_schedule_YYYYMMDD.csv` - 室内機別スケジュール

### 中間データ

- `data/02_PreprocessedData/{Store}/ac_control_processed_{Store}.csv` - 前処理済み AC 制御データ
- `data/02_PreprocessedData/{Store}/power_meter_processed_{Store}.csv` - 前処理済み電力データ
- `data/02_PreprocessedData/{Store}/features_processed_{Store}.csv` - 特徴量データ

### 学習済みモデル

- `data/03_Models/{Store}/models_{Zone}.pkl` - 制御エリア別学習済みモデル

## 特徴

1. **プロセス別設計** - 処理プロセスに応じた階層分離
2. **オブジェクト指向設計** - 各機能をクラスに分離
3. **設定項目の明確化** - パラメータを冒頭で設定
4. **エラーハンドリング** - 適切なエラーメッセージ
5. **進捗表示** - 処理状況をリアルタイム表示
6. **柔軟性** - 設定項目の変更が容易
7. **セキュリティ** - プライベート情報の分離管理

## 注意事項

- **プライベート情報**: `config/private_information.py` を必ず作成してください
- **データファイル**: `data/00_InputData/` ディレクトリに配置
- **マスターデータ**: `data/01_MasterData/` ディレクトリに配置
- **前処理済みデータ**: `data/02_PreprocessedData/` に保存
- **学習済みモデル**: `data/03_Models/` に保存
- **計画データ**: `data/04_PlanningData/` に出力

## トラブルシューティング

### よくある問題

1. **`private_information.py` が見つからない**

   - `config/private_information.py` ファイルを作成してください
   - 必要な変数を設定してください

2. **天気データが取得できない**

   - Visual Crossing Weather API のキーが正しく設定されているか確認
   - インターネット接続を確認

3. **データファイルが見つからない**

   - `data/` フォルダがプロジェクトのルートディレクトリに配置されているか確認
   - `data/00_InputData/{Store}/` に必要な CSV ファイルが存在するか確認
   - ファイル名が正しいか確認（ac-control-_.csv, ac-power-meter-_.csv）
   - データフォルダの構造が正しいか確認（上記の「ルートディレクトリの構造」を参照）

4. **パス設定の問題**

   - ローカルパスを使用する場合：`data/` フォルダが正しく配置されているか確認
   - リモートパスを使用する場合：Google Drive の接続とアクセス権限を確認
