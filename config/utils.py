import os
import platform
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from config import ACCESS_INFORMATION


# 現在のディレクトリを変更し、元のディレクトリを保存する関数
def ch_base_dir():
    global original_dir
    original_dir = os.getcwd()  # 現在のディレクトリを保存
    home_dir = os.path.expanduser("~")  # ホームディレクトリのパスを取得
    os.chdir(home_dir)  # ホームディレクトリに移動
    return os.getcwd()  # 新しいカレントディレクトリを返す


# 元のディレクトリに戻る関数
def reverse_dir():
    os.chdir(original_dir)  # 保存していた元のディレクトリに戻る


# 指定されたディレクトリに移動し、元のディレクトリを保存する関数
def change_dir(temp_folder_path):
    global original_dir
    original_dir = os.getcwd()  # 現在のディレクトリを保存
    os.chdir(temp_folder_path)  # 指定されたディレクトリに移動


# DataFrameをCSVファイルとして保存する関数
def upload_file(df: pd.DataFrame, name, temp_folder_path):
    change_dir(temp_folder_path)  # 指定されたディレクトリに移動
    df.to_csv(f"{name}.csv", index=False, encoding="utf-8")  # CSVとして保存
    reverse_dir()  # 元のディレクトリに戻る


def join_paths(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)


# 設定ファイルを読み込む関数
def load_config():
    root_path = os.path.dirname(os.path.dirname(__file__))  # ルートパスを取得
    config_path = "./config/config.yml"  # 設定ファイルのパス
    abs_path = Path(root_path, config_path).absolute()  # 絶対パスを生成

    # カスタムタグを登録
    yaml.SafeLoader.add_constructor("!join", join_paths)

    with open(abs_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)  # YAMLファイルを読み込む

    # プレースホルダーを実際の値で置換
    config["data_path"] = {
        k: v.format(ACCESS_INFORMATION=ACCESS_INFORMATION) if isinstance(v, str) else v
        for k, v in config["data_path"].items()
    }

    # 環境変数を展開
    for key, value in config["data_path"].items():
        if isinstance(value, str):
            config["data_path"][key] = os.path.expandvars(value)

    return config  # 設定を返す


# Google Driveの言語を検出する関数
def detect_google_drive_language():
    possible_drive_letters = [
        "G:",
        "H:",
        "I:",
        "J:",
        "K:",
    ]  # 可能性のあるドライブレター
    jp_folder = "共有ドライブ"  # 日本語フォルダ名
    en_folder = "Shared drives"  # 英語フォルダ名

    for drive in possible_drive_letters:
        jp_path = os.path.join(drive, "\\", jp_folder)
        en_path = os.path.join(drive, "\\", en_folder)

        if os.path.exists(jp_path):
            return "JP", drive  # 日本語が見つかった場合
        elif os.path.exists(en_path):
            return "EN", drive  # 英語が見つかった場合

    raise ValueError(
        "Google Drive shared folder not found. Please check your Google Drive installation."
    )


# OSと言語を検出する関数
def detect_os_and_language():
    os_name = platform.system().lower()  # OSの名前を取得
    if os_name == "darwin":
        return "mac", None  # Macの場合
    elif os_name == "windows":
        lang, drive = (
            detect_google_drive_language()
        )  # Windowsの場合、Google Driveの言語も検出
        return "win", lang, drive
    else:
        raise ValueError(f"Unsupported OS: {os_name}")


# データパスを取得する関数
def get_data_path(path_key: str) -> str:
    """データパスを取得する関数

    path_key: config.ymlのdata_pathのキー
    """

    root_path = ch_base_dir()  # ベースディレクトリを変更
    config = load_config()  # 設定を読み込む

    os_type, *extra = detect_os_and_language()  # OSと言語を検出
    if os_type == "win":
        # reverse_dir()  # 元のディレクトリに戻る
        lang, drive = extra
        base_path = f"{drive}\\"
        lang_path = (
            config["data_path"]["EN"] if lang == "EN" else config["data_path"]["JP"]
        )
        base_path_components = config["data_path"]["base_path_components"]
        full_base_path = os.path.join(base_path, lang_path, *base_path_components)
        data_path = os.path.join(full_base_path, config["data_path"][path_key])
    else:  # Mac
        lang = extra
        base_path = config["data_path"]["mac_base_path"]
        base_path_components = config["data_path"]["base_path_components"]
        lang_path = (
            config["data_path"]["EN"] if lang == "EN" else config["data_path"]["JP"]
        )
        full_base_path = os.path.join(
            base_path + ACCESS_INFORMATION, lang_path, *base_path_components
        )
        data_path = os.path.join(full_base_path, config["data_path"][path_key])

    # パスの正規化と絶対パスへの変換
    abs_path = os.path.abspath(os.path.normpath(data_path))
    return str(abs_path)  # 文字列として返す


def create_output_folder_paths(use_google_api=False):
    """出力フォルダのパスを作成します。"""

    trainValid_input_path = get_data_path(
        "trainValid_input_path",
    )

    trainValid_result_paths = {
        "result": get_data_path(
            "valid_out_result_path",
        ),
        "score": get_data_path(
            "score_report_path",
        ),
        "model": get_data_path(
            "trained_models_folder",
        ),
    }

    trained_models_folder = get_data_path(
        "trained_models_folder",
    )
    predict_out_path = get_data_path(
        "predict_out_path",
    )

    return (
        trainValid_input_path,
        trainValid_result_paths,
        trained_models_folder,
        predict_out_path,
    )


# Read the AC Master Data file
def read_ac_master_data() -> dict:
    """
    Reads the AC master data file and extracts relevant temperature thresholds.
    """
    ac_master_data_path = get_data_path("AC_master_data_path")
    ac_master_data = pd.read_excel(ac_master_data_path)
    master_start_mode_temperature = {
        "cooling_start_temp": ac_master_data.loc[0, "冷房開始温度"],
        "heating_start_temp": ac_master_data.loc[0, "暖房開始温度"],
    }

    print(f"ac_start_mode_temperature: {master_start_mode_temperature}")
    return master_start_mode_temperature


def get_and_extract_latest_zip(raw_data_path):
    """
    Find the latest zip file in raw_data_path, extract it to data/temp_raw_data,
    and return the path to the extracted folder.
    """
    # print all files in raw_data_path
    print(os.listdir(raw_data_path))

    # Get the latest zip file
    zip_files = [
        f for f in os.listdir(raw_data_path) if f.endswith(".zip") and f[:8].isdigit()
    ]
    if not zip_files:
        print("No zip files found in the raw_data_path")
        return None, None

    latest_zip = max(zip_files, key=lambda x: datetime.strptime(x[:8], "%Y%m%d"))
    latest_zip_path = os.path.join(raw_data_path, latest_zip)

    # Prepare extraction path - use Google Drive processed data path
    extract_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data",
        "temp_raw_data",
    )
    os.makedirs(extract_path, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(latest_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    # Get the path of the extracted folder (assuming zip contains a single folder)
    extracted_folder = os.path.join(extract_path, os.path.splitext(latest_zip)[0])

    # If it's not a folder, use the extract_path
    if not os.path.isdir(extracted_folder):
        extracted_folder = extract_path

    raw_data_date = latest_zip[:8]

    return extracted_folder, raw_data_date
