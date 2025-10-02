# -*- coding: utf-8 -*-
"""
レポーティング統合（ダッシュボードのみ生成）
 - 出力リセット
 - 既存の可視化呼び出しを一本化
 - 拡張分析HTMLの生成は停止
"""

import os
import shutil

from analysis.dashboards import (
    create_historical_dashboard,
    create_plan_validation_dashboard,
)


def reset_outputs(store_name: str = "Clea") -> None:
    """分析/可視化の出力をリセット（削除→フォルダ再作成）"""
    out_dir = "analysis/output"
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"🧹 出力をリセットしました: {out_dir}")


def generate_all_reports(store_name: str = "Clea"):
    """全レポート生成を統合呼び出し（ダッシュボードのみ）"""
    # 実績ダッシュボード（時別/日別）
    create_historical_dashboard(store_name, freq="H")
    create_historical_dashboard(store_name, freq="D")
    # 計画妥当性ダッシュボード
    create_plan_validation_dashboard(store_name, lookback_days=7)
    print("📦 全ダッシュボード生成が完了しました")
