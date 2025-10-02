# =============================================================================
# エアコン最適化システム - 実行サンプル
# =============================================================================

from analysis.reporting import reset_outputs, generate_all_reports
from config.private_information import WEATHER_API_KEY
from optimization.aircon_optimizer import AirconOptimizer


def run_optimization_for_store(
    store_name, temperature_std_multiplier=5.0, power_std_multiplier=5.0
):
    """
    指定されたストアの最適化を実行

    Args:
        store_name (str): 対象ストア名
        temperature_std_multiplier (float): 温度データの外れ値判定係数（デフォルト: 5.0）
        power_std_multiplier (float): 電力データの外れ値判定係数（デフォルト: 5.0）
    """
    print(f"🚀 {store_name}の最適化パイプライン開始")

    # 最適化システムの初期化（前処理をスキップ）
    enable_preprocessing = False
    optimizer = AirconOptimizer(
        store_name,
        enable_preprocessing=enable_preprocessing,
    )

    # フルパイプラインの実行（座標はマスタから自動取得）
    results = optimizer.run(
        weather_api_key=WEATHER_API_KEY,
        temperature_std_multiplier=temperature_std_multiplier,
        power_std_multiplier=power_std_multiplier,
        preference="energy",  # 電力優先で最適化
    )

    if results:
        print(f"🎉 {store_name}の最適化が完了しました")
        print("📁 結果ファイル:")
        print(f"   - data/04_OutputData/{store_name}/control_type_schedule.csv")
        print(f"   - data/04_OutputData/{store_name}/unit_schedule.csv")

        # 可視化の実行
        print(f"\n📊 {store_name}の結果可視化を開始...")
        try:
            # 出力をリセットしてから全レポート生成
            reset_outputs(store_name)
            stats_df = None
            try:
                generate_all_reports(store_name)
            except Exception as re:
                print(f"⚠️ レポート生成でエラー: {re}")

            print(f"✅ {store_name}の可視化が完了しました")
            print("📁 可視化ファイル:")
            print("   - analysis/output/*_analysis.html (各ゾーン分析)")
            print("   - analysis/output/summary_analysis.html (全体サマリー)")
            print("   - analysis/output/summary_statistics.csv (統計データ)")

            if stats_df is not None:
                print("\n📊 統計サマリー:")
                print(stats_df.to_string(index=False))

        except Exception as e:
            print(f"⚠️ 可視化でエラーが発生しました: {e}")
            print("最適化結果は正常に生成されています")

        return True
    else:
        print(f"❌ {store_name}の最適化に失敗しました")
        return False


def main():
    """メイン実行関数"""
    # 対象ストアのリスト（Cleaのみ）
    target_stores = ["Clea"]

    # 各ストアの最適化を実行
    for store_name in target_stores:
        print(f"\n{'='*50}")
        print(f"🏢 {store_name} の最適化開始")
        print(f"{'='*50}")

        success = run_optimization_for_store(
            store_name=store_name,
            temperature_std_multiplier=5.0,
            power_std_multiplier=5.0,
        )

        if success:
            print(f"✅ {store_name} の最適化完了")
        else:
            print(f"❌ {store_name} の最適化失敗")

    print(f"\n{'='*50}")
    print("🎯 全ストアの最適化処理完了")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
