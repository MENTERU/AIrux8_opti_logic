# =============================================================================
# エアコン最適化システム - 実行サンプル
# =============================================================================

from aircon_optimizer import AirconOptimizer


def run_optimization_for_store(
    store_name, std_multiplier_temp=3.0, std_multiplier_power=3.0
):
    """
    指定されたストアの最適化を実行

    Args:
        store_name (str): 対象ストア名
        std_multiplier_temp (float): 温度データの外れ値判定係数
        std_multiplier_power (float): 電力データの外れ値判定係数
    """
    print(f"🚀 {store_name}の最適化パイプライン開始")

    # 最適化システムの初期化（前処理を実行）
    enable_preprocessing = True
    optimizer = AirconOptimizer(store_name, enable_preprocessing=enable_preprocessing)

    # フルパイプラインの実行
    results = optimizer.run_full_pipeline(
        std_multiplier_temp=std_multiplier_temp,
        std_multiplier_power=std_multiplier_power,
    )

    if results:
        print(f"🎉 {store_name}の最適化が完了しました")
        print("📁 結果ファイル:")
        print(f"   - planning/{store_name}/control_type_schedule.csv")
        print(f"   - planning/{store_name}/unit_schedule.csv")
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
            std_multiplier_temp=3.0,
            std_multiplier_power=3.0,
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
