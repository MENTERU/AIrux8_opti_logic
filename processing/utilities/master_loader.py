import json
import os
from typing import Optional

import pandas as pd


# =============================
# IF: マスタ読み込み
# =============================
class MasterLoader:
    def __init__(self, store_name: str):
        self.store_name = store_name

    def load(self) -> Optional[dict]:
        from config.utils import get_data_path

        master_dir = get_data_path("master_data_path")
        p = os.path.join(master_dir, f"MASTER_{self.store_name}.json")
        if not os.path.exists(p):
            print(f"[Master] ファイルなし: {p}")
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                m = json.load(f)
            print(f"[Master] 読込OK: keys={list(m.keys())}")
            return m
        except Exception as e:
            print(f"[Master] 読込エラー: {e}")
            return None
