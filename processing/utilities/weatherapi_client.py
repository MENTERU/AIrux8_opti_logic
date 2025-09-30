from typing import Optional

import pandas as pd
import requests


# =============================
# IF: 天候データ取得
# =============================
class VisualCrossingWeatherAPIDataFetcher:
    """Visual Crossing Weather API で時別の天気（気温・湿度）を取得"""

    def __init__(
        self,
        coordinates: str,
        start_date: str,
        end_date: str,
        unit: str,
        api_key: str,
        temperature_col_name: str = "Outdoor Temp.",
        humidity_col_name: str = "Outdoor Humidity",
    ):
        self.coordinates = coordinates
        self.start_date = start_date
        self.end_date = end_date
        self.unit = unit
        self.api_key = api_key
        self.temperature_col_name = temperature_col_name
        self.humidity_col_name = humidity_col_name

    def fetch(self) -> Optional[pd.DataFrame]:
        try:
            url = (
                f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
                f"{self.coordinates}/{self.start_date}/{self.end_date}?unitGroup={self.unit}&key={self.api_key}&include=hours"
            )
            print(f"[Weather] Making API request to: {url}")

            res = requests.get(url, timeout=30)
            print(f"[Weather] HTTP Status Code: {res.status_code}")

            if res.status_code != 200:
                print(f"[Weather] HTTP Error: {res.status_code}")
                print(f"[Weather] Response text: {res.text[:500]}")
                return None

            res.raise_for_status()
            data = res.json()

            print(f"[Weather] API Response keys: {list(data.keys())}")
            print(f"[Weather] Number of days in response: {len(data.get('days', []))}")

            rows = []
            for d in data.get("days", []):
                print(f"[Weather] Processing day: {d.get('datetime', 'Unknown')}")
                hours = d.get("hours", [])
                print(f"[Weather] Number of hours for this day: {len(hours)}")

                for h in hours:
                    rows.append(
                        {
                            "datetime": f"{d['datetime']} {h['datetime']}",
                            self.temperature_col_name: h.get("temp"),
                            self.humidity_col_name: h.get("humidity"),
                        }
                    )

            print(f"[Weather] Total rows collected: {len(rows)}")

            df = pd.DataFrame(rows)
            if df.empty:
                print("[Weather] DataFrame is empty after processing")
                return None

            df["datetime"] = pd.to_datetime(df["datetime"])
            print(f"[Weather] Final DataFrame shape: {df.shape}")
            print(
                f"[Weather] Date range: {df['datetime'].min()} to {df['datetime'].max()}"
            )

            return df

        except requests.exceptions.RequestException as e:
            print(f"[Weather] Request Error: {e}")
            return None
        except Exception as e:
            print(f"[Weather] Unexpected Error: {e}")
            print(f"[Weather] Error type: {type(e).__name__}")
            return None
