"""
Feature builder for optimization pipeline
Creates all features needed by the trained models during optimization
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


class OptimizationFeatureBuilder:
    """
    Builds features for optimization pipeline to match training features
    """

    def __init__(self):
        self.history_buffer = {}  # Store historical values for lag features

    def build_features(
        self,
        base_features: Dict[str, Any],
        timestamp: pd.Timestamp,
        zone_name: str,
        weather_history: Dict[pd.Timestamp, Dict[str, float]] = None,
        power_history: Dict[pd.Timestamp, float] = None,
    ) -> pd.DataFrame:
        """
        Build complete feature set for model prediction

        Args:
            base_features: Basic features (A/C controls, weather, time)
            timestamp: Current timestamp
            zone_name: Zone name for history tracking
            weather_history: Historical weather data
            power_history: Historical power data

        Returns:
            DataFrame with all required features
        """

        # Start with base features
        features = base_features.copy()

        # Create A/C Status from A/C ON/OFF and A/C Mode
        # Status mapping: OFF=0, COOL=1, HEAT=2, FAN=3
        ac_onoff = base_features.get("A/C ON/OFF", 0)
        ac_mode = base_features.get("A/C Mode", 1)  # Default to COOL
        if ac_onoff == 0:
            features["A/C Status"] = 0  # OFF
        else:
            features["A/C Status"] = int(ac_mode)  # Use mode value when ON

        # Add time-based features
        features.update(self._get_time_features(timestamp))

        # Add engineered features
        features.update(
            self._get_engineered_features(
                base_features, timestamp, zone_name, weather_history, power_history
            )
        )

        return pd.DataFrame([features])

    def _get_time_features(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Get time-based features"""
        return {
            "DayOfWeek": timestamp.weekday(),
            "Hour": timestamp.hour,
            "Month": timestamp.month,
            "IsWeekend": 1 if timestamp.weekday() >= 5 else 0,
            "IsHoliday": 0,  # Simplified - could use jpholiday if needed
            "HourOfWeek": timestamp.weekday() * 24 + timestamp.hour,
        }

    def _get_engineered_features(
        self,
        base_features: Dict[str, Any],
        timestamp: pd.Timestamp,
        zone_name: str,
        weather_history: Dict[pd.Timestamp, Dict[str, float]] = None,
        power_history: Dict[pd.Timestamp, float] = None,
    ) -> Dict[str, Any]:
        """Get engineered features"""

        features = {}

        # Wet Bulb Temperature (simplified calculation)
        outdoor_temp = base_features.get("Outdoor Temp.", 25.0)
        outdoor_humidity = base_features.get("Outdoor Humidity", 60.0)
        features["Wet Bulb Temp"] = self._calculate_wet_bulb_temp(
            outdoor_temp, outdoor_humidity
        )

        # Temperature differences
        indoor_temp_lag1 = base_features.get("Indoor Temp. Lag1", 25.0)
        set_temp = base_features.get("A/C Set Temperature", 26.0)

        features["Temp Diff (Outdoor - Indoor Lag1)"] = outdoor_temp - indoor_temp_lag1
        features["Temp Diff (Indoor Lag1 - Setpoint)"] = indoor_temp_lag1 - set_temp

        # OnRunLength (simplified - track consecutive ON periods)
        ac_onoff = base_features.get("A/C ON/OFF", 0)
        if zone_name not in self.history_buffer:
            self.history_buffer[zone_name] = {"on_run_length": 0, "last_onoff": 0}

        if ac_onoff == 1:
            if self.history_buffer[zone_name]["last_onoff"] == 1:
                self.history_buffer[zone_name]["on_run_length"] += 1
            else:
                self.history_buffer[zone_name]["on_run_length"] = 1
        else:
            self.history_buffer[zone_name]["on_run_length"] = 0

        features["OnRunLength"] = self.history_buffer[zone_name]["on_run_length"]
        self.history_buffer[zone_name]["last_onoff"] = ac_onoff

        # Lag features (simplified - use current values if no history)
        features.update(
            self._get_lag_features(
                timestamp, weather_history, power_history, base_features
            )
        )

        # Rolling mean features (simplified - use current values)
        features.update(
            self._get_rolling_features(
                timestamp, weather_history, power_history, base_features
            )
        )

        return features

    def _calculate_wet_bulb_temp(self, temp_c: float, humidity_pct: float) -> float:
        """Calculate wet bulb temperature (simplified)"""
        # Simplified wet bulb calculation
        # More accurate would use psychrometric formulas
        return temp_c - (100 - humidity_pct) / 5.0

    def _get_lag_features(
        self,
        timestamp: pd.Timestamp,
        weather_history: Dict[pd.Timestamp, Dict[str, float]] = None,
        power_history: Dict[pd.Timestamp, float] = None,
        base_features: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Get lag features"""

        features = {}

        # Lag 1 hour
        lag1_time = timestamp - pd.Timedelta(hours=1)
        lag24_time = timestamp - pd.Timedelta(hours=24)

        # Outdoor temperature lags
        if weather_history and lag1_time in weather_history:
            features["Outdoor Temp. lag1"] = weather_history[lag1_time].get(
                "outdoor_temp", 25.0
            )
        else:
            features["Outdoor Temp. lag1"] = base_features.get("Outdoor Temp.", 25.0)

        if weather_history and lag24_time in weather_history:
            features["Outdoor Temp. lag24"] = weather_history[lag24_time].get(
                "outdoor_temp", 25.0
            )
        else:
            features["Outdoor Temp. lag24"] = base_features.get("Outdoor Temp.", 25.0)

        # Outdoor humidity lags
        if weather_history and lag1_time in weather_history:
            features["Outdoor Humidity lag1"] = weather_history[lag1_time].get(
                "outdoor_humidity", 60.0
            )
        else:
            features["Outdoor Humidity lag1"] = base_features.get(
                "Outdoor Humidity", 60.0
            )

        if weather_history and lag24_time in weather_history:
            features["Outdoor Humidity lag24"] = weather_history[lag24_time].get(
                "outdoor_humidity", 60.0
            )
        else:
            features["Outdoor Humidity lag24"] = base_features.get(
                "Outdoor Humidity", 60.0
            )

        # Power lags (use 0 if no history)
        if power_history and lag1_time in power_history:
            features["adjusted_power lag1"] = power_history[lag1_time]
        else:
            features["adjusted_power lag1"] = 0.0

        if power_history and lag24_time in power_history:
            features["adjusted_power lag24"] = power_history[lag24_time]
        else:
            features["adjusted_power lag24"] = 0.0

        return features

    def _get_rolling_features(
        self,
        timestamp: pd.Timestamp,
        weather_history: Dict[pd.Timestamp, Dict[str, float]] = None,
        power_history: Dict[pd.Timestamp, float] = None,
        base_features: Dict[str, Any] = None,
    ) -> Dict[str, float]:
        """Get rolling mean features (simplified)"""

        features = {}

        # For rolling features, we'll use current values as approximation
        # In a full implementation, you'd calculate rolling means from history

        current_temp = base_features.get("Outdoor Temp.", 25.0)
        current_humidity = base_features.get("Outdoor Humidity", 60.0)
        current_power = 0.0  # No current power value available

        # Rolling means (simplified - use current values)
        features["Outdoor Temp. rolling_mean3"] = current_temp
        features["Outdoor Temp. rolling_mean24"] = current_temp
        features["Outdoor Humidity rolling_mean3"] = current_humidity
        features["Outdoor Humidity rolling_mean24"] = current_humidity
        features["adjusted_power rolling_mean3"] = current_power
        features["adjusted_power rolling_mean24"] = current_power

        return features
