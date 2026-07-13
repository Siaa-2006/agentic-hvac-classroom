import pandas as pd
import numpy as np

class FuzzySymbolicTranslator:
    def __init__(self):
        pass

    def evaluate_temperature_fuzzy(self, temp: float) -> dict:
        """
        Calculates continuous membership degrees [0, 1] for temperature bands
        using formal triangular and trapezoidal membership functions.
        """
        memberships = {}
        
        # 1. Cold Set (Left-open Trapezoid)
        if temp <= 15.0: 
            memberships["Cold"] = 1.0
        elif 15.0 < temp < 18.0: 
            memberships["Cold"] = (18.0 - temp) / 3.0
        else: 
            memberships["Cold"] = 0.0
        
        # 2. Chilly Set (Triangle)
        if 15.0 < temp <= 18.0: 
            memberships["Chilly"] = (temp - 15.0) / 3.0
        elif 18.0 < temp < 21.0: 
            memberships["Chilly"] = (21.0 - temp) / 3.0
        else: 
            memberships["Chilly"] = 0.0
        
        # 3. Comfortable Set (Trapezoid - matches formal LaTeX equation in report)
        if 18.0 <= temp < 21.0: 
            memberships["Comfortable"] = (temp - 18.0) / 3.0
        elif 21.0 <= temp <= 24.0: 
            memberships["Comfortable"] = 1.0
        elif 24.0 < temp <= 26.5: 
            memberships["Comfortable"] = (26.5 - temp) / 2.5
        else: 
            memberships["Comfortable"] = 0.0
        
        # 4. Mildly Warm Set (Triangle)
        if 24.0 < temp <= 26.5: 
            memberships["Mildly Warm"] = (temp - 24.0) / 2.5
        elif 26.5 < temp < 29.0: 
            memberships["Mildly Warm"] = (29.0 - temp) / 2.5
        else: 
            memberships["Mildly Warm"] = 0.0
        
        # 5. Overheated Set (Right-open Trapezoid)
        if temp >= 29.0: 
            memberships["Overheated"] = 1.0
        elif 26.5 < temp < 29.0: 
            memberships["Overheated"] = (temp - 26.5) / 2.5
        else: 
            memberships["Overheated"] = 0.0
        
        # Determine dominant state
        dominant_state = max(memberships, key=memberships.get)
        
        return {
            "dominant_state": dominant_state,
            "memberships": {k: round(v, 2) for k, v in memberships.items()}
        }

    def evaluate_air_quality_fuzzy(self, co2: float) -> dict:
        """
        Maps continuous NDIR CO2 ppm readings to cognitive impact membership sets.
        """
        memberships = {}
        
        # 1. Excellent/Fresh (Left-open Trapezoid)
        if co2 <= 400.0:
            memberships["Excellent"] = 1.0
        elif 400.0 < co2 < 600.0:
            memberships["Excellent"] = (600.0 - co2) / 200.0
        else:
            memberships["Excellent"] = 0.0
            
        # 2. Adequate (Triangle)
        if 500.0 < co2 <= 800.0:
            memberships["Adequate"] = (co2 - 500.0) / 300.0
        elif 800.0 < co2 < 1000.0:
            memberships["Adequate"] = (1000.0 - co2) / 200.0
        else:
            memberships["Adequate"] = 0.0
            
        # 3. Stuffy/Elevated (Triangle)
        if 900.0 < co2 <= 1200.0:
            memberships["Stuffy"] = (co2 - 900.0) / 300.0
        elif 1200.0 < co2 < 1500.0:
            memberships["Stuffy"] = (1500.0 - co2) / 300.0
        else:
            memberships["Stuffy"] = 0.0
            
        # 4. Hazardous/Critical (Right-open Trapezoid)
        if co2 >= 1500.0:
            memberships["Hazardous"] = 1.0
        elif 1200.0 < co2 < 1500.0:
            memberships["Hazardous"] = (co2 - 1200.0) / 300.0
        else:
            memberships["Hazardous"] = 0.0
            
        dominant_state = max(memberships, key=memberships.get)
        return {
            "dominant_state": dominant_state,
            "memberships": {k: round(v, 2) for k, v in memberships.items()}
        }

    def evaluate_humidity_fuzzy(self, humidity: float) -> dict:
        """Calculates membership levels for humidity."""
        memberships = {}
        
        # Dry
        if humidity <= 30.0: memberships["Dry"] = 1.0
        elif 30.0 < humidity < 40.0: memberships["Dry"] = (40.0 - humidity) / 10.0
        else: memberships["Dry"] = 0.0
        
        # Comfortable
        if 35.0 < humidity <= 55.0: memberships["Comfortable"] = 1.0
        elif 30.0 <= humidity <= 35.0: memberships["Comfortable"] = (humidity - 30.0) / 5.0
        elif 55.0 < humidity < 65.0: memberships["Comfortable"] = (65.0 - humidity) / 10.0
        else: memberships["Comfortable"] = 0.0
        
        # Humid
        if humidity >= 65.0: memberships["Humid"] = 1.0
        elif 55.0 < humidity < 65.0: memberships["Humid"] = (humidity - 55.0) / 10.0
        else: memberships["Humid"] = 0.0
        
        dominant_state = max(memberships, key=memberships.get)
        return {
            "dominant_state": dominant_state,
            "memberships": {k: round(v, 2) for k, v in memberships.items()}
        }

    def aggregate_human_sentiment(self, actual_temp: float, preferences: list) -> str:
        """
        Calculates the real-time gap between physical sensor temperatures
        and active comfort voting matrices.
        """
        if not preferences:
            return "Neutral (No Occupant Input)"
        
        avg_pref = sum(preferences) / len(preferences)
        temp_gap = actual_temp - avg_pref
        
        if temp_gap > 3.0: 
            return "Strong Cooling Demand"
        elif 1.0 < temp_gap <= 3.0: 
            return "Mild Cooling Demand"
        elif -1.0 <= temp_gap <= 1.0: 
            return "Satisfied"
        elif -3.0 <= temp_gap < -1.0: 
            return "Mild Heating Demand"
        else: 
            return "Strong Heating Demand"

    def translate_payload(self, physical_data: dict, occupant_preferences: list) -> dict:
        """Transforms raw telemetry into Fuzzy-Symbolic frames ready for Agent States."""
        actual_temp = physical_data["temperature"]
        co2 = physical_data["co2"]
        humidity = physical_data["humidity"]
        
        temp_eval = self.evaluate_temperature_fuzzy(actual_temp)
        co2_eval = self.evaluate_air_quality_fuzzy(co2)
        humid_eval = self.evaluate_humidity_fuzzy(humidity)
        sentiment = self.aggregate_human_sentiment(actual_temp, occupant_preferences)
        
        return {
            "semantic_state": {
                "thermal_condition": temp_eval["dominant_state"],
                "air_purity": co2_eval["dominant_state"],
                "humidity_condition": humid_eval["dominant_state"],
                "group_comfort_sentiment": sentiment,
                "fuzzy_memberships": {
                    "temperature": temp_eval["memberships"],
                    "co2": co2_eval["memberships"],
                    "humidity": humid_eval["memberships"]
                }
            },
            "meta": {
                "actual_temp_c": actual_temp,
                "actual_co2_ppm": co2,
                "avg_requested_temp": round(sum(occupant_preferences)/len(occupant_preferences), 1) if occupant_preferences else None
            }
        }

