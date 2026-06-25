import pandas as pd

class FuzzySymbolicTranslator:
    def __init__(self):
        pass

    def evaluate_temperature(self, temp):
        """Maps crisp temperature to symbolic comfort levels."""
        if temp < 18.0: return "Cold"
        elif 18.0 <= temp < 21.0: return "Chilly"
        elif 21.0 <= temp <= 24.0: return "Comfortable"
        elif 24.0 < temp <= 26.5: return "Mildly Warm"
        else: return "Overheated"

    def evaluate_air_quality(self, co2):
        """Maps crisp CO2 ppm readings to cognitive impact."""
        if co2 < 600: return "Excellent"
        elif 600 <= co2 <= 1000: return "Adequate"
        elif 1000 < co2 <= 1500: return "Stuffy"
        else: return "Hazardous"

    def evaluate_humidity(self, humidity):
        """Maps relative humidity percentages to comfort levels."""
        if humidity < 30.0: return "Dry"
        elif 30.0 <= humidity <= 60.0: return "Comfortable"
        else: return "Humid"

    def aggregate_human_sentiment(self, actual_temp, preferences):
        """
        Calculates the gap between what the room is physically, 
        and what the humans actually want.
        """
        if not preferences:
            return "Neutral (No Occupant Input)"
        
        avg_pref = sum(preferences) / len(preferences)
        temp_gap = actual_temp - avg_pref
        
        # Positive gap means room is hotter than desired -> They want cooling
        if temp_gap > 3.0: return "Strong Cooling Demand"
        elif 1.0 < temp_gap <= 3.0: return "Mild Cooling Demand"
        elif -1.0 <= temp_gap <= 1.0: return "Satisfied"
        elif -3.0 <= temp_gap < -1.0: return "Mild Heating Demand"
        else: return "Strong Heating Demand"

    def translate_payload(self, physical_data, occupant_preferences):
        """Transforms the raw sensor data and human app inputs into an Agent-Ready Frame."""
        actual_temp = physical_data["temperature"]
        co2 = physical_data["co2"]
        humidity = physical_data["humidity"]
        
        return {
            "semantic_state": {
                "thermal_condition": self.evaluate_temperature(actual_temp),
                "air_purity": self.evaluate_air_quality(co2),
                "humidity_condition": self.evaluate_humidity(humidity),
                "group_comfort_sentiment": self.aggregate_human_sentiment(actual_temp, occupant_preferences)
            },
            "meta": {
                "actual_temp_c": actual_temp,
                "actual_co2_ppm": co2,
                "avg_requested_temp": round(sum(occupant_preferences)/len(occupant_preferences), 1) if occupant_preferences else None
            }
        }

# --- Local Verification & Testing ---
if __name__ == "__main__":
    import json
    print("=== Testing Translator with Real IAQ Data ===")
    
    # 1. Load the cleaned real-world data
    try:
        df = pd.read_csv("cleaned_classrooms.csv")
        
        # Let's pick a random row from the dataset (e.g., row 1000)
        sample_row = df.iloc[1000].to_dict()
        print("\n[Physical Sensor Reality (Row 1000)]")
        print(f"Temperature: {sample_row['temperature']}°C")
        print(f"CO2 Level:   {sample_row['co2']} ppm")
        
        # 2. Inject Mock Human Input (Simulating 3 students voting on the app)
        mock_human_preferences = [21.0, 21.5, 22.0]  # They want it around 21.5°C
        print(f"Occupant Desires: {mock_human_preferences}")
        
        # 3. Run the Translator
        translator = FuzzySymbolicTranslator()
        agent_frame = translator.translate_payload(sample_row, mock_human_preferences)
        
        print("\n[Translated AI Agent Semantic Frame]")
        print(json.dumps(agent_frame, indent=2))

    except FileNotFoundError:
        print("ERROR: cleaned_classrooms.csv not found. Did you run data_cleaner.py?")
