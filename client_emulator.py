import os
import sys
import time
import random
import requests
import pandas as pd

# --- Configuration ---
API_URL = "https://agentic-hvac-classroom.onrender.com/api/v1/telemetry"
CSV_FILE = "cleaned_classrooms.csv"

def print_dashboard(step, raw_telemetry, response_data):
    """Prints a beautiful, clean CLI dashboard representing the virtual IoT device."""
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print(f"📡 VIRTUAL ESP32 IoT NODE EMULATOR | STEP {step}")
    print("=" * 70)
    
    # 1. Display Outgoing Telemetry
    print("\n📤 [1. SENDING TELEMETRY TO CLOUD API]")
    print(f"   Target URL:  {API_URL}")
    print(f"   Window Zone:   Temp: {raw_telemetry['window_zone']['air_temp_c']}°C | CO2: {raw_telemetry['window_zone']['co2_ppm']} ppm")
    print(f"                  Votes: {raw_telemetry['window_zone']['occupant_preferences']}")
    print(f"   Interior Zone: Temp: {raw_telemetry['interior_zone']['air_temp_c']}°C | CO2: {raw_telemetry['interior_zone']['co2_ppm']} ppm")
    print(f"                  Votes: {raw_telemetry['interior_zone']['occupant_preferences']}")

    # 2. Display Status & AI Transcripts
    print("\n🧠 [2. CLOUD AGENTIC TRANSCRIPTS]")
    print(f"   API Status:  {response_data.get('status', 'ERROR').upper()}")
    print("   Debate Log:")
    for msg in response_data.get("expert_messages", []):
         print(f"     🗣️  {msg}")
    print(f"   Supervisor Decision:\n     \"{response_data.get('supervisor_reasoning', '')}\"")

    # 3. Display Incoming Actuator Setpoints (What the physical relays/servos would do)
    signals = response_data.get("control_signals", {})
    print("\n⚙️  [3. MECHANICAL ACTUATORS EXECUTING AT EDGE]")
    print(f"   Central HVAC Mode:   [{signals.get('hvac_mode', 'OFF')}]")
    print(f"   Window VAV Damper:   [{signals.get('window_damper_pct', 0)}%] " + "▓" * (signals.get('window_damper_pct', 0) // 10))
    print(f"   Interior VAV Damper: [{signals.get('interior_damper_pct', 0)}%] " + "▓" * (signals.get('interior_damper_pct', 0) // 10))
    
    fan_speed = signals.get('fan_speed', 'OFF')
    fan_icons = {"OFF": "❌ OFF", "LOW": "🌀 LOW (15% Flow)", "MED": "🌀🌀 MEDIUM (35% Flow)", "HIGH": "🌀🌀🌀 HIGH (60% Flow)"}
    print(f"   Ventilation Fan:     [{fan_icons.get(fan_speed, fan_speed)}]")
    print("\n" + "=" * 70)


def run_emulator(total_intervals=5):
    # 1. Verify dataset presence
    if not os.path.exists(CSV_FILE):
        print(f"❌ Error: {CSV_FILE} not found. Please run your data_cleaner.py first.")
        sys.exit(1)

    print(f"Loading environment telemetry stream from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    
    # Let's pick a starting index in the afternoon where the classroom gets busy (e.g. index 2400)
    start_index = 2400
    
    print("\nInitiating continuous edge-to-gateway telemetry stream.")
    print("Make sure your FastAPI server is running in another terminal tab! (python app.py)")
    print("Waiting 3 seconds to start...\n")
    time.sleep(3)

    for step in range(1, total_intervals + 1):
        # Fetch consecutive rows from the CSV to simulate a real continuous timeline
        row_idx = start_index + (step * 2)
        if row_idx >= len(df):
             row_idx = start_index # Wrap around if limit exceeded
             
        row_data = df.iloc[row_idx].to_dict()
        
        # Build raw spatial offsets for our multi-zone inputs
        w_temp = round(row_data["temperature"] + 1.2, 2) # sun-drenched window zone
        w_co2 = round(row_data["co2"] - 60.0, 1)
        i_temp = round(row_data["temperature"] - 0.4, 2) # cooler, darker interior zone
        i_co2 = round(row_data["co2"] + 120.0, 1)

        # Mock occupant voting inputs via their phone app
        w_votes = [21.5, 22.0, 21.0] # Window occupants want cooling
        i_votes = [23.5, 24.0, 24.5] # Interior occupants want normal room temp
        
        # 2. Package into the strict JSON schema expected by app.py FastAPI
        payload = {
            "scenario_id": f"emu_step_{step}",
            "description": f"Live automated telemetry stream from physical classroom. Step {step}.",
            "window_zone": {
                "air_temp_c": w_temp,
                "relative_humidity": 50.0,
                "co2_ppm": w_co2,
                "occupant_preferences": w_votes
            },
            "interior_zone": {
                "air_temp_c": i_temp,
                "relative_humidity": 52.0,
                "co2_ppm": i_co2,
                "occupant_preferences": i_votes
            }
        }

        # 3. HTTP POST requests sent to FastAPI Server
        try:
            response = requests.post(API_URL, json=payload, timeout=15)
            if response.status_code == 200:
                print_dashboard(step, payload, response.json())
            else:
                print(f"⚠️ [HTTP Error {response.status_code}] server failed to process telemetry.")
                print(response.text)
        except requests.exceptions.ConnectionError:
            print("\n❌ Error: Cannot connect to the API Gateway.")
            print(f"Ensure that your local FastAPI server is active and listening at: {API_URL}")
            print("To start it, open a separate terminal tab and run: python app.py\n")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error encountered: {str(e)}")
            sys.exit(1)

        # Pacing wait to preserve Groq free-tier rate limits
        time.sleep(5.0)

    print("\n🎉 Emulator run complete! API gateway communication verified successfully.")


if __name__ == "__main__":
    # Let's run a 3-step continuous edge network simulation
    run_emulator(total_intervals=3)
