import os
import sys
import time
import re
import pandas as pd
from translator import FuzzySymbolicTranslator

try:
    from hvac_agents import hvac_app, AgentState
except ImportError:
    print("❌ Error: Could not import hvac_app or AgentState from hvac_agents.py.")
    sys.exit(1)

def parse_supervisor_decision(decision_text: str):
    text_lower = decision_text.lower()
    hvac_mode = "OFF"
    if "cool" in text_lower or "chilling" in text_lower:
        hvac_mode = "COOL"
    elif "heat" in text_lower or "warm" in text_lower:
        hvac_mode = "HEAT"

    clauses = re.split(r'[,.;]|\band\b|\bwhile\b|\bbut\b', text_lower)
    w_damper = 50  
    i_damper = 50  
    
    for clause in clauses:
        pct_match = re.search(r'(\d+)\s*%', clause)
        if pct_match:
            pct_val = int(pct_match.group(1))
            has_window = any(kw in clause for kw in ["window", "perimeter"])
            has_interior = any(kw in clause for kw in ["interior", "desk", "hallway"])
            
            if has_window and has_interior:
                w_pos = clause.find("window") if "window" in clause else clause.find("perimeter")
                i_pos = clause.find("interior") if "interior" in clause else clause.find("desk")
                pct_pos = pct_match.start()
                if abs(w_pos - pct_pos) < abs(i_pos - pct_pos):
                    w_damper = pct_val
                else:
                    i_damper = pct_val
            elif has_window:
                w_damper = pct_val
            elif has_interior:
                i_damper = pct_val

    fan_speed = "OFF"
    if re.search(r'\bhigh\b', text_lower):
        fan_speed = "HIGH"
    elif re.search(r'\b(?:medium|med)\b', text_lower):
        fan_speed = "MED"
    elif re.search(r'\blow\b', text_lower):
        fan_speed = "LOW"
    elif re.search(r'\boff\b', text_lower):
        fan_speed = "OFF"
    elif "ventilation" in text_lower or "fan" in text_lower:
        fan_speed = "LOW"

    return {
        "hvac_mode": hvac_mode,
        "window_damper_pct": max(0, min(100, w_damper)),
        "interior_damper_pct": max(0, min(100, i_damper)),
        "fan_speed": fan_speed
    }

def run_baseline_controller(room_state: dict, w_pref: list, i_pref: list):
    w_target = sum(w_pref) / len(w_pref) if w_pref else 22.0
    i_target = sum(i_pref) / len(i_pref) if i_pref else 22.0

    avg_error = ((room_state["w_temp"] - w_target) + (room_state["i_temp"] - i_target)) / 2.0
    hvac_mode = "COOL" if avg_error > 0.5 else "OFF"

    w_damper = max(10, min(100, int((room_state["w_temp"] - w_target) * 35))) if hvac_mode == "COOL" else 10
    i_damper = max(10, min(100, int((room_state["i_temp"] - i_target) * 35))) if hvac_mode == "COOL" else 10

    max_co2 = max(room_state["w_co2"], room_state["i_co2"])
    if max_co2 > 1200:
        fan_speed = "HIGH"
    elif max_co2 > 950:
        fan_speed = "MED"
    elif max_co2 > 650:
        fan_speed = "LOW"
    else:
        fan_speed = "OFF"

    return {
        "hvac_mode": hvac_mode,
        "window_damper_pct": w_damper,
        "interior_damper_pct": i_damper,
        "fan_speed": fan_speed
    }

def update_classroom_physics(current_state: dict, control: dict, occupancy: int, ambient_temp: float = 32.5):
    w_temp = current_state["w_temp"]
    w_co2 = current_state["w_co2"]
    i_temp = current_state["i_temp"]
    i_co2 = current_state["i_co2"]
    
    mode = control["hvac_mode"]
    w_damp = control["window_damper_pct"] / 100.0
    i_damp = control["interior_damper_pct"] / 100.0
    fan = control["fan_speed"]

    fan_flow_map = {"OFF": 0.0, "LOW": 0.15, "MED": 0.35, "HIGH": 0.60}
    fan_flow = fan_flow_map.get(fan, 0.0)

    occ_window = int(occupancy * 0.4)
    occ_interior = occupancy - occ_window

    # Window Zone Thermal Updates
    w_solar_gain = 0.52  
    w_metabolic_gain = occ_window * 0.012  
    w_thermal_exchange = 0.06 * (ambient_temp - w_temp)
    w_cooling = w_damp * 1.42 if mode == "COOL" else 0.0
    w_temp_next = w_temp + w_solar_gain + w_metabolic_gain + w_thermal_exchange - w_cooling

    # Window Zone Air Volume CO2
    w_co2_generation = 40.0 + (occ_window * 13.5)
    w_co2_dilution = w_damp * fan_flow * (w_co2 - 400.0)
    w_co2_next = max(400.0, w_co2 + w_co2_generation - w_co2_dilution)

    # Interior Zone Thermal Updates
    i_solar_gain = 0.05
    i_metabolic_gain = occ_interior * 0.015
    i_thermal_exchange = 0.04 * (26.0 - i_temp)
    i_cooling = i_damp * 1.42 if mode == "COOL" else 0.0
    i_temp_next = i_temp + i_solar_gain + i_metabolic_gain + i_thermal_exchange - i_cooling

    # Interior Zone Air Volume CO2
    i_co2_generation = 40.0 + (occ_interior * 16.0)
    i_co2_dilution = i_damp * fan_flow * (i_co2 - 400.0)
    i_co2_next = max(400.0, i_co2 + i_co2_generation - i_co2_dilution)

    return {
        "w_temp": round(w_temp_next, 2),
        "w_co2": round(w_co2_next, 1),
        "i_temp": round(i_temp_next, 2),
        "i_co2": round(i_co2_next, 1)
    }

def run_simulation_campaign(steps=12):
    print("================================================================")
    print("🌅 STARTING DYNAMIC 12-HOUR BENCHMARK CO-SIMULATION")
    print("================================================================")

    try:
        df = pd.read_csv("cleaned_classrooms.csv")
        print(f"Successfully loaded {len(df)} rows of Bangladesh classroom telemetry.")
    except FileNotFoundError:
        print("❌ Error: cleaned_classrooms.csv not found. Run data_cleaner.py first.")
        return

    start_state = {
        "w_temp": 24.5,   
        "w_co2": 450.0,   
        "i_temp": 23.5,
        "i_co2": 480.0
    }

    schedule = {
        1:  (40, "Lecture 1 Begins (High Density)"),
        2:  (40, "Lecture 1 Ongoing"),
        3:  (5,  "10:30 AM Mid-morning Break"),
        4:  (25, "Lecture 2 Begins (Medium Density)"),
        5:  (25, "Lecture 2 Ongoing"),
        6:  (0,  "12:30 PM Lunch Break (Vacant)"),
        7:  (45, "Lecture 3 Begins (Peak Class Density)"),
        8:  (45, "Lecture 3 Ongoing"),
        9:  (10, "3:00 PM Short Recess"),
        10: (20, "Lab Session (Medium Density)"),
        11: (20, "Lab Session Ongoing"),
        12: (0,  "5:00 PM Class Dismissed")
    }

    w_pref = [22.0, 21.5, 22.0]  
    i_pref = [24.0, 24.5, 23.5]  

    experiment_logs = []
    agent_state = start_state.copy()
    translator = FuzzySymbolicTranslator()

    print("\n[Executing Simulation Run A: Fuzzy-Symbolic Agentic Control (Groq/Llama)]")
    for step in range(1, steps + 1):
        occ, phase = schedule[step]
        print(f"  Step {step}/12 - {phase} | Students: {occ} ...")
        
        w_trans = translator.translate_payload({"temperature": agent_state["w_temp"], "co2": agent_state["w_co2"], "humidity": 55.0}, w_pref)
        i_trans = translator.translate_payload({"temperature": agent_state["i_temp"], "co2": agent_state["i_co2"], "humidity": 55.0}, i_pref)

        payload = AgentState(
            scenario_id=f"step_{step}",
            description=phase,
            window_zone=w_trans,
            interior_zone=i_trans,
            messages=[],
            final_decision=""
        )

        result = hvac_app.invoke(payload)
        signals = parse_supervisor_decision(result["final_decision"])

        ambient = df.iloc[1200 + step]["temperature"]
        next_state = update_classroom_physics(agent_state, signals, occupancy=occ, ambient_temp=ambient)

        experiment_logs.append({
            "step": step, "controller": "Agentic_LangGraph", "occupancy": occ,
            "w_temp": agent_state["w_temp"], "w_co2": agent_state["w_co2"],
            "i_temp": agent_state["i_temp"], "i_co2": agent_state["i_co2"],
            "hvac_mode": signals["hvac_mode"], "vav_window": signals["window_damper_pct"],
            "vav_interior": signals["interior_damper_pct"], "vent_fan": signals["fan_speed"]
        })
        agent_state = next_state
        time.sleep(2.0)  

    print("\n[Executing Simulation Run B: Hysteresis Controller Baseline]")
    base_state = start_state.copy()
    for step in range(1, steps + 1):
        occ, phase = schedule[step]
        print(f"  Step {step}/12 - {phase} | Students: {occ} ...")
        
        signals = run_baseline_controller(base_state, w_pref, i_pref)

        ambient = df.iloc[1200 + step]["temperature"]
        next_state = update_classroom_physics(base_state, signals, occupancy=occ, ambient_temp=ambient)

        experiment_logs.append({
            "step": step, "controller": "Baseline_Rule_Based", "occupancy": occ,
            "w_temp": base_state["w_temp"], "w_co2": base_state["w_co2"],
            "i_temp": base_state["i_temp"], "i_co2": base_state["i_co2"],
            "hvac_mode": signals["hvac_mode"], "vav_window": signals["window_damper_pct"],
            "vav_interior": signals["interior_damper_pct"], "vent_fan": signals["fan_speed"]
        })
        base_state = next_state

    pd.DataFrame(experiment_logs).to_csv("simulation_results.csv", index=False)
    print("\n✅ Simulation cycle completed. Compiled telemetry exported to 'simulation_results.csv'!")

if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️ ERROR: No GROQ_API_KEY environment variable detected.")
        sys.exit(1)
    run_simulation_campaign()
