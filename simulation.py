import os
import sys
import time
import re
import pandas as pd
from translator import FuzzySymbolicTranslator

# Import the compiled LangGraph workflow from your hvac_agents script
try:
    from hvac_agents import hvac_app, AgentState
except ImportError:
    print("❌ Error: Could not import hvac_app or AgentState from hvac_agents.py.")
    print("Ensure hvac_agents.py is in the same folder and compiles successfully.")
    sys.exit(1)


# --- 1. Robust Natural Language Controller Parser ---
def parse_supervisor_decision(decision_text: str):
    """
    Parses the Supervisor's raw English paragraph to extract numerical control signals.
    Employs clause-based segmentation and strict word boundaries to eliminate matching overlaps.
    """
    text_lower = decision_text.lower()
    
    # 1. Extract Central HVAC Mode
    hvac_mode = "OFF"
    if "cool" in text_lower or "chilling" in text_lower:
        hvac_mode = "COOL"
    elif "heat" in text_lower or "warm" in text_lower:
        hvac_mode = "HEAT"

    # 2. Extract Zone VAV Damper opening percentages (0 to 100%)
    clauses = re.split(r'[,.;]|\band\b|\bwhile\b|\bbut\b', text_lower)
    
    w_damper = 50  # Default middle-ground fallback
    i_damper = 50  # Default middle-ground fallback
    
    for clause in clauses:
        pct_match = re.search(r'(\d+)\s*%', clause)
        if pct_match:
            pct_val = int(pct_match.group(1))
            
            # Identify which zone the current clause primarily discusses
            has_window = any(kw in clause for kw in ["window", "perimeter"])
            has_interior = any(kw in clause for kw in ["interior", "desk", "hallway"])
            
            if has_window and has_interior:
                # Find which keyword is closer to the percentage
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

    # 3. Extract Central Ventilation Fan Speed with Strict Word Boundaries (\b)
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

    # Constrain dampers to physical hardware limits (0 - 100%)
    w_damper = max(0, min(100, w_damper))
    i_damper = max(0, min(100, i_damper))

    return {
        "hvac_mode": hvac_mode,
        "window_damper_pct": w_damper,
        "interior_damper_pct": i_damper,
        "fan_speed": fan_speed
    }


# --- 2. Traditional Rule-Based Baseline Controller ---
def run_baseline_controller(room_state: dict, w_pref: list, i_pref: list):
    """
    Standard industrial hysteresis rule-based HVAC controller.
    Used as an experimental baseline control benchmark.
    """
    w_target = sum(w_pref) / len(w_pref) if w_pref else 22.0
    i_target = sum(i_pref) / len(i_pref) if i_pref else 22.0

    # HVAC mode selection based on average error
    avg_error = ((room_state["w_temp"] - w_target) + (room_state["i_temp"] - i_target)) / 2.0
    hvac_mode = "COOL" if avg_error > 0.5 else "OFF"

    # VAV damper openings proportional to temperature offset (min 10% to ensure ventilation bypass)
    w_damper = max(10, min(100, int((room_state["w_temp"] - w_target) * 30))) if hvac_mode == "COOL" else 10
    i_damper = max(10, min(100, int((room_state["i_temp"] - i_target) * 30))) if hvac_mode == "COOL" else 10

    # Ventilation fan speed proportional to highest CO2 concentration
    max_co2 = max(room_state["w_co2"], room_state["i_co2"])
    if max_co2 > 1500:
        fan_speed = "HIGH"
    elif max_co2 > 1000:
        fan_speed = "MED"
    elif max_co2 > 700:
        fan_speed = "LOW"
    else:
        fan_speed = "OFF"

    return {
        "hvac_mode": hvac_mode,
        "window_damper_pct": w_damper,
        "interior_damper_pct": i_damper,
        "fan_speed": fan_speed
    }


# --- 3. Dynamic Micro-Climate Physics Engine ---
def update_classroom_physics(current_state: dict, control: dict, ambient_temp: float = 32.5):
    """
    Simulates physical thermal changes and gas mass balances in the room for step t -> t+1.
    """
    w_temp = current_state["w_temp"]
    w_co2 = current_state["w_co2"]
    i_temp = current_state["i_temp"]
    i_co2 = current_state["i_co2"]
    
    mode = control["hvac_mode"]
    w_damp = control["window_damper_pct"] / 100.0
    i_damp = control["interior_damper_pct"] / 100.0
    fan = control["fan_speed"]

    # Map fan speeds to air-flow exchange multipliers
    fan_flow_map = {"OFF": 0.0, "LOW": 0.15, "MED": 0.35, "HIGH": 0.60}
    fan_flow = fan_flow_map.get(fan, 0.0)

    # --- Zone 1: Window Zone Physics ---
    w_solar_gain = 0.4  # Solar radiation multiplier
    w_thermal_loss_or_gain = 0.05 * (ambient_temp - w_temp)
    w_cooling = w_damp * 1.2 if mode == "COOL" else 0.0
    w_temp_next = w_temp + w_solar_gain + w_thermal_loss_or_gain - w_cooling

    # CO2 mass balance
    w_co2_generation = 120.0
    w_co2_dilution = w_damp * fan_flow * (w_co2 - 400.0)
    w_co2_next = max(400.0, w_co2 + w_co2_generation - w_co2_dilution)

    # --- Zone 2: Interior Zone Physics ---
    i_metabolic_gain = 0.15
    i_thermal_loss_or_gain = 0.05 * (26.0 - i_temp)
    i_cooling = i_damp * 1.2 if mode == "COOL" else 0.0
    i_temp_next = i_temp + i_metabolic_gain + i_thermal_loss_or_gain - i_cooling

    # CO2 mass balance
    i_co2_generation = 150.0
    i_co2_dilution = i_damp * fan_flow * (i_co2 - 400.0)
    i_co2_next = max(400.0, i_co2 + i_co2_generation - i_co2_dilution)

    return {
        "w_temp": round(w_temp_next, 2),
        "w_co2": round(w_co2_next, 1),
        "i_temp": round(i_temp_next, 2),
        "i_co2": round(i_co2_next, 1)
    }


# --- 4. Main Time-Series Execution Loop with Benchmarking ---
def run_timeseries_simulation(steps=3):
    print("================================================================")
    print("🚀 STARTING CO-SIMULATION COMPARATIVE BENCHMARK")
    print("================================================================")

    try:
        df = pd.read_csv("cleaned_classrooms.csv")
        print(f"Loaded {len(df)} historical rows of Bangladesh classroom data.")
    except FileNotFoundError:
        print("❌ Error: cleaned_classrooms.csv not found. Run data_cleaner.py first.")
        return

    # Initialize exact identical starting states for both systems
    init_row = df.iloc[500].to_dict()
    start_state = {
        "w_temp": init_row["temperature"] + 1.5,
        "w_co2": init_row["co2"] - 50.0,
        "i_temp": init_row["temperature"] - 0.5,
        "i_co2": init_row["co2"] + 100.0
    }

    window_preferences = [22.0, 21.5, 22.0]  # Target: 21.8°C
    interior_preferences = [24.0, 24.5, 23.5] # Target: 24.0°C

    # Execution logs for CSV export
    experiment_logs = []

    # Initialize tracking metrics
    agent_metrics = {"comfort_violations": 0.0, "co2_violations": 0.0, "estimated_energy": 0.0}
    base_metrics = {"comfort_violations": 0.0, "co2_violations": 0.0, "estimated_energy": 0.0}

    # ==================== RUN SIMULATION A: AGENTIC CONTROLLER ====================
    print("\n[Executing Simulation A: LangGraph Agentic Controller]")
    room_state_agent = start_state.copy()
    translator = FuzzySymbolicTranslator()

    for step in range(1, steps + 1):
        print(f"  -> Agent Control Step {step}/{steps}...")
        w_translated = translator.translate_payload({"temperature": room_state_agent["w_temp"], "co2": room_state_agent["w_co2"], "humidity": 55.0}, window_preferences)
        i_translated = translator.translate_payload({"temperature": room_state_agent["i_temp"], "co2": room_state_agent["i_co2"], "humidity": 55.0}, interior_preferences)

        state_payload = AgentState(
            scenario_id=f"sim_step_{step}",
            description="Continuous agentic time-series benchmark step.",
            window_zone=w_translated,
            interior_zone=i_translated,
            messages=[],
            final_decision=""
        )

        result = hvac_app.invoke(state_payload)
        signals = parse_supervisor_decision(result["final_decision"])

        # Track Metrics
        w_target = 21.8
        i_target = 24.0
        agent_metrics["comfort_violations"] += abs(room_state_agent["w_temp"] - w_target) + abs(room_state_agent["i_temp"] - i_target)
        agent_metrics["co2_violations"] += max(0, room_state_agent["w_co2"] - 1000) + max(0, room_state_agent["i_co2"] - 1000)
        
        # Energy proxy calculation: Mode Power + Damper openings + Fan speed status
        energy_step = (1.5 if signals["hvac_mode"] == "COOL" else 0.0) + (signals["window_damper_pct"] + signals["interior_damper_pct"])/100.0 + (1.0 if signals["fan_speed"] == "HIGH" else 0.5 if signals["fan_speed"] == "MED" else 0.2 if signals["fan_speed"] == "LOW" else 0.0)
        agent_metrics["estimated_energy"] += energy_step

        # Update environment state
        ambient_temp = df.iloc[500 + step].to_dict()["temperature"]
        next_state = update_classroom_physics(room_state_agent, signals, ambient_temp=ambient_temp)

        # Log Data Row
        experiment_logs.append({
            "step": step, "controller": "Agentic_LangGraph",
            "w_temp": room_state_agent["w_temp"], "w_co2": room_state_agent["w_co2"],
            "i_temp": room_state_agent["i_temp"], "i_co2": room_state_agent["i_co2"],
            "actuator_mode": signals["hvac_mode"], "vav_window": signals["window_damper_pct"],
            "vav_interior": signals["interior_damper_pct"], "vent_fan": signals["fan_speed"]
        })
        room_state_agent = next_state
        time.sleep(2.0) # Pacing

    # ==================== RUN SIMULATION B: RULE-BASED BASELINE ====================
    print("\n[Executing Simulation B: Traditional Rule-Based Baseline]")
    room_state_base = start_state.copy()

    for step in range(1, steps + 1):
        print(f"  -> Baseline Control Step {step}/{steps}...")
        signals = run_baseline_controller(room_state_base, window_preferences, interior_preferences)

        # Track Metrics
        w_target = 21.8
        i_target = 24.0
        base_metrics["comfort_violations"] += abs(room_state_base["w_temp"] - w_target) + abs(room_state_base["i_temp"] - i_target)
        base_metrics["co2_violations"] += max(0, room_state_base["w_co2"] - 1000) + max(0, room_state_base["i_co2"] - 1000)
        
        energy_step = (1.5 if signals["hvac_mode"] == "COOL" else 0.0) + (signals["window_damper_pct"] + signals["interior_damper_pct"])/100.0 + (1.0 if signals["fan_speed"] == "HIGH" else 0.5 if signals["fan_speed"] == "MED" else 0.2 if signals["fan_speed"] == "LOW" else 0.0)
        base_metrics["estimated_energy"] += energy_step

        # Update environment state
        ambient_temp = df.iloc[500 + step].to_dict()["temperature"]
        next_state = update_classroom_physics(room_state_base, signals, ambient_temp=ambient_temp)

        # Log Data Row
        experiment_logs.append({
            "step": step, "controller": "Baseline_Rule_Based",
            "w_temp": room_state_base["w_temp"], "w_co2": room_state_base["w_co2"],
            "i_temp": room_state_base["i_temp"], "i_co2": room_state_base["i_co2"],
            "actuator_mode": signals["hvac_mode"], "vav_window": signals["window_damper_pct"],
            "vav_interior": signals["interior_damper_pct"], "vent_fan": signals["fan_speed"]
        })
        room_state_base = next_state

    # ==================== PRINT EVALUATION ANALYSIS ====================
    print("\n" + "="*60)
    print("📊 COMPARATIVE PERFORMANCE SCORECARD (Lower is Better)")
    print("="*60)
    print(f"  Metric                     | Baseline (Rules) | Agentic (LangGraph) | Delta")
    print(f"  ---------------------------|------------------|---------------------|-------")
    
    cv_diff = agent_metrics['comfort_violations'] - base_metrics['comfort_violations']
    print(f"  Thermal Discomfort Index   | {base_metrics['comfort_violations']:.2f}             | {agent_metrics['comfort_violations']:.2f}              | {cv_diff:+.2f}")
    
    co2_diff = agent_metrics['co2_violations'] - base_metrics['co2_violations']
    print(f"  IAQ Violation Index (CO2)  | {base_metrics['co2_violations']:.1f}           | {agent_metrics['co2_violations']:.1f}            | {co2_diff:+.1f}")
    
    nrg_diff = agent_metrics['estimated_energy'] - base_metrics['estimated_energy']
    nrg_pct = (nrg_diff / base_metrics['estimated_energy']) * 100.0 if base_metrics['estimated_energy'] > 0 else 0
    print(f"  System Energy Consumption  | {base_metrics['estimated_energy']:.2f}             | {agent_metrics['estimated_energy']:.2f}              | {nrg_pct:+.1f}%")
    print("="*60)

    # Export to CSV
    res_df = pd.DataFrame(experiment_logs)
    res_df.to_csv("simulation_results.csv", index=False)
    print("\n✅ Successfully compiled comparative records and exported to 'simulation_results.csv'!")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY") and not os.environ.get("GEMINI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  ERROR: No active LLM provider found.")
        print("Please load your Groq API key in the terminal first:")
        print("export GROQ_API_KEY='your-key-here'\n")
        sys.exit(1)

    # Let's run a 3-step benchmark evaluation
    run_timeseries_simulation(steps=3)
