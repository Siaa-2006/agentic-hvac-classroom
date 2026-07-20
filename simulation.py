import json
import os
import sys
import time

import pandas as pd

from translator import FuzzySymbolicTranslator

try:
    from hvac_agents import AgentState, hvac_app
except ImportError:
    print(
        "Could not import hvac_app or AgentState from hvac_agents.py."
    )
    sys.exit(1)


# =========================================================
# RULE-BASED BASELINE
# =========================================================

def run_baseline_controller(
    room_state: dict,
    window_preferences: list[float],
    interior_preferences: list[float],
) -> dict:
    window_target = (
        sum(window_preferences) / len(window_preferences)
        if window_preferences
        else 22.0
    )
    interior_target = (
        sum(interior_preferences) / len(interior_preferences)
        if interior_preferences
        else 22.0
    )

    average_error = (
        (room_state["w_temp"] - window_target)
        + (room_state["i_temp"] - interior_target)
    ) / 2.0

    hvac_mode = "COOL" if average_error > 0.5 else "OFF"

    if hvac_mode == "COOL":
        window_damper = max(
            10,
            min(
                100,
                int((room_state["w_temp"] - window_target) * 35),
            ),
        )
        interior_damper = max(
            10,
            min(
                100,
                int((room_state["i_temp"] - interior_target) * 35),
            ),
        )
    else:
        window_damper = 10
        interior_damper = 10

    maximum_co2 = max(
        room_state["w_co2"],
        room_state["i_co2"],
    )

    if maximum_co2 > 1200:
        fan_speed = "HIGH"
    elif maximum_co2 > 950:
        fan_speed = "MED"
    elif maximum_co2 > 650:
        fan_speed = "LOW"
    else:
        fan_speed = "OFF"

    return {
        "hvac_mode": hvac_mode,
        "window_damper_pct": window_damper,
        "interior_damper_pct": interior_damper,
        "fan_speed": fan_speed,
    }


# =========================================================
# HEURISTIC CLASSROOM UPDATE MODEL
# =========================================================

def update_classroom_physics(
    current_state: dict,
    control: dict,
    occupancy: int,
    ambient_temp: float = 32.5,
) -> dict:
    window_temp = current_state["w_temp"]
    window_co2 = current_state["w_co2"]
    interior_temp = current_state["i_temp"]
    interior_co2 = current_state["i_co2"]

    mode = control["hvac_mode"]
    window_damper = control["window_damper_pct"] / 100.0
    interior_damper = control["interior_damper_pct"] / 100.0

    fan_flow_map = {
        "OFF": 0.0,
        "LOW": 0.15,
        "MED": 0.35,
        "HIGH": 0.60,
    }
    fan_flow = fan_flow_map.get(control["fan_speed"], 0.0)

    window_occupancy = int(occupancy * 0.4)
    interior_occupancy = occupancy - window_occupancy

    window_temp_next = (
        window_temp
        + 0.52
        + window_occupancy * 0.012
        + 0.06 * (ambient_temp - window_temp)
        - (
            window_damper * 1.42
            if mode == "COOL"
            else 0.0
        )
    )

    window_co2_next = max(
        400.0,
        window_co2
        + 40.0
        + window_occupancy * 13.5
        - window_damper
        * fan_flow
        * (window_co2 - 400.0),
    )

    interior_temp_next = (
        interior_temp
        + 0.05
        + interior_occupancy * 0.015
        + 0.04 * (26.0 - interior_temp)
        - (
            interior_damper * 1.42
            if mode == "COOL"
            else 0.0
        )
    )

    interior_co2_next = max(
        400.0,
        interior_co2
        + 40.0
        + interior_occupancy * 16.0
        - interior_damper
        * fan_flow
        * (interior_co2 - 400.0),
    )

    return {
        "w_temp": round(window_temp_next, 2),
        "w_co2": round(window_co2_next, 1),
        "i_temp": round(interior_temp_next, 2),
        "i_co2": round(interior_co2_next, 1),
    }


def create_agent_state(
    step: int,
    phase: str,
    occupancy: int,
    ambient_temp_c: float,
    window_translated: dict,
    interior_translated: dict,
) -> AgentState:
    return AgentState(
        scenario_id=f"step_{step}",
        description=phase,
        occupancy=int(occupancy),
        ambient_temp_c=float(ambient_temp_c),
        window_zone=window_translated,
        interior_zone=interior_translated,
        messages=[],
        final_decision="",
        control_signals={},
        decision_source="",
        initial_supervisor_output="",
        raw_supervisor_output="",
        schema_retry_output="",
        constraint_retry_output="",
        validation_error="",
        constraint_metadata={},
        schema_retry_used=False,
        constraint_retry_used=False,
        first_pass_constraint_compliant=False,
        first_pass_constraint_violations=[],
        post_retry_constraint_violations=[],
        rule_adjustments=[],
    )


# =========================================================
# MAIN EXPERIMENT
# =========================================================

def run_simulation_campaign(steps: int = 12) -> None:
    if not 1 <= steps <= 12:
        raise ValueError("steps must be between 1 and 12.")

    if not any(
        os.environ.get(key)
        for key in [
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
        ]
    ):
        print("No LLM API key is configured.")
        sys.exit(1)

    try:
        classroom_data = pd.read_csv("cleaned_classrooms.csv")
    except FileNotFoundError:
        print("cleaned_classrooms.csv was not found.")
        sys.exit(1)

    if "temperature" not in classroom_data.columns:
        print(
            "cleaned_classrooms.csv has no 'temperature' column."
        )
        sys.exit(1)

    if len(classroom_data) <= 1212:
        print("cleaned_classrooms.csv needs at least 1213 rows.")
        sys.exit(1)

    initial_state = {
        "w_temp": 24.5,
        "w_co2": 450.0,
        "i_temp": 23.5,
        "i_co2": 480.0,
    }

    schedule = {
        1: (40, "Lecture 1 Begins (High Density)"),
        2: (40, "Lecture 1 Ongoing"),
        3: (5, "Mid-Morning Break"),
        4: (25, "Lecture 2 Begins (Medium Density)"),
        5: (25, "Lecture 2 Ongoing"),
        6: (0, "Lunch Break (Vacant)"),
        7: (45, "Lecture 3 Begins (Peak Density)"),
        8: (45, "Lecture 3 Ongoing"),
        9: (10, "Short Recess"),
        10: (20, "Lab Session"),
        11: (20, "Lab Session Ongoing"),
        12: (0, "Class Dismissed"),
    }

    window_preferences = [22.0, 21.5, 22.0]
    interior_preferences = [24.0, 24.5, 23.5]

    translator = FuzzySymbolicTranslator()
    experiment_logs: list[dict] = []
    reasoning_logs: list[dict] = []

    # =====================================================
    # AGENTIC CONTROLLER
    # =====================================================

    print("RUN A: Constraint-Guided Agentic Controller")
    agent_state = initial_state.copy()

    for step in range(1, steps + 1):
        occupancy, phase = schedule[step]
        print(f"Agentic step {step}/{steps}: {phase}")

        window_translated = translator.translate_payload(
            {
                "temperature": agent_state["w_temp"],
                "co2": agent_state["w_co2"],
                "humidity": 55.0,
            },
            window_preferences,
        )
        interior_translated = translator.translate_payload(
            {
                "temperature": agent_state["i_temp"],
                "co2": agent_state["i_co2"],
                "humidity": 55.0,
            },
            interior_preferences,
        )

        ambient_temperature = float(
            classroom_data.iloc[1200 + step]["temperature"]
        )

        result = hvac_app.invoke(
            create_agent_state(
                step=step,
                phase=phase,
                occupancy=occupancy,
                ambient_temp_c=ambient_temperature,
                window_translated=window_translated,
                interior_translated=interior_translated,
            )
        )
        signals = result["control_signals"]

        experiment_logs.append(
            {
                "step": step,
                "controller": "Agentic_LangGraph",
                "phase": phase,
                "occupancy": occupancy,
                "ambient_temp": ambient_temperature,
                "w_temp": agent_state["w_temp"],
                "w_co2": agent_state["w_co2"],
                "i_temp": agent_state["i_temp"],
                "i_co2": agent_state["i_co2"],
                "hvac_mode": signals["hvac_mode"],
                "vav_window": signals["window_damper_pct"],
                "vav_interior": signals["interior_damper_pct"],
                "vent_fan": signals["fan_speed"],
                "decision_source": result["decision_source"],
            }
        )

        reasoning_logs.append(
            {
                "step": step,
                "phase": phase,
                "occupancy": occupancy,
                "window_agent": result["messages"][0],
                "interior_agent": result["messages"][1],
                "constraint_metadata": json.dumps(
                    result.get("constraint_metadata", {}),
                    sort_keys=True,
                ),
                "recommended_hvac_mode": result.get(
                    "constraint_metadata",
                    {},
                ).get("recommended_hvac_mode", ""),
                "recommended_window_damper_pct": result.get(
                    "constraint_metadata",
                    {},
                ).get("recommended_damper_pct", {}).get(
                    "window",
                    "",
                ),
                "recommended_interior_damper_pct": result.get(
                    "constraint_metadata",
                    {},
                ).get("recommended_damper_pct", {}).get(
                    "interior",
                    "",
                ),
                "initial_supervisor_output": result.get(
                    "initial_supervisor_output",
                    "",
                ),
                "schema_retry_output": result.get(
                    "schema_retry_output",
                    "",
                ),
                "constraint_retry_output": result.get(
                    "constraint_retry_output",
                    "",
                ),
                "raw_supervisor_output": result.get(
                    "raw_supervisor_output",
                    "",
                ),
                "supervisor_reasoning": result["final_decision"],
                "hvac_mode": signals["hvac_mode"],
                "window_damper_pct": signals[
                    "window_damper_pct"
                ],
                "interior_damper_pct": signals[
                    "interior_damper_pct"
                ],
                "fan_speed": signals["fan_speed"],
                "decision_source": result["decision_source"],
                "schema_retry_used": result.get(
                    "schema_retry_used",
                    False,
                ),
                "constraint_retry_used": result.get(
                    "constraint_retry_used",
                    False,
                ),
                "first_pass_constraint_compliant": result.get(
                    "first_pass_constraint_compliant",
                    False,
                ),
                "first_pass_constraint_violations": " | ".join(
                    result.get(
                        "first_pass_constraint_violations",
                        [],
                    )
                ),
                "post_retry_constraint_violations": " | ".join(
                    result.get(
                        "post_retry_constraint_violations",
                        [],
                    )
                ),
                "rule_adjustments": " | ".join(
                    result.get("rule_adjustments", [])
                ),
                "validation_error": result.get(
                    "validation_error",
                    "",
                ),
            }
        )

        agent_state = update_classroom_physics(
            current_state=agent_state,
            control=signals,
            occupancy=occupancy,
            ambient_temp=ambient_temperature,
        )

        time.sleep(1.0)

    # =====================================================
    # RULE-BASED BASELINE
    # =====================================================

    print("RUN B: Rule-Based Baseline")
    baseline_state = initial_state.copy()

    for step in range(1, steps + 1):
        occupancy, phase = schedule[step]

        signals = run_baseline_controller(
            room_state=baseline_state,
            window_preferences=window_preferences,
            interior_preferences=interior_preferences,
        )

        ambient_temperature = float(
            classroom_data.iloc[1200 + step]["temperature"]
        )

        experiment_logs.append(
            {
                "step": step,
                "controller": "Baseline_Rule_Based",
                "phase": phase,
                "occupancy": occupancy,
                "ambient_temp": ambient_temperature,
                "w_temp": baseline_state["w_temp"],
                "w_co2": baseline_state["w_co2"],
                "i_temp": baseline_state["i_temp"],
                "i_co2": baseline_state["i_co2"],
                "hvac_mode": signals["hvac_mode"],
                "vav_window": signals["window_damper_pct"],
                "vav_interior": signals["interior_damper_pct"],
                "vent_fan": signals["fan_speed"],
                "decision_source": "direct_rule_based",
            }
        )

        baseline_state = update_classroom_physics(
            current_state=baseline_state,
            control=signals,
            occupancy=occupancy,
            ambient_temp=ambient_temperature,
        )

    simulation_dataframe = pd.DataFrame(experiment_logs)
    reasoning_dataframe = pd.DataFrame(reasoning_logs)

    simulation_dataframe.to_csv(
        "simulation_results.csv",
        index=False,
    )
    reasoning_dataframe.to_csv(
        "reasoning_audit.csv",
        index=False,
    )

    source_counts = reasoning_dataframe[
        "decision_source"
    ].value_counts()
    first_pass_count = int(
        reasoning_dataframe[
            "first_pass_constraint_compliant"
        ].sum()
    )
    schema_retry_count = int(
        reasoning_dataframe["schema_retry_used"].sum()
    )
    constraint_retry_count = int(
        reasoning_dataframe["constraint_retry_used"].sum()
    )
    final_rule_count = int(
        reasoning_dataframe["rule_adjustments"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )

    print("\nSimulation completed.")
    print(
        f"simulation_results.csv: "
        f"{len(simulation_dataframe)} rows"
    )
    print(
        f"reasoning_audit.csv: "
        f"{len(reasoning_dataframe)} rows"
    )
    print("\nAgentic decision sources:")
    print(source_counts.to_string())
    print(
        f"\nFirst-pass constraint compliant: "
        f"{first_pass_count}/{steps}"
    )
    print(f"Schema retries used: {schema_retry_count}/{steps}")
    print(
        f"Constraint retries used: "
        f"{constraint_retry_count}/{steps}"
    )
    print(
        f"Final deterministic modifications: "
        f"{final_rule_count}/{steps}"
    )


if __name__ == "__main__":
    run_simulation_campaign()
