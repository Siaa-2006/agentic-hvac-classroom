import json
import os
import time
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from decision_output import (
    ControlDecisionError,
    apply_rule_based_command_checks,
    build_constraint_metadata,
    deterministic_fallback,
    find_constraint_violations,
    validate_control_decision,
)
from translator import FuzzySymbolicTranslator


class AgentState(TypedDict, total=False):
    scenario_id: str
    description: str
    occupancy: int
    ambient_temp_c: float

    window_zone: dict
    interior_zone: dict
    messages: List[str]

    final_decision: str
    control_signals: dict

    decision_source: str
    initial_supervisor_output: str
    raw_supervisor_output: str
    schema_retry_output: str
    constraint_retry_output: str
    validation_error: str

    constraint_metadata: dict
    schema_retry_used: bool
    constraint_retry_used: bool
    first_pass_constraint_compliant: bool
    first_pass_constraint_violations: List[str]
    post_retry_constraint_violations: List[str]
    rule_adjustments: List[str]


# =========================================================
# LLM SETUP
# =========================================================

llm = None
provider_name = ""


if os.environ.get("GROQ_API_KEY"):
    try:
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_retries=3,
        )
        provider_name = "Groq (Llama-3.3-70b)"

    except ImportError:
        print("Install Groq support using: pip install langchain-groq")


elif os.environ.get("GEMINI_API_KEY"):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,
            max_retries=0,
        )
        provider_name = "Google Gemini"

    except ImportError:
        print(
            "Install Gemini support using: "
            "pip install langchain-google-genai"
        )


elif os.environ.get("OPENAI_API_KEY"):
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.0,
            max_retries=0,
        )
        provider_name = "OpenAI"

    except ImportError:
        print(
            "Install OpenAI support using: "
            "pip install langchain-openai"
        )


def invoke_with_retry(
    active_llm,
    messages,
    max_retries: int = 5,
):
    time.sleep(1.0)
    delays = [2, 4, 8, 16, 32]

    for index, delay in enumerate(delays):
        try:
            return active_llm.invoke(messages)

        except Exception as error:
            error_text = str(error).lower()
            is_rate_limit = any(
                term in error_text
                for term in [
                    "429",
                    "resource_exhausted",
                    "rate_limit",
                    "quota",
                    "exhausted",
                ]
            )

            if is_rate_limit and index < max_retries - 1:
                print(
                    f"Rate limit reached. Retrying after "
                    f"{delay} seconds..."
                )
                time.sleep(delay)
                continue

            raise

    return active_llm.invoke(messages)


def append_error(current_error: str, new_error: str) -> str:
    if not current_error:
        return new_error
    return f"{current_error} | {new_error}"


# =========================================================
# WINDOW-ZONE AGENT
# =========================================================

def window_zone_expert(state: AgentState):
    if not llm:
        raise ValueError("No LLM provider configured.")

    semantic_data = state["window_zone"]["semantic_state"]
    metadata = state["window_zone"]["meta"]

    messages = [
        SystemMessage(
            content=(
                "You are the Window Zone Expert. Advocate only for "
                "the thermal comfort and air-quality needs of the "
                "window zone. Describe the thermal need and the "
                "ventilation need separately. Do not choose numerical "
                "damper percentages or a final classroom-level fan "
                "speed; the supervisor will coordinate those values."
            )
        ),
        HumanMessage(
            content=(
                "Window Zone Physical State:\n"
                f"Temperature: {metadata['actual_temp_c']} °C\n"
                f"CO2: {metadata['actual_co2_ppm']} ppm\n"
                f"Requested temperature: "
                f"{metadata['avg_requested_temp']} °C\n\n"
                f"Thermal condition: "
                f"{semantic_data['thermal_condition']}\n"
                f"Air purity: {semantic_data['air_purity']}\n"
                f"Occupant demand: "
                f"{semantic_data['group_comfort_sentiment']}\n\n"
                "Recommend the zone's thermal action and ventilation "
                "priority in two clear sentences."
            )
        ),
    ]

    response = invoke_with_retry(llm, messages)
    state.setdefault("messages", []).append(
        f"Window Zone Expert: {response.content}"
    )
    return state


# =========================================================
# INTERIOR-ZONE AGENT
# =========================================================

def interior_zone_expert(state: AgentState):
    if not llm:
        raise ValueError("No LLM provider configured.")

    semantic_data = state["interior_zone"]["semantic_state"]
    metadata = state["interior_zone"]["meta"]

    messages = [
        SystemMessage(
            content=(
                "You are the Interior Zone Expert. Advocate only for "
                "the thermal comfort and air-quality needs of the "
                "interior zone. Describe the thermal need and the "
                "ventilation need separately. Do not choose numerical "
                "damper percentages or a final classroom-level fan "
                "speed; the supervisor will coordinate those values."
            )
        ),
        HumanMessage(
            content=(
                "Interior Zone Physical State:\n"
                f"Temperature: {metadata['actual_temp_c']} °C\n"
                f"CO2: {metadata['actual_co2_ppm']} ppm\n"
                f"Requested temperature: "
                f"{metadata['avg_requested_temp']} °C\n\n"
                f"Thermal condition: "
                f"{semantic_data['thermal_condition']}\n"
                f"Air purity: {semantic_data['air_purity']}\n"
                f"Occupant demand: "
                f"{semantic_data['group_comfort_sentiment']}\n\n"
                "Recommend the zone's thermal action and ventilation "
                "priority in two clear sentences."
            )
        ),
    ]

    response = invoke_with_retry(llm, messages)
    state.setdefault("messages", []).append(
        f"Interior Zone Expert: {response.content}"
    )
    return state


# =========================================================
# SUPERVISOR JSON FORMAT
# =========================================================

JSON_OUTPUT_INSTRUCTION = """
Return ONLY one valid JSON object with exactly these fields:
- hvac_mode
- window_damper_pct
- interior_damper_pct
- fan_speed
- reasoning

Field rules:
1. hvac_mode must be one of the values supplied in allowed_hvac_modes.
2. window_damper_pct must be an integer inside the inclusive range
   supplied for the selected HVAC mode.
3. interior_damper_pct must be an integer inside the inclusive range
   supplied for the selected HVAC mode.
4. fan_speed must be one of the values supplied in allowed_fan_levels.
5. reasoning must briefly explain the conflict resolution.
6. When a damper range has min equal to max, use that exact value.
7. Do not invent values outside the supplied feasible ranges.
8. Do not include Markdown fences.
9. Do not include any text outside the JSON object.
""".strip()


def build_supervisor_prompt(
    window_metadata: dict,
    interior_metadata: dict,
    debate_log: str,
    constraint_metadata: dict,
) -> str:
    constraint_text = json.dumps(constraint_metadata, indent=2)

    return (
        "Classroom Spatial Telemetry:\n\n"
        "Window Zone:\n"
        f"- Temperature: {window_metadata['actual_temp_c']} °C\n"
        f"- CO2: {window_metadata['actual_co2_ppm']} ppm\n"
        f"- Requested temperature: "
        f"{window_metadata['avg_requested_temp']} °C\n\n"
        "Interior Zone:\n"
        f"- Temperature: {interior_metadata['actual_temp_c']} °C\n"
        f"- CO2: {interior_metadata['actual_co2_ppm']} ppm\n"
        f"- Requested temperature: "
        f"{interior_metadata['avg_requested_temp']} °C\n\n"
        f"Zone recommendations:\n{debate_log}\n\n"
        "Deterministic Feasible-Command Metadata:\n"
        f"{constraint_text}\n\n"
        "Mandatory selection procedure:\n"
        "1. Choose hvac_mode only from allowed_hvac_modes. HEAT is "
        "not allowed unless it appears in that list.\n"
        "2. After choosing the mode, use only that mode's damper "
        "ranges: cool_mode_damper_ranges for COOL or "
        "off_mode_damper_ranges for OFF.\n"
        "3. Every range is inclusive. A range [40, 40] means the "
        "only valid value is exactly 40.\n"
        "4. Select fan_speed only from allowed_fan_levels. Prefer "
        "required_minimum_fan unless a stronger fan is clearly "
        "needed.\n"
        "5. Satisfy required_damper_order exactly. The ranges have "
        "already been coupled so that this ordering is feasible.\n"
        "6. Use recommended_damper_pct for each zone by default. "
        "Do not automatically choose the minimum of a range.\n"
        "7. Deviate from a recommended damper only when the local "
        "zone recommendations justify it, and remain inside the "
        "narrow recommended-mode range.\n"
        "8. A recommendation of 100% is intentional when the "
        "performance model indicates a strong active demand; do not "
        "replace it with an arbitrary lower value.\n"
        "9. OFF is a ventilation-only state in this simulator: "
        "thermal cooling is disabled, while the fan and dampers may "
        "remain active.\n"
        "10. Before returning JSON, check the selected mode, both "
        "ranges, the recommended values, the damper order, and the "
        "fan level one final time."
        "\n\nResolve the conflict and produce one classroom-level "
        "command.\n\n"
        f"{JSON_OUTPUT_INSTRUCTION}"
    )


# =========================================================
# SUPERVISORY AGENT
# =========================================================

def multi_zone_supervisor(state: AgentState):
    if not llm:
        raise ValueError("No LLM provider configured.")

    debate_log = "\n".join(state.get("messages", []))
    window_metadata = state["window_zone"]["meta"]
    interior_metadata = state["interior_zone"]["meta"]
    occupancy = int(state.get("occupancy", 0))
    ambient_temp_c = float(
        state.get(
            "ambient_temp_c",
            window_metadata["actual_temp_c"],
        )
    )

    constraint_metadata = build_constraint_metadata(
        window_meta=window_metadata,
        interior_meta=interior_metadata,
        occupancy=occupancy,
        ambient_temp_c=ambient_temp_c,
    )
    constraint_text = json.dumps(constraint_metadata, indent=2)

    supervisor_messages = [
        SystemMessage(
            content=(
                "You are the Head Multi-Zone HVAC Supervisor. "
                "Combine the two local recommendations into one "
                "classroom-level command. The deterministic metadata "
                "contains a performance-aware recommended mode, an "
                "exact required fan level, recommended damper values, "
                "and narrow feasible damper ranges. Treat the mode, "
                "fan, and ranges as mandatory. Use the recommended "
                "damper values by default rather than automatically "
                "selecting the lower bounds. Before returning JSON, "
                "verify the mode, fan, both recommended values, both "
                "ranges, and the required damper order."
            )
        ),
        HumanMessage(
            content=build_supervisor_prompt(
                window_metadata=window_metadata,
                interior_metadata=interior_metadata,
                debate_log=debate_log,
                constraint_metadata=constraint_metadata,
            )
        ),
    ]

    first_response = invoke_with_retry(llm, supervisor_messages)
    initial_output = first_response.content
    raw_output = initial_output
    schema_retry_output = ""
    constraint_retry_output = ""
    validation_error = ""
    schema_retry_used = False
    constraint_retry_used = False

    try:
        decision = validate_control_decision(raw_output)
        decision_source = "llm_validated"

    except ControlDecisionError as first_error:
        schema_retry_used = True
        validation_error = str(first_error)

        corrective_messages = [
            SystemMessage(
                content=(
                    "Repair invalid HVAC control JSON. Preserve the "
                    "intended command where possible, obey all supplied "
                    "constraints, and return only the corrected JSON."
                )
            ),
            HumanMessage(
                content=(
                    "The previous supervisor response failed schema "
                    "validation.\n\n"
                    f"Invalid response:\n{raw_output}\n\n"
                    f"Validation error:\n{validation_error}\n\n"
                    f"Constraint metadata:\n{constraint_text}\n\n"
                    f"{JSON_OUTPUT_INSTRUCTION}"
                )
            ),
        ]

        retry_response = invoke_with_retry(llm, corrective_messages)
        schema_retry_output = retry_response.content
        raw_output = schema_retry_output

        try:
            decision = validate_control_decision(raw_output)
            decision_source = "llm_schema_retry"

        except ControlDecisionError as retry_error:
            validation_error = append_error(
                validation_error,
                f"Schema retry failed: {retry_error}",
            )
            decision = deterministic_fallback(
                window_meta=window_metadata,
                interior_meta=interior_metadata,
                occupancy=occupancy,
                ambient_temp_c=ambient_temp_c,
            )
            decision_source = "rule_based_fallback"

    first_pass_violations: list[str] = []

    if decision_source != "rule_based_fallback":
        first_pass_violations = find_constraint_violations(
            decision=decision,
            window_meta=window_metadata,
            interior_meta=interior_metadata,
            occupancy=occupancy,
            ambient_temp_c=ambient_temp_c,
        )

        if first_pass_violations:
            constraint_retry_used = True
            constraint_retry_messages = [
                SystemMessage(
                    content=(
                        "Repair the HVAC command by selecting values directly "
                        "from the supplied mode-specific feasible "
                        "ranges. When min equals max, use that exact "
                        "value. Return only one valid JSON object."
                    )
                ),
                HumanMessage(
                    content=(
                        "The command passed the JSON schema but violated "
                        "one or more mandatory control constraints.\n\n"
                        f"Current command:\n"
                        f"{json.dumps(decision, indent=2)}\n\n"
                        f"Constraint metadata:\n{constraint_text}\n\n"
                        "Detected violations:\n"
                        f"{json.dumps(first_pass_violations, indent=2)}"
                        "\n\nCorrect the command using the selected "
                        "mode's exact inclusive damper ranges. Do "
                        "not solve the minimum, maximum, and ordering "
                        "rules separately; the supplied ranges already "
                        "combine them. Use recommended_damper_pct by "
                        "default and stay inside the narrow ranges.\n\n"
                        f"{JSON_OUTPUT_INSTRUCTION}"
                    )
                ),
            ]

            semantic_response = invoke_with_retry(
                llm,
                constraint_retry_messages,
            )
            constraint_retry_output = semantic_response.content

            try:
                repaired_decision = validate_control_decision(
                    constraint_retry_output
                )
                decision = repaired_decision
                raw_output = constraint_retry_output
                decision_source = "llm_constraint_retry"

            except ControlDecisionError as semantic_error:
                validation_error = append_error(
                    validation_error,
                    f"Constraint retry produced invalid JSON: "
                    f"{semantic_error}",
                )

    post_retry_violations = find_constraint_violations(
        decision=decision,
        window_meta=window_metadata,
        interior_meta=interior_metadata,
        occupancy=occupancy,
        ambient_temp_c=ambient_temp_c,
    )

    decision, rule_adjustments = apply_rule_based_command_checks(
        decision=decision,
        window_meta=window_metadata,
        interior_meta=interior_metadata,
        occupancy=occupancy,
        ambient_temp_c=ambient_temp_c,
    )

    if rule_adjustments:
        decision_source = f"{decision_source}_rule_checked"

    state["final_decision"] = decision["reasoning"]
    state["control_signals"] = {
        "hvac_mode": decision["hvac_mode"],
        "window_damper_pct": decision["window_damper_pct"],
        "interior_damper_pct": decision["interior_damper_pct"],
        "fan_speed": decision["fan_speed"],
    }

    state["decision_source"] = decision_source
    state["initial_supervisor_output"] = initial_output
    state["raw_supervisor_output"] = raw_output
    state["schema_retry_output"] = schema_retry_output
    state["constraint_retry_output"] = constraint_retry_output
    state["validation_error"] = validation_error

    state["constraint_metadata"] = constraint_metadata
    state["schema_retry_used"] = schema_retry_used
    state["constraint_retry_used"] = constraint_retry_used
    state["first_pass_constraint_compliant"] = (
        not decision_source.startswith("rule_based_fallback")
        and not bool(first_pass_violations)
    )
    state["first_pass_constraint_violations"] = (
        first_pass_violations
    )
    state["post_retry_constraint_violations"] = (
        post_retry_violations
    )
    state["rule_adjustments"] = rule_adjustments

    return state


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(AgentState)
workflow.add_node("window_zone_expert", window_zone_expert)
workflow.add_node("interior_zone_expert", interior_zone_expert)
workflow.add_node("supervisor", multi_zone_supervisor)

workflow.set_entry_point("window_zone_expert")
workflow.add_edge("window_zone_expert", "interior_zone_expert")
workflow.add_edge("interior_zone_expert", "supervisor")
workflow.add_edge("supervisor", END)

hvac_app = workflow.compile()


def map_json_telemetry_to_translator(telemetry: dict) -> dict:
    return {
        "temperature": telemetry["air_temp_c"],
        "co2": telemetry["co2_ppm"],
        "humidity": telemetry["relative_humidity"],
    }


# =========================================================
# OPTIONAL DIRECT TEST
# =========================================================

if __name__ == "__main__":
    print("=== LangGraph Multi-Zone HVAC Control ===")

    if not llm:
        print("No active LLM provider was found.")
        print(
            "Set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY."
        )
        raise SystemExit(1)

    print(f"Using Provider: {provider_name}")

    try:
        with open("dataset.json", "r", encoding="utf-8") as file:
            scenarios_data = json.load(file)
    except FileNotFoundError:
        print("dataset.json was not found.")
        raise SystemExit(1)

    translator = FuzzySymbolicTranslator()

    for index, scenario in enumerate(scenarios_data["scenarios"]):
        if index > 0:
            time.sleep(1.0)

        window_raw = map_json_telemetry_to_translator(
            scenario["window_zone"]["telemetry"]
        )
        interior_raw = map_json_telemetry_to_translator(
            scenario["interior_zone"]["telemetry"]
        )

        window_translated = translator.translate_payload(
            window_raw,
            scenario["window_zone"]["occupant_preferences"],
        )
        interior_translated = translator.translate_payload(
            interior_raw,
            scenario["interior_zone"]["occupant_preferences"],
        )

        initial_state = AgentState(
            scenario_id=scenario["scenario_id"],
            description=scenario["description"],
            occupancy=int(scenario.get("occupancy", 0)),
            ambient_temp_c=float(
                scenario.get(
                    "ambient_temp_c",
                    window_raw["temperature"],
                )
            ),
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

        final_state = hvac_app.invoke(initial_state)

        print("\n--- EXPERT RECOMMENDATIONS ---")
        for message in final_state["messages"]:
            print(message)

        print("\n--- FINAL CONTROL SIGNALS ---")
        print(final_state["control_signals"])
        print("Decision source:", final_state["decision_source"])
        print(
            "First-pass compliant:",
            final_state["first_pass_constraint_compliant"],
        )
        print(
            "First-pass violations:",
            final_state["first_pass_constraint_violations"],
        )
        print("Rule adjustments:", final_state["rule_adjustments"])
        print("Reasoning:", final_state["final_decision"])
