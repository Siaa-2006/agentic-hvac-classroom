import json
import os
import time
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from decision_output import (
    ControlDecisionError,
    deterministic_fallback,
    validate_control_decision,
)
from translator import FuzzySymbolicTranslator


class AgentState(TypedDict, total=False):
    scenario_id: str
    description: str

    window_zone: dict
    interior_zone: dict

    messages: List[str]

    final_decision: str
    control_signals: dict

    decision_source: str
    raw_supervisor_output: str
    validation_error: str


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
            temperature=0.2,
            max_retries=3,
        )

        provider_name = "Groq (Llama-3.3-70b)"

    except ImportError:
        print(
            "Install Groq support using: "
            "pip install langchain-groq"
        )


elif os.environ.get("GEMINI_API_KEY"):
    try:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
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
            temperature=0.2,
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


# =========================================================
# WINDOW-ZONE AGENT
# =========================================================

def window_zone_expert(state: AgentState):
    if not llm:
        raise ValueError(
            "No LLM provider configured."
        )

    semantic_data = state["window_zone"]["semantic_state"]
    metadata = state["window_zone"]["meta"]

    messages = [
        SystemMessage(
            content=(
                "You are the Window Zone Expert. Advocate only "
                "for the comfort and air quality of occupants "
                "near the windows. Ignore the interior zone."
            )
        ),

        HumanMessage(
            content=(
                f"Window Zone Physical State:\n"
                f"Temperature: "
                f"{metadata['actual_temp_c']} °C\n"
                f"CO2: "
                f"{metadata['actual_co2_ppm']} ppm\n\n"

                f"Thermal condition: "
                f"{semantic_data['thermal_condition']}\n"
                f"Air purity: "
                f"{semantic_data['air_purity']}\n"
                f"Occupant demand: "
                f"{semantic_data['group_comfort_sentiment']}\n\n"

                "Recommend a specific HVAC action and support "
                "your recommendation in two clear sentences."
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
        raise ValueError(
            "No LLM provider configured."
        )

    semantic_data = state["interior_zone"]["semantic_state"]
    metadata = state["interior_zone"]["meta"]

    messages = [
        SystemMessage(
            content=(
                "You are the Interior Zone Expert. Advocate "
                "only for the comfort and air quality of "
                "occupants at the interior desks. Ignore the "
                "window zone."
            )
        ),

        HumanMessage(
            content=(
                f"Interior Zone Physical State:\n"
                f"Temperature: "
                f"{metadata['actual_temp_c']} °C\n"
                f"CO2: "
                f"{metadata['actual_co2_ppm']} ppm\n\n"

                f"Thermal condition: "
                f"{semantic_data['thermal_condition']}\n"
                f"Air purity: "
                f"{semantic_data['air_purity']}\n"
                f"Occupant demand: "
                f"{semantic_data['group_comfort_sentiment']}\n\n"

                "Recommend a specific HVAC action and support "
                "your recommendation in two clear sentences."
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

{
  "hvac_mode": "COOL",
  "window_damper_pct": 70,
  "interior_damper_pct": 60,
  "fan_speed": "MED",
  "reasoning": "Brief explanation of how both zone demands were balanced."
}

Rules:

1. hvac_mode must be exactly:
   COOL, HEAT, or OFF.

2. window_damper_pct must be an integer from 0 to 100.

3. interior_damper_pct must be an integer from 0 to 100.

4. fan_speed must be exactly:
   OFF, LOW, MED, or HIGH.

5. reasoning must briefly explain the conflict resolution.

6. Do not include Markdown fences.

7. Do not include any text outside the JSON object.
""".strip()


# =========================================================
# SUPERVISORY AGENT
# =========================================================

def multi_zone_supervisor(state: AgentState):
    if not llm:
        raise ValueError(
            "No LLM provider configured."
        )

    debate_log = "\n".join(
        state.get("messages", [])
    )

    window_metadata = state["window_zone"]["meta"]
    interior_metadata = state["interior_zone"]["meta"]

    supervisor_messages = [
        SystemMessage(
            content=(
                "You are the Head Multi-Zone HVAC Supervisor. "
                "Read both zone recommendations and coordinate "
                "one classroom-level control plan. Select one "
                "central HVAC mode, one window-zone damper "
                "percentage, one interior-zone damper "
                "percentage, and one ventilation-fan level."
            )
        ),

        HumanMessage(
            content=(
                "Classroom Spatial Telemetry:\n\n"

                f"Window Zone:\n"
                f"- Temperature: "
                f"{window_metadata['actual_temp_c']} °C\n"
                f"- CO2: "
                f"{window_metadata['actual_co2_ppm']} ppm\n"
                f"- Requested temperature: "
                f"{window_metadata['avg_requested_temp']} °C\n\n"

                f"Interior Zone:\n"
                f"- Temperature: "
                f"{interior_metadata['actual_temp_c']} °C\n"
                f"- CO2: "
                f"{interior_metadata['actual_co2_ppm']} ppm\n"
                f"- Requested temperature: "
                f"{interior_metadata['avg_requested_temp']} °C\n\n"

                f"Zone recommendations:\n"
                f"{debate_log}\n\n"

                "Resolve any conflict between the zones, "
                "including cases where one zone requests "
                "heating and the other requests cooling.\n\n"

                f"{JSON_OUTPUT_INSTRUCTION}"
            )
        ),
    ]

    first_response = invoke_with_retry(
        llm,
        supervisor_messages,
    )

    raw_output = first_response.content
    validation_error = ""

    try:
        decision = validate_control_decision(
            raw_output
        )

        decision_source = "llm_validated"

    except ControlDecisionError as first_error:
        validation_error = str(first_error)

        corrective_messages = [
            SystemMessage(
                content=(
                    "You repair invalid HVAC control JSON. "
                    "Return only the corrected JSON object "
                    "and no additional text."
                )
            ),

            HumanMessage(
                content=(
                    "The previous supervisor response failed "
                    "validation.\n\n"

                    f"Invalid response:\n"
                    f"{raw_output}\n\n"

                    f"Validation error:\n"
                    f"{validation_error}\n\n"

                    f"{JSON_OUTPUT_INSTRUCTION}"
                )
            ),
        ]

        retry_response = invoke_with_retry(
            llm,
            corrective_messages,
        )

        raw_output = retry_response.content

        try:
            decision = validate_control_decision(
                raw_output
            )

            decision_source = "llm_corrective_retry"

        except ControlDecisionError as retry_error:
            validation_error = (
                f"First attempt: {validation_error} | "
                f"Corrective retry: {retry_error}"
            )

            decision = deterministic_fallback(
                window_meta=window_metadata,
                interior_meta=interior_metadata,
            )

            decision_source = "rule_based_fallback"

    state["final_decision"] = decision["reasoning"]

    state["control_signals"] = {
        "hvac_mode": decision["hvac_mode"],

        "window_damper_pct":
            decision["window_damper_pct"],

        "interior_damper_pct":
            decision["interior_damper_pct"],

        "fan_speed": decision["fan_speed"],
    }

    state["decision_source"] = decision_source
    state["raw_supervisor_output"] = raw_output
    state["validation_error"] = validation_error

    return state


# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(AgentState)

workflow.add_node(
    "window_zone_expert",
    window_zone_expert,
)

workflow.add_node(
    "interior_zone_expert",
    interior_zone_expert,
)

workflow.add_node(
    "supervisor",
    multi_zone_supervisor,
)

workflow.set_entry_point(
    "window_zone_expert"
)

workflow.add_edge(
    "window_zone_expert",
    "interior_zone_expert",
)

workflow.add_edge(
    "interior_zone_expert",
    "supervisor",
)

workflow.add_edge(
    "supervisor",
    END,
)

hvac_app = workflow.compile()


def map_json_telemetry_to_translator(
    telemetry: dict,
) -> dict:
    return {
        "temperature": telemetry["air_temp_c"],
        "co2": telemetry["co2_ppm"],
        "humidity": telemetry["relative_humidity"],
    }


# =========================================================
# OPTIONAL DIRECT TEST
# =========================================================

if __name__ == "__main__":
    print(
        "=== LangGraph Multi-Zone HVAC Control ==="
    )

    if not llm:
        print(
            "No active LLM provider was found."
        )

        print(
            "Set GROQ_API_KEY, GEMINI_API_KEY, "
            "or OPENAI_API_KEY."
        )

        raise SystemExit(1)

    print(f"Using Provider: {provider_name}")

    try:
        with open(
            "dataset.json",
            "r",
            encoding="utf-8",
        ) as file:
            scenarios_data = json.load(file)

    except FileNotFoundError:
        print(
            "dataset.json was not found."
        )

        raise SystemExit(1)

    translator = FuzzySymbolicTranslator()

    for index, scenario in enumerate(
        scenarios_data["scenarios"]
    ):
        if index > 0:
            time.sleep(1.0)

        window_raw = (
            map_json_telemetry_to_translator(
                scenario["window_zone"]["telemetry"]
            )
        )

        interior_raw = (
            map_json_telemetry_to_translator(
                scenario["interior_zone"]["telemetry"]
            )
        )

        window_translated = translator.translate_payload(
            window_raw,
            scenario["window_zone"][
                "occupant_preferences"
            ],
        )

        interior_translated = translator.translate_payload(
            interior_raw,
            scenario["interior_zone"][
                "occupant_preferences"
            ],
        )

        initial_state = AgentState(
            scenario_id=scenario["scenario_id"],
            description=scenario["description"],

            window_zone=window_translated,
            interior_zone=interior_translated,

            messages=[],

            final_decision="",
            control_signals={},

            decision_source="",
            raw_supervisor_output="",
            validation_error="",
        )

        final_state = hvac_app.invoke(
            initial_state
        )

        print("\n--- EXPERT RECOMMENDATIONS ---")

        for message in final_state["messages"]:
            print(message)

        print("\n--- VALIDATED CONTROL SIGNALS ---")
        print(final_state["control_signals"])

        print(
            "Decision source:",
            final_state["decision_source"],
        )

        print(
            "Reasoning:",
            final_state["final_decision"],
        )
