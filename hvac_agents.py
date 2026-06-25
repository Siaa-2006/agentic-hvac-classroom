import os
import json
import time
import pandas as pd
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from translator import FuzzySymbolicTranslator

# --- 1. Define the State (Multi-Zone Classroom Memory) ---
class AgentState(TypedDict):
    scenario_id: str
    description: str
    window_zone: dict        # Translated window zone data
    interior_zone: dict      # Translated interior zone data
    messages: List[str]      # The multi-agent debate transcript
    final_decision: str      # The Supervisor's coordinated command

# --- 2. Initialize the AI Brain (Prioritizing Free Groq) ---
llm = None
provider_name = ""

# Check for Groq first (Highly recommended free alternative, 14,400 requests/day)
if os.environ.get("GROQ_API_KEY"):
    try:
        from langchain_groq import ChatGroq
        # Using Llama-3.3-70b for advanced reasoning and spatial synthesis
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2, max_retries=3)
        provider_name = "Groq (Llama-3.3-70b)"
    except ImportError:
        print("\n⚠️  To use Groq, please install: pip install langchain-groq")

# Fallback 1: Google Gemini (Free tier, but limited to 20 daily requests)
elif os.environ.get("GEMINI_API_KEY") and not llm:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_retries=0)
        provider_name = "Google Gemini (gemini-2.5-flash)"
    except ImportError:
        print("\n⚠️  To use Gemini, please install: pip install langchain-google-genai")

# Fallback 2: OpenAI (Paid developer account required)
elif os.environ.get("OPENAI_API_KEY") and not llm:
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_retries=0)
        provider_name = "OpenAI (gpt-4o-mini)"
    except ImportError:
        print("\n⚠️  To use OpenAI, please install: pip install langchain-openai")


# --- 3. Robust Invoke Helper with Exponential Backoff ---
def invoke_with_retry(llm, messages, max_retries=5):
    """
    Invokes the LLM with custom backoff.
    Ensures free-tier rate limits are handled gracefully with terminal alerts.
    """
    # Small 1-second delay to preserve request pacing
    time.sleep(1.0)
    
    delays = [2, 4, 8, 16, 32]
    for i, delay in enumerate(delays):
        try:
            return llm.invoke(messages)
        except Exception as e:
            err_str = str(e).lower()
            
            # Identify rate limits
            is_rate_limit = any(term in err_str for term in ["429", "resource_exhausted", "rate_limit", "quota", "exhausted"])
            if is_rate_limit and i < max_retries - 1:
                print(f"  ⚠️  [Rate Limit Alert] Speed limits reached. Pausing for {delay}s before retrying...")
                time.sleep(delay)
                continue
            raise e
            
    return llm.invoke(messages)


# --- 4. Define the Multi-Zone Expert Nodes ---

def window_zone_expert(state: AgentState):
    """Advocates strictly for the comfort and air quality of the window perimeter zone."""
    if not llm:
        raise ValueError("No LLM provider configured. Please set GROQ_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY.")
        
    data = state["window_zone"]["semantic_state"]
    meta = state["window_zone"]["meta"]
    
    messages = [
        SystemMessage(content="You are the Window Zone Expert. Your sole responsibility is to advocate for the comfort and safety of occupants sitting next to the windows. You ignore the interior of the room."),
        HumanMessage(content=f"""Window Zone Physical State: Temp {meta['actual_temp_c']}°C, CO2 {meta['actual_co2_ppm']} ppm.
Fuzzy Status: Thermal state is '{data['thermal_condition']}', Air Purity is '{data['air_purity']}', occupant demand is '{data['group_comfort_sentiment']}'.
What specific HVAC action does the Window Zone need? Support your claim in 2 clear sentences.""")
    ]
    
    response = invoke_with_retry(llm, messages)
    state["messages"].append(f"Window Zone Expert: {response.content}")
    return state


def interior_zone_expert(state: AgentState):
    """Advocates strictly for the comfort and air quality of the interior zone."""
    if not llm:
        raise ValueError("No LLM provider configured.")
        
    data = state["interior_zone"]["semantic_state"]
    meta = state["interior_zone"]["meta"]
    
    messages = [
        SystemMessage(content="You are the Interior Zone Expert. Your sole responsibility is to advocate for the comfort and safety of occupants sitting in the interior desks (away from windows). You ignore the window perimeter."),
        HumanMessage(content=f"""Interior Zone Physical State: Temp {meta['actual_temp_c']}°C, CO2 {meta['actual_co2_ppm']} ppm.
Fuzzy Status: Thermal state is '{data['thermal_condition']}', Air Purity is '{data['air_purity']}', occupant demand is '{data['group_comfort_sentiment']}'.
What specific HVAC action does the Interior Zone need? Support your claim in 2 clear sentences.""")
    ]
    
    response = invoke_with_retry(llm, messages)
    state["messages"].append(f"Interior Zone Expert: {response.content}")
    return state


def multi_zone_supervisor(state: AgentState):
    """Resolves spatial conflicts, balances ventilation, and coordinates central HVAC hardware actuators."""
    if not llm:
        raise ValueError("No LLM provider configured.")
        
    debate_log = "\n".join(state["messages"])
    w_meta = state["window_zone"]["meta"]
    i_meta = state["interior_zone"]["meta"]
    
    messages = [
        SystemMessage(content="You are the Head Multi-Zone HVAC Supervisor. Your job is to read recommendations from both Zone Experts and coordinate a central control plan. You must decide: Central heating/cooling mode, Window Zone VAV damper position (0-100%), Interior Zone VAV damper position (0-100%), and Ventilation Fan speed (Off/Low/Med/High)."),
        HumanMessage(content=f"""Classroom Spatial Telemetry:
- Window Zone: {w_meta['actual_temp_c']}°C, {w_meta['actual_co2_ppm']} ppm CO2 (Requested: {w_meta['avg_requested_temp']}°C)
- Interior Zone: {i_meta['actual_temp_c']}°C, {i_meta['actual_co2_ppm']} ppm CO2 (Requested: {i_meta['avg_requested_temp']}°C)

Here is the debate from the Zone Experts:
{debate_log}

Resolve any conflicts (such as one zone needing heat while the other needs cooling) and issue a final, coordinated control plan. Keep your decision to exactly 3 sentences.""")
    ]
    
    response = invoke_with_retry(llm, messages)
    state["final_decision"] = response.content
    return state


# --- 5. Build the Graph (The Multi-Zone Workflow) ---
workflow = StateGraph(AgentState)

workflow.add_node("window_zone_expert", window_zone_expert)
workflow.add_node("interior_zone_expert", interior_zone_expert)
workflow.add_node("supervisor", multi_zone_supervisor)

workflow.set_entry_point("window_zone_expert")
workflow.add_edge("window_zone_expert", "interior_zone_expert")
workflow.add_edge("interior_zone_expert", "supervisor")
workflow.add_edge("supervisor", END)

hvac_app = workflow.compile()


# --- 6. Run the Simulation ---
def map_json_telemetry_to_translator(telemetry):
    """Maps scenario telemetry keys to keys expected by translator.py."""
    return {
        "temperature": telemetry["air_temp_c"],
        "co2": telemetry["co2_ppm"],
        "humidity": telemetry["relative_humidity"]
    }

if __name__ == "__main__":
    print("=== LangGraph Multi-Zone HVAC Control Chamber ===")
    
    # Validation Warning
    if not llm:
        print("\n⚠️  ERROR: No active LLM provider found.")
        print("Please configure your Groq key in your terminal:")
        print("export GROQ_API_KEY='your-key-here' && pip install langchain-groq\n")
        exit()

    print(f"Using Provider: {provider_name}\n")

    # Load scenarios
    try:
        with open("dataset.json", "r") as f:
            scenarios_data = json.load(f)
    except FileNotFoundError:
        print("ERROR: dataset.json not found in this folder.")
        exit()

    translator = FuzzySymbolicTranslator()

    # Loop and run the debate for each distinct scenario
    for index, scenario in enumerate(scenarios_data["scenarios"]):
        # Add a light pacing delay
        if index > 0:
            time.sleep(1.0)

        print(f"\n" + "="*60)
        print(f"SCENARIO: {scenario['scenario_id']}")
        print(f"Description: {scenario['description']}")
        print("="*60)

        w_raw = map_json_telemetry_to_translator(scenario["window_zone"]["telemetry"])
        w_prefs = scenario["window_zone"]["occupant_preferences"]
        w_translated = translator.translate_payload(w_raw, w_prefs)

        i_raw = map_json_telemetry_to_translator(scenario["interior_zone"]["telemetry"])
        i_prefs = scenario["interior_zone"]["occupant_preferences"]
        i_translated = translator.translate_payload(i_raw, i_prefs)

        initial_state = AgentState(
            scenario_id=scenario["scenario_id"],
            description=scenario["description"],
            window_zone=w_translated,
            interior_zone=i_translated,
            messages=[],
            final_decision=""
        )

        print("Running spatial consensus debate...")
        try:
            final_state = hvac_app.invoke(initial_state)
            
            print("\n--- EXPERT RECOMMENDATIONS ---")
            for msg in final_state["messages"]:
                print(f"🗣️  {msg}")
            
            print("\n⚖️  COORDINATED SUPERVISOR CONTROL ACTION:")
            print(final_state["final_decision"])
            print("-"*60)
            
        except Exception as e:
            print("\n❌ Error during graph execution:")
            print(f"Details: {str(e)}")
            break
