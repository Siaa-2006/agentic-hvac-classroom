import os
import sys
import re
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure our local agent and translator files are importable
try:
    from translator import FuzzySymbolicTranslator
    from hvac_agents import hvac_app, AgentState
except ImportError:
    print("❌ Error: Could not import translator.py or hvac_agents.py.")
    print("Please make sure both files are present in the current working directory.")
    sys.exit(1)

# --- 1. Define Request & Response Schemas ---

class ZoneTelemetry(BaseModel):
    air_temp_c: float = Field(..., description="Current temperature of the zone in degrees Celsius")
    relative_humidity: float = Field(50.0, description="Relative humidity percentage")
    co2_ppm: float = Field(..., description="Carbon dioxide concentration in ppm")
    occupant_preferences: List[float] = Field(default_factory=list, description="Recent occupant temperature preferences voted via app")

class ClassroomPayload(BaseModel):
    scenario_id: str = Field("live_stream", description="Identifier for tracking the telemetry event")
    description: str = Field("Live classroom spatial sensor input", description="Context metadata description")
    window_zone: ZoneTelemetry = Field(..., description="Telemetry and preferences from the window row perimeter")
    interior_zone: ZoneTelemetry = Field(..., description="Telemetry and preferences from the interior core row")

class ControlCommands(BaseModel):
    hvac_mode: str = Field(..., description="Central system thermal mode (COOL, HEAT, or OFF)")
    window_damper_pct: int = Field(..., description="Target open position for the window VAV damper (0-100%)")
    interior_damper_pct: int = Field(..., description="Target open position for the interior VAV damper (0-100%)")
    fan_speed: str = Field(..., description="Target ventilation fan speed (OFF, LOW, MED, or HIGH)")

class ClassroomResponse(BaseModel):
    status: str = Field("success", description="API execution status")
    scenario_id: str
    expert_messages: List[str] = Field(..., description="Transcript of the individual specialized agent recommendations")
    supervisor_reasoning: str = Field(..., description="The raw natural language explanation from the supervisor")
    control_signals: ControlCommands = Field(..., description="Parsed structured mechanical instructions")


# --- 2. Robust Natural Language Decision Parser ---

def parse_supervisor_decision(decision_text: str) -> dict:
    """
    Decodes the Supervisor's natural language summary into strict numeric control targets.
    Utilizes segment clause isolation and strict word boundaries to eliminate overlap bugs.
    """
    text_lower = decision_text.lower()
    
    # 1. Parse central heating/cooling mode
    hvac_mode = "OFF"
    if "cool" in text_lower or "chilling" in text_lower:
        hvac_mode = "COOL"
    elif "heat" in text_lower or "warm" in text_lower:
        hvac_mode = "HEAT"

    # 2. Segment text into clauses to isolate zone targets
    clauses = re.split(r'[,.;]|\band\b|\bwhile\b|\bbut\b', text_lower)
    w_damper = 50  # Balanced default fallbacks
    i_damper = 50
    
    for clause in clauses:
        pct_match = re.search(r'(\d+)\s*%', clause)
        if pct_match:
            pct_val = int(pct_match.group(1))
            
            # Map the percentage to the correct zone based on keyword proximity
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

    # 3. Parse ventilation fan speeds with strict word boundary checks
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

    # Enforce strict physical hardware constraints
    w_damper = max(0, min(100, w_damper))
    i_damper = max(0, min(100, i_damper))

    return {
        "hvac_mode": hvac_mode,
        "window_damper_pct": w_damper,
        "interior_damper_pct": i_damper,
        "fan_speed": fan_speed
    }


# --- 3. Instantiate FastAPI Server ---

app = FastAPI(
    title="Agentic Spatial HVAC API",
    description="FastAPI gateway managing multi-zone environment controls via LangGraph and Llama-3.3-70b",
    version="1.0.0"
)

# Initialize the shared translator
translator = FuzzySymbolicTranslator()

@app.get("/")
def read_root():
    """Service status health-check endpoint."""
    return {
        "service": "Dhaka Classroom Agentic Controller API",
        "status": "healthy",
        "configured_llm": os.environ.get("GROQ_API_KEY", "Not Set")[:8] + "..." if os.environ.get("GROQ_API_KEY") else "None (Need Key)"
    }

@app.post("/api/v1/telemetry", response_model=ClassroomResponse)
async def process_telemetry(payload: ClassroomPayload):
    """
    Accepts spatial telemetry, runs the fuzzy symbol translator,
    executes the multi-agent LangGraph consensus debate, and returns 
    structured physical control signals for edge hardware actuators.
    """
    
    # 1. Translate Window Zone raw state
    w_raw = {
        "temperature": payload.window_zone.air_temp_c,
        "co2": payload.window_zone.co2_ppm,
        "humidity": payload.window_zone.relative_humidity
    }
    w_translated = translator.translate_payload(w_raw, payload.window_zone.occupant_preferences)

    # 2. Translate Interior Zone raw state
    i_raw = {
        "temperature": payload.interior_zone.air_temp_c,
        "co2": payload.interior_zone.co2_ppm,
        "humidity": payload.interior_zone.relative_humidity
    }
    i_translated = translator.translate_payload(i_raw, payload.interior_zone.occupant_preferences)

    # 3. Construct AgentState payload
    initial_state = AgentState(
        scenario_id=payload.scenario_id,
        description=payload.description,
        window_zone=w_translated,
        interior_zone=i_translated,
        messages=[],
        final_decision=""
    )

    # 4. Trigger LangGraph flow with runtime safety bounds
    try:
        final_state = hvac_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent reasoning flow: {str(e)}"
        )

    # 5. Extract results and parse supervisor decision text
    raw_decision = final_state["final_decision"]
    control_signals_dict = parse_supervisor_decision(raw_decision)

    # 6. Format and validate final JSON output
    control_signals = ControlCommands(
        hvac_mode=control_signals_dict["hvac_mode"],
        window_damper_pct=control_signals_dict["window_damper_pct"],
        interior_damper_pct=control_signals_dict["interior_damper_pct"],
        fan_speed=control_signals_dict["fan_speed"]
    )

    return ClassroomResponse(
        status="success",
        scenario_id=payload.scenario_id,
        expert_messages=final_state["messages"],
        supervisor_reasoning=raw_decision,
        control_signals=control_signals
    )


# --- 4. Main Entry Point (Local Testing) ---
if __name__ == "__main__":
    import uvicorn
    # Check if a LLM API key is registered in working directory
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  Warning: GROQ_API_KEY environment variable is missing.")
        print("Set your key in your terminal before running:")
        print("export GROQ_API_KEY='your-key'\n")

    print("Launching Local API gateway on port 8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
