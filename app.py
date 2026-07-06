import os
import sys
import re
from typing import List, Optional
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


# --- 2. ESP32 Single-Zone Hardware Integration Schemas ---

class IoTNodePayload(BaseModel):
    zone_id: str = Field(..., description="Must be 'window_zone' or 'interior_zone' matching your ESP32 config")
    air_temp_c: float = Field(..., description="Crisp temperature reading from DHT22")
    relative_humidity: float = Field(..., description="Relative humidity from DHT22")
    co2_ppm: float = Field(..., description="PPM estimate from NDIR analog sensor")
    occupant_preferences: List[float] = Field(default_factory=list, description="Mobile comfort vote array")

class IoTNodeResponse(BaseModel):
    status: str = "success"
    cooling_mode: str = Field(..., description="Either 'ON' or 'OFF' matching ESP32 controller.ino expectations")
    damper_angle: int = Field(..., description="Calculated servo actuator angle between 0 and 180 degrees")
    fan_speed: str = Field(..., description="Fan operational target ('OFF', 'LOW', 'MEDIUM', 'HIGH')")
    supervisor_decision: str = Field(..., description="Coordinated supervisor reasoning log")


# --- 3. Robust Natural Language Decision Parser ---

def parse_supervisor_decision(decision_text: str) -> dict:
    """
    Decodes the Supervisor's natural language summary into strict numeric control targets.
    Employs clause-based segmentation and strict word boundaries to eliminate overlap bugs.
    """
    text_lower = decision_text.lower()
    
    # 1. Parse central heating/cooling mode using expanded robust keywords
    hvac_mode = "OFF"
    cooling_keywords = ["cool", "chilling", "ac", "air condition", "aircon", "chiller", "cooling"]
    heating_keywords = ["heat", "warm", "heating", "warming", "heater"]
    
    if any(word in text_lower for word in cooling_keywords):
        hvac_mode = "COOL"
    elif any(word in text_lower for word in heating_keywords):
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

    # 3. Parse ventilation fan speeds with clause-isolated extraction.
    fan_clause = None
    for clause in clauses:
        if any(kw in clause for kw in ["fan", "ventilation", "blower", "speed"]):
            fan_clause = clause
            break

    search_scope = fan_clause if fan_clause else text_lower

    fan_speed = "OFF"
    if re.search(r'\bhigh\b', search_scope):
        fan_speed = "HIGH"
    elif re.search(r'\b(?:medium|med)\b', search_scope):
        fan_speed = "MED"
    elif re.search(r'\blow\b', search_scope):
        fan_speed = "LOW"
    elif re.search(r'\boff\b', search_scope):
        fan_speed = "OFF"
    elif "ventilation" in search_scope or "fan" in search_scope:
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


# --- 4. Instantiate FastAPI Server & Core In-Memory Cache ---

app = FastAPI(
    title="Agentic Spatial HVAC API",
    description="FastAPI gateway managing multi-zone environment controls via LangGraph and Llama-3.3-70b",
    version="1.0.0"
)

# Initialize the shared translator
translator = FuzzySymbolicTranslator()

# In-memory telemetry cache to preserve state between async ESP32 single-zone uploads
iot_classroom_cache = {
    "window_zone": {
        "air_temp_c": 26.5,
        "relative_humidity": 50.0,
        "co2_ppm": 450.0,
        "occupant_preferences": [22.0, 21.5, 22.0]
    },
    "interior_zone": {
        "air_temp_c": 24.5,
        "relative_humidity": 52.0,
        "co2_ppm": 550.0,
        "occupant_preferences": [24.0, 24.5, 23.5]
    }
}


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
    Accepts complete dual-zone spatial telemetry, runs the fuzzy translator,
    executes the LangGraph consensus debate, and returns structured control signals.
    """
    w_raw = {
        "temperature": payload.window_zone.air_temp_c,
        "co2": payload.window_zone.co2_ppm,
        "humidity": payload.window_zone.relative_humidity
    }
    w_translated = translator.translate_payload(w_raw, payload.window_zone.occupant_preferences)

    i_raw = {
        "temperature": payload.interior_zone.air_temp_c,
        "co2": payload.interior_zone.co2_ppm,
        "humidity": payload.interior_zone.relative_humidity
    }
    i_translated = translator.translate_payload(i_raw, payload.interior_zone.occupant_preferences)

    initial_state = AgentState(
        scenario_id=payload.scenario_id,
        description=payload.description,
        window_zone=w_translated,
        interior_zone=i_translated,
        messages=[],
        final_decision=""
    )

    try:
        final_state = hvac_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent reasoning flow: {str(e)}"
        )

    raw_decision = final_state["final_decision"]
    control_signals_dict = parse_supervisor_decision(raw_decision)

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


@app.post("/api/v1/iot/node", response_model=IoTNodeResponse)
async def process_iot_node_telemetry(payload: IoTNodePayload):
    """
    Dedicated endpoint matching the single-zone post payload from esp32_controller.ino.
    Handles caching, multi-agent co-simulation, and returns flat, hardware-actuator instructions.
    """
    zone = payload.zone_id.strip().lower()
    if zone not in ["window_zone", "interior_zone"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid zone_id. Pin to 'window_zone' or 'interior_zone' to align with hardware config."
        )

    # 1. Update our in-memory cache with the uploading ESP32's latest telemetry
    iot_classroom_cache[zone] = {
        "air_temp_c": payload.air_temp_c,
        "relative_humidity": payload.relative_humidity,
        "co2_ppm": payload.co2_ppm,
        "occupant_preferences": payload.occupant_preferences
    }

    # 2. Extract both cache entries to construct the full two-zone classroom state
    w_cached = iot_classroom_cache["window_zone"]
    i_cached = iot_classroom_cache["interior_zone"]

    # 3. Apply Fuzzy Translation
    w_translated = translator.translate_payload(
        {"temperature": w_cached["air_temp_c"], "co2": w_cached["co2_ppm"], "humidity": w_cached["relative_humidity"]},
        w_cached["occupant_preferences"]
    )
    i_translated = translator.translate_payload(
        {"temperature": i_cached["air_temp_c"], "co2": i_cached["co2_ppm"], "humidity": i_cached["relative_humidity"]},
        i_cached["occupant_preferences"]
    )

    # 4. Trigger LangGraph workflow
    initial_state = AgentState(
        scenario_id="live_iot_hardware_stream",
        description=f"Automated execution triggered by edge node: {payload.zone_id}",
        window_zone=w_translated,
        interior_zone=i_translated,
        messages=[],
        final_decision=""
    )

    try:
        final_state = hvac_app.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing agent reasoning flow: {str(e)}"
        )

    raw_decision = final_state["final_decision"]
    control_signals_dict = parse_supervisor_decision(raw_decision)

    # 5. Map central HVAC status to the ESP32's simple "cooling_mode" ("ON" or "OFF")
    cooling_on_off = "ON" if control_signals_dict["hvac_mode"] == "COOL" else "OFF"

    # 6. Extract and scale VAV Damper percentage to target Servo Motor angles (0 to 180 degrees)
    damper_pct = control_signals_dict["window_damper_pct"] if zone == "window_zone" else control_signals_dict["interior_damper_pct"]
    mapped_angle = int((damper_pct / 100.0) * 180.0)

    # 7. Format fan speed outputs to conform with C++ comparison arrays ("MED" mapped to "MEDIUM")
    fan_speed_mapped = "MEDIUM" if control_signals_dict["fan_speed"] == "MED" else control_signals_dict["fan_speed"]

    return IoTNodeResponse(
        status="success",
        cooling_mode=cooling_on_off,
        damper_angle=mapped_angle,
        fan_speed=fan_speed_mapped,
        supervisor_decision=raw_decision
    )


# --- 5. Main Entry Point ---
if __name__ == "__main__":
    import uvicorn
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  Warning: GROQ_API_KEY environment variable is missing.")
        print("Set your key in your terminal before running:")
        print("export GROQ_API_KEY='your-key'\n")

    print("Launching Local API gateway on port 8000...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
