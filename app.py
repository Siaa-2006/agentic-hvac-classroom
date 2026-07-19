import os
import sys
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from translator import FuzzySymbolicTranslator
    from hvac_agents import hvac_app, AgentState

except ImportError:
    print(
        "Could not import translator.py "
        "or hvac_agents.py."
    )

    sys.exit(1)


# =========================================================
# REQUEST AND RESPONSE MODELS
# =========================================================

class ZoneTelemetry(BaseModel):
    air_temp_c: float

    relative_humidity: float = 50.0

    co2_ppm: float

    occupant_preferences: List[float] = (
        Field(default_factory=list)
    )


class ClassroomPayload(BaseModel):
    scenario_id: str = "live_stream"

    description: str = (
        "Live classroom spatial sensor input"
    )

    window_zone: ZoneTelemetry

    interior_zone: ZoneTelemetry


class ControlCommands(BaseModel):
    hvac_mode: Literal[
        "COOL",
        "HEAT",
        "OFF",
    ]

    window_damper_pct: int = Field(
        ge=0,
        le=100,
    )

    interior_damper_pct: int = Field(
        ge=0,
        le=100,
    )

    fan_speed: Literal[
        "OFF",
        "LOW",
        "MED",
        "HIGH",
    ]


class ClassroomResponse(BaseModel):
    status: str = "success"

    scenario_id: str

    expert_messages: List[str]

    supervisor_reasoning: str

    control_signals: ControlCommands

    decision_source: str

    validation_error: Optional[str] = None


class IoTNodePayload(BaseModel):
    zone_id: str

    air_temp_c: float

    relative_humidity: float

    co2_ppm: float

    occupant_preferences: List[float] = (
        Field(default_factory=list)
    )


class IoTNodeResponse(BaseModel):
    status: str = "success"

    cooling_mode: Literal[
        "ON",
        "OFF",
    ]

    damper_angle: int = Field(
        ge=0,
        le=180,
    )

    fan_speed: Literal[
        "OFF",
        "LOW",
        "MEDIUM",
        "HIGH",
    ]

    supervisor_decision: str

    decision_source: str


# =========================================================
# FASTAPI SETUP
# =========================================================

app = FastAPI(
    title="Agentic Spatial HVAC API",

    description=(
        "FastAPI gateway for validated "
        "multi-zone HVAC recommendations"
    ),

    version="2.0.0",
)


translator = FuzzySymbolicTranslator()


iot_classroom_cache = {
    "window_zone": {
        "air_temp_c": 26.5,

        "relative_humidity": 50.0,

        "co2_ppm": 450.0,

        "occupant_preferences": [
            22.0,
            21.5,
            22.0,
        ],
    },

    "interior_zone": {
        "air_temp_c": 24.5,

        "relative_humidity": 52.0,

        "co2_ppm": 550.0,

        "occupant_preferences": [
            24.0,
            24.5,
            23.5,
        ],
    },
}


def build_initial_state(
    scenario_id: str,
    description: str,
    window_translated: dict,
    interior_translated: dict,
) -> AgentState:
    return AgentState(
        scenario_id=scenario_id,

        description=description,

        window_zone=window_translated,

        interior_zone=interior_translated,

        messages=[],

        final_decision="",

        control_signals={},

        decision_source="",

        raw_supervisor_output="",

        validation_error="",
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def read_root():
    return {
        "service":
            "Agentic Spatial HVAC API",

        "status":
            "healthy",

        "validated_structured_output":
            True,
    }


# =========================================================
# COMPLETE TWO-ZONE ENDPOINT
# =========================================================

@app.post(
    "/api/v1/telemetry",
    response_model=ClassroomResponse,
)
async def process_telemetry(
    payload: ClassroomPayload,
):
    window_raw = {
        "temperature":
            payload.window_zone.air_temp_c,

        "co2":
            payload.window_zone.co2_ppm,

        "humidity":
            payload.window_zone.relative_humidity,
    }

    interior_raw = {
        "temperature":
            payload.interior_zone.air_temp_c,

        "co2":
            payload.interior_zone.co2_ppm,

        "humidity":
            payload.interior_zone.relative_humidity,
    }

    window_translated = (
        translator.translate_payload(
            window_raw,

            payload.window_zone.
                occupant_preferences,
        )
    )

    interior_translated = (
        translator.translate_payload(
            interior_raw,

            payload.interior_zone.
                occupant_preferences,
        )
    )

    initial_state = build_initial_state(
        scenario_id=payload.scenario_id,

        description=payload.description,

        window_translated=
            window_translated,

        interior_translated=
            interior_translated,
    )

    try:
        final_state = hvac_app.invoke(
            initial_state
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Error executing agent "
                f"reasoning flow: {error}"
            ),
        ) from error

    control_signals = ControlCommands(
        **final_state["control_signals"]
    )

    return ClassroomResponse(
        status="success",

        scenario_id=payload.scenario_id,

        expert_messages=
            final_state["messages"],

        supervisor_reasoning=
            final_state["final_decision"],

        control_signals=
            control_signals,

        decision_source=
            final_state["decision_source"],

        validation_error=(
            final_state.get(
                "validation_error"
            )
            or None
        ),
    )


# =========================================================
# OPTIONAL IOT ENDPOINT
# =========================================================

@app.post(
    "/api/v1/iot/node",
    response_model=IoTNodeResponse,
)
async def process_iot_node_telemetry(
    payload: IoTNodePayload,
):
    zone = payload.zone_id.strip().lower()

    if zone not in {
        "window_zone",
        "interior_zone",
    }:
        raise HTTPException(
            status_code=400,

            detail=(
                "zone_id must be "
                "'window_zone' or "
                "'interior_zone'."
            ),
        )

    iot_classroom_cache[zone] = {
        "air_temp_c":
            payload.air_temp_c,

        "relative_humidity":
            payload.relative_humidity,

        "co2_ppm":
            payload.co2_ppm,

        "occupant_preferences":
            payload.occupant_preferences,
    }

    window_cached = (
        iot_classroom_cache[
            "window_zone"
        ]
    )

    interior_cached = (
        iot_classroom_cache[
            "interior_zone"
        ]
    )

    window_translated = (
        translator.translate_payload(
            {
                "temperature":
                    window_cached[
                        "air_temp_c"
                    ],

                "co2":
                    window_cached[
                        "co2_ppm"
                    ],

                "humidity":
                    window_cached[
                        "relative_humidity"
                    ],
            },

            window_cached[
                "occupant_preferences"
            ],
        )
    )

    interior_translated = (
        translator.translate_payload(
            {
                "temperature":
                    interior_cached[
                        "air_temp_c"
                    ],

                "co2":
                    interior_cached[
                        "co2_ppm"
                    ],

                "humidity":
                    interior_cached[
                        "relative_humidity"
                    ],
            },

            interior_cached[
                "occupant_preferences"
            ],
        )
    )

    initial_state = build_initial_state(
        scenario_id=
            "live_iot_hardware_stream",

        description=(
            "Telemetry received from "
            f"{payload.zone_id}"
        ),

        window_translated=
            window_translated,

        interior_translated=
            interior_translated,
    )

    try:
        final_state = hvac_app.invoke(
            initial_state
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,

            detail=(
                "Error executing agent "
                f"reasoning flow: {error}"
            ),
        ) from error

    controls = final_state[
        "control_signals"
    ]

    cooling_mode = (
        "ON"
        if controls["hvac_mode"] == "COOL"
        else "OFF"
    )

    if zone == "window_zone":
        damper_percent = controls[
            "window_damper_pct"
        ]

    else:
        damper_percent = controls[
            "interior_damper_pct"
        ]

    damper_angle = int(
        (damper_percent / 100.0)
        * 180.0
    )

    fan_speed = (
        "MEDIUM"
        if controls["fan_speed"] == "MED"
        else controls["fan_speed"]
    )

    return IoTNodeResponse(
        status="success",

        cooling_mode=cooling_mode,

        damper_angle=damper_angle,

        fan_speed=fan_speed,

        supervisor_decision=
            final_state["final_decision"],

        decision_source=
            final_state["decision_source"],
    )


# =========================================================
# LOCAL SERVER
# =========================================================

if __name__ == "__main__":
    import uvicorn

    if not any(
        os.environ.get(key)
        for key in [
            "GROQ_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
        ]
    ):
        print(
            "Warning: no LLM API key "
            "is configured."
        )

    uvicorn.run(
        "app:app",

        host="127.0.0.1",

        port=8000,

        reload=True,
    )
