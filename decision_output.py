import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

try:
    from pydantic import ConfigDict
except ImportError:
    ConfigDict = None


class ControlDecision(BaseModel):
    hvac_mode: Literal["COOL", "HEAT", "OFF"]

    window_damper_pct: int = Field(ge=0, le=100)
    interior_damper_pct: int = Field(ge=0, le=100)

    fan_speed: Literal["OFF", "LOW", "MED", "HIGH"]

    reasoning: str = Field(min_length=1)

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


class ControlDecisionError(ValueError):
    pass


def extract_json_object(raw_text: str) -> str:
    cleaned = raw_text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ControlDecisionError(
            "No JSON object was found in the supervisor response."
        )

    return cleaned[start:end + 1]


def validate_control_decision(raw_text: str) -> dict:
    json_text = extract_json_object(raw_text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as error:
        raise ControlDecisionError(
            f"Invalid JSON: {error}"
        ) from error

    try:
        if hasattr(ControlDecision, "model_validate"):
            decision = ControlDecision.model_validate(payload)
        else:
            decision = ControlDecision.parse_obj(payload)

    except ValidationError as error:
        raise ControlDecisionError(str(error)) from error

    if hasattr(decision, "model_dump"):
        return decision.model_dump()

    return decision.dict()


def deterministic_fallback(
    window_meta: dict,
    interior_meta: dict,
) -> dict:
    window_temp = float(window_meta["actual_temp_c"])
    interior_temp = float(interior_meta["actual_temp_c"])

    window_target = float(
        window_meta.get("avg_requested_temp") or 22.0
    )
    interior_target = float(
        interior_meta.get("avg_requested_temp") or 22.0
    )

    average_error = (
        (window_temp - window_target)
        + (interior_temp - interior_target)
    ) / 2.0

    hvac_mode = "COOL" if average_error > 0.5 else "OFF"

    if hvac_mode == "COOL":
        window_damper = max(
            10,
            min(
                100,
                int((window_temp - window_target) * 35),
            ),
        )

        interior_damper = max(
            10,
            min(
                100,
                int((interior_temp - interior_target) * 35),
            ),
        )

    else:
        window_damper = 10
        interior_damper = 10

    maximum_co2 = max(
        float(window_meta["actual_co2_ppm"]),
        float(interior_meta["actual_co2_ppm"]),
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
        "reasoning": (
            "The supervisory output remained invalid after one "
            "corrective retry, so the deterministic rule-based "
            "fallback was applied."
        ),
    }
