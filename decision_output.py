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


FAN_LEVEL_ORDER = {
    "OFF": 0,
    "LOW": 1,
    "MED": 2,
    "HIGH": 3,
}

# All values below are experiment-specific heuristic settings.
# They are not calibrated physical HVAC parameters.
COLD_ZONE_PROTECTION_ERROR_C = -0.25
VENTILATION_ONLY_COLD_ERROR_C = -1.25
SEVERE_COLD_ERROR_C = -1.50
NO_COOLING_REQUIRED_ERROR_C = 0.50
THERMAL_RESPONSE_FRACTION = 0.80
COOLING_GAIN_PER_FULL_DAMPER = 1.42
PERFORMANCE_RANGE_HALF_WIDTH_PCT = 3
MINIMUM_ACTIVE_DAMPER_PCT = 10


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


def required_fan_speed(maximum_co2: float) -> str:
    """Return the minimum fan level required by the experiment rules."""
    if maximum_co2 > 1200:
        return "HIGH"
    if maximum_co2 > 950:
        return "MED"
    if maximum_co2 > 650:
        return "LOW"
    return "OFF"


def minimum_damper_for_co2(co2_ppm: float) -> int:
    """Return the minimum damper opening required by the experiment rules."""
    if co2_ppm > 1500:
        return 50
    if co2_ppm > 1200:
        return 40
    if co2_ppm > 950:
        return 25
    return 0


def recommended_ventilation_damper(co2_ppm: float) -> int:
    """
    Return a performance-aware ventilation recommendation.

    This is deliberately above the hard minimum at elevated CO2 so that
    the controller does not merely sit at the admissible lower bound.
    """
    if co2_ppm > 1500:
        return 65
    if co2_ppm > 1200:
        return 55
    if co2_ppm > 950:
        return 40
    if co2_ppm > 650:
        return 20
    return 10


def _clamp_int(value: float, minimum: int = 0, maximum: int = 100) -> int:
    return int(round(max(minimum, min(maximum, value))))


def _range(minimum: int, maximum: int) -> dict:
    if minimum > maximum:
        raise ValueError(
            f"Infeasible damper range: minimum {minimum} exceeds "
            f"maximum {maximum}."
        )
    return {"min": int(minimum), "max": int(maximum)}


def _clamp(value: int, bounds: dict) -> int:
    return max(
        int(bounds["min"]),
        min(int(bounds["max"]), int(value)),
    )


def _performance_band(
    recommendation: int,
    hard_minimum: int,
    hard_maximum: int,
) -> dict:
    lower = max(
        hard_minimum,
        recommendation - PERFORMANCE_RANGE_HALF_WIDTH_PCT,
    )
    upper = min(
        hard_maximum,
        recommendation + PERFORMANCE_RANGE_HALF_WIDTH_PCT,
    )

    if lower > upper:
        lower = upper = _clamp_int(
            recommendation,
            hard_minimum,
            hard_maximum,
        )

    return _range(lower, upper)


def _thermal_damper_recommendation(
    zone: str,
    actual_temp_c: float,
    requested_temp_c: float,
    occupancy: int,
    ambient_temp_c: float,
) -> tuple[int, float]:
    """
    Estimate a one-step damper recommendation using the same heuristic
    temperature terms as simulation.py.

    The controller compensates for the estimated background gain and
    removes 80% of the current positive temperature error. A negative
    error reduces the recommendation and allows the zone to warm.
    """
    occupancy = max(0, int(occupancy))
    window_occupancy = int(occupancy * 0.4)
    interior_occupancy = occupancy - window_occupancy
    temperature_error = actual_temp_c - requested_temp_c

    if zone == "window":
        background_gain = (
            0.52
            + window_occupancy * 0.012
            + 0.06 * (ambient_temp_c - actual_temp_c)
        )
    elif zone == "interior":
        background_gain = (
            0.05
            + interior_occupancy * 0.015
            + 0.04 * (26.0 - actual_temp_c)
        )
    else:
        raise ValueError(f"Unknown zone: {zone}")

    requested_cooling_effect = (
        background_gain
        + THERMAL_RESPONSE_FRACTION * temperature_error
    )

    recommended_pct = _clamp_int(
        100.0
        * max(0.0, requested_cooling_effect)
        / COOLING_GAIN_PER_FULL_DAMPER
    )

    return recommended_pct, background_gain


def _maximum_damper_while_cooling(
    temperature_error_c: float,
    minimum_damper_pct: int,
) -> int:
    """Protect a zone that is already colder than requested."""
    if temperature_error_c <= COLD_ZONE_PROTECTION_ERROR_C:
        return max(
            minimum_damper_pct,
            MINIMUM_ACTIVE_DAMPER_PCT,
        )
    return 100


def _enforce_recommended_order(
    window_recommendation: int,
    interior_recommendation: int,
    window_maximum: int,
    interior_maximum: int,
    required_order: str,
) -> tuple[int, int]:
    window_value = window_recommendation
    interior_value = interior_recommendation

    if (
        required_order == "INTERIOR_GTE_WINDOW"
        and interior_value < window_value
    ):
        if interior_maximum >= window_value:
            interior_value = window_value
        else:
            window_value = interior_value

    elif (
        required_order == "WINDOW_GTE_INTERIOR"
        and window_value < interior_value
    ):
        if window_maximum >= interior_value:
            window_value = interior_value
        else:
            interior_value = window_value

    return window_value, interior_value


def _couple_ranges_for_order(
    window_range: dict,
    interior_range: dict,
    required_order: str,
) -> tuple[dict, dict]:
    """Make every value pair in the two ranges satisfy the order."""
    window_min = int(window_range["min"])
    window_max = int(window_range["max"])
    interior_min = int(interior_range["min"])
    interior_max = int(interior_range["max"])

    if required_order == "INTERIOR_GTE_WINDOW":
        interior_min = max(interior_min, window_max)
        if interior_min > interior_max:
            common_value = min(window_max, interior_max)
            window_max = common_value
            window_min = min(window_min, window_max)
            interior_min = common_value
            interior_max = max(interior_max, interior_min)

    elif required_order == "WINDOW_GTE_INTERIOR":
        window_min = max(window_min, interior_max)
        if window_min > window_max:
            common_value = min(window_max, interior_max)
            interior_max = common_value
            interior_min = min(interior_min, interior_max)
            window_min = common_value
            window_max = max(window_max, window_min)

    return (
        _range(window_min, window_max),
        _range(interior_min, interior_max),
    )


def build_constraint_metadata(
    window_meta: dict,
    interior_meta: dict,
    occupancy: int = 0,
    ambient_temp_c: float | None = None,
) -> dict:
    """
    Build performance-aware feasible ranges before LLM arbitration.

    The metadata combines:
    - CO2 minimum ventilation rules,
    - cold-zone protection,
    - a model-aware thermal recommendation,
    - a stronger CO2 recommendation when air quality is poor,
    - a narrow +/-3 percentage-point selection band,
    - and the coupled zone-priority ordering.
    """
    window_co2 = float(window_meta["actual_co2_ppm"])
    interior_co2 = float(interior_meta["actual_co2_ppm"])
    maximum_co2 = max(window_co2, interior_co2)
    co2_difference = abs(window_co2 - interior_co2)

    window_temp = float(window_meta["actual_temp_c"])
    interior_temp = float(interior_meta["actual_temp_c"])
    window_target = float(
        window_meta.get("avg_requested_temp") or 22.0
    )
    interior_target = float(
        interior_meta.get("avg_requested_temp") or 22.0
    )

    ambient_temp = (
        float(ambient_temp_c)
        if ambient_temp_c is not None
        else window_temp
    )
    occupancy = max(0, int(occupancy))

    window_error = window_temp - window_target
    interior_error = interior_temp - interior_target

    window_minimum = minimum_damper_for_co2(window_co2)
    interior_minimum = minimum_damper_for_co2(interior_co2)
    window_ventilation_recommendation = recommended_ventilation_damper(
        window_co2
    )
    interior_ventilation_recommendation = recommended_ventilation_damper(
        interior_co2
    )

    window_thermal_recommendation, window_background_gain = (
        _thermal_damper_recommendation(
            zone="window",
            actual_temp_c=window_temp,
            requested_temp_c=window_target,
            occupancy=occupancy,
            ambient_temp_c=ambient_temp,
        )
    )
    interior_thermal_recommendation, interior_background_gain = (
        _thermal_damper_recommendation(
            zone="interior",
            actual_temp_c=interior_temp,
            requested_temp_c=interior_target,
            occupancy=occupancy,
            ambient_temp_c=ambient_temp,
        )
    )

    window_cool_maximum = _maximum_damper_while_cooling(
        window_error,
        window_minimum,
    )
    interior_cool_maximum = _maximum_damper_while_cooling(
        interior_error,
        interior_minimum,
    )

    priority_rule_active = (
        maximum_co2 > 1200 and co2_difference >= 300
    )
    priority_zone = "NONE"
    required_order = "NONE"

    if priority_rule_active:
        if window_co2 > interior_co2:
            priority_zone = "WINDOW"
            required_order = "WINDOW_GTE_INTERIOR"
        else:
            priority_zone = "INTERIOR"
            required_order = "INTERIOR_GTE_WINDOW"

    warmest_error = max(window_error, interior_error)
    coldest_error = min(window_error, interior_error)

    ventilation_only_required = (
        warmest_error <= NO_COOLING_REQUIRED_ERROR_C
        or coldest_error <= SEVERE_COLD_ERROR_C
        or (
            coldest_error <= VENTILATION_ONLY_COLD_ERROR_C
            and maximum_co2 > 950
        )
    )

    recommended_mode = (
        "OFF" if ventilation_only_required else "COOL"
    )

    # A single recommended mode is supplied. This avoids a nominally
    # valid but performance-inappropriate mode choice by the LLM.
    allowed_modes = [recommended_mode]

    cool_window_recommendation = _clamp_int(
        max(
            window_thermal_recommendation,
            window_ventilation_recommendation,
            window_minimum,
        ),
        window_minimum,
        window_cool_maximum,
    )
    cool_interior_recommendation = _clamp_int(
        max(
            interior_thermal_recommendation,
            interior_ventilation_recommendation,
            interior_minimum,
        ),
        interior_minimum,
        interior_cool_maximum,
    )

    cool_window_recommendation, cool_interior_recommendation = (
        _enforce_recommended_order(
            cool_window_recommendation,
            cool_interior_recommendation,
            window_cool_maximum,
            interior_cool_maximum,
            required_order,
        )
    )

    off_window_recommendation = max(
        window_minimum,
        window_ventilation_recommendation,
    )
    off_interior_recommendation = max(
        interior_minimum,
        interior_ventilation_recommendation,
    )

    off_window_recommendation, off_interior_recommendation = (
        _enforce_recommended_order(
            off_window_recommendation,
            off_interior_recommendation,
            100,
            100,
            required_order,
        )
    )

    cool_window_range = _performance_band(
        cool_window_recommendation,
        window_minimum,
        window_cool_maximum,
    )
    cool_interior_range = _performance_band(
        cool_interior_recommendation,
        interior_minimum,
        interior_cool_maximum,
    )
    cool_window_range, cool_interior_range = _couple_ranges_for_order(
        cool_window_range,
        cool_interior_range,
        required_order,
    )

    off_window_range = _performance_band(
        off_window_recommendation,
        window_minimum,
        100,
    )
    off_interior_range = _performance_band(
        off_interior_recommendation,
        interior_minimum,
        100,
    )
    off_window_range, off_interior_range = _couple_ranges_for_order(
        off_window_range,
        off_interior_range,
        required_order,
    )

    cool_ranges = {
        "window": cool_window_range,
        "interior": cool_interior_range,
    }
    off_ranges = {
        "window": off_window_range,
        "interior": off_interior_range,
    }

    if recommended_mode == "COOL":
        recommended_dampers = {
            "window": cool_window_recommendation,
            "interior": cool_interior_recommendation,
        }
        active_ranges = cool_ranges
    else:
        recommended_dampers = {
            "window": off_window_recommendation,
            "interior": off_interior_recommendation,
        }
        active_ranges = off_ranges

    required_fan = required_fan_speed(maximum_co2)

    if recommended_mode == "OFF":
        thermal_reason = (
            "Ventilation-only mode is recommended because cooling is "
            "not required or one zone is sufficiently colder than its "
            "requested temperature. Dampers may remain open for CO2 "
            "control without applying the simulator's cooling term."
        )
    else:
        thermal_reason = (
            "Cooling is recommended. Each zone receives a narrow "
            "performance band centred on the larger of its thermal and "
            "ventilation recommendations, subject to cold-zone "
            "protection."
        )

    return {
        "occupancy": occupancy,
        "ambient_temp_c": round(ambient_temp, 2),
        "allowed_hvac_modes": allowed_modes,
        "recommended_hvac_mode": recommended_mode,
        "required_minimum_fan": required_fan,
        "allowed_fan_levels": [required_fan],
        "required_damper_order": required_order,
        "co2_priority_rule_active": priority_rule_active,
        "co2_priority_zone": priority_zone,
        "co2_difference_ppm": round(co2_difference, 1),
        "window_temperature_error_c": round(window_error, 2),
        "interior_temperature_error_c": round(interior_error, 2),
        "window_background_gain_c_per_step": round(
            window_background_gain,
            3,
        ),
        "interior_background_gain_c_per_step": round(
            interior_background_gain,
            3,
        ),
        "thermal_damper_recommendations": {
            "window": window_thermal_recommendation,
            "interior": interior_thermal_recommendation,
        },
        "ventilation_damper_recommendations": {
            "window": window_ventilation_recommendation,
            "interior": interior_ventilation_recommendation,
        },
        "recommended_damper_pct": recommended_dampers,
        "cool_mode_damper_ranges": cool_ranges,
        "off_mode_damper_ranges": off_ranges,
        "recommended_mode_damper_ranges": active_ranges,
        "ventilation_only_required": ventilation_only_required,
        "thermal_protection_reason": thermal_reason,
        "selection_rule": (
            "Use the recommended HVAC mode and exact required fan. "
            "Choose each damper at its recommended_damper_pct by "
            "default. A deviation is allowed only inside the narrow "
            "recommended-mode range and must be justified by the two "
            "zone demands."
        ),
    }


def _ranges_for_mode(constraints: dict, mode: str) -> dict:
    if mode == "COOL":
        return constraints["cool_mode_damper_ranges"]
    if mode == "OFF":
        return constraints["off_mode_damper_ranges"]

    recommended_mode = constraints["recommended_hvac_mode"]
    if recommended_mode == "COOL":
        return constraints["cool_mode_damper_ranges"]
    return constraints["off_mode_damper_ranges"]


def find_constraint_violations(
    decision: dict,
    window_meta: dict,
    interior_meta: dict,
    occupancy: int = 0,
    ambient_temp_c: float | None = None,
) -> list[str]:
    """Return command violations without changing the command."""
    constraints = build_constraint_metadata(
        window_meta=window_meta,
        interior_meta=interior_meta,
        occupancy=occupancy,
        ambient_temp_c=ambient_temp_c,
    )
    violations: list[str] = []

    mode = decision["hvac_mode"]
    allowed_modes = constraints["allowed_hvac_modes"]

    if mode not in allowed_modes:
        violations.append(
            f"hvac_mode {mode} is not allowed; allowed modes are "
            f"{allowed_modes}"
        )

    fan = decision["fan_speed"]
    allowed_fans = constraints["allowed_fan_levels"]
    if fan not in allowed_fans:
        violations.append(
            f"fan_speed {fan} is not allowed; allowed fan levels are "
            f"{allowed_fans}"
        )

    ranges = _ranges_for_mode(constraints, mode)
    window_range = ranges["window"]
    interior_range = ranges["interior"]

    window_damper = int(decision["window_damper_pct"])
    interior_damper = int(decision["interior_damper_pct"])

    if not (
        int(window_range["min"])
        <= window_damper
        <= int(window_range["max"])
    ):
        violations.append(
            f"window damper {window_damper}% is outside the "
            f"inclusive {mode} range "
            f"[{window_range['min']}, {window_range['max']}]"
        )

    if not (
        int(interior_range["min"])
        <= interior_damper
        <= int(interior_range["max"])
    ):
        violations.append(
            f"interior damper {interior_damper}% is outside the "
            f"inclusive {mode} range "
            f"[{interior_range['min']}, {interior_range['max']}]"
        )

    required_order = constraints["required_damper_order"]

    if (
        required_order == "WINDOW_GTE_INTERIOR"
        and window_damper < interior_damper
    ):
        violations.append(
            "required damper order is WINDOW_GTE_INTERIOR but the "
            "window damper is smaller"
        )

    if (
        required_order == "INTERIOR_GTE_WINDOW"
        and interior_damper < window_damper
    ):
        violations.append(
            "required damper order is INTERIOR_GTE_WINDOW but the "
            "interior damper is smaller"
        )

    return violations


def apply_rule_based_command_checks(
    decision: dict,
    window_meta: dict,
    interior_meta: dict,
    occupancy: int = 0,
    ambient_temp_c: float | None = None,
) -> tuple[dict, list[str]]:
    """Clamp the proposal to the performance-aware feasible region."""
    corrected = decision.copy()
    adjustments: list[str] = []

    constraints = build_constraint_metadata(
        window_meta=window_meta,
        interior_meta=interior_meta,
        occupancy=occupancy,
        ambient_temp_c=ambient_temp_c,
    )

    allowed_modes = constraints["allowed_hvac_modes"]
    if corrected["hvac_mode"] not in allowed_modes:
        old_mode = corrected["hvac_mode"]
        corrected["hvac_mode"] = constraints[
            "recommended_hvac_mode"
        ]
        adjustments.append(
            f"thermal mode changed from {old_mode} to "
            f"{corrected['hvac_mode']}"
        )

    allowed_fans = constraints["allowed_fan_levels"]
    if corrected["fan_speed"] not in allowed_fans:
        old_fan = corrected["fan_speed"]
        corrected["fan_speed"] = constraints[
            "required_minimum_fan"
        ]
        adjustments.append(
            f"fan changed from {old_fan} to "
            f"{corrected['fan_speed']}"
        )

    ranges = _ranges_for_mode(
        constraints,
        corrected["hvac_mode"],
    )

    for zone, field in [
        ("window", "window_damper_pct"),
        ("interior", "interior_damper_pct"),
    ]:
        old_value = int(corrected[field])
        new_value = _clamp(old_value, ranges[zone])

        if new_value != old_value:
            corrected[field] = new_value
            adjustments.append(
                f"{zone} damper clamped from {old_value}% to "
                f"{new_value}% using range "
                f"[{ranges[zone]['min']}, {ranges[zone]['max']}]"
            )

    required_order = constraints["required_damper_order"]

    if (
        required_order == "INTERIOR_GTE_WINDOW"
        and corrected["interior_damper_pct"]
        < corrected["window_damper_pct"]
    ):
        old_window = corrected["window_damper_pct"]
        corrected["window_damper_pct"] = min(
            corrected["window_damper_pct"],
            corrected["interior_damper_pct"],
        )
        adjustments.append(
            f"window damper reduced from {old_window}% to "
            f"{corrected['window_damper_pct']}% to enforce "
            "INTERIOR_GTE_WINDOW"
        )

    elif (
        required_order == "WINDOW_GTE_INTERIOR"
        and corrected["window_damper_pct"]
        < corrected["interior_damper_pct"]
    ):
        old_interior = corrected["interior_damper_pct"]
        corrected["interior_damper_pct"] = min(
            corrected["interior_damper_pct"],
            corrected["window_damper_pct"],
        )
        adjustments.append(
            f"interior damper reduced from {old_interior}% to "
            f"{corrected['interior_damper_pct']}% to enforce "
            "WINDOW_GTE_INTERIOR"
        )

    if adjustments:
        corrected["reasoning"] = (
            f"{decision['reasoning']} "
            "Deterministic performance-range checks adjusted the "
            "final command: "
            + "; ".join(adjustments)
            + "."
        )

    return corrected, adjustments


def deterministic_fallback(
    window_meta: dict,
    interior_meta: dict,
    occupancy: int = 0,
    ambient_temp_c: float | None = None,
) -> dict:
    """Return the exact performance-aware recommendation."""
    constraints = build_constraint_metadata(
        window_meta=window_meta,
        interior_meta=interior_meta,
        occupancy=occupancy,
        ambient_temp_c=ambient_temp_c,
    )

    return {
        "hvac_mode": constraints["recommended_hvac_mode"],
        "window_damper_pct": constraints[
            "recommended_damper_pct"
        ]["window"],
        "interior_damper_pct": constraints[
            "recommended_damper_pct"
        ]["interior"],
        "fan_speed": constraints["required_minimum_fan"],
        "reasoning": (
            "The supervisory output remained invalid after one "
            "schema-correction attempt, so the deterministic "
            "fallback used the precomputed performance-aware "
            "recommendation."
        ),
    }
