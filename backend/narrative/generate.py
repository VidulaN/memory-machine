"""
narrative/generate.py
Produces human-readable drought condition summaries from feature values
and model predictions. Used by both app.py and predict_live.py.
"""

from __future__ import annotations


# ── Public interface (called by app.py) ──────────────────────────────────────

def generate_narrative(
    country_or_features,
    score_or_prediction,
    avg_temp_or_probs=None,
    avg_precip: float | None = None,
    trend: float | None = None,
    prob: float | None = None,
) -> str:
    """
    Two call signatures are supported:

    1. Legacy (app.py):
       generate_narrative(country, score, avg_temp, avg_precip, trend, prob)

    2. Feature-dict (predict_live.py):
       generate_narrative(features_dict, prediction, probabilities_list)
    """
    if isinstance(country_or_features, dict):
        return _from_features(
            country_or_features,
            score_or_prediction,
            avg_temp_or_probs,
        )
    return _from_summary(
        country_or_features,
        score_or_prediction,
        avg_temp_or_probs,
        avg_precip,
        trend,
        prob,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _rain_sentence(avg_precip: float) -> str:
    if avg_precip < 200:
        return f"Annual precipitation is critically low at {round(avg_precip)} mm."
    if avg_precip < 600:
        return f"Rainfall is below average at {round(avg_precip)} mm per year."
    return f"Rainfall is relatively stable at {round(avg_precip)} mm annually."


def _temp_sentence(avg_temp: float) -> str:
    if avg_temp > 35:
        return (
            f"Temperatures are extreme, averaging {avg_temp:.1f}°C, "
            "accelerating moisture loss from soil."
        )
    if avg_temp > 28:
        return (
            f"Temperatures are elevated at {avg_temp:.1f}°C, "
            "increasing evaporative stress on crops."
        )
    return f"Temperatures average {avg_temp:.1f}°C — within a moderate range."


def _trend_sentence(trend: float) -> str:
    if trend > 3:
        return "Drought conditions have worsened significantly over the historical record."
    if trend < -3:
        return "Conditions have improved, with declining drought pressure over time."
    return "Drought patterns have remained broadly stable across the recorded period."


def _risk_word(prob: float) -> str:
    if prob >= 60:
        return "high"
    if prob >= 35:
        return "moderate"
    return "low"


def _from_summary(
    country: str,
    _score: float,
    avg_temp: float,
    avg_precip: float,
    trend: float,
    prob: float,
) -> str:
    risk = _risk_word(prob)
    return (
        f"{_rain_sentence(avg_precip)} "
        f"{_temp_sentence(avg_temp)} "
        f"{_trend_sentence(trend)} "
        f"The ML model rates overall drought probability at {prob:.0f}% "
        f"for {country}, indicating {risk} risk."
    )


def _from_features(
    features: dict,
    prediction: int,
    probabilities: list[float],
) -> str:
    precip_30 = features.get("precip_30day", 0)
    precip_7  = features.get("precip_7day",  0)
    temp_max  = features.get("temp_max",     0)
    temp_7    = features.get("temp_7day",    0)
    evap      = features.get("evapotranspiration", 0)
    precip    = features.get("precipitation", 0)

    drought_prob = probabilities[1] * 100 if len(probabilities) > 1 else 0
    risk         = _risk_word(drought_prob)
    status       = "**Drought conditions detected.**" if prediction == 1 else "No active drought detected."

    rain_line = (
        f"30-day accumulated rainfall is critically low at {precip_30:.1f} mm."
        if precip_30 < 5 else
        f"30-day rolling precipitation stands at {precip_30:.1f} mm."
    )

    temp_line = (
        f"Today's maximum temperature reached {temp_max:.1f}°C "
        f"(7-day mean {temp_7:.1f}°C), "
        + (
            "placing severe heat stress on crops."
            if temp_max > 35 else
            "within elevated but manageable bounds."
            if temp_max > 28 else
            "within a moderate range."
        )
    )

    evap_line = (
        f"Evapotranspiration is running at {evap:.2f} mm/day, "
        + (
            "far exceeding moisture inputs — soil water deficit is critical."
            if evap > 5 else
            "exceeding recent rainfall — monitor soil moisture closely."
            if evap > 3 else
            "broadly balanced with moisture inputs."
        )
    )

    return (
        f"{status} {rain_line} {temp_line} {evap_line} "
        f"Model confidence: {drought_prob:.0f}% drought probability ({risk} risk)."
    )
