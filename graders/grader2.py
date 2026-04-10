"""Grader 2: Weighted scoring for delay, fuel-critical handling, and separation compliance."""
from typing import Dict, Any


def grade(episode_log: Dict[str, Any]) -> float:
    """
    Weighted score: 40% delay, 40% fuel-critical handling, 20% separation compliance.

    FIX (Bug 8): fuel_critical_count now uses 'initial_fuel_critical_ids' passed from the
    environment, which records aircraft that were FUEL_CRITICAL at spawn time.
    Previously it used final fuel_remaining_min < 15, which includes any aircraft that
    naturally depleted fuel over a long episode — massively inflating the denominator
    and making fuel_score collapse to near-zero even with perfect play.
    """
    # Updated bounds to guarantee scores are strictly between (0, 1) at 4 decimal places
    MIN_SCORE = 0.001
    MAX_SCORE = 0.999

    if not episode_log:
        return MIN_SCORE

    aircraft = episode_log.get("aircraft", [])
    total_aircraft = len(aircraft)
    if total_aircraft == 0:
        return MIN_SCORE

    # 40%: Landings completed
    landed = sum(1 for ac in aircraft if ac.get("status") == "LANDED")
    delay_score = landed / total_aircraft

    # 40%: Fuel-critical aircraft all land
    # Use initial_fuel_critical_ids if available (set at episode start before any fuel depletion).
    # Fall back to priority flag only (not fuel_remaining_min) to avoid counting aircraft
    # that aged into fuel-critical status during a long episode.
    initial_fc_ids = set(episode_log.get("initial_fuel_critical_ids", []))
    if initial_fc_ids:
        fuel_critical_count = len(initial_fc_ids)
        fuel_landed = sum(
            1 for ac in aircraft
            if ac.get("id") in initial_fc_ids and ac.get("status") == "LANDED"
        )
    else:
        # Fallback: count only aircraft whose priority is FUEL_CRITICAL (set at spawn
        # or upgraded during tick — but not using final fuel level which degrades over time)
        fuel_critical_count = sum(
            1 for ac in aircraft if ac.get("priority") == "FUEL_CRITICAL"
        )
        fuel_landed = sum(
            1 for ac in aircraft
            if ac.get("priority") == "FUEL_CRITICAL" and ac.get("status") == "LANDED"
        )
    fuel_score = fuel_landed / fuel_critical_count if fuel_critical_count > 0 else 1.0

    # 20%: Separation compliance (duplicate sequence positions = violations)
    separation_violations = episode_log.get("separation_violations", 0)
    max_violations = 10
    separation_score = max(0.0, 1.0 - separation_violations / max_violations)

    final_score = 0.4 * delay_score + 0.4 * fuel_score + 0.2 * separation_score
    
    # Enforce safe boundaries on the final returned score
    return max(MIN_SCORE, min(MAX_SCORE, final_score))
