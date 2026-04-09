"""Grader 2: Weighted scoring for delay, fuel-critical handling, and separation compliance."""
from typing import Dict, Any

def fix_score(score: float) -> float:
    return max(0.01, min(0.99, float(score)))


def grade(episode_log: Dict[str, Any]) -> float:
    if not episode_log:
        return 0.01

    aircraft = episode_log.get("aircraft", [])
    total_aircraft = len(aircraft)

    if total_aircraft == 0:
        return 0.01

    # 40%: Landings completed
    landed = sum(1 for ac in aircraft if ac.get("status") == "LANDED")
    delay_score = landed / total_aircraft

    # 40%: Fuel-critical handling
    initial_fc_ids = set(episode_log.get("initial_fuel_critical_ids", []))

    if initial_fc_ids:
        fuel_critical_count = len(initial_fc_ids)
        fuel_landed = sum(
            1 for ac in aircraft
            if ac.get("id") in initial_fc_ids and ac.get("status") == "LANDED"
        )
    else:
        fuel_critical_count = sum(
            1 for ac in aircraft if ac.get("priority") == "FUEL_CRITICAL"
        )
        fuel_landed = sum(
            1 for ac in aircraft
            if ac.get("priority") == "FUEL_CRITICAL" and ac.get("status") == "LANDED"
        )

    fuel_score = fuel_landed / fuel_critical_count if fuel_critical_count > 0 else 1.0

    # 20%: Separation compliance
    separation_violations = episode_log.get("separation_violations", 0)
    max_violations = 10
    separation_score = max(0.0, 1.0 - separation_violations / max_violations)

    # final score must be strictly between (0,1)
    final_score = 0.4 * delay_score + 0.4 * fuel_score + 0.2 * separation_score
    return fix_score(final_score)
