"""Grader 3: Composite scoring for complex scenarios."""
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

    # 25%: Landings completed
    landed = sum(1 for ac in aircraft if ac.get("status") == "LANDED")
    landing_score = landed / total_aircraft

    # 25%: Emergency handling
    emergency_count = sum(1 for ac in aircraft if ac.get("priority") == "EMERGENCY")
    emergency_landed = sum(
        1 for ac in aircraft
        if ac.get("priority") == "EMERGENCY" and ac.get("status") == "LANDED"
    )
    emergency_score = emergency_landed / emergency_count if emergency_count > 0 else 1.0

    # 25%: Fuel exhaustion
    fuel_exhausted = episode_log.get("fuel_exhausted_count", 0)
    fuel_score = max(0.0, 1.0 - fuel_exhausted / max(1, total_aircraft))

    # 25%: Queue management
    max_queue = episode_log.get("max_queue_size", 0)
    queue_score = 1.0 if max_queue <= 12 else max(0.0, 1.0 - (max_queue - 12) / 20)

    # final score must be strictly between (0,1)
    final_score = (
        0.25 * landing_score
        + 0.25 * emergency_score
        + 0.25 * fuel_score
        + 0.25 * queue_score
    )

    return fix_score(final_score)
