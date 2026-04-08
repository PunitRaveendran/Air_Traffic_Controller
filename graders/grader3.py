"""Grader 3: Composite scoring for complex scenarios."""
from typing import Dict, Any


def grade(episode_log: Dict[str, Any]) -> float:
    """
    Composite score:
    - 25% landings completed
    - 25% emergency handling
    - 25% no fuel exhaustions (proportional, not binary)
    - 25% queue never exceeds 12 aircraft (queue = INBOUND+HOLDING only)
    """
    if not episode_log:
        return 0.0

    aircraft = episode_log.get("aircraft", [])
    total_aircraft = len(aircraft)
    if total_aircraft == 0:
        return 0.0

    # 25%: Landings completed
    landed = sum(1 for ac in aircraft if ac.get("status") == "LANDED")
    landing_score = landed / total_aircraft

    # 25%: Emergency aircraft all land
    emergency_count = sum(1 for ac in aircraft if ac.get("priority") == "EMERGENCY")
    emergency_landed = sum(
        1 for ac in aircraft
        if ac.get("priority") == "EMERGENCY" and ac.get("status") == "LANDED"
    )
    emergency_score = emergency_landed / emergency_count if emergency_count > 0 else 1.0

    # 25%: Fuel exhaustion score (proportional — was binary 0/1, too punishing)
    # Previously: any exhaustion = 0. Now: proportional penalty per exhausted aircraft.
    # This way a well-played episode with 1 unlucky exhaustion still gets partial credit.
    fuel_exhausted = episode_log.get("fuel_exhausted_count", 0)
    fuel_score = max(0.0, 1.0 - fuel_exhausted / max(1, total_aircraft))

    # 25%: Queue management (queue = INBOUND+HOLDING only, per env fix Bug 3)
    max_queue = episode_log.get("max_queue_size", 0)
    queue_score = 1.0 if max_queue <= 12 else max(0.0, 1.0 - (max_queue - 12) / 20)

    final_score = 0.25 * landing_score + 0.25 * emergency_score + 0.25 * fuel_score + 0.25 * queue_score
    return max(0.0, min(1.0, final_score))