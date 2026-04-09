"""Grader 1: Basic landing completion and delay scoring."""
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

    landed_count = sum(1 for ac in aircraft if ac.get("status") == "LANDED")
    delay_score = landed_count / total_aircraft

    delay_penalty = 0.0
    waiting_times = episode_log.get("holding_steps", {})

    if waiting_times:
        avg_delay = sum(waiting_times.values()) / len(waiting_times)
        delay_penalty = min(0.3, avg_delay * 0.01)

    # final score must be strictly between (0,1)
    return fix_score(delay_score - delay_penalty)
