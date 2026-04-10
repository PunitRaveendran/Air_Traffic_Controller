"""Grader 1: Basic landing completion and delay scoring."""
from typing import Dict, Any

def grade(episode_log: Dict[str, Any]) -> float:
    """
    Score based on landing completion with delay penalty.

    score = landings_completed / total_aircraft
    penalty for avg delay
    """
    # Increased the margin to avoid rounding to 1.0000 or 0.0000
    MIN_SCORE = 0.001
    MAX_SCORE = 0.999

    if not episode_log:
        return MIN_SCORE

    aircraft = episode_log.get("aircraft", [])
    
    total_aircraft = len(aircraft)
    if total_aircraft == 0:
        return MIN_SCORE

    landed_count = sum(1 for ac in aircraft if ac.get("status") == "LANDED")

    delay_score = landed_count / total_aircraft

    delay_penalty = 0.0
    waiting_times = episode_log.get("holding_steps", {})
    if waiting_times:
        avg_delay = sum(waiting_times.values()) / len(waiting_times)
        delay_penalty = min(0.3, avg_delay * 0.01)

    final_score = max(MIN_SCORE, min(MAX_SCORE, delay_score - delay_penalty))
    return final_score
