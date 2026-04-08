"""Reward computation for ATC environment."""
from typing import Any, Dict


class RewardEvent:
    LANDING_SUCCESS = "LANDING_SUCCESS"
    DELAY_PER_STEP = "DELAY_PER_STEP"
    FUEL_CRITICAL_LATE = "FUEL_CRITICAL_LATE"
    FUEL_EXHAUSTED = "FUEL_EXHAUSTED"
    EMERGENCY_NOT_PRIORITIZED = "EMERGENCY_NOT_PRIORITIZED"
    SEPARATION_VIOLATION = "SEPARATION_VIOLATION"
    RUNWAY_CONFLICT = "RUNWAY_CONFLICT"
    HOLDING_PENALTY = "HOLDING_PENALTY"


def compute_reward(event: str, context: Dict[str, Any]) -> float:
    """Compute reward based on event and context."""
    reward_values = {
        RewardEvent.LANDING_SUCCESS: 1.0,
        RewardEvent.DELAY_PER_STEP: -0.05,
        RewardEvent.FUEL_CRITICAL_LATE: -2.0,
        RewardEvent.FUEL_EXHAUSTED: -10.0,
        RewardEvent.EMERGENCY_NOT_PRIORITIZED: -10.0,
        RewardEvent.SEPARATION_VIOLATION: -5.0,
        RewardEvent.RUNWAY_CONFLICT: -20.0,
        RewardEvent.HOLDING_PENALTY: -0.1,
    }

    base_reward = reward_values.get(event, 0.0)

    if event == RewardEvent.DELAY_PER_STEP:
        return base_reward * context.get("waiting_count", 1)

    if event == RewardEvent.FUEL_CRITICAL_LATE:
        steps_waited = context.get("steps_waited", 5)
        if steps_waited > 5:
            return base_reward * (steps_waited - 5)
        return 0.0

    if event == RewardEvent.EMERGENCY_NOT_PRIORITIZED:
        steps_waited = context.get("steps_waited", 0)
        if steps_waited > 3:
            return base_reward
        return 0.0

    return base_reward


def get_reward_breakdown() -> Dict[str, float]:
    """Return breakdown structure for rewards."""
    return {
        "landing_success": 0.0,
        "delay_penalty": 0.0,
        "fuel_critical_penalty": 0.0,
        "fuel_exhausted_penalty": 0.0,
        "emergency_penalty": 0.0,
        "separation_violation": 0.0,
        "runway_conflict": 0.0,
        "holding_penalty": 0.0,
    }