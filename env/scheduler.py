"""Wake turbulence separation and sequencing rules."""
from typing import List, Optional, Tuple

from env.aircraft import Aircraft, AircraftType
from env.runway import Runway, RunwayStatus


class WakeTurbulenceRule:
    """Separation requirements based on wake turbulence categories."""

    SEPARATION_NM = {
        (AircraftType.HEAVY, AircraftType.HEAVY): 4.0,
        (AircraftType.HEAVY, AircraftType.MEDIUM): 5.0,
        (AircraftType.HEAVY, AircraftType.LIGHT): 6.0,
        (AircraftType.MEDIUM, AircraftType.LIGHT): 3.0,
        (AircraftType.MEDIUM, AircraftType.HEAVY): 4.0,
        (AircraftType.LIGHT, AircraftType.HEAVY): 4.0,
        (AircraftType.LIGHT, AircraftType.MEDIUM): 2.0,
    }

    @classmethod
    def get_separation(cls, lead_type: AircraftType, trail_type: AircraftType) -> float:
        """Get required separation in nautical miles."""
        return cls.SEPARATION_NM.get(
            (lead_type, trail_type),
            cls.SEPARATION_NM.get((trail_type, lead_type), 2.0)
        )

    @classmethod
    def can_land(
        cls,
        aircraft: Aircraft,
        runway: Runway,
        last_landed_type: Optional[AircraftType],
        current_step: int,
    ) -> bool:
        """Check if aircraft can land given wake turbulence rules."""
        if not runway.is_available:
            return False
        if current_step < runway.next_available_step:
            return False
        if last_landed_type is None:
            return True
        separation_nm = cls.get_separation(last_landed_type, aircraft.aircraft_type)
        return aircraft.distance_nm <= separation_nm + 5


class Scheduler:
    """Manages aircraft sequencing and action validation."""

    @staticmethod
    def get_sequence(aircraft_list: List[Aircraft]) -> List[Aircraft]:
        """Sort aircraft by priority, then fuel, then ETA."""
        def sort_key(a: Aircraft):
            priority_order = {
                "EMERGENCY": 0,
                "FUEL_CRITICAL": 1,
                "NORMAL": 2,
            }
            return (
                priority_order.get(a.priority.value, 2),
                a.fuel_remaining_min,
                a.eta_steps,
            )

        return sorted(aircraft_list, key=sort_key)

    @staticmethod
    def validate_action(
        action: dict, state: dict
    ) -> Tuple[bool, str]:
        """Validate an action against current state."""
        aircraft_id = action.get("aircraft_id")
        action_type = action.get("action_type")

        aircraft_map = {ac["id"]: ac for ac in state.get("aircraft", [])}
        runway_map = {rw["id"]: rw for rw in state.get("runways", [])}

        if aircraft_id not in aircraft_map:
            return False, f"Aircraft {aircraft_id} not found"

        aircraft = aircraft_map[aircraft_id]

        if aircraft["status"] == "LANDED":
            return False, f"Aircraft {aircraft_id} already landed"

        if aircraft["status"] == "LANDING":
            return False, f"Aircraft {aircraft_id} is landing"

        if action_type == "assign":
            runway_id = action.get("runway_id")
            if not runway_id:
                return False, "assign action requires runway_id"
            if runway_id not in runway_map:
                return False, f"Runway {runway_id} not found"
            if runway_map[runway_id]["status"] == "CLOSED":
                return False, f"Runway {runway_id} is closed"
            return True, "valid"
        elif action_type == "hold":
            return True, "valid"
        elif action_type == "expedite":
            return True, "valid"
        else:
            return False, f"Unknown action type: {action_type}"