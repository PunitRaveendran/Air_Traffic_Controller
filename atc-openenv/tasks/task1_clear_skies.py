"""Task 1: Clear Skies - Basic scenario with calm conditions."""
import random
from typing import List, Optional

from env.aircraft import Aircraft, AircraftType, Priority, AircraftStatus
from env.runway import Runway, RunwayStatus


class Task1ClearSkies:
    """5 aircraft, 2 runways (both OPEN), all fuel > 40min, no emergencies."""

    @staticmethod
    def get_config() -> dict:
        return {
            "max_steps": 20,
            "aircraft_count": 5,
            "runway_count": 2,
        }

    @staticmethod
    def create_aircraft(rng: Optional[random.Random] = None) -> List[Aircraft]:
        if rng is None:
            import random
            rng = random.Random(42)

        distances = [5, 8, 12, 15, 20]
        speeds = [180, 200, 220, 160, 190]
        types = [
            AircraftType.HEAVY,
            AircraftType.HEAVY,
            AircraftType.MEDIUM,
            AircraftType.MEDIUM,
            AircraftType.LIGHT,
        ]
        fuels = [60, 55, 50, 65, 45]

        aircraft = []
        for i in range(5):
            aircraft.append(Aircraft(
                id=f"AC{i+1}",
                aircraft_type=types[i],
                priority=Priority.NORMAL,
                status=AircraftStatus.INBOUND,
                distance_nm=float(distances[i]),
                speed_knots=float(speeds[i]),
                fuel_remaining_min=float(fuels[i]),
            ))
        return aircraft

    @staticmethod
    def create_runways() -> List[Runway]:
        return [
            Runway(id="RW01", status=RunwayStatus.OPEN),
            Runway(id="RW02", status=RunwayStatus.OPEN),
        ]

    @staticmethod
    def get_pending_arrivals() -> List[dict]:
        return []