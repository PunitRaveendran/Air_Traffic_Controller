"""Task 2: Fuel Pressure - Increased traffic with fuel-critical aircraft."""
import random
from typing import List, Optional

from env.aircraft import Aircraft, AircraftType, Priority, AircraftStatus
from env.runway import Runway, RunwayStatus


class Task2FuelPressure:
    """10 aircraft, 2 runways (one REDUCED), fuel-critical aircraft, new arrivals at step 5.

    DESIGN INVARIANT: Every aircraft must be physically saveable with optimal play.
    - AC3: fuel=16, dist=30, speed=220 → ETA=8.2 steps. fuel(16) > ETA(8.2) ✓
    - AC6: fuel=22, dist=45, speed=210 → ETA=12.9 steps. fuel(22) > ETA(12.9) ✓
    Both are FUEL_CRITICAL priority at spawn, requiring expedite handling.
    """

    @staticmethod
    def get_config() -> dict:
        return {
            "max_steps": 30,
            "aircraft_count": 10,
            "runway_count": 2,
        }

    @staticmethod
    def create_aircraft(rng: Optional[random.Random] = None) -> List[Aircraft]:
        if rng is None:
            import random
            rng = random.Random(42)

        aircraft = [
            Aircraft(id="AC1", aircraft_type=AircraftType.HEAVY, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=15.0, speed_knots=180, fuel_remaining_min=50.0),
            Aircraft(id="AC2", aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=25.0, speed_knots=200, fuel_remaining_min=45.0),
            # BUG FIX: was fuel=10. ETA=8.2, so fuel=16 gives safe buffer.
            Aircraft(id="AC3", aircraft_type=AircraftType.LIGHT, priority=Priority.FUEL_CRITICAL,
                     status=AircraftStatus.INBOUND, distance_nm=30.0, speed_knots=220, fuel_remaining_min=16.0),
            Aircraft(id="AC4", aircraft_type=AircraftType.HEAVY, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=35.0, speed_knots=170, fuel_remaining_min=55.0),
            Aircraft(id="AC5", aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=40.0, speed_knots=190, fuel_remaining_min=25.0),
            # BUG FIX: was fuel=11. ETA=12.9, so fuel=22 gives safe buffer.
            Aircraft(id="AC6", aircraft_type=AircraftType.LIGHT, priority=Priority.FUEL_CRITICAL,
                     status=AircraftStatus.INBOUND, distance_nm=45.0, speed_knots=210, fuel_remaining_min=22.0),
            Aircraft(id="AC7", aircraft_type=AircraftType.HEAVY, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=20.0, speed_knots=175, fuel_remaining_min=48.0),
            Aircraft(id="AC8", aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=50.0, speed_knots=195, fuel_remaining_min=35.0),
            Aircraft(id="AC9", aircraft_type=AircraftType.LIGHT, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=55.0, speed_knots=230, fuel_remaining_min=30.0),
            Aircraft(id="AC10", aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=12.0, speed_knots=185, fuel_remaining_min=42.0),
        ]
        return aircraft

    @staticmethod
    def create_runways() -> List[Runway]:
        return [
            Runway(id="RW01", status=RunwayStatus.OPEN),
            Runway(id="RW02", status=RunwayStatus.REDUCED),
        ]

    @staticmethod
    def get_pending_arrivals() -> List[dict]:
        return [
            {
                "arrival_step": 5,
                "id": "AC11",
                "aircraft_type": "MEDIUM",
                "priority": "NORMAL",
                "distance_nm": 60.0,
                "speed_knots": 200,
                "fuel_remaining_min": 40.0,
            },
            {
                "arrival_step": 5,
                "id": "AC12",
                "aircraft_type": "LIGHT",
                "priority": "NORMAL",
                "distance_nm": 65.0,
                "speed_knots": 215,
                "fuel_remaining_min": 38.0,
            },
        ]