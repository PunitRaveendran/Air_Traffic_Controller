"""Task 3: Full Emergency - Complex scenario with emergencies and runway closure."""
from __future__ import annotations
from typing import TYPE_CHECKING
import random
from typing import List, Optional

if TYPE_CHECKING:
    from env.atc_env import ATCEnv

from env.aircraft import Aircraft, AircraftType, Priority, AircraftStatus
from env.runway import Runway, RunwayStatus


class Task3FullEmergency:
    """15 aircraft, 2 runways, runway closes at step 3, emergency at step 5, continuous arrivals.

    DESIGN INVARIANTS:
    1. Every aircraft has enough fuel to survive the full 40-step episode (fuel >= 42).
       This prevents unavoidable fuel exhaustions from killing the fuel_score component.
    2. RW02 closes at step 3. On closure, aircraft ASSIGNED to RW02 are reset to HOLDING
       so the controller can immediately reassign them — no aircraft gets permanently stranded.
    3. Total aircraft = 23. With fixed runway throughput (sep=2 for RW01), up to 20 can land
       on RW01 in steps 3-40, plus 2 on RW02 in steps 1-2 = 22 possible landings.
    4. Queue measurement counts only INBOUND+HOLDING. New arrivals come in pairs of 2,
       so max queue spike = 2 per batch, well under the 12-aircraft threshold.
    """

    @staticmethod
    def get_config() -> dict:
        return {
            "max_steps": 40,
            "aircraft_count": 15,
            "runway_count": 2,
        }

    @staticmethod
    def create_aircraft(rng: Optional[random.Random] = None) -> List[Aircraft]:
        if rng is None:
            import random
            rng = random.Random(42)

        # All aircraft have fuel >= 42 so they can survive the full 40-step episode
        # even if they end up in queue position 15+ and don't land before the episode ends.
        # Previously AC14 (fuel=30), AC15 (fuel=28), AC11 (fuel=35), AC12 (fuel=32) etc.
        # would exhaust fuel mid-episode making fuel_score = 0 unavoidable.
        aircraft = [
            Aircraft(id="AC1",  aircraft_type=AircraftType.HEAVY,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=12.0, speed_knots=175, fuel_remaining_min=55.0),
            Aircraft(id="AC2",  aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=18.0, speed_knots=195, fuel_remaining_min=55.0),
            Aircraft(id="AC3",  aircraft_type=AircraftType.LIGHT,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=22.0, speed_knots=220, fuel_remaining_min=55.0),
            Aircraft(id="AC4",  aircraft_type=AircraftType.HEAVY,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=28.0, speed_knots=170, fuel_remaining_min=60.0),
            Aircraft(id="AC5",  aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=32.0, speed_knots=200, fuel_remaining_min=55.0),
            Aircraft(id="AC6",  aircraft_type=AircraftType.LIGHT,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=38.0, speed_knots=230, fuel_remaining_min=55.0),
            Aircraft(id="AC7",  aircraft_type=AircraftType.HEAVY,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=42.0, speed_knots=165, fuel_remaining_min=58.0),
            Aircraft(id="AC8",  aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=48.0, speed_knots=190, fuel_remaining_min=55.0),
            Aircraft(id="AC9",  aircraft_type=AircraftType.LIGHT,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=52.0, speed_knots=225, fuel_remaining_min=55.0),
            # AC10: will become EMERGENCY at step 5 — closest aircraft, lands early
            Aircraft(id="AC10", aircraft_type=AircraftType.HEAVY,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=8.0,  speed_knots=180, fuel_remaining_min=52.0),
            # BUG FIX: was fuel=35. Needs >= 42 to survive full 40 steps.
            Aircraft(id="AC11", aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=55.0, speed_knots=205, fuel_remaining_min=55.0),
            # BUG FIX: was fuel=32. Needs >= 42 to survive full 40 steps.
            Aircraft(id="AC12", aircraft_type=AircraftType.LIGHT,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=60.0, speed_knots=235, fuel_remaining_min=55.0),
            Aircraft(id="AC13", aircraft_type=AircraftType.HEAVY,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=15.0, speed_knots=172, fuel_remaining_min=55.0),
            # BUG FIX: was fuel=30. ETA ~20 steps + queue wait would exceed 30 fuel.
            Aircraft(id="AC14", aircraft_type=AircraftType.MEDIUM, priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=65.0, speed_knots=192, fuel_remaining_min=55.0),
            # BUG FIX: was fuel=28. ETA ~18 steps + queue wait would exceed 28 fuel.
            Aircraft(id="AC15", aircraft_type=AircraftType.LIGHT,  priority=Priority.NORMAL,
                     status=AircraftStatus.INBOUND, distance_nm=70.0, speed_knots=228, fuel_remaining_min=55.0),
        ]
        return aircraft

    @staticmethod
    def create_runways() -> List[Runway]:
        return [
            Runway(id="RW01", status=RunwayStatus.OPEN),
            Runway(id="RW02", status=RunwayStatus.OPEN),
        ]

    @staticmethod
    def get_pending_arrivals() -> List[dict]:
        """4 batches of 2 aircraft arriving at steps 4, 8, 12, 16.
        All have fuel >= 42 to survive until episode end at step 40.
        """
        arrivals = []
        for batch in range(4):
            step = 4 + batch * 4
            arrivals.extend([
                {
                    "arrival_step": step,
                    "id": f"AC{16 + batch * 2}",
                    "aircraft_type": "MEDIUM",
                    "priority": "NORMAL",
                    "distance_nm": 75.0 + batch * 5,
                    "speed_knots": 200,
                    # Fuel = 45 - batch*3. Arrives at step 4,8,12,16.
                    # Worst case: arrives at step 16 with fuel=33, episode ends at 40 (24 more steps). 33>24 ✓
                    "fuel_remaining_min": 45.0 - batch * 3,
                },
                {
                    "arrival_step": step,
                    "id": f"AC{17 + batch * 2}",
                    "aircraft_type": "LIGHT",
                    "priority": "NORMAL",
                    "distance_nm": 80.0 + batch * 5,
                    "speed_knots": 220,
                    # Fuel = 42 - batch*3. Worst: step 16 with fuel=30, 24 steps remain. 30>24 ✓
                    "fuel_remaining_min": 42.0 - batch * 3,
                },
            ])
        return arrivals

    def process_step_event(self, env: "ATCEnv", step: int) -> None:
        """Handle dynamic events: runway closure at step 3, emergency declaration at step 5."""
        if step == 3:
            for runway in env._runways:
                if runway.id == "RW02":
                    runway.close()
                    # BUG FIX (Bug 6): When RW02 closes, reset any aircraft ASSIGNED to it
                    # back to HOLDING so the controller can immediately reassign them to RW01.
                    # Without this, they are permanently stranded on the closed runway.
                    for aircraft in env._aircraft:
                        if (aircraft.assigned_runway == "RW02"
                                and aircraft.status == AircraftStatus.ASSIGNED):
                            aircraft.status = AircraftStatus.HOLDING
                            aircraft.assigned_runway = None
                    break

        if step == 5:
            for aircraft in env._aircraft:
                if aircraft.id == "AC10":
                    aircraft.priority = Priority.EMERGENCY
                    break