"""Aircraft models and simulation logic."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AircraftType(str, Enum):
    HEAVY = "HEAVY"
    MEDIUM = "MEDIUM"
    LIGHT = "LIGHT"


class Priority(str, Enum):
    NORMAL = "NORMAL"
    FUEL_CRITICAL = "FUEL_CRITICAL"
    EMERGENCY = "EMERGENCY"


class AircraftStatus(str, Enum):
    INBOUND = "INBOUND"
    HOLDING = "HOLDING"
    ASSIGNED = "ASSIGNED"
    LANDING = "LANDING"
    LANDED = "LANDED"


class Aircraft(BaseModel):
    id: str
    aircraft_type: AircraftType
    priority: Priority = Priority.NORMAL
    status: AircraftStatus = AircraftStatus.INBOUND
    distance_nm: float
    speed_knots: float
    fuel_remaining_min: float
    assigned_runway: Optional[str] = None
    sequence_position: Optional[int] = None

    @property
    def eta_steps(self) -> int:
        """Estimated time of arrival in simulation steps (1 step = 1 minute)."""
        if self.speed_knots <= 0:
            return 999
        return max(1, int(self.distance_nm / self.speed_knots * 60))

    @property
    def is_fuel_critical(self) -> bool:
        return self.fuel_remaining_min < 15.0

    @property
    def is_emergency(self) -> bool:
        return self.priority == Priority.EMERGENCY

    def tick(self) -> None:
        """Advance simulation by one step - update distance and fuel."""
        if self.status == AircraftStatus.LANDED:
            return
        if self.status == AircraftStatus.LANDING:
            self.distance_nm = max(0.0, self.distance_nm - 3.0)
        elif self.speed_knots > 0:
            self.distance_nm = max(0.0, self.distance_nm - self.speed_knots / 60.0)
        self.fuel_remaining_min = max(0.0, self.fuel_remaining_min - 1.0)
        if self.fuel_remaining_min < 15.0 and self.priority == Priority.NORMAL:
            self.priority = Priority.FUEL_CRITICAL