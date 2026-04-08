"""Runway models and operations."""
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class RunwayStatus(str, Enum):
    OPEN = "OPEN"
    REDUCED = "REDUCED"
    CLOSED = "CLOSED"


class Runway(BaseModel):
    id: str
    status: RunwayStatus = RunwayStatus.OPEN
    next_available_step: int = 0
    current_occupant: Optional[str] = None

    @property
    def separation_steps(self) -> int:
        """Separation steps required between aircraft."""
        return 2 if self.status == RunwayStatus.OPEN else 4

    @property
    def is_available(self) -> bool:
        """Check if runway is available for new assignment."""
        return self.status != RunwayStatus.CLOSED

    def occupy(self, aircraft_id: str, current_step: int) -> None:
        """Assign aircraft to runway and set next available time."""
        self.current_occupant = aircraft_id
        self.next_available_step = current_step + self.separation_steps

    def tick(self, current_step: int) -> None:
        """Update runway availability based on current step."""
        if (
            self.current_occupant is not None
            and current_step >= self.next_available_step
        ):
            self.current_occupant = None

    def close(self) -> None:
        """Close the runway."""
        self.status = RunwayStatus.CLOSED
        self.current_occupant = None

    def reduce(self) -> None:
        """Reduce runway capacity (increase separation)."""
        if self.status == RunwayStatus.OPEN:
            self.status = RunwayStatus.REDUCED