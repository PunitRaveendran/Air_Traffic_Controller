"""
Professional ATC System - Physics & Logic Core
Real-time movement, fuel consumption, conflict detection, and snapshot history for animation.
"""

import math
import heapq
import random
import threading
import time as time_module
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

try:
    import winsound
    SOUND_AVAILABLE = True
except ImportError:
    SOUND_AVAILABLE = False

# ============ CONSTANTS ============
NM_TO_KM = 1.852
FUEL_BURN_RATE_APPROACH = 0.8   # Min/min during approach
FUEL_BURN_RATE_HOLD    = 1.5    # Min/min during hold
MIN_SEPARATION_NM       = 3.0
MIN_VERTICAL_SEP_FT     = 1000

SNAPSHOT_INTERVAL_STEPS = 10   # Capture state every N simulation steps for animation playback

# Wake Turbulence Separation (Minutes between landings)
WAKE_SEP = {
    ("HEAVY",  "HEAVY"):  2.0,
    ("HEAVY",  "MEDIUM"): 2.5,
    ("HEAVY",  "LIGHT"):  3.0,
    ("MEDIUM", "HEAVY"):  1.5,
    ("MEDIUM", "MEDIUM"): 2.0,
    ("MEDIUM", "LIGHT"):  2.5,
    ("LIGHT",  "HEAVY"):  1.5,
    ("LIGHT",  "MEDIUM"): 2.0,
    ("LIGHT",  "LIGHT"):  2.0,
}


class Status(Enum):
    APPROACHING    = "APPROACHING"
    HOLDING        = "HOLDING"
    FINAL_APPROACH = "FINAL_APPROACH"
    LANDING        = "LANDING"
    LANDED         = "LANDED"
    TAXIING        = "TAXIING"
    EXITED         = "EXITED"
    CRASHED        = "CRASHED"
    GO_AROUND      = "GO_AROUND"


class Priority(Enum):
    MAYDAY        = 10000
    PAN_PAN       = 5000
    CRITICAL_FUEL = 2000
    LOW_FUEL      = 1000
    NORMAL        = 100


@dataclass
class Weather:
    condition:  str   = "CLEAR"
    wind_speed: float = 5.0
    wind_dir:   float = 270.0
    visibility: float = 10.0

    def get_separation_mult(self) -> float:
        if self.condition == "STORM": return 1.5
        if self.condition == "RAIN":  return 1.2
        return 1.0


@dataclass
class Aircraft:
    id:            str
    fuel_remaining: float   # Minutes
    wake_category:  str
    is_emergency:   bool

    # Physics State
    distance_nm:  float = 40.0   # Distance from threshold
    altitude_ft:  float = 4000.0
    speed_kts:    float = 250.0
    heading_deg:  float = 90.0   # Fixed inbound heading (aircraft approach from this direction)

    # Scheduling
    arrival_time:    float = 0.0
    landing_time:    Optional[float] = None
    assigned_runway: Optional[str]  = None
    wait_time:       float = 0.0

    # Status
    status:         Status = Status.APPROACHING
    priority_score: float  = 0.0

    # Internal
    initial_fuel: float = field(init=False)
    initial_dist: float = field(init=False)

    def __post_init__(self):
        self.initial_fuel = self.fuel_remaining
        self.initial_dist = self.distance_nm
        self.wake_category = self.wake_category.upper()
        # Give each aircraft a stable, unique inbound heading (spread around compass)
        rng = random.Random(hash(self.id))
        self.heading_deg = rng.uniform(0, 360)
        self.calculate_priority()

    @property
    def eta_minutes(self) -> float:
        if self.speed_kts > 0:
            return (self.distance_nm / self.speed_kts) * 60.0
        return float("inf")

    def calculate_priority(self):
        score = 0
        if self.is_emergency:            score += Priority.MAYDAY.value
        elif self.fuel_remaining < 10:   score += Priority.CRITICAL_FUEL.value
        elif self.fuel_remaining < 20:   score += Priority.LOW_FUEL.value

        score += max(0, (100 - self.eta_minutes)) * 10

        if self.wake_category == "HEAVY":
            score += 50
        self.priority_score = score

    def update_physics(self, dt_seconds: float, weather: Weather):
        """Advance aircraft state by dt_seconds."""
        if self.status in (Status.EXITED, Status.CRASHED):
            return

        dt_min = dt_seconds / 60.0
        weather_factor = 1.0 + (weather.wind_speed / 100.0)

        # --- Fuel Consumption ---
        burn = FUEL_BURN_RATE_HOLD if self.status == Status.HOLDING else FUEL_BURN_RATE_APPROACH
        self.fuel_remaining -= burn * dt_min * weather_factor

        if self.fuel_remaining <= 0:
            self.fuel_remaining = 0.0
            self.status = Status.CRASHED
            return

        # --- Movement ---
        if self.status == Status.APPROACHING:
            move = (self.speed_kts / 60.0) * dt_min
            self.distance_nm = max(0.0, self.distance_nm - move)
            # Gradually descend toward pattern altitude
            target_alt = max(500.0, self.distance_nm * 300.0)
            if self.altitude_ft > target_alt:
                self.altitude_ft = max(target_alt, self.altitude_ft - 500.0 * dt_min)

        elif self.status == Status.HOLDING:
            # Orbit: heading rotates, distance stays roughly constant (slight variation)
            self.heading_deg = (self.heading_deg + 6.0 * dt_min) % 360.0

        elif self.status == Status.FINAL_APPROACH:
            self.speed_kts    = max(130.0, self.speed_kts - 5.0 * dt_min)
            move = (self.speed_kts / 60.0) * dt_min
            self.distance_nm  = max(0.0, self.distance_nm - move)
            self.altitude_ft  = max(0.0, self.altitude_ft - 400.0 * dt_min)

        elif self.status == Status.LANDING:
            # Decelerating roll-out
            self.speed_kts    = max(0.0, self.speed_kts - 20.0 * dt_min)
            self.distance_nm  = max(0.0, self.distance_nm - (self.speed_kts / 60.0) * dt_min)
            self.altitude_ft  = 0.0

        elif self.status == Status.TAXIING:
            self.altitude_ft = 0.0
            self.speed_kts   = 20.0

        # Auto-upgrade to emergency on critically low fuel
        if self.fuel_remaining < 5.0 and not self.is_emergency:
            self.is_emergency = True

        self.calculate_priority()

    def get_cartesian_pos(self) -> Tuple[float, float]:
        """
        Radar position: aircraft approaches from heading_deg toward (0,0).
        As distance_nm decreases the blip moves toward the airport centre.
        """
        rad = math.radians(self.heading_deg)
        x = self.distance_nm * math.sin(rad)
        y = self.distance_nm * math.cos(rad)
        return x, y

    def snapshot(self) -> Dict:
        """Return a lightweight state snapshot for animation playback."""
        x, y = self.get_cartesian_pos()
        return {
            "id":           self.id,
            "x":            x,
            "y":            y,
            "fuel":         self.fuel_remaining,
            "fuel_pct":     (self.fuel_remaining / self.initial_fuel * 100) if self.initial_fuel > 0 else 0,
            "distance":     self.distance_nm,
            "altitude":     self.altitude_ft,
            "speed":        self.speed_kts,
            "heading":      self.heading_deg,
            "status":       self.status.value,
            "priority_score": self.priority_score,
            "wake":         self.wake_category,
            "emergency":    self.is_emergency,
            "critical":     self.fuel_remaining < 10.0,
            "low_fuel":     self.fuel_remaining < 20.0,
            "runway":       self.assigned_runway,
            "landing_time": self.landing_time,
            "wait_time":    self.wait_time,
        }


@dataclass
class RunwaySlot:
    start_time: float   # Simulation seconds
    end_time:   float
    aircraft_id: str


class Runway:
    def __init__(self, name: str, heading: float):
        self.name          = name
        self.heading       = heading
        self.is_active     = True
        self.slots: List[RunwaySlot] = []
        self.next_available: float = 0.0
        self.last_wake:    Optional[str] = None

    def get_next_slot(self, ac: Aircraft, current_time: float,
                      weather: Weather) -> Tuple[float, float]:
        if not self.is_active:
            return float("inf"), 0.0

        base_sep = WAKE_SEP.get((self.last_wake or "MEDIUM", ac.wake_category), 2.0)
        sep_min  = base_sep * weather.get_separation_mult()
        sep_secs = sep_min * 60.0

        slot_start = max(current_time, self.next_available)
        return slot_start, sep_secs

    def book_slot(self, ac: Aircraft, start: float, duration_secs: float, sep_secs: float):
        self.slots.append(RunwaySlot(start, start + duration_secs, ac.id))
        self.next_available = start + sep_secs
        self.last_wake      = ac.wake_category

    def current_occupant(self, sim_time: float) -> Optional[str]:
        for slot in self.slots:
            if slot.start_time <= sim_time <= slot.end_time:
                return slot.aircraft_id
        return None

    def runway_progress(self, sim_time: float) -> Optional[float]:
        """0.0 → 1.0 progress of current occupant through rollout."""
        for slot in self.slots:
            if slot.start_time <= sim_time <= slot.end_time:
                dur = slot.end_time - slot.start_time
                return (sim_time - slot.start_time) / dur if dur > 0 else 0.0
        return None


class ConflictDetector:
    @staticmethod
    def check_conflicts(ac_list: List[Aircraft]) -> List[Tuple[str, str, float]]:
        conflicts = []
        active = [a for a in ac_list
                  if a.status not in (Status.EXITED, Status.CRASHED, Status.LANDED, Status.TAXIING)]
        for i, a1 in enumerate(active):
            for a2 in active[i + 1:]:
                p1, p2 = a1.get_cartesian_pos(), a2.get_cartesian_pos()
                dist_nm = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                v_dist  = abs(a1.altitude_ft - a2.altitude_ft)
                if dist_nm < MIN_SEPARATION_NM and v_dist < MIN_VERTICAL_SEP_FT:
                    conflicts.append((a1.id, a2.id, dist_nm))
        return conflicts


class ATCScheduler:
    LANDING_DURATION_SECS = 90.0  # Seconds for one landing + roll-out

    def __init__(self, task_level: int = 1):
        self.task_level   = task_level
        self.runways      = [Runway("RWY-09L", 90), Runway("RWY-27R", 270)]
        self.weather      = Weather("CLEAR", 5, 270, 10)
        self.current_time = 0.0        # Simulation seconds
        self.aircraft: List[Aircraft] = []
        self.landed:   List[Aircraft] = []
        self.events:   List[str]      = []
        self.metrics   = {"crashes": 0, "efficiency": 0.0, "total_delay": 0.0}

        if task_level == 2:
            self.weather = Weather("RAIN", 15, 300, 5)
        elif task_level == 3:
            self.weather = Weather("STORM", 25, 290, 2)
            self.runways[1].is_active = False

    def add_aircraft(self, ac: Aircraft):
        self.aircraft.append(ac)

    def schedule_step(self, dt: float):
        """Advance simulation by dt seconds."""
        self.current_time += dt

        # 1. Update physics
        for ac in self.aircraft:
            ac.update_physics(dt, self.weather)

        # 2. Conflict detection
        for c in ConflictDetector.check_conflicts(self.aircraft):
            msg = f"CONFLICT: {c[0]} & {c[1]} ({c[2]:.1f}nm)"
            if not self.events or self.events[-1] != msg:
                self.events.append(msg)

        # 3. Assign unscheduled aircraft to runways
        schedulable = [
            a for a in self.aircraft
            if a.status not in (Status.EXITED, Status.CRASHED, Status.LANDED,
                                Status.TAXIING, Status.FINAL_APPROACH,
                                Status.LANDING)
            and a.landing_time is None
        ]
        schedulable.sort(key=lambda x: x.priority_score, reverse=True)

        for ac in schedulable:
            best_rwy, best_time, best_sep = None, float("inf"), 0.0
            for rwy in self.runways:
                t, s = rwy.get_next_slot(ac, self.current_time, self.weather)
                if t < best_time:
                    best_time, best_sep, best_rwy = t, s, rwy

            if best_rwy:
                ac.assigned_runway = best_rwy.name
                ac.landing_time    = best_time
                ac.wait_time       = max(0.0, best_time - self.current_time)
                ac.status          = Status.FINAL_APPROACH
                best_rwy.book_slot(ac, best_time, self.LANDING_DURATION_SECS, best_sep)

                phrase = "EXPEDITE LANDING" if ac.is_emergency else "CLEARED TO LAND"
                self.events.append(f"{ac.id}: {phrase} {best_rwy.name}")

                if ac.is_emergency and SOUND_AVAILABLE:
                    try:
                        t = threading.Thread(target=lambda: winsound.Beep(1000, 300), daemon=True)
                        t.start()
                    except Exception:
                        pass

        # 4. State transitions driven by landing_time
        for ac in self.aircraft:
            if ac.landing_time is None:
                continue
            lt = ac.landing_time
            t  = self.current_time
            if ac.status == Status.FINAL_APPROACH and t >= lt:
                ac.status = Status.LANDING
            elif ac.status == Status.LANDING and t >= lt + self.LANDING_DURATION_SECS * 0.5:
                ac.status    = Status.LANDED
                ac.speed_kts = 30.0
            elif ac.status == Status.LANDED and t >= lt + self.LANDING_DURATION_SECS:
                ac.status = Status.TAXIING
            elif ac.status == Status.TAXIING and t >= lt + self.LANDING_DURATION_SECS * 2:
                ac.status = Status.EXITED
                if ac not in self.landed:
                    self.landed.append(ac)

        # 5. Metrics
        self.metrics["crashes"] = sum(1 for a in self.aircraft if a.status == Status.CRASHED)
        self.metrics["total_delay"] = sum(a.wait_time for a in self.aircraft if a.wait_time > 0)
        processed = len(self.landed) + self.metrics["crashes"]
        if processed > 0:
            on_time = sum(1 for a in self.landed if a.wait_time < 300)   # < 5 min wait
            self.metrics["efficiency"] = (on_time / processed) * 100.0

    def get_scores(self) -> Dict:
        n = len(self.aircraft)
        if n == 0:
            return {"score": 0.0, "reward": 0.0}
        crash_pen = self.metrics["crashes"] / n
        delay_pen = min(1.0, self.metrics["total_delay"] / (n * 1800))   # 30 min baseline
        score = max(0.0, 1.0 - (crash_pen * 0.5 + delay_pen * 0.5))
        if self.task_level == 3:
            score *= 0.9
        return {"score": round(score, 4), "reward": round(score * 10.0, 2)}

    def snapshot(self) -> Dict:
        """Capture full state for a single animation frame."""
        return {
            "sim_time":  self.current_time,
            "aircraft":  [ac.snapshot() for ac in self.aircraft],
            "events":    list(self.events[-8:]),
            "metrics":   dict(self.metrics),
            "runways": [
                {
                    "name":      rw.name,
                    "is_active": rw.is_active,
                    "occupant":  rw.current_occupant(self.current_time),
                    "progress":  rw.runway_progress(self.current_time),
                    "slots": [{"start": s.start_time, "end": s.end_time, "id": s.aircraft_id}
                              for s in rw.slots],
                }
                for rw in self.runways
            ],
        }


# ─────────────────────────────────────────────
# Task Scenario Definitions
# ─────────────────────────────────────────────

def get_task_data(level: int) -> List[Dict]:
    """Return aircraft spawn data for each task level."""
    data = []
    if level == 1:
        configs = [
            ("FL101", 55, "HEAVY",  False, 25),
            ("FL102", 48, "MEDIUM", False, 32),
            ("FL103", 60, "LIGHT",  False, 18),
            ("FL104", 52, "MEDIUM", False, 40),
            ("FL105", 45, "HEAVY",  False, 15),
        ]
    elif level == 2:
        configs = [
            ("FL201", 50, "HEAVY",  False, 20),
            ("FL202", 45, "MEDIUM", False, 28),
            ("FL203", 16, "LIGHT",  False, 30),   # FUEL_CRITICAL (ETA ~8 min, fuel=16)
            ("FL204", 55, "HEAVY",  False, 35),
            ("FL205", 25, "MEDIUM", False, 40),
            ("FL206", 22, "LIGHT",  False, 45),   # FUEL_CRITICAL (ETA ~13 min, fuel=22)
            ("FL207", 48, "HEAVY",  False, 20),
            ("FL208", 35, "MEDIUM", False, 50),
            ("FL209", 30, "LIGHT",  False, 55),
            ("FL210", 42, "MEDIUM", False, 12),
        ]
    else:
        configs = [
            ("FL301", 55, "HEAVY",  False, 12),
            ("FL302", 55, "MEDIUM", False, 18),
            ("FL303", 55, "LIGHT",  False, 22),
            ("FL304", 60, "HEAVY",  False, 28),
            ("FL305", 55, "MEDIUM", False, 32),
            ("FL306", 55, "LIGHT",  False, 38),
            ("FL307", 58, "HEAVY",  False, 42),
            ("FL308", 55, "MEDIUM", False, 48),
            ("FL309", 55, "LIGHT",  False, 52),
            ("FL310", 52, "HEAVY",  True,  8),    # EMERGENCY
            ("FL311", 55, "MEDIUM", False, 55),
            ("FL312", 55, "LIGHT",  False, 60),
            ("FL313", 55, "HEAVY",  False, 15),
            ("FL314", 55, "MEDIUM", False, 65),
            ("FL315", 55, "LIGHT",  False, 70),
        ]
    return [
        {"id": c[0], "fuel": c[1], "wake": c[2], "emerg": c[3], "dist": c[4]}
        for c in configs
    ]


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run_simulation(level: int, sim_duration_secs: float = 3600.0,
                   dt: float = 1.0) -> Dict:
    """
    Run the full ATC simulation and return timestep snapshots for animation.

    Returns:
        {
          "snapshots": List[Dict],   # One per SNAPSHOT_INTERVAL_STEPS steps
          "weather":   Weather,
          "scores":    Dict,
          "final_metrics": Dict,
          "task_level": int,
        }
    """
    sched = ATCScheduler(level)
    for d in get_task_data(level):
        sched.add_aircraft(Aircraft(
            id=d["id"], fuel_remaining=d["fuel"], wake_category=d["wake"],
            is_emergency=d["emerg"], distance_nm=d["dist"],
        ))

    snapshots: List[Dict] = []
    steps = int(sim_duration_secs / dt)

    for step in range(steps):
        sched.schedule_step(dt)

        # Capture snapshot every N steps
        if step % SNAPSHOT_INTERVAL_STEPS == 0:
            snapshots.append(sched.snapshot())

        # Early exit if all aircraft resolved
        all_done = all(
            a.status in (Status.EXITED, Status.CRASHED)
            for a in sched.aircraft
        )
        if all_done:
            snapshots.append(sched.snapshot())
            break

    return {
        "snapshots":     snapshots,
        "weather":       sched.weather,
        "scores":        sched.get_scores(),
        "final_metrics": sched.metrics,
        "task_level":    level,
    }
