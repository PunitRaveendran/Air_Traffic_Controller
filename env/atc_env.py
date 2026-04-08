"""Core ATC Environment implementation."""
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from env.aircraft import Aircraft, AircraftStatus, AircraftType, Priority
from env.runway import Runway, RunwayStatus
from env.scheduler import Scheduler, WakeTurbulenceRule
from env.reward import RewardEvent, compute_reward, get_reward_breakdown


class Action(BaseModel):
    aircraft_id: str
    action_type: str = Field(description="assign/hold/expedite")
    runway_id: Optional[str] = None
    sequence_position: Optional[int] = None


class Observation(BaseModel):
    aircraft: List[Dict[str, Any]]
    runways: List[Dict[str, Any]]
    timestep: int
    new_arrivals_count: int
    episode_done: bool


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float] = Field(default_factory=get_reward_breakdown)


class State(BaseModel):
    aircraft: List[Dict[str, Any]]
    runways: List[Dict[str, Any]]
    timestep: int
    task_id: int
    score_so_far: float = 0.0


class ATCEnv:
    """Air Traffic Control runway sequencing environment."""

    def __init__(self, seed: Optional[int] = None):
        self._seed = seed
        self._rng = None
        self._aircraft: List[Aircraft] = []
        self._runways: List[Runway] = []
        self._timestep: int = 0
        self._task_id: int = 1
        self._max_steps: int = 40
        self._total_reward: float = 0.0
        self._new_arrivals_count: int = 0
        self._episode_done: bool = False
        self._score_so_far: float = 0.0
        self._last_landed_types: Dict[str, Optional[AircraftType]] = {}
        self._fuel_exhausted_count: int = 0
        self._separation_violations: int = 0
        self._runway_conflicts: int = 0
        self._holding_steps: Dict[str, int] = {}
        self._emergency_wait_steps: Dict[str, int] = {}
        self._pending_arrivals: List[Dict[str, Any]] = []
        self._task_config: Dict[str, Any] = {}
        self._task_instance: Any = None
        self._max_queue_size: int = 0
        # Tracks which aircraft IDs were fuel-critical at episode start (for grader2)
        self._initial_fuel_critical_ids: List[str] = []

    def _seed_random(self, seed: int) -> None:
        import random
        self._rng = random.Random(seed)
        self._rng.seed(seed)

    def reset(self, task_id: int = 1) -> "Observation":
        """Reset environment with specified task."""
        self._seed_random(self._seed or task_id * 42)
        self._timestep = 0
        self._total_reward = 0.0
        self._episode_done = False
        self._score_so_far = 0.0
        self._new_arrivals_count = 0
        self._fuel_exhausted_count = 0
        self._separation_violations = 0
        self._runway_conflicts = 0
        self._last_landed_types = {}
        self._holding_steps = {}
        self._emergency_wait_steps = {}
        self._pending_arrivals = []
        self._max_queue_size = 0
        self._initial_fuel_critical_ids = []
        self._task_id = task_id

        task_class = self._get_task_class(task_id)
        self._task_config = task_class.get_config()
        self._max_steps = self._task_config.get("max_steps", 40)
        self._task_instance = task_class()

        self._aircraft = self._initialize_aircraft(task_id)
        self._runways = self._initialize_runways(task_id)
        self._pending_arrivals = self._get_pending_arrivals(task_id)

        # Record which aircraft are fuel-critical at spawn (before any simulation)
        self._initial_fuel_critical_ids = [
            ac.id for ac in self._aircraft
            if ac.priority == Priority.FUEL_CRITICAL or ac.fuel_remaining_min < 15.0
        ]

        return self._get_observation()

    def _get_task_class(self, task_id: int):
        from tasks.task1_clear_skies import Task1ClearSkies
        from tasks.task2_fuel_pressure import Task2FuelPressure
        from tasks.task3_full_emergency import Task3FullEmergency
        tasks = {1: Task1ClearSkies, 2: Task2FuelPressure, 3: Task3FullEmergency}
        return tasks.get(task_id, Task1ClearSkies)

    def _initialize_aircraft(self, task_id: int) -> List[Aircraft]:
        task_class = self._get_task_class(task_id)
        return task_class.create_aircraft(self._rng)

    def _initialize_runways(self, task_id: int) -> List[Runway]:
        task_class = self._get_task_class(task_id)
        return task_class.create_runways()

    def _get_pending_arrivals(self, task_id: int) -> List[Dict[str, Any]]:
        task_class = self._get_task_class(task_id)
        return task_class.get_pending_arrivals()

    # GRADER REQUIREMENT: Parameter MUST be exactly named 'action'
    def step(self, action: List[Action]) -> Tuple["Observation", "Reward", bool, Dict[str, Any]]:
        """Execute one simulation step with provided actions."""
        if self._episode_done:
            return self._get_observation(), Reward(value=0.0), True, {}

        reward_breakdown = get_reward_breakdown()
        self._timestep += 1

        # Apply agent actions first
        self._apply_actions(action)

        # FIX (Bug 1): Tick runways BEFORE processing landings so a runway that
        # becomes free this step is immediately usable — corrects a 1-step throughput loss.
        self._tick_runways()

        self._process_landings(reward_breakdown)

        self._process_new_arrivals()

        self._tick_aircraft(reward_breakdown)

        self._apply_holding_penalty(reward_breakdown)

        reward_value = sum(reward_breakdown.values())
        self._total_reward += reward_value
        self._score_so_far = self._total_reward

        self._process_task_events()

        self._check_episode_end()

        observation = self._get_observation()
        reward = Reward(value=reward_value, breakdown=reward_breakdown)
        info = {"score_so_far": self._score_so_far}

        return observation, reward, self._episode_done, info

    # INTERNAL HELPER: uses 'actions' to avoid loop conflicts
    def _apply_actions(self, actions: List[Action]) -> None:
        state_dict = self.state().model_dump()
        # Track sequence positions used per runway in this step to detect conflicts
        runway_seq_used: Dict[str, set] = {}

        for action in actions:
            valid, reason = Scheduler.validate_action(action.model_dump(), state_dict)
            if not valid:
                continue

            aircraft = self._find_aircraft(action.aircraft_id)
            if aircraft is None:
                continue

            if action.action_type in ("assign", "expedite"):
                aircraft.status = AircraftStatus.ASSIGNED
                aircraft.assigned_runway = action.runway_id
                if action.sequence_position is not None:
                    # FIX (Bug 4): Count duplicate sequence positions as separation violations
                    rw = action.runway_id or ""
                    if rw not in runway_seq_used:
                        runway_seq_used[rw] = set()
                    if action.sequence_position in runway_seq_used[rw]:
                        self._separation_violations += 1
                    else:
                        runway_seq_used[rw].add(action.sequence_position)
                    aircraft.sequence_position = action.sequence_position

            elif action.action_type == "hold":
                aircraft.status = AircraftStatus.HOLDING
                self._holding_steps[aircraft.id] = self._holding_steps.get(aircraft.id, 0) + 1

    def _process_landings(self, reward_breakdown: Dict[str, float]) -> None:
        for runway in self._runways:
            if runway.status == RunwayStatus.CLOSED:
                continue
            if runway.current_occupant is not None:
                continue

            # Find all assigned aircraft ready to land on this runway
            ready_aircraft = [
                ac for ac in self._aircraft
                if ac.status == AircraftStatus.ASSIGNED
                and ac.assigned_runway == runway.id
                and ac.distance_nm <= 5.0
            ]

            if not ready_aircraft:
                continue

            # Pick aircraft with lowest sequence position (or closest if not set)
            ready_aircraft.sort(key=lambda x: (x.sequence_position or 999, x.distance_nm))
            aircraft = ready_aircraft[0]

            # No incorrect separation violation check here (Bug 4 fixed via _apply_actions)
            aircraft.status = AircraftStatus.LANDING
            runway.occupy(aircraft.id, self._timestep)
            self._last_landed_types[runway.id] = aircraft.aircraft_type
            reward_breakdown["landing_success"] += 1.0

        # Transition LANDING → LANDED once aircraft reaches threshold
        for aircraft in self._aircraft:
            if aircraft.status == AircraftStatus.LANDING and aircraft.distance_nm <= 2.0:
                aircraft.status = AircraftStatus.LANDED
                aircraft.sequence_position = None

    def _process_new_arrivals(self) -> None:
        new_arrivals = []
        for pending in self._pending_arrivals[:]:
            if pending["arrival_step"] == self._timestep:
                aircraft = Aircraft(
                    id=pending["id"],
                    aircraft_type=AircraftType(pending["aircraft_type"]),
                    priority=Priority(pending.get("priority", "NORMAL")),
                    status=AircraftStatus.INBOUND,
                    distance_nm=pending["distance_nm"],
                    speed_knots=pending["speed_knots"],
                    fuel_remaining_min=pending["fuel_remaining_min"],
                )
                new_arrivals.append(aircraft)
                self._pending_arrivals.remove(pending)
                self._new_arrivals_count += 1

        self._aircraft.extend(new_arrivals)

    def _tick_aircraft(self, reward_breakdown: Dict[str, float]) -> None:
        for aircraft in self._aircraft:
            if aircraft.status == AircraftStatus.LANDED:
                continue

            prev_fuel = aircraft.fuel_remaining_min
            aircraft.tick()
            new_fuel = aircraft.fuel_remaining_min

            if prev_fuel > 0 and new_fuel <= 0:
                self._fuel_exhausted_count += 1
                reward_breakdown["fuel_exhausted_penalty"] += 10.0

            if aircraft.is_fuel_critical:
                if aircraft.status in (AircraftStatus.HOLDING, AircraftStatus.ASSIGNED):
                    steps_waited = self._holding_steps.get(aircraft.id, 0)
                    if steps_waited > 5:
                        reward_breakdown["fuel_critical_penalty"] += 2.0 * (steps_waited - 5)

            if aircraft.is_emergency:
                if aircraft.status == AircraftStatus.HOLDING:
                    self._emergency_wait_steps[aircraft.id] = self._emergency_wait_steps.get(aircraft.id, 0) + 1
                    if self._emergency_wait_steps[aircraft.id] > 3:
                        reward_breakdown["emergency_penalty"] += 10.0

            if aircraft.status != AircraftStatus.LANDED:
                waiting = aircraft.status in (AircraftStatus.HOLDING, AircraftStatus.ASSIGNED)
                if waiting:
                    reward_breakdown["delay_penalty"] -= 0.05

        # FIX (Bug 3): Count only INBOUND + HOLDING as the "queue" — aircraft waiting
        # for assignment. ASSIGNED/LANDING aircraft are already being handled and should
        # not inflate the queue counter, making Task 3 unwinnable.
        current_queue = sum(
            1 for ac in self._aircraft
            if ac.status in (AircraftStatus.INBOUND, AircraftStatus.HOLDING)
        )
        if current_queue > self._max_queue_size:
            self._max_queue_size = current_queue

    def _apply_holding_penalty(self, reward_breakdown: Dict[str, float]) -> None:
        holding_count = sum(
            1 for ac in self._aircraft if ac.status == AircraftStatus.HOLDING
        )
        reward_breakdown["holding_penalty"] = -0.1 * holding_count

    def _tick_runways(self) -> None:
        for runway in self._runways:
            runway.tick(self._timestep)

    def _process_task_events(self) -> None:
        """Dispatch step events to task instance for runway closures, emergencies, etc."""
        if self._task_instance is not None and hasattr(self._task_instance, "process_step_event"):
            self._task_instance.process_step_event(self, self._timestep)

    def _check_episode_end(self) -> None:
        # FIX (Bug 2): Do NOT terminate episode on fuel exhaustion. Fuel exhaustion is
        # already penalized via reward and grader score. Terminating instantly when one
        # physically-impossible aircraft crashes is too punishing and collapses all
        # other aircraft scores. The episode runs to natural completion.

        # All aircraft landed (and no more pending arrivals)
        all_landed = all(ac.status == AircraftStatus.LANDED for ac in self._aircraft)
        no_more_arrivals = len(self._pending_arrivals) == 0
        if all_landed and no_more_arrivals:
            self._episode_done = True
            return

        if self._timestep >= self._max_steps:
            self._episode_done = True
            return

    def _find_aircraft(self, aircraft_id: str) -> Optional[Aircraft]:
        for aircraft in self._aircraft:
            if aircraft.id == aircraft_id:
                return aircraft
        return None

    def _get_observation(self) -> "Observation":
        aircraft_data = []
        for ac in self._aircraft:
            d = ac.model_dump()
            d["eta_steps"] = ac.eta_steps
            d["is_fuel_critical"] = ac.is_fuel_critical
            d["is_emergency"] = ac.is_emergency
            aircraft_data.append(d)

        runway_data = [rw.model_dump() for rw in self._runways]

        return Observation(
            aircraft=aircraft_data,
            runways=runway_data,
            timestep=self._timestep,
            new_arrivals_count=self._new_arrivals_count,
            episode_done=self._episode_done,
        )

    def state(self) -> "State":
        """Return full environment state."""
        aircraft_data = []
        for ac in self._aircraft:
            d = ac.model_dump()
            d["eta_steps"] = ac.eta_steps
            d["is_fuel_critical"] = ac.is_fuel_critical
            d["is_emergency"] = ac.is_emergency
            aircraft_data.append(d)

        return State(
            aircraft=aircraft_data,
            runways=[rw.model_dump() for rw in self._runways],
            timestep=self._timestep,
            task_id=self._task_id,
            score_so_far=self._score_so_far,
        )

    def get_grader_input(self) -> Dict[str, Any]:
        """Get episode data for grading."""
        return {
            "aircraft": [ac.model_dump() for ac in self._aircraft],
            "timestep": self._timestep,
            "fuel_exhausted_count": self._fuel_exhausted_count,
            "separation_violations": self._separation_violations,
            "runway_conflicts": self._runway_conflicts,
            "total_reward": self._total_reward,
            "holding_steps": self._holding_steps,
            "emergency_wait_steps": self._emergency_wait_steps,
            "max_queue_size": self._max_queue_size,
            # FIX (Bug 8): Pass initial fuel-critical IDs so graders can correctly
            # distinguish aircraft that were BORN fuel-critical vs those that aged into it.
            "initial_fuel_critical_ids": self._initial_fuel_critical_ids,
        }