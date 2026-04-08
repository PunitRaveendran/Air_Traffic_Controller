# ATC OpenEnv - Air Traffic Control Runway Sequencing Environment
---
title: ATC Agentic Inference
emoji: ✈️
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---
# ATC Agent Hackathon Submission
This Space runs the inference script for the ATC environment.
Production-ready OpenEnv-compliant environment for AI agents to act as ATC sequencing controllers.

## Overview & Motivation

**ATC OpenEnv** is an Air Traffic Control sequencing environment simulator explicitly designed for AI agents. The primary motivation is to evaluate reinforcement learning and LLM-based agents in an operational safety-critical setting where delays cause compounding penalties and physical errors (like fuel starvation) lead to episode-terminating crashes. 

An AI agent acts as the primary ATC sequencing controller, observing inbound flights and making continuous decisions on runway assignments, landing order, and flow expediting. This is a physics-grounded workflow simulation, capturing real ATC operational pressure.

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| aircraft | List[dict] | All aircraft in the sector |
| aircraft[].id | str | Unique aircraft identifier |
| aircraft[].aircraft_type | str | HEAVY, MEDIUM, or LIGHT |
| aircraft[].priority | str | NORMAL, FUEL_CRITICAL, or EMERGENCY |
| aircraft[].status | str | INBOUND, HOLDING, ASSIGNED, LANDING, LANDED |
| aircraft[].distance_nm | float | Distance to runway in nautical miles |
| aircraft[].speed_knots | float | Current speed in knots |
| aircraft[].fuel_remaining_min | float | Fuel remaining in minutes |
| aircraft[].assigned_runway | str | Assigned runway ID (if any) |
| aircraft[].sequence_position | int | Position in landing sequence |
| runways | List[dict] | All runways at the airport |
| runways[].id | str | Runway identifier |
| runways[].status | str | OPEN, REDUCED, or CLOSED |
| runways[].next_available_step | int | Step when runway is next available |
| runways[].current_occupant | str | Aircraft ID currently on runway |
| timestep | int | Current simulation step |
| new_arrivals_count | int | Number of new arrivals this episode |
| episode_done | bool | Whether episode has ended |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| aircraft_id | str | Target aircraft identifier |
| action_type | str | assign, hold, or expedite |
| runway_id | str | Runway to assign (for assign action) |
| sequence_position | int | Landing sequence position |

## Tasks

### Task 1: Clear Skies
- **Difficulty**: Easy
- **Aircraft**: 5
- **Runways**: 2 (both OPEN)
- **Fuel**: All > 40 min
- **Emergencies**: None
- **Max Steps**: 20
- **Initial Distances**: 10-50 nm spread

### Task 2: Fuel Pressure
- **Difficulty**: Medium
- **Aircraft**: 10 initial, 2 arrive at step 5
- **Runways**: 1 OPEN, 1 REDUCED
- **Fuel Critical**: 2 aircraft with < 12 min
- **Max Steps**: 30

### Task 3: Full Emergency
- **Difficulty**: Hard
- **Aircraft**: 15 initial, 2 every 4 steps
- **Runways**: 2 (one closes at step 3)
- **Emergency**: One aircraft at step 5
- **Max Steps**: 40

## Scoring

### Grader 1 (Task 1)
- **Formula**: landings_completed / total_aircraft - avg_delay_penalty
- **Focus**: Basic landing completion efficiency

### Grader 2 (Task 2)
- **Formula**: 40% delay score + 40% fuel-critical handling + 20% separation compliance
- **Focus**: Handling fuel-critical situations

### Grader 3 (Task 3)
- **Formula**: 25% landings + 25% emergency + 25% no fuel exhaust + 25% queue < 12
- **Focus**: Multi-objective complex scenario handling

## Setup Instructions

### Local Setup

### Local LLM Inference

To run the inference script locally, you **must** supply environment variables pointing to your Language Model:

```bash
cd atc-openenv
pip install -r requirements.txt

# Linux/macOS
export API_BASE_URL="https://api.openai.com/v1" # Or Hugging Face Router, local vLLM etc.
export MODEL_NAME="gpt-4o"
export HF_TOKEN="your_hugging_face_or_api_key"

# Windows (Command Prompt)
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set HF_TOKEN=hf_YourTokenHere

# Run validation inference
python inference.py
```

### Local API Server

```bash
# Run the FastAPI Web API server locally
python app.py
```

### Docker Setup

```bash
cd atc-openenv
docker build -t atc-openenv .
docker run -p 7860:7860 atc-openenv
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset?task_id={1,2,3}` | POST | Reset environment |
| `/step` | POST | Execute actions |
| `/state` | GET | Get full state |

## Baseline Scores (from inference.py)

| Task | Score | Total Reward |
|------|-------|--------------|
| 1 - Clear Skies | 1.000 | 4.35 |
| 2 - Fuel Pressure | 1.000 | 5.35 |
| 3 - Full Emergency | 0.967 | -2.00 |

## Example Interaction

```python
from env.atc_env import ATCEnv, Action

# Initialize environment
env = ATCEnv()
obs = env.reset(task_id=1)

# Observe current state
print(f"Timestep: {obs.timestep}")
print(f"Aircraft: {len(obs.aircraft)}")
print(f"Runways: {len(obs.runways)}")

# Submit actions (assign aircraft to runways)
actions = [
    Action(aircraft_id="AC1", action_type="assign", runway_id="RW01", sequence_position=1),
    Action(aircraft_id="AC2", action_type="assign", runway_id="RW02", sequence_position=1),
]

# Step simulation
obs, reward, done, info = env.step(actions)
print(f"Reward: {reward.value}")
print(f"Done: {done}")

# Get final state
state = env.state()
print(f"Score so far: {state.score_so_far}")
```

## Wake Turbulence Separation Rules

| Lead Type | Trail Type | Separation (nm) |
|-----------|------------|-----------------|
| HEAVY | HEAVY | 4 |
| HEAVY | MEDIUM | 5 |
| HEAVY | LIGHT | 6 |
| MEDIUM | LIGHT | 3 |
| OTHER | OTHER | 2 |

## Reward Events

| Event | Value |
|-------|-------|
| LANDING_SUCCESS | +1.0 |
| DELAY_PER_STEP | -0.05 per waiting aircraft |
| FUEL_CRITICAL_LATE | -2.0 per step after threshold |
| FUEL_EXHAUSTED | -10.0 |
| EMERGENCY_NOT_PRIORITIZED | -10.0 |
| SEPARATION_VIOLATION | -5.0 |
| RUNWAY_CONFLICT | -20.0 |
| HOLDING_PENALTY | -0.1 per step per holding aircraft |

## Environment Implementation

All models use Pydantic v2 for data validation. The environment is fully deterministic given a random seed. Graders are stateless and accept episode_log as input.