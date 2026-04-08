"""FastAPI wrapper for ATC OpenEnv environment."""
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.atc_env import ATCEnv, Action, Observation, State


app = FastAPI(title="ATC OpenEnv", version="1.0.0")

_env: Optional[ATCEnv] = None


class ResetRequest(BaseModel):
    task_id: int = 1


class StepRequest(BaseModel):
    actions: list[Action]


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/reset")
def reset(task_id: int = 1) -> Observation:
    """Reset environment with specified task."""
    global _env
    _env = ATCEnv(seed=42)
    return _env.reset(task_id)


@app.post("/step")
def step(request: StepRequest) -> tuple[Observation, float, bool, dict]:
    """Execute actions and return next observation."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    obs, reward, done, info = _env.step(request.actions)
    return obs, reward.value, done, info


@app.get("/state")
def state() -> State:
    """Get current environment state."""
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return _env.state()


@app.get("/")
def read_root():
    """Root endpoint to verify the server is running."""
    return {"message": "Server is running! The API is ready for the model."}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()