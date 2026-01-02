from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .sim import Simulation, SimSettings

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
DATA_DIR = ROOT / "data"

async def _run_sim_loop(stop_event: asyncio.Event) -> None:
    # Drive the simulation with real wall-clock time so sim seconds match real seconds.
    # This avoids "slow motion" drift caused by asyncio.sleep + processing overhead.
    import time
    # Adaptive sim tick rate: higher fleets make each step heavier.
    # Dropping the sim Hz a bit keeps the UI responsive at 1000-3000 cars.
    base_hz = 60.0
    min_dt = 1.0 / 240.0   # clamp very small deltas
    max_dt = 1.0 / 10.0    # clamp huge spikes (tab sleep / breakpoint)
    last = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()
        dt = now - last
        last = now
        dt = max(min_dt, min(max_dt, dt))

        # Choose an update rate based on current fleet size.
        n = len(sim.vehicles)
        target_hz = base_hz
        if n >= 900:
            target_hz = 45.0
        if n >= 1800:
            target_hz = 30.0
        if n >= 2600:
            target_hz = 24.0

        try:
            sim.step(dt)
        except Exception:
            # Never let the background loop die silently (this would freeze the UI).
            import traceback
            traceback.print_exc()
            # If NNUE caused the issue, disable it so the sim can keep running.
            settings.nnue_enabled = False

        # Yield control; we aim for ~60Hz but dt above already accounts for drift.
        await asyncio.sleep(max(0.0, (1.0 / target_hz) - (time.perf_counter() - now)))



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the simulation loop in the background (FastAPI lifespan replaces on_event).
    stop = asyncio.Event()
    task = asyncio.create_task(_run_sim_loop(stop))
    try:
        yield
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Prishtina Traffic Sim", version="1.0", lifespan=lifespan)

# Static frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Single global simulation instance
settings = SimSettings()
sim = Simulation(data_dir=DATA_DIR, settings=settings)

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

@app.get("/api/bootstrap")
def bootstrap() -> Dict[str, Any]:
    """
    Initial data: roads polylines, traffic lights, and current settings.
    """
    return {
        "roads": sim.road_export(),
        "lights": sim.lights_export(),
        "settings": settings.model_dump(),
    }

@app.get("/api/metrics")
def metrics() -> Dict[str, Any]:
    return sim.metrics_export()

@app.post("/api/settings")
def update_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    settings.apply_patch(payload)
    # Apply the new fleet target immediately so the UI updates without a refresh.
    sim.reconcile_fleet()
    return {"ok": True, "settings": settings.model_dump()}

@app.websocket("/ws")
async def ws_updates(ws: WebSocket) -> None:
    await ws.accept()
    try:
        # Send an immediate snapshot so the UI can render instantly
        await ws.send_text(json.dumps({"type": "snapshot", "payload": sim.snapshot_export()}))
        # Stream updates
        async for msg in sim.stream_updates():
            await ws.send_text(msg)
    except WebSocketDisconnect:
        return

 

def main() -> None:
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=9999, reload=False, log_level="info")

if __name__ == "__main__":
    main()
