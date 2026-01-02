# Prishtina Traffic Simulation + NNUE (Localhost:9999)

This project runs a real-time traffic simulation over a lightweight road network for **Prishtina** and overlays **traffic lights from GeoJSON**.
A small "NNUE-style" incremental neural model can (optionally) re-route vehicles away from congested roads.

## What you get
- Leaflet UI with:
  - Traffic lights (all features from `data/traffic_lights.geojson`, including duplicates at same coordinates)
  - Vehicles with colored icons:
    - Moving: **blue**
    - Waiting / queued: **pink**
    - Arrived: **purple**
  - Click vehicle / light to see details + a glowing circle that follows the object
  - Legend, settings panel (NNUE toggle + sim speed slider), and average waiting-time graph
- Backend sim ticks at **>=30Hz** and streams updates via WebSocket.

## Run (Debian)
```bash
cd pristina_traffic_sim
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.main
```

Then open: `http://localhost:9999`

## Notes about the road network (no SUMO / no osmnx)
- On first run, the backend tries to download Prishtina roads via **Overpass API** and caches them to `data/roads_cache.json`.
- If Overpass is unreachable (no internet), the app falls back to a small built-in demo network centered on Prishtina so the UI still runs.

## Files
- `backend/` FastAPI server + simulation
- `frontend/` static UI (Leaflet + Chart.js)
- `data/traffic_lights.geojson` your exported lights
- `data/roads_cache.json` generated on first successful Overpass fetch
