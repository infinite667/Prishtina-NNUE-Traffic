# Prishtina NNUE Traffic

This project runs a high-fidelity real-time traffic simulation over a large-scale road network for **Prishtina** (40,000+ nodes), overlaid with live traffic lights and autonomous vehicles.

It features a cutting-edge **Hybrid AI System** that combines classical pathfinding with a **Neural Network (NNUE)** to learn and avoid traffic jams in real-time.

## Some of the key Features

### 1. Smart Traffic AI (The "Dreamer")
-   **Neural Network Update Engine (NNUE)**: A vectorized Numpy-based neural network that learns to predict edge delays.
-   **Background "Dreamer" Mode**: The AI trains on a **background thread** (Multi-threaded Double Buffering), allowing it to "dream" (replay) past traffic scenarios and optimize its brain 100x/sec without slowing down the simulation.
-   **Live Learning**: Watch the AI get smarter over time! When enabled, cars dynamically reroute to avoid developing jams, utilizing side streets effectively.

### 2. Intelligent Routing
-   **Aggressive Rerouting**: cars hate waiting! The AI penalizes localized congestion 6x higher than free-flow travel, forcing immediate detours.
-   **Dynamic Gap Scaling**: The simulation engine automatically adjusts safety gaps for small residential roads (<6m), preventing deadlocks in tight neighborhoods while maintaining safety on highways.

### 3. Split Lane Logic
-   **Slip Lanes**: Vehicles detecting a **Right Turn** are granted a temporary **20% Speed Boost** and effectively "ignore" red lights (simulating protected slip lanes), improving intersection throughput.
-   *Note*: This behavior is AI-controlled and only active when NNUE is enabled.

### 4. Interactive Frontend
-   **Leaflet Map**: Renders thousands of moving vehicles (Blue = Moving, Pink = Queued, Purple = Arrived).
-   **Live Metrics**: Real-time graphs for "Average Waiting Time" and active car counts.
-   **Controls**: Toggle AI On/Off, adjust Simulation Speed, and change Fleet Size (up to 2500+ cars) on the fly.

---

##  Backend Architecture

- Python 3.11+ / FastAPI / WebSockets.

---

## Installation & Run

### Prerequisites
-   Python 3.9+

### 1. Setup
```bash
# Clone the repo (if using git)
# cd pristina_traffic_sim

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run
```bash
python -m backend.main
```

### 3. Play
Open your browser to: **[http://localhost:9999](http://localhost:9999)**

*Note: On the first run, the simulation will download ~15MB of map data from Overpass API. This may take 10-20 seconds ( or maybe a touch more ). Subsequent runs will use the cached map.*

---

## Controls

| Control | Function |
| :--- | :--- |
| **NNUE / AI Rerouting** | **Enable** to wake up the "Dreamer". The AI will start learning from traffic history and cars will begin taking smarter routes. **Disable** to use standard shortest-path logic. |
| **Speed Multiplier** | Speed up time (up to 5x). Useful for gathering training data quickly. |
| **Fleet Size** | Adjust the number of cars (200 - 1000+). |

---

## Project Structure

-   `backend/` - Core simulation logic.
    -   `sim.py`: Physics engine, gap logic, and main loop.
    -   `ai/traffic_ai.py`: Defined the "Dreamer" thread and Experience Replay buffer.
    -   `routing.py`: Graph loader (Overpass API) and A* pathfinding.
    -   `nnue.py`: Neural Network architecture.
-   `frontend/` - HTML/JS/CSS (Leaflet & Chart.js).
-   `data/` - Caches and assets.
    -   `ai_store/`: Where the AI saves its learned brain (`nnue_weights.pkl`).

---

**Built with ❤️ for Prishtina from Murat Mehmeti.**
