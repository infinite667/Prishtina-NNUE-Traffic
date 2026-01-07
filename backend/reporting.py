from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# matplotlib is used to generate a local "full history" graph alongside the CSV.
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


@dataclass
class SimReporter:
    """
    Writes a CSV row every second and regenerates a simple PNG chart.

    Files are created per launch under <project_root>/sim_saves/Sim-Results-YYYY-MM-DD_HH-MM-SS:
      - sim_results_YYYY-MM-DD_HH-MM-SS.csv
      - sim_results_YYYY-MM-DD_HH-MM-SS_waiting.png
      - sim_results_YYYY-MM-DD_HH-MM-SS_cars.png
    """
    project_root: Path
    out_dir: Path = field(init=False)
    csv_path: Path = field(init=False)
    wait_png_path: Path = field(init=False)
    cars_png_path: Path = field(init=False)

    # buffers
    t: List[float] = field(default_factory=list)
    cars: List[int] = field(default_factory=list)
    avg_wait: List[float] = field(default_factory=list)
    ai_on: List[int] = field(default_factory=list)

    _last_logged_sec: int = field(default=-1, init=False)
    _header_written: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Base folder for all runs
        self.out_dir = self.project_root / "sim_saves"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Per-run folder
        ts = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = self.out_dir / f"Sim-Results-{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        base = f"sim_results_{ts}"
        self.csv_path = run_dir / f"{base}.csv"
        self.wait_png_path = run_dir / f"{base}_waiting.png"
        self.cars_png_path = run_dir / f"{base}_cars.png"

    def maybe_log(self, sim_time_s: float, cars_in_city: int, avg_waiting_s: float, ai_enabled: bool) -> None:
        """
        Append one row per integer second of simulation time.
        """
        sec = int(sim_time_s)
        if sec == self._last_logged_sec:
            return
        self._last_logged_sec = sec

        # Reporting format/units:
        # - sim_time_s: whole seconds only (no sub-second drift like 1.0154...)
        # - avg_waiting_s: keep only 3 decimals; scale from internal ticks to seconds.
        sim_time_out = float(sec)
        # The simulation internally accumulates waiting time in real seconds.
        # Just round it.
        avg_wait_out = round(float(avg_waiting_s), 3)

        self.t.append(sim_time_out)
        self.cars.append(int(cars_in_city))
        self.avg_wait.append(avg_wait_out)
        self.ai_on.append(1 if ai_enabled else 0)

        # Write a single-row dataframe (fast enough at 1Hz; keeps pandas in the pipeline)
        row = pd.DataFrame([{
            "timestamp_iso": pd.Timestamp.now().isoformat(),
            "sim_time_s": sim_time_out,
            "cars_in_city": int(cars_in_city),
            "avg_waiting_s": avg_wait_out,
            "ai_enabled": int(1 if ai_enabled else 0),
        }])
        row.to_csv(self.csv_path, mode="a", header=(not self._header_written), index=False)
        self._header_written = True

        # Regenerate charts (also 1Hz; small and safe for typical run lengths)
        self._write_charts()

    def _write_charts(self) -> None:
        if not self.t:
            return

        t = np.asarray(self.t, dtype=float)
        cars = np.asarray(self.cars, dtype=float)
        w = np.asarray(self.avg_wait, dtype=float)
        ai = np.asarray(self.ai_on, dtype=int)

        # Waiting chart with two datasets (red for AI off, green for AI on), like the UI.
        w_off = np.where(ai == 0, w, np.nan)
        w_on = np.where(ai == 1, w, np.nan)

        plt.figure()
        plt.plot(t, w_off, linewidth=2)  # default color overridden below
        plt.plot(t, w_on, linewidth=2)
        ax = plt.gca()
        # Force red/green to match request
        ax.lines[0].set_color("#ef4444")  # red
        ax.lines[1].set_color("#22c55e")  # green
        plt.xlabel("Sim time (s)")
        plt.ylabel("Avg waiting (s)")
        plt.title("Avg waiting time (full history)")
        plt.tight_layout()
        plt.savefig(self.wait_png_path, dpi=140)
        plt.close()

        # Cars chart (single line; full history)
        plt.figure()
        plt.plot(t, cars, linewidth=2)
        plt.xlabel("Sim time (s)")
        plt.ylabel("Cars in city")
        plt.title("Cars in city (full history)")
        plt.tight_layout()
        plt.savefig(self.cars_png_path, dpi=140)
        plt.close()
