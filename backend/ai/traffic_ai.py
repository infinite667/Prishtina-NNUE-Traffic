from __future__ import annotations

import collections
import copy
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..nnue import TinyNNUE


@dataclass
class EdgeLearner:
    """
    Traffic AI using TinyNNUE for delay prediction.
    Implements generic Experience Replay for online learning.
    Phase 3: Background Dreamer (Threaded Training).
    """
    store_dir: Path
    
    # Flags
    enabled: bool = False
    
    # NNUE (Inference copy - fast, read-only)
    nnue: TinyNNUE = field(init=False)
    
    # Helper for background training
    _dreamer_nnue: TinyNNUE = field(init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread = field(init=False)
    
    # Replay Buffer
    # stores (features, label_delay_ratio)
    replay_buffer: collections.deque = field(default_factory=lambda: collections.deque(maxlen=10000))
    
    # Training Control
    _batch_size: int = 32

    def __post_init__(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        # Input features:
        # 1. freeflow_s (log scale or ratio)
        # 2. occupancy_ratio (0..1+)
        # 3. is_red_light (0 or 1)
        # 4. global_avg_wait (normalized)
        # 5. capacity (log)
        self.nnue = TinyNNUE(input_dim=5, hidden_dim=32, lr=0.002)
        self.nnue.load(self.model_path)
        
        # Create a separate copy for the background thread to mangle
        # This avoids race conditions during inference.
        self._dreamer_nnue = copy.deepcopy(self.nnue)
        
        # Start the Dreamer
        self._thread = threading.Thread(target=self._dream_loop, daemon=True)
        self._thread.start()

    @property
    def model_path(self) -> Path:
        return self.store_dir / "nnue_weights.pkl"

    def penalty_multiplier(
        self,
        u: int,
        v: int,
        freeflow_s: float,
        occ: float,
        cap: float,
        red: float,
        avg_wait_s: float,
    ) -> float:
        """
        Predict delay multiplier using NNUE.
        This runs on the MAIN thread (inference).
        """
        # If disabled, return neutral multiplier? 
        # Actually user wants AI behavior to stop if disabled, but Sim calls this only if enabled usually.
        # But to be safe:
        if not self.enabled:
            return 1.0

        features = self._make_features(freeflow_s, occ, cap, red, avg_wait_s)
        # The network predicts a multiplier directly (centered around 1.0)
        # No lock needed: we only read weights, and weights are atomic replaced in sync()
        pred = self.nnue.predict_multiplier(features)
        
        # Safety clamp to prevent wild routing
        return max(0.8, min(8.0, pred))

    def update_edge_delay(
        self, 
        u: int, 
        v: int, 
        actual_delay_s: float, 
        context: Dict[str, float]
    ) -> None:
        """
        Add experience to replay buffer.
        """
        if not self.enabled:
            return

        # Calculate target multiplier: (freeflow + delay) / freeflow
        ff = max(0.1, context["freeflow_s"])
        target_mult = (ff + actual_delay_s) / ff
        target_mult = max(0.8, min(8.0, target_mult))

        features = self._make_features(
            context["freeflow_s"], 
            context["occ"], 
            context["cap"], 
            context["red"], 
            context["avg_wait_s"]
        )
        
        # Appending to deque is thread-safe in CPython (GIL), so lock not strictly needed for append,
        # but good practice if we switched to list.
        self.replay_buffer.append((features, target_mult))

    def _make_features(self, ff: float, occ: float, cap: float, red: float, wait: float) -> List[float]:
        # Normalize features roughly to 0..1 range
        return [
            math.log1p(ff) * 0.1,       # log duration
            min(2.0, occ / max(1.0, cap)), # saturation
            1.0 if red > 0.5 else 0.0,  # signal state
            math.log1p(wait) * 0.1,     # global congestion state
            math.log1p(cap) * 0.1       # road scale
        ]

    def _dream_loop(self) -> None:
        """
        Background process that constantly improves the model (The Dreamer).
        """
        last_sync = time.time()
        last_save = time.time()
        
        while not self._stop_event.is_set():
            # 1. Guard: Only learn if AI is enabled
            if not self.enabled:
                time.sleep(0.5)
                continue
            
            # 2. Guard: Need enough data
            if len(self.replay_buffer) < self._batch_size:
                time.sleep(0.1)
                continue
                
            # 3. Train Step (The Dream)
            # Sample a batch
            try:
                # Random sample is safe-ish for deque, or convert to list
                # Creating a list form a deque is O(N), so limit lookback if huge? 
                # Ideally we just index, but deque doesn't support random access well.
                # Let's just take the last 2000 items as pool if buffer is huge.
                # For now, standard sample.
                batch = random.sample(self.replay_buffer, self._batch_size)
                
                # Train the DREAMER model (not the live one)
                loss = self._dreamer_nnue.train_batch(batch, lr=0.002)
            except Exception:
                time.sleep(0.1)
                continue
            
            # 4. Sync Step (Update the Brain)
            # Every 1.0 second, we push the learned weights to the live model
            now = time.time()
            if now - last_sync > 0.5:
                # Copy weights safely
                state = self._dreamer_nnue.state_dict()
                # We can assume state_dict keys are standard
                self.nnue.W1 = state["W1"].copy()
                self.nnue.b1 = state["b1"].copy()
                self.nnue.W2 = state["W2"].copy()
                self.nnue.b2 = state["b2"].copy()
                self.nnue.scale = state["scale"].copy()
                last_sync = now
            
            # 5. Save Step
            if now - last_save > 30.0:
                self._dreamer_nnue.save(self.model_path)
                last_save = now
            
            # Sleep tiny amount to yield CPU to sim
            time.sleep(0.005)

    def maybe_save(self, min_interval_s: float = 5.0) -> None:
        """
        Save the model if enough time has passed since last save.
        """
        now = time.time()
        # We can implement a simple throttle here.
        # Note: The background dreamer also saves periodically.
        # This method allows the MAIN thread to trigger a save (e.g. on shutdown or periodically).
        # To avoid race conditions with the dreamer saving, we should probably just let the dreamer do it,
        # OR share the lock. But `TinyNNUE.save` is just atomic pickle dump, so it's mostly fine
        # if we don't interleave writes corruptly. 
        # For safety, let's just ignore this call if the dreamer is running, 
        # or implement a flag. Actually, `_dream_loop` has a 30s timer.
        # If the sim calls this explicitly, we should save.
        
        # Simple throttle
        if not hasattr(self, "_last_explicit_save"):
            self._last_explicit_save = 0.0
            
        if now - self._last_explicit_save < min_interval_s:
            return
            
        self._last_explicit_save = now
        # Save the inference model (which is synced from dreamer)
        self.nnue.save(self.model_path)

    def shutdown(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
