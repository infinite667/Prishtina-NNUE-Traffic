from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _drelu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def _softplus(x: np.ndarray) -> np.ndarray:
    # stable softplus
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class TinyNNUE:
    """
    Numpy-optimized Tiny MLP for traffic delay estimation.
    """
    input_dim: int
    hidden_dim: int = 32
    lr: float = 0.002

    # Weights
    W1: np.ndarray = field(init=False)  # (in, hidden)
    b1: np.ndarray = field(init=False)  # (hidden,)
    W2: np.ndarray = field(init=False)  # (hidden, 1)
    b2: np.ndarray = field(init=False)  # (1,)

    # Feature scaler (running mean/std could be better, here just max-scale)
    scale: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(1337)
        # Xavier-ish init
        limit1 = np.sqrt(6 / (self.input_dim + self.hidden_dim))
        self.W1 = rng.uniform(-limit1, limit1, (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        limit2 = np.sqrt(6 / (self.hidden_dim + 1))
        self.W2 = rng.uniform(-limit2, limit2, (self.hidden_dim, 1)).astype(np.float32)
        self.b2 = np.array([1.0], dtype=np.float32)  # start bias around 1.0 (multiplier)

        self.scale = np.ones(self.input_dim, dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # Simple running max tracking for stability
        # x shape: (B, D) or (D,)
        if x.ndim == 1:
            self.scale = np.maximum(self.scale * 0.995, np.abs(x) + 1e-6)
            return x / self.scale
        else:
            # Batch update
            batch_max = np.max(np.abs(x), axis=0)
            self.scale = np.maximum(self.scale * 0.995, batch_max + 1e-6)
            return x / self.scale

    def predict_multiplier(self, x: List[float]) -> float:
        """Single-sample inference (fast path for sim)."""
        x_in = np.array(x, dtype=np.float32)
        xn = self._normalize(x_in)
        
        # Forward
        z1 = xn @ self.W1 + self.b1
        h1 = _relu(z1)
        z2 = h1 @ self.W2 + self.b2
        
        # Softplus + 0.5 to keep it positive and centered around 1.0
        # This acts as a cost multiplier.
        out = 0.5 + _softplus(z2)
        return float(out[0])

    def train_batch(self, batch: List[Tuple[List[float], float]]) -> float:
        """SGD step on a batch. Returns mean loss."""
        if not batch:
            return 0.0

        X_raw = np.array([b[0] for b in batch], dtype=np.float32)
        Y = np.array([[b[1]] for b in batch], dtype=np.float32)  # (B, 1)

        X = self._normalize(X_raw)

        # Forward
        z1 = X @ self.W1 + self.b1     # (B, H)
        h1 = _relu(z1)                 # (B, H)
        z2 = h1 @ self.W2 + self.b2    # (B, 1)
        
        pred = 0.5 + _softplus(z2)     # (B, 1)

        # Loss = MSE
        diff = pred - Y
        loss = np.mean(diff ** 2)

        # Backward
        # dL/dpred = 2 * diff / B
        d_pred = (2.0 / len(batch)) * diff
        
        # d_pred/dz2 = sigmoid(z2)
        d_z2 = d_pred * _sigmoid(z2)
        
        # Grads 2
        d_W2 = h1.T @ d_z2             # (H, B) @ (B, 1) -> (H, 1)
        d_b2 = np.sum(d_z2, axis=0)    # (1,)
        
        # Grads 1
        d_h1 = d_z2 @ self.W2.T        # (B, 1) @ (1, H) -> (B, H)
        d_z1 = d_h1 * _drelu(z1)       # (B, H)
        
        d_W1 = X.T @ d_z1              # (D, B) @ (B, H) -> (D, H)
        d_b1 = np.sum(d_z1, axis=0)    # (H,)

        # Update
        self.W1 -= self.lr * d_W1
        self.b1 -= self.lr * d_b1
        self.W2 -= self.lr * d_W2
        self.b2 -= self.lr * d_b2

        return float(loss)

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.W1 = state["W1"]
            self.b1 = state["b1"]
            self.W2 = state["W2"]
            self.b2 = state["b2"]
            self.scale = state["scale"]
        except Exception:
            pass

    def state_dict(self):
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "scale": self.scale
        }
