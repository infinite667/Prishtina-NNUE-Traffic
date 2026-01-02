from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

def _relu(x: float) -> float:
    return x if x > 0 else 0.0

def _drelu(x: float) -> float:
    return 1.0 if x > 0 else 0.0

def _softplus_stable(x: float) -> float:
    """Numerically stable softplus: log(1+exp(x))."""
    # Avoid overflow for large positive x.
    if x > 30.0:
        return x
    # For large negative x, exp(x) is tiny.
    if x < -30.0:
        return math.exp(x)
    return math.log1p(math.exp(x))

def _sigmoid_stable(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)

@dataclass
class TinyNNUE:
    """
    A tiny, fast, incrementally trainable MLP.
    It's "NNUE-like" in spirit: small, cheap forward pass, and can be updated online.

    Output is a multiplier around ~1.0.
    """
    input_dim: int
    hidden_dim: int = 32
    # weights: hidden (H x D), bias hidden (H), out (H), bias out (1)
    def __post_init__(self) -> None:
        rng = random.Random(1337)
        self.W1 = [[(rng.random() - 0.5) * 0.1 for _ in range(self.input_dim)] for _ in range(self.hidden_dim)]
        self.b1 = [(rng.random() - 0.5) * 0.1 for _ in range(self.hidden_dim)]
        self.W2 = [(rng.random() - 0.5) * 0.1 for _ in range(self.hidden_dim)]
        self.b2 = 1.0  # start near multiplier=1

        # Running feature scale for stability
        self._scale = [1.0 for _ in range(self.input_dim)]

    def _normalize(self, x: Sequence[float]) -> List[float]:
        # Update simple running scale (robust-ish)
        out = []
        for i, v in enumerate(x):
            av = abs(v)
            self._scale[i] = max(self._scale[i] * 0.999, av * 0.001 + self._scale[i] * 0.999)
            s = self._scale[i] if self._scale[i] > 1e-6 else 1.0
            out.append(v / s)
        return out

    def predict_multiplier(self, x: Sequence[float]) -> float:
        xn = self._normalize(x)
        h = []
        for j in range(self.hidden_dim):
            z = self.b1[j]
            w = self.W1[j]
            for i in range(self.input_dim):
                z += w[i] * xn[i]
            h.append(_relu(z))
        y = self.b2
        for j in range(self.hidden_dim):
            y += self.W2[j] * h[j]
        # Softplus-ish to keep positive, then shift near 1 (stable)
        mult = 0.5 + _softplus_stable(y)
        return mult

    def train_batch(self, batch: List[Tuple[List[float], float]], lr: float = 0.003) -> None:
        # Simple SGD on MSE loss: (pred - target)^2
        for x, target in batch:
            xn = self._normalize(x)
            # forward
            z1 = [0.0] * self.hidden_dim
            h = [0.0] * self.hidden_dim
            for j in range(self.hidden_dim):
                z = self.b1[j]
                w = self.W1[j]
                for i in range(self.input_dim):
                    z += w[i] * xn[i]
                z1[j] = z
                h[j] = _relu(z)
            y_lin = self.b2
            for j in range(self.hidden_dim):
                y_lin += self.W2[j] * h[j]
            # Stable softplus and sigmoid to avoid overflow when NNUE is enabled
            pred = 0.5 + _softplus_stable(y_lin)
            sig = _sigmoid_stable(y_lin)
            d_pred_dy = sig
            # loss
            err = (pred - target)
            dL_dpred = 2.0 * err
            dL_dy = dL_dpred * d_pred_dy

            # grads for output layer
            for j in range(self.hidden_dim):
                self.W2[j] -= lr * (dL_dy * h[j])
            self.b2 -= lr * dL_dy

            # hidden layer grads
            for j in range(self.hidden_dim):
                dh = dL_dy * self.W2[j]
                dz = dh * _drelu(z1[j])
                self.b1[j] -= lr * dz
                for i in range(self.input_dim):
                    self.W1[j][i] -= lr * (dz * xn[i])
