import numpy as np
import torch
from .node import Node
import contextlib
import math
import warnings

@contextlib.contextmanager
def nan_trap(field, tag: str = "", **kwargs):
    """A context manager to trap NaN/Inf values in tensors for debugging."""
    try:
        yield
    finally:
        tensors_to_check = {
            "states": field.states,
            "pressures": field.pressures,
        }
        tensors_to_check.update(kwargs)

        for name, tensor in tensors_to_check.items():
            if tensor is not None and tensor.numel() > 0:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"\n[NaN-TRAP] {tag} → {name} contains NaN/Inf!")
                    bad_indices = ((torch.isnan(tensor) | torch.isinf(tensor)).nonzero(as_tuple=True)[0]).tolist()
                    print(f"  First bad indices: {bad_indices[:10]}")
                    raise RuntimeError(f"NaN detected in '{name}' — aborting step.")

class Field:
    def __init__(self, n=10, device='cpu', dtype=torch.float64, K=1.0, f_max=8.0, g_mean=0.01):
        self.n = n
        self.device = device
        self.dtype = dtype
        
        # Physics parameters
        self.K = K              # Spring constant for pairwise forces
        self.f_max = f_max      # Absolute force cap per edge
        self.g_mean = g_mean    # Mean-field damping gain

        assert self.dtype == torch.float64, "Field must use float64 for stability."

        self.states = torch.rand(n, device=device, dtype=dtype)
        self.pressures = torch.zeros(n, device=device, dtype=dtype)

    def step(self, dt=0.01, friction=0.02):
        if self.n == 0:
            return

        # 1. Pairwise spring forces, clipped to prevent explosion
        state_diffs = self.states.unsqueeze(1) - self.states.unsqueeze(0)
        pairwise_forces = torch.clamp(self.K * state_diffs, -self.f_max, self.f_max)
        total_pairwise_forces = torch.sum(pairwise_forces, dim=1)

        # 2. Mean-field damping (pulls nodes toward the mean state)
        mean_state = torch.mean(self.states)
        mean_field_damping_forces = self.g_mean * (self.states - mean_state)

        # 3. Total force on each node
        total_forces = total_pairwise_forces - mean_field_damping_forces

        # 4. Semi-implicit Euler integration
        # Update pressure (velocity) first...
        self.pressures += total_forces * dt
        self.pressures *= (1.0 - friction)

        # ...then update state (position) using the *new* pressure.
        self.states += self.pressures * dt

    @property
    def cpu_states(self):
        return self.states.cpu().numpy() 