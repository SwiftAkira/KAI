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
    def __init__(self, n=10, device='cpu', dtype=torch.float64, global_gain=0.15, impulse_gain=0.4):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.global_gain = global_gain
        self.impulse_gain = impulse_gain

        self.states = torch.rand(n, device=device, dtype=dtype)
        self.pressures = torch.zeros(n, device=device, dtype=dtype)

    def step(self, dt=0.01, friction=0.025):
        if self.n == 0:
            return

        # Vectorized state differences
        state_diffs = self.states.unsqueeze(1) - self.states.unsqueeze(0)

        # Calculate pairwise forces
        pairwise_forces = torch.sum(state_diffs, dim=1)
        
        if pairwise_forces.numel() == 0:
            return

        global_mean = pairwise_forces.mean()
        broadcast_forces = self.global_gain * (global_mean - pairwise_forces)
        total_forces = pairwise_forces + broadcast_forces

        # Update pressures based on total forces
        self.pressures += total_forces * dt

        # Apply friction to pressure
        self.pressures -= friction * self.pressures
        
        # CRITICAL: Clamp pressure *before* it updates state to prevent explosion
        self.pressures.clamp_(-1, 1)

        # Update states based on pressure
        self.states += self.pressures * dt

    @property
    def cpu_states(self):
        return self.states.cpu().numpy() 