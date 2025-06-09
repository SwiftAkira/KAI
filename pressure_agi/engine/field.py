import numpy as np
import torch
from .node import Node

class Field:
    def __init__(self, n=10, device='cpu', dtype=torch.float32, global_gain=0.01, impulse_gain=0.1):
        self.n = n
        self.device = device
        self.dtype = dtype
        self.global_gain = global_gain
        self.impulse_gain = impulse_gain

        self.states = torch.rand(n, device=device, dtype=dtype)
        self.pressures = torch.zeros(n, device=device, dtype=dtype)

    def step(self, dt=0.01, friction=0.05):
        if self.n == 0:
            return

        # Vectorized state differences
        state_diffs = self.states.unsqueeze(1) - self.states.unsqueeze(0)

        # Calculate pairwise forces
        pairwise_forces = torch.sum(state_diffs, dim=1)

        # Apply global broadcast term (attraction to herd)
        global_mean = pairwise_forces.mean()
        broadcast_forces = self.global_gain * (global_mean - pairwise_forces)
        total_forces = pairwise_forces + broadcast_forces

        # Update pressures based on total forces
        self.pressures += total_forces * dt

        # Apply friction to pressure
        self.pressures -= friction * self.pressures

        # Update states based on pressure
        self.states += self.pressures * dt

        # Clip only pressures to prevent explosion, let states rise
        self.pressures.clamp_(-1, 1)

    @property
    def cpu_states(self):
        return self.states.cpu().numpy() 