import numpy as np
import torch
from .node import Node

class Field:
    def __init__(self, n=100, k=0.05, damp=0.02, device='cpu'):
        self.nodes = [Node() for _ in range(n)]
        self.K = k
        self.damp = damp
        self.n = n
        self.dtype = torch.float32

        if device == 'gpu' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
            self.dtype = torch.float64

        self.states = torch.zeros(n, dtype=self.dtype, device=self.device)
        self.pressures = torch.zeros(n, dtype=self.dtype, device=self.device)

    def step(self, dt=0.01):
        total_state = torch.sum(self.states)
        forces = self.K * (total_state - self.n * self.states)

        self.pressures = forces - self.damp * self.pressures
        self.states += self.pressures * dt
        self.pressures *= 0.8  # Decay from original propagate

    @property
    def cpu_states(self):
        return self.states.cpu().numpy() 