import numpy as np
from .node import Node

class Field:
    def __init__(self, n=100, k=0.05, damp=0.02):
        self.nodes = [Node() for _ in range(n)]
        self.K = k
        self.damp = damp

    def step(self, dt=0.01):
        forces = np.zeros(len(self.nodes))
        # naive O(N²) pairwise spring — optimise later
        for i, a in enumerate(self.nodes):
            for j, b in enumerate(self.nodes[i+1:], i+1):
                d = b.state - a.state
                f = self.K * d
                forces[i] += f
                forces[j] -= f
        for node, f in zip(self.nodes, forces):
            node.pressure = f - self.damp * node.pressure
            node.propagate(dt) 