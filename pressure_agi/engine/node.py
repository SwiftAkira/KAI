import numpy as np

class Node:
    __slots__ = ("valence", "tags")
    def __init__(self, valence=0.0, tags=None):
        self.valence  = float(valence)
        self.tags     = set(tags or ())

    def propagate(self, dt: float):
        self.state += self.pressure * dt
        # simple decay so energy doesn't explode
        self.pressure *= 0.8 