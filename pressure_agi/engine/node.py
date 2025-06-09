import numpy as np

class Node:
    __slots__ = ("state", "valence", "pressure", "tags")
    def __init__(self, state=0.0, valence=0.0, tags=None):
        self.state    = float(state)
        self.valence  = float(valence)
        self.pressure = 0.0
        self.tags     = set(tags or ())

    def propagate(self, dt: float):
        self.state += self.pressure * dt
        # simple decay so energy doesn't explode
        self.pressure *= 0.8 