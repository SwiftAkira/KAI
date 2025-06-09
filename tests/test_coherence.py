import numpy as np, pytest
from pressure_agi.engine.field import Field

def coherence(field):
    states = field.cpu_states
    return np.mean(np.cos(states - states.mean()))

def test_stability():
    f = Field(n=100); [f.step(0.02) for _ in range(400)]
    assert coherence(f) >= 0.85

def test_stability_gpu():
    f = Field(n=100, device='gpu'); [f.step(0.02) for _ in range(400)]
    assert coherence(f) >= 0.85 