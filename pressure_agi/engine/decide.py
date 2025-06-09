import torch
from pressure_agi.engine.field import Field

def decide(field: Field):
    """
    Decides on an action based on the mean state of the field.
    """
    mean_state = torch.mean(field.states).item()
    return "go" if mean_state > 0 else "stop" 