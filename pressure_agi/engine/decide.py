import torch
from pressure_agi.engine.field import Field

def decide(field: Field, theta_pos: float, theta_neg: float):
    """
    Decides on an action based on the mean state of the field and configured thresholds.
    """
    mean_state = torch.mean(field.states).item()
    if mean_state > theta_pos:
        return "say_positive"
    elif mean_state < theta_neg:
        return "say_negative"
    else:
        return "say_neutral" 