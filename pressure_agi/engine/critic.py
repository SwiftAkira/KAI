import torch
from .field import Field

class Critic:
    """
    The Critic evaluates the state of the field and applies a penalty
    to prevent runaway energy (high entropy).
    """
    def __init__(self, entropy_threshold=0.5, penalty_factor=0.1):
        self.entropy_threshold = entropy_threshold
        self.penalty_factor = penalty_factor
        self.last_entropy = 0.0

    def evaluate(self, field: Field):
        """
        Calculates the entropy of the field and applies a pressure penalty
        if it exceeds the threshold.
        """
        if field.n == 0:
            return

        # Calculate entropy (standard deviation of states)
        entropy = torch.std(field.states).item()
        self.last_entropy = entropy

        # Calculate penalty only if entropy is above the threshold
        if entropy > self.entropy_threshold:
            penalty = self.penalty_factor * (entropy - self.entropy_threshold)
            
            # Apply penalty to the pressure of all nodes
            field.pressures -= penalty 