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

    def calculate_entropy(self, field: Field) -> float:
        """Calculates and returns the entropy of the field (std dev of states)."""
        if field.n == 0:
            return 0.0
        entropy = torch.std(field.states).item()
        self.last_entropy = entropy
        return entropy

    def apply_penalty(self, field: Field):
        """Applies a pressure penalty if entropy is above the threshold."""
        entropy = self.last_entropy # Use the last calculated entropy
        if entropy > self.entropy_threshold:
            # Apply a uniform negative pressure to cool the system down
            penalty = self.penalty_factor * (entropy - self.entropy_threshold)
            field.pressures -= penalty

    def evaluate(self, field: Field):
        """
        Calculates the entropy of the field and applies a penalty if it's too high.
        This is a convenience method that combines calculation and application.
        """
        self.calculate_entropy(field)
        self.apply_penalty(field) 