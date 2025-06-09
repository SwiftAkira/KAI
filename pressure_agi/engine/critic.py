import torch
from .field import Field

class Critic:
    """
    Evaluates the state of the Field, providing metrics for stability and performance.
    """
    def __init__(self, entropy_threshold=0.9, penalty_factor=0.1):
        self.entropy_threshold = entropy_threshold
        self.penalty_factor = penalty_factor
        self.last_entropy = 0.0
        self.last_score = 0.0
        self.frozen_steps = 0

    def freeze(self, steps: int):
        """Freezes the critic's reward signal for a number of steps."""
        self.frozen_steps = steps

    def calculate_entropy(self, field: Field) -> float:
        """Calculates and returns the entropy of the field (std dev of states)."""
        if field.n == 0:
            return 0.0
        entropy = torch.std(field.states).item()
        self.last_entropy = entropy
        return entropy

    def calculate_energy(self, field: Field) -> float:
        """Calculates the total kinetic energy of the field's pressures."""
        if field.n == 0:
            return 0.0
        # Energy = Σ ½·pressure²
        return torch.sum(0.5 * field.pressures.pow(2)).item()

    def get_composite_score(self, field: Field, with_grad: bool = False) -> tuple[float, torch.Tensor | None]:
        """
        Calculates a composite score based on negative entropy and energy.
        Lower is better. The planner will try to maximize this value.
        Optionally returns gradients for logging.
        """
        if field.n == 0:
            return 0.0, None

        states = field.states.clone()
        pressures = field.pressures.clone()

        if with_grad:
            states.requires_grad_(True)
            pressures.requires_grad_(True)

        entropy = torch.std(states)
        energy = torch.sum(0.5 * pressures.pow(2))
        
        # Composite score = -entropy - 0.1 * energy
        score = -entropy - (0.1 * energy)

        grad = None
        if with_grad:
            score.backward()
            grad = states.grad.clone()

        self.last_score = score.item() # Cache the new score
        return self.last_score, grad

    def apply_penalty(self, field: Field):
        """Applies a pressure penalty if entropy is above the threshold."""
        # First, check if the critic is frozen
        if self.frozen_steps > 0:
            self.frozen_steps -= 1
            # Return the last known score without recalculating
            return self.last_score, None

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