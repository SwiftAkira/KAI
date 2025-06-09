import torch
from pressure_agi.engine.field import Field
from pressure_agi.engine.critic import Critic

def test_critic_dampens_pressure_on_high_entropy():
    """
    Tests that the Critic applies a negative pressure penalty when the
    field's state entropy is high.
    """
    # 1. Setup
    field = Field(n=10, device='cpu')
    critic = Critic()

    # Artificially create a high-entropy state
    # A simple way is to set half the nodes to a high positive state and half to a high negative state
    high_entropy_states = torch.tensor([-5.0] * 5 + [5.0] * 5, dtype=field.dtype, device=field.device)
    field.states = high_entropy_states

    # Ensure initial pressure is zero
    initial_pressure_sum = torch.sum(field.pressures).item()
    assert initial_pressure_sum == 0.0

    # 2. Action
    critic.evaluate(field)

    # 3. Assert
    # The pressure should now be negative because of the penalty
    final_pressure_sum = torch.sum(field.pressures).item()
    assert final_pressure_sum < initial_pressure_sum
    assert final_pressure_sum < 0

def test_critic_does_nothing_on_low_entropy():
    """
    Tests that the Critic does not apply any penalty when the field's
    state entropy is low.
    """
    # 1. Setup
    field = Field(n=10, device='cpu')
    critic = Critic()

    # Artificially create a low-entropy state (all nodes are the same)
    low_entropy_states = torch.tensor([0.1] * 10, dtype=field.dtype, device=field.device)
    field.states = low_entropy_states

    initial_pressure_sum = torch.sum(field.pressures).item()
    assert initial_pressure_sum == 0.0

    # 2. Action
    critic.evaluate(field)

    # 3. Assert
    # The pressure should remain unchanged
    final_pressure_sum = torch.sum(field.pressures).item()
    assert final_pressure_sum == initial_pressure_sum
    assert final_pressure_sum == 0.0 