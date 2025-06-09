import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import copy

# Note: We are no longer strictly dependent on Gymnasium.
# The planner can work with any object that has a 'step' method and can be deep-copied.
# import gymnasium as gym

from .field import Field
from .critic import Critic

@dataclass
class GoalNode:
    """Represents a potential goal state for the planner."""
    vector: torch.Tensor
    reward_est: float = 0.0
    status: str = "pending" # "pending", "in_progress", "achieved", "failed"
    parent: Optional['GoalNode'] = None
    children: List['GoalNode'] = field(default_factory=list)

def monte_pressure_rollout(
    field_state: Field,
    goal_vector: torch.Tensor,
    critic: Critic,
    settle_steps: int = 80,
    env_clone: Optional[object] = None,
    action_to_take: Optional[object] = None,
) -> float:
    """
    Simulates injecting a goal and settling the field to evaluate the outcome.
    The reward can be a combination of internal score and external environment reward.
    """
    # --- Internal Simulation ---
    sim_field = copy.deepcopy(field_state)

    min_dim = min(sim_field.n, len(goal_vector))
    if min_dim > 0:
        sim_field.pressures[:min_dim] += goal_vector[:min_dim]

    for _ in range(settle_steps):
        sim_field.step()
    
    # The primary score comes from the critic's evaluation of the final state
    internal_reward, _ = critic.get_composite_score(sim_field)
    total_reward = internal_reward
    
    # --- Optional: External Simulation ---
    if env_clone is not None and action_to_take is not None:
        _obs, external_reward, _done = env_clone.step(action_to_take)
        # The external reward is added to the internal score
        total_reward += external_reward

    return total_reward

class Planner:
    """A simple Monte Carlo Tree Search-style planner."""
    def __init__(self, critic: Critic, epsilon: float = 0.1, expansion_threshold: float = 0.05):
        self.root = GoalNode(vector=torch.empty(0), status="root")
        self.critic = critic
        self.epsilon = epsilon # Chance to explore a random goal
        self.expansion_threshold = expansion_threshold

    def evaluate_candidates(
        self,
        field: Field,
        candidates: List[Tuple[torch.Tensor, Optional[object]]],
        env: Optional[object] = None,
        max_candidates: int = 5
    ):
        """
        Evaluates a list of candidate goals using rollouts.
        Each candidate is a tuple of (goal_vector, action_for_env).
        The 'action_for_env' can be None if there is no environment.
        """
        for goal_vector, action_name in candidates[:max_candidates]:
            env_clone = copy.deepcopy(env) if env else None
            
            delta_reward = monte_pressure_rollout(
                field_state=field,
                goal_vector=goal_vector,
                critic=self.critic,
                env_clone=env_clone,
                action_to_take=action_name,
                settle_steps=20
            )
            
            if delta_reward > self.expansion_threshold:
                new_node = GoalNode(vector=goal_vector, reward_est=delta_reward, parent=self.root)
                self.root.children.append(new_node)
    
    def select_action(self) -> Optional[GoalNode]:
        """
        Selects the best goal to pursue based on estimated reward (ε-greedy).
        """
        if not self.root.children:
            return None

        # Epsilon-greedy selection
        if torch.rand(1).item() < self.epsilon:
            # Explore: pick a random child
            rand_idx = torch.randint(0, len(self.root.children), (1,)).item()
            return self.root.children[rand_idx]
        else:
            # Exploit: pick the best child
            best_child = max(self.root.children, key=lambda node: node.reward_est)
            return best_child
            
    def prune_goals(self):
        """Clears the current tree for the next planning cycle."""
        self.root.children = [] 