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
    env_clone: object, # Changed from gym.Env to object for more flexibility
    action_to_take: int,
    settle_steps: int = 80
) -> float:
    """
    Simulates injecting a goal into a clone of the field AND taking an action
    in a clone of the environment to estimate the reward.
    The reward is the external reward penalized by internal instability.
    """
    # --- Internal Simulation: Calculate instability penalty ---
    sim_field = copy.deepcopy(field_state)

    # Inject the candidate goal's vector as a pressure impulse
    min_dim = min(sim_field.n, len(goal_vector))
    if min_dim > 0:
        sim_field.pressures[:min_dim] += goal_vector[:min_dim]

    # Settle the simulated field
    for _ in range(settle_steps):
        sim_field.step()
    entropy_penalty = critic.calculate_entropy(sim_field)

    # --- External Simulation: Get reward from the environment ---
    # The environment clone is already at the correct state.
    # We don't need the observation, just the reward from the step.
    # The action is now the name of the action, not an index.
    _obs, reward, done = env_clone.step(action_to_take)

    # The final estimated reward is the external reward minus the internal penalty.
    final_reward = reward - (entropy_penalty * 0.1) # Scale penalty to be competitive with reward
    return final_reward

class Planner:
    """A simple Monte Carlo Tree Search-style planner."""
    def __init__(self, critic: Critic, epsilon: float = 0.1, expansion_threshold: float = 0.05):
        self.root = GoalNode(vector=torch.empty(0), status="root")
        self.critic = critic
        self.epsilon = epsilon # Chance to explore a random goal
        self.expansion_threshold = expansion_threshold

    def evaluate_candidates(self, field: Field, candidates: List[Tuple[torch.Tensor, object]], env: object, max_candidates: int = 5):
        """
        Evaluates a list of candidate goals (vector, action_name) using rollouts.
        """
        for goal_vector, action_name in candidates[:max_candidates]:
            # Each rollout needs a fresh clone of the environment at the current state.
            env_clone = copy.deepcopy(env)
            
            delta_reward = monte_pressure_rollout(
                field, goal_vector, self.critic, env_clone, action_name, settle_steps=20 # Less steps for speed
            )
            
            # Note: The original expansion threshold was designed for coherence,
            # it might be too high for env rewards. We'll keep it for now.
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