import numpy as np
import torch
import pytest

from pressure_agi.environments.grid_world import GridWorld
from pressure_agi.engine.field import Field
from pressure_agi.engine.planner import Planner
from pressure_agi.engine.critic import Critic
from pressure_agi.io.adapter import GridWorldAdapter

def get_reward_curve(seed, max_steps=10):
    """
    Runs a simulation and returns the sequence of composite scores from the critic.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = GridWorld(size=5)
    device = 'cpu'
    critic = Critic()
    planner = Planner(critic=critic, epsilon=0.1, expansion_threshold=-1.0)
    
    goal_vectors = {
        name: torch.tensor([1 if i == idx else 0 for i, _ in enumerate(env.action_space)], dtype=torch.float64, device=device)
        for idx, name in enumerate(env.action_space)
    }
    adapter = GridWorldAdapter(goal_vectors)

    obs = env.reset()
    field = Field(n=len(obs), device=device)
    field.states = obs.clone()

    reward_curve = []
    for _ in range(max_steps):
        # Log the current score
        score, _ = critic.get_composite_score(field)
        reward_curve.append(score)

        candidate_tuples = [(vector, name) for name, vector in goal_vectors.items()]
        planner.evaluate_candidates(field, candidate_tuples, env)
        
        selected_goal = planner.select_action()
        
        if selected_goal is None:
            action_name = np.random.choice(env.action_space)
        else:
            action_name = adapter.adapt(selected_goal)
        
        new_obs, _, done = env.step(action_name)
        field.states = new_obs.clone()

        if done:
            break
            
    return reward_curve

def test_determinism_with_fixed_seed():
    """
    Tests that rerunning a simulation with the same seed produces identical reward curves.
    Pass criteria: pytest -q --reruns 3 -> identical reward curves (assert max Δ < 1e-6).
    
    The pytest --reruns flag can be used with this test. Here, we simulate
    it by running it multiple times in a loop.
    """
    num_reruns = 3
    fixed_seed = 42
    
    # Run once to get the baseline curve
    baseline_curve = get_reward_curve(seed=fixed_seed)
    
    for i in range(num_reruns - 1):
        rerun_curve = get_reward_curve(seed=fixed_seed)
        
        # Check that the curves are the same length
        assert len(baseline_curve) == len(rerun_curve), \
            f"Run {i+2} produced a reward curve of different length."
            
        # Check that the values are almost identical
        max_delta = np.max(np.abs(np.array(baseline_curve) - np.array(rerun_curve)))
        
        print(f"Determinism check run {i+2}: Max delta = {max_delta}")
        assert max_delta < 1e-6, \
            f"Run {i+2} was not deterministic. Max reward delta was {max_delta}" 