import numpy as np
import torch
import time
import copy
from rich.console import Console
from rich.text import Text

from pressure_agi.engine.field import Field
from pressure_agi.engine.planner import Planner, GoalNode
from pressure_agi.engine.critic import Critic
from pressure_agi.io.adapter import GridWorldAdapter
from pressure_agi.environments.grid_world import GridWorld

# --- Benchmark Harness ---

def run_benchmark(max_steps=50):
    """Runs the GridWorld navigation benchmark."""
    
    # --- Setup ---
    env = GridWorld(size=5)
    device = 'cpu'
    critic = Critic()
    planner = Planner(critic=critic, epsilon=0.2, expansion_threshold=-1.0)

    # --- Agent Setup ---
    goal_vectors = {
        name: torch.tensor([1 if i == idx else 0 for i, _ in enumerate(env.action_space)], dtype=torch.float64, device=device)
        for idx, name in enumerate(env.action_space)
    }
    adapter = GridWorldAdapter(goal_vectors)
    
    start_time = time.time()
    
    # --- Run Loop ---
    obs = env.reset()
    field = Field(n=len(obs), device=device)
    field.states = obs.clone()

    print("Initial State:")
    env.render()
    
    success = False
    for step in range(max_steps):
        # --- Planning ---
        candidate_tuples = [
            (vector, name) # Pass the action name (string) now
            for name, vector in goal_vectors.items()
        ]
        
        planner.evaluate_candidates(field, candidate_tuples, env)
        
        selected_goal = planner.select_action()
        
        if selected_goal is None:
            # If planner fails to produce a goal, choose a random action
            action_name = np.random.choice(env.action_space)
        else:
            # Use the adapter to convert the planner's goal into an env action
            action_name = adapter.adapt(selected_goal)
        
        print(f"\n--- Step {step+1}/{max_steps} ---")
        print(f"Agent chose action: '{action_name}' (Reward Est: {selected_goal.reward_est if selected_goal else 'N/A'})")

        # --- Environment Step ---
        new_obs, reward, done = env.step(action_name)
        field.states = new_obs.clone() # Update field with new state
        
        env.render()

        if done:
            success = True
            print(f"\n[bold green]Success![/bold green] Agent reached the goal in {step + 1} steps.")
            break

    if not success:
        print(f"\n[bold red]Failure.[/bold red] Agent did not reach the goal within {max_steps} steps.")

    end_time = time.time()
    print(f"\nBenchmark finished in {end_time - start_time:.2f} seconds.")
    return success

if __name__ == "__main__":
    np.random.seed(0) # Make the benchmark deterministic
    torch.manual_seed(0)
    run_benchmark() 