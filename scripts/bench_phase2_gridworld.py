import numpy as np
import torch
import time
import copy
from rich.console import Console
from rich.text import Text

from pressure_agi.engine.field import Field
from pressure_agi.engine.planner import Planner
from pressure_agi.engine.critic import Critic

# --- A Simple Grid World Environment ---

class GridWorld:
    """
    A simple, cloneable grid world environment for testing the planner.
    The agent ('A') must navigate to the goal ('G').
    """
    def __init__(self, size=5):
        self.size = size
        self.action_space = ['move_north', 'move_south', 'move_east', 'move_west']
        self.action_map = {
            'move_north': np.array([-1, 0]),
            'move_south': np.array([1, 0]),
            'move_east':  np.array([0, 1]),
            'move_west':  np.array([0, -1]),
        }
        self.reset()

    def reset(self):
        """Places agent and goal at random non-overlapping positions."""
        self.agent_pos = np.random.randint(0, self.size, size=2)
        self.goal_pos = np.random.randint(0, self.size, size=2)
        while np.array_equal(self.agent_pos, self.goal_pos):
            self.goal_pos = np.random.randint(0, self.size, size=2)
        return self._get_obs()

    def step(self, action_name):
        """Executes an action and returns obs, reward, done."""
        if action_name not in self.action_map:
            raise ValueError(f"Unknown action: {action_name}")

        # Calculate new position and clip it to stay within bounds
        move = self.action_map[action_name]
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)

        # Check for goal
        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if done else -0.1 # Reward for success, small penalty for each step

        return self._get_obs(), reward, done

    def _get_obs(self):
        """Returns the state vector [agent_y, agent_x, goal_y, goal_x]."""
        return torch.tensor(np.concatenate([self.agent_pos, self.goal_pos]), dtype=torch.float64)

    def render(self):
        """Prints the grid to the console."""
        grid = np.full((self.size, self.size), " . ", dtype=object)
        grid[self.agent_pos[0], self.agent_pos[1]] = Text(" A ", style="bold red")
        grid[self.goal_pos[0], self.goal_pos[1]] = Text(" G ", style="bold green")
        console = Console()
        for row in grid:
            console.print("".join(str(cell) for cell in row))

# --- Benchmark Harness ---

def run_benchmark(max_steps=50):
    """Runs the GridWorld navigation benchmark."""
    
    # --- Setup ---
    env = GridWorld(size=5)
    device = 'cpu'
    critic = Critic()
    # Epsilon needs to be high enough to explore, but not so high it's always random
    planner = Planner(critic=critic, epsilon=0.2, expansion_threshold=-1.0) # Allow negative rewards

    # --- Agent Setup ---
    # Goal vectors are one-hot encodings for each possible action
    goal_vectors = {
        name: torch.tensor([1 if i == idx else 0 for i, _ in enumerate(env.action_space)], dtype=torch.float64, device=device)
        for idx, name in enumerate(env.action_space)
    }
    
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
            (vector, name)
            for name, vector in goal_vectors.items()
        ]
        
        # This is where the problematic deepcopy was. It should now work.
        planner.evaluate_candidates(field, candidate_tuples, env)
        
        selected_goal = planner.select_action()
        
        if selected_goal is None:
            action_name = np.random.choice(env.action_space)
        else:
            best_vec = selected_goal.vector
            for name, vec in goal_vectors.items():
                if torch.equal(best_vec, vec):
                    action_name = name
                    break
        
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