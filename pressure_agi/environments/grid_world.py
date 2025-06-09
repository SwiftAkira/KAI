import numpy as np
import torch
from rich.console import Console
from rich.text import Text

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
        self.agent_pos = None
        self.goal_pos = None
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

        move = self.action_map[action_name]
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)

        done = np.array_equal(self.agent_pos, self.goal_pos)
        reward = 1.0 if done else -0.1

        return self._get_obs(), reward, done

    def _get_obs(self):
        """Returns the state vector [agent_y, agent_x, goal_y, goal_x]."""
        return torch.tensor(np.concatenate([self.agent_pos, self.goal_pos]), dtype=torch.float64)

    def render(self):
        """Prints the grid to the console."""
        grid = np.full((self.size, self.size), " . ", dtype=object)
        if self.agent_pos is not None:
            grid[self.agent_pos[0], self.agent_pos[1]] = Text(" A ", style="bold red")
        if self.goal_pos is not None:
            grid[self.goal_pos[0], self.goal_pos[1]] = Text(" G ", style="bold green")
        console = Console()
        for row in grid:
            console.print("".join(str(cell) for cell in row)) 