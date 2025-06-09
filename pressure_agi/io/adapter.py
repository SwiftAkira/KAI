from abc import ABC, abstractmethod
import torch
from typing import Any, Dict
import random

# Import the types that represent internal actions
from pressure_agi.engine.decide import Action
from pressure_agi.engine.planner import GoalNode

class Adapter(ABC):
    """Base class for action adapters that convert internal actions to external commands."""
    @abstractmethod
    def adapt(self, action_object: Any) -> Any:
        """Converts an internal action object (like Action or GoalNode) to an external command."""
        pass

class TextAdapter(Adapter):
    """Adapts an internal Action into a human-readable text command for the REPL."""

    def __init__(self):
        self.positive_responses = [
            "I'm glad to hear that!",
            "That's great!",
            "Wonderful!",
            "That's good to know."
        ]
        self.negative_responses = [
            "I'm sorry to hear that.",
            "That's unfortunate.",
            "Oh no.",
            "That sounds difficult."
        ]

    def adapt(self, action_object: Any) -> str:
        """Encodes the action type into a string."""
        if isinstance(action_object, Action):
            if action_object.type == "positive":
                return random.choice(self.positive_responses)
            elif action_object.type == "negative":
                return random.choice(self.negative_responses)
            else:
                return "..." # More subtle than 'do_nothing'
        elif action_object is None:
            return "..."
        # The REPL doesn't use the planner, so we don't expect GoalNodes here.
        # We can add that logic if the REPL gets more complex.
        else:
            # Fallback for unexpected types
            return "..."

class GridWorldAdapter(Adapter):
    """Adapts a GoalNode from the planner to a command for the GridWorld environment."""
    def __init__(self, goal_vectors: Dict[str, torch.Tensor]):
        """
        Initializes with a dictionary mapping action names to their corresponding
        goal vectors.
        """
        self.goal_vectors = goal_vectors

    def adapt(self, action_object: Any) -> str:
        """
        Finds the action name whose vector is closest to the GoalNode's vector.
        """
        if not isinstance(action_object, GoalNode):
            raise TypeError(f"GridWorldAdapter requires a GoalNode, but received {type(action_object)}")

        winning_vector = action_object.vector
        best_match_name = None
        min_dist = float('inf')

        # Find the candidate vector that is most similar to the winning vector
        for name, vec in self.goal_vectors.items():
            dist = torch.norm(winning_vector - vec)
            if dist < min_dist:
                min_dist = dist
                best_match_name = name
        
        if best_match_name is None:
            # This should not happen if goal_vectors is not empty.
            raise ValueError("Could not find a matching action for the given GoalNode.")
            
        return best_match_name 