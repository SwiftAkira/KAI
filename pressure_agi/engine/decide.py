from dataclasses import dataclass
from typing import Optional
import torch

from .field import Field

@dataclass
class Action:
    """Represents a simple, decided action."""
    type: str  # "positive", "negative", "neutral"
    vector: Optional[torch.Tensor] = None

def decide(field: Field, pos_threshold: float, neg_threshold: float) -> Optional[Action]:
    """
    Analyzes the field's coherence and decides on a simple action.
    """
    if field.n == 0:
        return None

    coherence = torch.mean(field.states).item()

    if coherence > pos_threshold:
        return Action(type="positive")
    elif coherence < neg_threshold:
        return Action(type="negative")
    else:
        return None 