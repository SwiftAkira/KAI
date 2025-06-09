import torch
from pressure_agi.engine.node import Node
from pressure_agi.engine.field import Field

def inject(field: Field, percepts: list[dict]):
    """
    Injects new percepts into the field by adding nodes.
    """
    if not percepts:
        return
        
    num_new_nodes = len(percepts)
    
    new_states = torch.tensor([p['valence'] for p in percepts], dtype=field.dtype, device=field.device)
    new_pressures = torch.zeros(num_new_nodes, dtype=field.dtype, device=field.device)

    field.states = torch.cat((field.states, new_states))
    field.pressures = torch.cat((field.pressures, new_pressures))

    field.n += num_new_nodes 