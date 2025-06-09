import torch
from pressure_agi.engine.node import Node
from pressure_agi.engine.field import Field

def inject(field: Field, packets: list):
    """
    Injects new percepts into the field.
    This involves adding new Node objects and resizing the state and pressure tensors.
    The initial state of the new nodes is set to their valence.
    """
    num_new_nodes = len(packets)
    if num_new_nodes == 0:
        return

    new_valences = []
    for p in packets:
        valence = p.get("valence", 0.0)
        node = Node(valence=valence, tags={p["type"], "percept"})
        field.nodes.append(node)
        new_valences.append(valence)

    # Resize field tensors to accommodate the new nodes
    # The initial state of the new nodes is their valence
    new_states = torch.tensor(new_valences, dtype=field.dtype, device=field.device)
    new_pressures = torch.zeros(num_new_nodes, dtype=field.dtype, device=field.device)

    field.states = torch.cat((field.states, new_states))
    field.pressures = torch.cat((field.pressures, new_pressures))

    field.n += num_new_nodes 