from pressure_agi.engine.memory import EpisodicMemory
import pytest
import torch

def test_memory_growth():
    mem = EpisodicMemory()
    assert mem.size == 0
    for i in range(10):
        # The new store method expects 'vector' and 'decision' keys.
        mem.store({"t": i, "vector": torch.tensor([float(i)]), "decision": f"action_{i}"})
    assert mem.size == 10

def test_retrieve_last():
    mem = EpisodicMemory()
    for i in range(10):
        mem.store({"t": i, "vector": torch.tensor([float(i)]), "decision": f"snapshot_{i}"})

    last_one = mem.retrieve_last(k=1)
    assert len(last_one) == 1
    assert last_one[0]['t'] == 9
    
    last_five = mem.retrieve_last(k=5)
    assert len(last_five) == 5
    assert last_five[0]['t'] == 9
    assert last_five[4]['t'] == 5

def test_memory_storage_and_retrieval():
    memory = EpisodicMemory()
    snapshot1 = {"t": 1, "vector": torch.tensor([1.0, 0.0]), "decision": "action1"}
    snapshot2 = {"t": 2, "vector": torch.tensor([0.0, 1.0]), "decision": "action2"}
    
    memory.store(snapshot1)
    memory.store(snapshot2)
    
    last = memory.retrieve_last(k=1)
    assert len(last) == 1
    assert last[0]["t"] == 2
    assert last[0]["decision"] == "action2"

    two_last = memory.retrieve_last(k=2)
    assert len(two_last) == 2
    # retrieval order is last-to-first
    assert two_last[0]["t"] == 2
    assert two_last[1]["t"] == 1

def test_tensor_memory_retrieval():
    """Tests storing and retrieving from the new tensor-based memory."""
    memory = EpisodicMemory(max_episodes=10, max_dim=5)
    
    # Create 5 orthogonal vectors
    v1 = torch.tensor([1.0, 0, 0, 0, 0], dtype=torch.float64)
    v2 = torch.tensor([0, 1.0, 0, 0, 0], dtype=torch.float64)
    v3 = torch.tensor([0, 0, 1.0, 0, 0], dtype=torch.float64)
    v4 = torch.tensor([0, 0, 0, 1.0, 0], dtype=torch.float64)
    v5 = torch.tensor([0, 0, 0, 0, 1.0], dtype=torch.float64)
    
    vectors = [v1, v2, v3, v4, v5]
    
    # Store them
    for i, v in enumerate(vectors):
        memory.store({"t": i, "vector": v, "decision": f"vector_{i+1}"})

    assert memory.size == 5

    # Query for v3
    query_vector = v3
    retrieved = memory.retrieve(query_vector, k=1)

    # Verification
    assert len(retrieved) == 1
    retrieved_snapshot = retrieved[0]
    
    assert retrieved_snapshot["decision"] == "vector_3"
    assert torch.allclose(retrieved_snapshot["vector"], v3)

    # Test retrieval with a slightly noisy vector
    noisy_query = v5 + torch.tensor([0.01, 0, -0.02, 0, 0.05])
    retrieved_noisy = memory.retrieve(noisy_query, k=1)
    assert len(retrieved_noisy) == 1
    assert retrieved_noisy[0]["decision"] == "vector_5" 