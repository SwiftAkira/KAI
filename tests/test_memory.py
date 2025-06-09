from pressure_agi.engine.memory import EpisodicMemory

def test_memory_growth():
    mem = EpisodicMemory()
    assert len(mem.G) == 0
    for i in range(5):
        mem.store({"t":i})
    assert len(mem.G) == 5

def test_retrieve_last():
    mem = EpisodicMemory()
    for i in range(10):
        mem.store({"t": i, "value": f"snapshot_{i}"})
    
    # Retrieve last one
    last_one = mem.retrieve_last(k=1)
    assert len(last_one) == 1
    assert last_one[0]['value'] == 'snapshot_9'

    # Retrieve last three
    last_three = mem.retrieve_last(k=3)
    assert len(last_three) == 3
    assert last_three[0]['value'] == 'snapshot_7'
    assert last_three[2]['value'] == 'snapshot_9'

    # Retrieve more than available
    all_nodes = mem.retrieve_last(k=20)
    assert len(all_nodes) == 10
    assert all_nodes[0]['value'] == 'snapshot_0' 