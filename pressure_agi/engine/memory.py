import networkx as nx

class EpisodicMemory:
    def __init__(self):
        self.G = nx.DiGraph()
        self.node_counter = 0

    def store(self, snapshot):
        idx = self.node_counter
        self.G.add_node(idx, **snapshot)
        if idx > 0:  # link to previous
            self.G.add_edge(idx-1, idx, weight=1.0)
        self.node_counter += 1

    def retrieve_last(self, k: int = 1):
        """
        Retrieves the last k snapshots from memory.
        Returns a list of snapshots, or an empty list if memory is empty.
        """
        if not self.G:
            return []
        
        # Nodes are indexed from 0 to node_counter - 1
        start_node = max(0, self.node_counter - k)
        return [self.G.nodes[i] for i in range(start_node, self.node_counter)] 