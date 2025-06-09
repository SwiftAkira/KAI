import networkx as nx

class EpisodicMemory:
    def __init__(self):
        self.G = nx.DiGraph()

    def store(self, snapshot):
        idx = len(self.G)
        self.G.add_node(idx, **snapshot)
        if idx:  # link to previous
            self.G.add_edge(idx-1, idx, weight=1.0) 