import torch
import pathlib

class EpisodicMemory:
    """
    A memory system that stores episodes as flat tensors for efficient similarity search.
    It uses a fixed-size circular buffer to store memories.
    """
    def __init__(self, max_episodes: int = 2000, max_dim: int = 512, device: str = 'cpu'):
        """
        Initializes the memory store.
        Args:
            max_episodes: The maximum number of episodes to store.
            max_dim: The maximum dimension of the state vectors to be stored (padding).
            device: The torch device to store tensors on.
        """
        self.max_episodes = max_episodes
        self.max_dim = max_dim
        self.device = device

        # Initialize the tensor stores
        self.mem_states = torch.zeros((max_episodes, max_dim), device=device, dtype=torch.float64)
        self.mem_time = torch.zeros(max_episodes, device=device, dtype=torch.int64)
        self.mem_tags = ["" for _ in range(max_episodes)]
        
        self.pointer = 0
        self.size = 0

    def save_to_disk(self, file_path: str):
        """Saves the current memory state to a file."""
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'mem_states': self.mem_states,
            'mem_time': self.mem_time,
            'mem_tags': self.mem_tags,
            'pointer': self.pointer,
            'size': self.size
        }, file_path)
        print(f"[Memory] Snapshot saved to {file_path}")

    def load_from_disk(self, file_path: str):
        """Loads a memory state from a file."""
        path = pathlib.Path(file_path)
        if not path.exists():
            print(f"[Memory] Snapshot file not found: {file_path}")
            return

        data = torch.load(file_path)
        self.mem_states = data['mem_states'].to(self.device)
        self.mem_time = data['mem_time'].to(self.device)
        self.mem_tags = data['mem_tags']
        self.pointer = data['pointer']
        self.size = data['size']
        print(f"[Memory] Snapshot loaded from {file_path}")

    def _pad_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Pads a vector to the max_dim size."""
        padded = torch.zeros(self.max_dim, device=self.device, dtype=torch.float64)
        # Handle vectors larger than max_dim by truncating
        current_dim = min(vector.shape[0], self.max_dim)
        padded[:current_dim] = vector[:current_dim]
        return padded

    def store(self, snapshot: dict):
        """
        Stores a new episode snapshot in memory. Overwrites the oldest entry if full.
        Args:
            snapshot (dict): A dictionary containing 't', 'vector', and 'decision'.
        """
        # Pad the state vector to the fixed dimension
        padded_vector = self._pad_vector(snapshot['vector'])

        # Store the data at the current pointer location
        self.mem_states[self.pointer] = padded_vector
        self.mem_time[self.pointer] = snapshot['t']
        self.mem_tags[self.pointer] = snapshot['decision']

        # Update pointer and size
        self.pointer = (self.pointer + 1) % self.max_episodes
        self.size = min(self.size + 1, self.max_episodes)

    def retrieve(self, query_vector: torch.Tensor, k: int) -> list[dict]:
        """
        Retrieves the top k most similar episodes from memory.
        Args:
            query_vector: The current state vector of the field to compare against.
            k: The number of top similar episodes to retrieve.
        Returns:
            A list of the top k snapshot dictionaries.
        """
        if self.size == 0:
            return []

        # Pad the query vector
        padded_query = self._pad_vector(query_vector).unsqueeze(0) # Shape [1, max_dim]

        # Get the currently stored memories
        active_mem_states = self.mem_states[:self.size] # Shape [size, max_dim]

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(padded_query, active_mem_states)

        # Get the top k indices
        k = min(k, self.size)
        top_k_indices = torch.topk(similarities, k=k).indices

        # Retrieve the corresponding snapshots
        retrieved = []
        for idx in top_k_indices:
            retrieved.append({
                "t": self.mem_time[idx].item(),
                "vector": self.mem_states[idx],
                "decision": self.mem_tags[idx]
            })
        
        return retrieved

    def retrieve_last(self, k: int = 1) -> list[dict]:
        """
        A convenience method to retrieve the last k stored items.
        NOTE: This is not based on similarity, but on storage order.
        """
        if self.size == 0:
            return []
        
        k = min(k, self.size)
        
        # Get the indices of the last k items
        indices = [(self.pointer - 1 - i + self.max_episodes) % self.max_episodes for i in range(k)]
        
        retrieved = []
        for idx in indices:
            retrieved.append({
                "t": self.mem_time[idx].item(),
                "vector": self.mem_states[idx],
                "decision": self.mem_tags[idx]
            })
            
        return retrieved 