import torch
import time

from pressure_agi.engine.field import Field

def run_latency_benchmark(device: str, n_nodes: int = 1000, n_steps: int = 1000):
    """
    Measures the average time per step of the Field simulation.
    """
    print(f"\n--- Latency Benchmark ---")
    print(f"Device: {device}, Nodes: {n_nodes}, Steps: {n_steps}")
    
    try:
        field = Field(n=n_nodes, device=device)
    except Exception as e:
        print(f"Could not initialize field on device '{device}': {e}")
        return

    # Warm-up phase
    for _ in range(100):
        field.step()

    # Measurement phase
    start_time = time.time()
    for _ in range(n_steps):
        field.step()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_step = (total_time / n_steps) * 1000 # Convert to milliseconds

    print(f"Total time for {n_steps} steps: {total_time:.3f} s")
    print(f"Average latency per step: {avg_time_per_step:.4f} ms")
    
    # The user's target is < 300ms, but that was for a full settle-cycle,
    # not a single step. This is just a performance data point.
    if avg_time_per_step < 10: # A single step should be very fast
        print("Latency check: [green]PASSED[/green]")
    else:
        print("Latency check: [red]FAILED[/red]")


if __name__ == "__main__":
    run_latency_benchmark(device='cpu')
    
    if torch.cuda.is_available():
        run_latency_benchmark(device='cuda')
    
    if torch.backends.mps.is_available():
        run_latency_benchmark(device='mps') 