import time
import typer
import torch
import numpy as np
from pressure_agi.engine.field import Field
from pressure_agi.engine.decide import decide

app = typer.Typer()

@app.command()
def run(nodes: int = 100, iterations: int = 1000, gpu: bool = False):
    """
    Measures the decision latency for the pressure-AGI field.
    """
    device_name = 'gpu' if gpu else 'cpu'
    field = Field(n=nodes, device=device_name)

    # Initialize field with random states to ensure non-trivial computation
    field.states += (2 * torch.rand(field.n, dtype=field.dtype, device=field.device) - 1)
    field.step()

    latencies = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        decide(field)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds

    avg_latency = np.mean(latencies)
    print(f"Device: {field.device.type}")
    print(f"Average decision latency over {iterations} iterations: {avg_latency:.4f} ms")

    if avg_latency <= 250:
        print("✅ Latency is within the acceptable range (<= 250 ms).")
    else:
        print("❌ Latency exceeds the acceptable range (> 250 ms).")

if __name__ == "__main__":
    app() 