import typer
import matplotlib.pyplot as plt
from pressure_agi.engine.field import Field
from tests.test_coherence import coherence

app = typer.Typer()

@app.command()
def plot(nodes: int = 100, steps: int = 400, gpu: bool = False, output: str = "docs/coherence_curve.png"):
    """
    Runs the simulation and plots the coherence of the field over time.
    """
    device = 'gpu' if gpu else 'cpu'
    field = Field(n=nodes, device=device)
    
    coherence_values = []
    for _ in range(steps):
        field.step(0.02)
        coherence_values.append(coherence(field))

    plt.figure(figsize=(10, 6))
    plt.plot(coherence_values)
    plt.title("Coherence Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Coherence")
    plt.grid(True)
    plt.savefig(output)
    print(f"Coherence plot saved to {output}")

if __name__ == "__main__":
    app() 