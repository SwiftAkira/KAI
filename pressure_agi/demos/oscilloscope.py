import time, typer, random
import torch
from rich.live import Live
from rich.table import Table
from pressure_agi.engine.field import Field
from pressure_agi.engine.memory import EpisodicMemory
from pressure_agi.engine.decide import decide

app = typer.Typer()

@app.command()
def run(nodes: int = 30, steps: int = 500, gpu: bool = False):
    device = 'gpu' if gpu else 'cpu'
    field = Field(n=nodes, device=device)
    memory = EpisodicMemory()
    # Random initial states
    field.states = field.states + (2 * torch.rand(field.n, dtype=field.dtype, device=field.device) - 1)

    with Live(auto_refresh=False) as live:
        for t in range(steps):
            field.step(0.02)
            snapshot = {
                't': t,
                'state_vector': field.cpu_states.tolist()
            }
            memory.store(snapshot)
            
            decision = decide(field)
            table = Table(
                title=f"t={t} ({field.device})",
                caption=f"Decision: {decision}"
            )
            for i, state in enumerate(field.cpu_states):
                table.add_row(f"{i}", f"{state:+.3f}")
            live.update(table, refresh=True)
            time.sleep(0.01)

if __name__ == "__main__":
    app() 