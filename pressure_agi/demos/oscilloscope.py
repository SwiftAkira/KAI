import time, typer, random
from rich.live import Live
from rich.table import Table
from pressure_agi.engine.field import Field
from pressure_agi.engine.memory import EpisodicMemory

app = typer.Typer()

@app.command()
def run(nodes: int = 30, steps: int = 500):
    field = Field(n=nodes)
    memory = EpisodicMemory()
    for node in field.nodes:
        node.state = random.uniform(-1, 1)

    with Live(auto_refresh=False) as live:
        for t in range(steps):
            field.step(0.02)
            snapshot = {
                't': t,
                'state_vector': [n.state for n in field.nodes]
            }
            memory.store(snapshot)
            table = Table(title=f"t={t}")
            for i, n in enumerate(field.nodes):
                table.add_row(f"{i}", f"{n.state:+.3f}")
            live.update(table, refresh=True)
            time.sleep(0.01)

if __name__ == "__main__":
    app() 