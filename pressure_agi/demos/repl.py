import typer
import yaml
from rich import print
from rich.live import Live
from rich.table import Table
import asyncio

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject
from pressure_agi.engine.decide import decide
from pressure_agi.io.codec_text import decode, encode
from pressure_agi.engine.memory import EpisodicMemory
from pressure_agi.engine.critic import Critic

app = typer.Typer()

def load_config():
    """Loads configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

async def step_once(
    text: str,
    field: Field,
    critic: Critic,
    memory: EpisodicMemory,
    loop_count: int,
    settle_steps: int,
    pos_threshold: float,
    neg_threshold: float,
    verbose: bool = True
) -> str:
    """Processes a single turn of the agent's loop."""
    # 1. Decode input text into percepts
    percepts = decode(text)
    if verbose: print(f"[yellow]Decoded {len(percepts)} percepts...[/yellow]")

    # 2. Inject percepts into the field
    inject(field, percepts)
    if verbose: print(f"[yellow]Injected. Field now has {field.n} nodes.[/yellow]")

    # 3. Critic evaluates the field before settling
    critic.evaluate(field)

    # 4. Settle the field by stepping the simulation
    # A short settle after injection is critical for stability
    for _ in range(10):
        field.step()

    if verbose: print(f"[yellow]Settling field for {settle_steps} steps...[/yellow]")
    for _ in range(settle_steps):
        field.step()

    # 5. Decide on an action
    action = decide(field, pos_threshold, neg_threshold)

    # 6. Store snapshot in memory
    snapshot = {
        "t": loop_count,
        "vector": field.cpu_states,
        "decision": action
    }
    memory.store(snapshot)
    if verbose: print(f"[blue]Stored snapshot {loop_count} in memory. Last decision was '{memory.retrieve_last()[0]['decision']}'.[/blue]")
    
    return action

def generate_dashboard(loop_count, entropy, decision) -> Table:
    """Creates a Rich table for the dashboard."""
    table = Table(title="Pressure-AGI State")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_row("Turn", str(loop_count))
    table.add_row("Field Entropy", f"{entropy:.4f}")
    table.add_row("Decision", decision)
    return table

@app.command()
def run(
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration."),
    settle_steps: int = typer.Option(100, help="Number of simulation steps to settle the field."),
    theta_pos: float = typer.Option(None, help="Positive decision threshold. Overrides config."),
    theta_neg: float = typer.Option(None, help="Negative decision threshold. Overrides config.")
):
    """
    A simple REPL to interact with the pressure-AGI agent.
    """
    print("[bold green]Starting Pressure-AGI REPL...[/bold green]")
    print("Enter text to be perceived by the agent. Type 'exit' to quit.")

    config = load_config()
    pos_threshold = theta_pos if theta_pos is not None else config.get('decision_thresholds', {}).get('positive', 0.1)
    neg_threshold = theta_neg if theta_neg is not None else config.get('decision_thresholds', {}).get('negative', -0.1)

    print(f"Using thresholds: [cyan]positive > {pos_threshold}[/cyan], [cyan]negative < {neg_threshold}[/cyan]\n")

    device = 'gpu' if gpu else 'cpu'
    memory = EpisodicMemory()
    critic = Critic()
    loop_count = 0

    while True:
        try:
            field = Field(n=0, device=device)
            text = input("> ")
            if text.lower() == 'exit':
                break
            
            loop_count += 1

            table = generate_dashboard(loop_count, critic.last_entropy, "...")
            with Live(table, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                action = asyncio.run(step_once(
                    text, field, critic, memory, loop_count,
                    settle_steps, pos_threshold, neg_threshold,
                    verbose=False # Suppress step_once prints for clean dashboard
                ))
                
                output = encode(action)
                
                # Update dashboard with final values
                table = generate_dashboard(loop_count, critic.last_entropy, action)
                live.update(table)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[bold red]An error occurred: {e}[/bold red]")

    print("[bold green]Exiting REPL.[/bold green]")

if __name__ == "__main__":
    app() 