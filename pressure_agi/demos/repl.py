import typer
import yaml
from rich import print
from rich.live import Live
from rich.table import Table
import asyncio
from typing import Optional
import torch

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject
from pressure_agi.engine.decide import decide
from pressure_agi.io.codec_text import decode, encode
from pressure_agi.engine.memory import EpisodicMemory
from pressure_agi.engine.critic import Critic
from pressure_agi.engine.planner import Planner

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
    critic: Optional[Critic],
    memory: Optional[EpisodicMemory],
    loop_count: int,
    settle_steps: int,
    pos_threshold: float,
    neg_threshold: float,
    resonance_gain: float = 0.0,
    k_resonance: int = 0,
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
    if critic:
        # HACK: The critic's intervention destabilizes the new physics model.
        # The new model is self-regulating, so the critic is disabled for now.
        # critic.evaluate(field)
        pass # Explicitly doing nothing with the critic for now

    # 4. Settle the field by stepping the simulation
    # A short settle after injection is critical for stability
    for _ in range(10):
        field.step()

    if verbose: print(f"[yellow]Settling field for {settle_steps} steps...[/yellow]")
    for _ in range(settle_steps):
        field.step()

    # 5. Memory Resonance
    if memory and k_resonance > 0 and resonance_gain > 0.0:
        if verbose: print(f"[cyan]Retrieving {k_resonance} memories for resonance...[/cyan]")
        retrieved_snapshots = memory.retrieve(field.states, k=k_resonance)
        
        for snapshot in retrieved_snapshots:
            retrieved_state = snapshot['vector']
            
            # Calculate delta and apply resonance to pressure
            min_dim = min(field.n, len(retrieved_state))
            delta = retrieved_state[:min_dim] - field.states[:min_dim]
            field.pressures[:min_dim] += resonance_gain * delta
            if verbose: print(f"  [cyan]Applying resonance from snapshot t={snapshot['t']}...[/cyan]")
        
        # Settle the field again after applying resonance
        if verbose and retrieved_snapshots: print(f"[yellow]Re-settling field after resonance...[/yellow]")
        for _ in range(settle_steps // 2):
            field.step()

    # 6. Decide on an action (Planner replaces simple 'decide')
    # The planner will be used here to select a high-level action/goal
    # For now, we will keep the simple decide for the REPL loop
    action = decide(field, pos_threshold, neg_threshold)

    # 7. Store snapshot in memory
    if memory:
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
    theta_neg: float = typer.Option(None, help="Negative decision threshold. Overrides config."),
    resonance_gain: float = typer.Option(0.07, help="Gain for memory resonance pressure."),
    k_resonance: int = typer.Option(3, help="Number of memories to use for resonance."),
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
    memory = EpisodicMemory(device=device)
    critic = Critic()
    planner = Planner(critic=critic)
    loop_count = 0

    while True:
        try:
            field = Field(n=0, device=device)
            text = input("> ")
            if text.lower() == 'exit':
                break
            
            loop_count += 1

            # --- Planning Step ---
            # 1. Generate candidate goals (for now, random vectors)
            num_nodes = field.n if field.n > 0 else 1 # Avoid size 0
            candidate_goals = [torch.randn(num_nodes, device=device, dtype=torch.float64) for _ in range(5)]
            
            # 2. Planner evaluates candidates
            planner.evaluate_candidates(field, candidate_goals)

            # 3. Select best action
            selected_goal_node = planner.select_action()

            # 4. Inject selected action into the live field
            if selected_goal_node:
                action_vector = selected_goal_node.vector
                min_dim = min(field.n, len(action_vector))
                if min_dim > 0:
                    field.pressures[:min_dim] += action_vector[:min_dim]

            # 5. Prune planner tree for next cycle
            planner.prune_goals()
            
            # --- Main REPL Step ---
            # We still run the original step_once to process the user input
            # and get a low-level action for the dashboard. The planner runs in parallel.
            table = generate_dashboard(loop_count, critic.last_entropy, "...")
            with Live(table, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                action = asyncio.run(step_once(
                    text, field, critic, memory, loop_count,
                    settle_steps, pos_threshold, neg_threshold,
                    resonance_gain=resonance_gain,
                    k_resonance=k_resonance,
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