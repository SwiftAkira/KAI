import typer
import yaml
from rich import print
from rich.live import Live
from rich.table import Table
import asyncio
from typing import Optional, List
import torch
from rich.layout import Layout
from rich.panel import Panel
from rich.bar import Bar

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject
from pressure_agi.engine.decide import decide, Action
from pressure_agi.io.codec_text import decode
from pressure_agi.engine.memory import EpisodicMemory
from pressure_agi.engine.critic import Critic
from pressure_agi.engine.planner import Planner
from pressure_agi.io.adapter import TextAdapter

app = typer.Typer()

def load_config():
    """Loads configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

async def step_once(
    field: Field,
    text: str,
    memory: EpisodicMemory,
    critic: Critic,
    loop_count: int,
    settle_steps: int,
    pos_threshold: float,
    neg_threshold: float,
    k_resonance: float,
    verbose: bool = False
) -> tuple[Action, float]:
    """
    Performs one full step of the agent's sense-plan-act loop.
    This version includes resonance damping.
    """
    # 1. Encode and inject text input
    if text:
        percepts = decode(text)
        if verbose: print(f"Injecting {len(percepts)} percepts.")
        inject(field, percepts)

    # 2. Settle the field
    if verbose: print(f"Settling field for {settle_steps} steps.")
    for _ in range(settle_steps):
        field.step()

    # 3. Critic evaluates the current state
    critic.evaluate(field)

    # 4. Episodic Memory Resonance with Damping
    new_k_resonance = k_resonance
    if memory.size > 0:
        entropy_before = critic.last_entropy
        
        retrieved_snapshots = memory.retrieve(field.states, k=3)
        if verbose and retrieved_snapshots: print(f"Retrieved {len(retrieved_snapshots)} memories.")

        if retrieved_snapshots:
            total_resonance_delta = torch.zeros_like(field.pressures)
            for snapshot in retrieved_snapshots:
                field_dim = field.n
                resonance_vector = snapshot['vector'][:field_dim] * new_k_resonance
                total_resonance_delta += resonance_vector

            # --- Damping Logic ---
            max_delta_norm = 1.0
            current_delta_norm = torch.norm(total_resonance_delta)
            if current_delta_norm > max_delta_norm:
                total_resonance_delta = total_resonance_delta * (max_delta_norm / current_delta_norm)

            field.pressures += total_resonance_delta

            entropy_after = critic.calculate_entropy(field)
            if entropy_after > entropy_before:
                new_k_resonance *= 0.9 # Decay gain
            else:
                new_k_resonance *= 1.05 # Increase gain
                new_k_resonance = min(new_k_resonance, 1.0)
            if verbose: print(f"k_resonance updated to {new_k_resonance:.3f}")

            # Settle field again after resonance
            for _ in range(settle_steps // 2):
                field.step()

    # 5. Decide on an action
    action = decide(field, pos_threshold, neg_threshold)
    if action is None:
        action = Action(type="neutral", vector=torch.empty(0, device=field.device, dtype=field.dtype))

    # 6. Store snapshot in memory
    snapshot = {
        "t": loop_count,
        "text": text,
        "vector": action.vector,
        "decision": action.type
    }
    memory.store(snapshot)

    return action, new_k_resonance

def generate_dashboard(
    loop_count: int,
    entropy: float,
    action: str,
    rollout_rewards: Optional[List[float]] = None
) -> Panel:
    """Generates the Rich layout for the dashboard."""
    
    layout = Layout()
    layout.split_column(
        Layout(name="header"),
        Layout(name="main")
    )

    header_table = Table.grid(expand=True)
    header_table.add_column(justify="left")
    header_table.add_column(justify="right")
    header_table.add_row(
        f"[bold]Loop Step:[/bold] {loop_count}",
        f"[bold]System Entropy:[/bold] {entropy:.4f}"
    )

    main_layout = Layout()
    main_layout.split_row(
        Layout(name="left"),
        Layout(name="right")
    )

    action_panel = Panel(
        f"[cyan]{action}[/cyan]",
        title="[bold]Action Taken[/bold]",
        border_style="green"
    )

    # --- Rollout Rewards Panel ---
    if rollout_rewards:
        reward_bars = ""
        # Normalize rewards for display if they vary widely, but for now, just clip them
        min_reward = min(rollout_rewards) if rollout_rewards else 0
        max_reward = max(rollout_rewards) if rollout_rewards else 0
        
        for i, reward in enumerate(rollout_rewards):
            # Simple scaling for the bar
            bar_fraction = (reward - min_reward) / (max_reward - min_reward + 1e-6)
            reward_bars += f"Candidate {i+1}: {reward:.3f}\n"
            reward_bars += str(Bar(size=50, begin=0, end=1, fraction=bar_fraction)) + "\n"
            
        rewards_panel = Panel(
            reward_bars,
            title="[bold]Planner Rollout Rewards[/bold]",
            border_style="magenta"
        )
    else:
        rewards_panel = Panel("N/A", title="[bold]Planner Rollout Rewards[/bold]", border_style="magenta")

    main_layout["left"].update(action_panel)
    main_layout["right"].update(rewards_panel)

    layout["header"].update(header_table)
    layout["main"].update(main_layout)

    return Panel(layout, title="[bold yellow]Pressure-AGI Dashboard[/bold yellow]")

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
    adapter = TextAdapter()
    k_res = k_resonance # Use a different name to avoid shadowing
    loop_count = 0
    snapshot_file = "memory_snapshot.pt"

    # Load memory from last session if available
    memory.load_from_disk(snapshot_file)

    while True:
        try:
            field = Field(n=0, device=device)
            text = input("> ")
            if text.lower() == 'exit':
                break
            
            loop_count += 1

            # --- Snapshotting ---
            if loop_count % 500 == 0:
                memory.save_to_disk(snapshot_file)

            # --- Planning Step (Not used in REPL, but runs in parallel) ---
            if field.n > 0:
                candidate_goals = [torch.randn(field.n, device=device, dtype=torch.float64) for _ in range(5)]
                candidate_tuples = [(vec, None) for vec in candidate_goals]
                planner.evaluate_candidates(field, candidate_tuples, env=None)
                selected_goal_node = planner.select_action()
                if selected_goal_node:
                    # Inject the selected goal into the live field
                    action_vector = selected_goal_node.vector
                    min_dim = min(field.n, len(action_vector))
                    if min_dim > 0:
                        field.pressures[:min_dim] += action_vector[:min_dim]
                    
                    # Freeze the critic to prevent oscillation
                    critic.freeze(10)

                planner.prune_goals()
            
            # --- Main REPL Step ---
            table = generate_dashboard(loop_count, critic.last_entropy, "...", planner.last_rollout_rewards)
            with Live(table, screen=True, redirect_stderr=False, vertical_overflow="visible") as live:
                action, k_res = asyncio.run(step_once(
                    field, text, memory, critic,
                    loop_count=loop_count,
                    settle_steps=settle_steps,
                    pos_threshold=pos_threshold,
                    neg_threshold=neg_threshold,
                    k_resonance=k_res,
                    verbose=False
                ))
                
                output = adapter.adapt(action)
                
                table = generate_dashboard(loop_count, critic.last_entropy, output, planner.last_rollout_rewards)
                live.update(table)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[bold red]An error occurred: {e}[/bold red]")

    print("[bold green]Exiting REPL.[/bold green]")

if __name__ == "__main__":
    app() 