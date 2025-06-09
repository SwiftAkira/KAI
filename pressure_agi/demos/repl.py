import typer
import yaml
from rich import print

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject
from pressure_agi.engine.decide import decide
from pressure_agi.io.codec_text import decode, encode

app = typer.Typer()

def load_config():
    """Loads configuration from config.yaml"""
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return {}

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
    # Prioritize CLI options, then config file, then defaults
    pos_threshold = theta_pos if theta_pos is not None else config.get('decision_thresholds', {}).get('positive', 0.1)
    neg_threshold = theta_neg if theta_neg is not None else config.get('decision_thresholds', {}).get('negative', -0.1)

    print(f"Using thresholds: [cyan]positive > {pos_threshold}[/cyan], [cyan]negative < {neg_threshold}[/cyan]\n")

    device = 'gpu' if gpu else 'cpu'

    while True:
        try:
            # Start with a fresh, empty field for each input
            field = Field(n=0, device=device)
            text = input("> ")
            if text.lower() == 'exit':
                break

            # 1. Decode input text into percepts
            percepts = decode(text)
            print(f"[yellow]Decoded {len(percepts)} percepts...[/yellow]")

            # 2. Inject percepts into the field
            inject(field, percepts)
            print(f"[yellow]Injected. Field now has {field.n} nodes.[/yellow]")

            # 3. Settle the field by stepping the simulation
            print(f"[yellow]Settling field for {settle_steps} steps...[/yellow]")
            for _ in range(settle_steps):
                field.step()

            # 4. Decide on an action
            action = decide(field, pos_threshold, neg_threshold)
            
            # 5. Encode and print the action
            output = encode(action)
            print(f"[bold magenta]{output}[/bold magenta]\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[bold red]An error occurred: {e}[/bold red]")

    print("[bold green]Exiting REPL.[/bold green]")

if __name__ == "__main__":
    app() 