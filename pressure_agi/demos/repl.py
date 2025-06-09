import typer
from rich import print

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject
from pressure_agi.engine.decide import decide
from pressure_agi.io.codec_text import decode, encode

app = typer.Typer()

@app.command()
def run(gpu: bool = False, settle_steps: int = 100):
    """
    A simple REPL to interact with the pressure-AGI agent.
    """
    print("[bold green]Starting Pressure-AGI REPL...[/bold green]")
    print("Enter text to be perceived by the agent. Type 'exit' to quit.")

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
            action = decide(field)
            
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