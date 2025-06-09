import asyncio
import numpy as np
import torch
import itertools

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject

# --- Search Space ---
GLOBAL_GAINS = np.linspace(0.01, 0.2, 5)
IMPULSE_GAINS = np.linspace(0.1, 0.5, 5)
FRICTIONS = np.linspace(0.01, 0.05, 3)
DT = 0.01
CONVERSATION_TURNS = 100
SETTLE_STEPS = 200
COHERENCE_TARGET = 0.85

def generate_positive_scripted_conversation(turns: int):
    """Generates a consistently positive conversation."""
    base_phrases = [
        {"valence": 0.8}, {"valence": 0.9}, {"valence": 0.7}
    ]
    return [base_phrases[i % len(base_phrases)] for i in range(turns)]

async def test_params(params):
    """Tests a single combination of parameters."""
    global_gain, impulse_gain, friction = params
    
    field = Field(
        n=0, 
        device='cpu', 
        global_gain=global_gain, 
        impulse_gain=impulse_gain
    )
    
    conversation = generate_positive_scripted_conversation(CONVERSATION_TURNS)

    try:
        # --- Main Loop ---
        for i, percept_data in enumerate(conversation):
            # Inject just one percept per "turn" for this test
            percepts = [percept_data]
            inject(field, percepts)
            # Settle field after each injection
            for _ in range(10): # Mini-settle
                 field.step(dt=DT, friction=friction)

        # --- Final Settle ---
        for _ in range(SETTLE_STEPS):
            field.step(dt=DT, friction=friction)

        # --- Check Coherence ---
        if field.n > 0:
            final_coherence = torch.mean(field.states).item()
            if np.isnan(final_coherence):
                return False, 0 # Unstable
            if final_coherence >= COHERENCE_TARGET:
                return True, final_coherence # Success
        
        return False, final_coherence # Failed to meet target

    except Exception:
        return False, 0 # Unstable

async def run_sweep():
    """Runs the parameter sweep."""
    print("--- Starting Parameter Sweep ---")
    param_combinations = list(itertools.product(GLOBAL_GAINS, IMPULSE_GAINS, FRICTIONS))
    print(f"Testing {len(param_combinations)} combinations...")

    for i, params in enumerate(param_combinations):
        gg, ig, f = params
        print(f"  Testing ({i+1}/{len(param_combinations)}): global_gain={gg:.3f}, impulse_gain={ig:.3f}, friction={f:.3f}...", end="")
        
        success, coherence = await test_params(params)
        
        if success:
            print(f"\n[bold green]✔ SUCCESS![/bold green]")
            print(f"  Coherence: {coherence:.4f}")
            print(f"  Winning Parameters:")
            print(f"    global_gain: {gg}")
            print(f"    impulse_gain: {ig}")
            print(f"    friction: {f}")
            
            # Store to file
            with open("docs/tuned_params.yml", "w") as f:
                f.write(f"global_gain: {gg}\n")
                f.write(f"impulse_gain: {ig}\n")
                f.write(f"friction: {f}\n")
            print("Parameters saved to docs/tuned_params.yml")
            return

        else:
            print(f" Failed. Coherence: {coherence:.4f}")

    print("\n[bold red]--- Sweep finished. No suitable parameters found. ---[/bold red]")


if __name__ == "__main__":
    from rich import print
    asyncio.run(run_sweep()) 