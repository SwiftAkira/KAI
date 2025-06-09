import asyncio
import numpy as np
import torch
import itertools
from rich import print

from pressure_agi.engine.field import Field
from pressure_agi.engine.injector import inject

# --- Search Space ---
F_MAX_RANGE = np.linspace(8.0, 12.0, 2) # Narrow this as it seems less sensitive
G_MEAN_RANGE = np.linspace(0.01, 0.05, 3) # Try lower values
FRICTION_RANGE = np.linspace(0.02, 0.04, 2) # Narrow this
K_RANGE = np.linspace(1.0, 2.0, 3) # Widen this to increase force
SEEDS = range(20) # Test each param set against 20 seeds for robustness

# --- Simulation Constants ---
DT = 0.01
CONVERSATION_TURNS = 100
SETTLE_STEPS = 200
COHERENCE_TARGET = 0.90 # Aim high for a robust margin

def generate_positive_scripted_conversation(turns: int):
    """Generates a consistently positive conversation."""
    base_percepts = [{"valence": 0.95}, {"valence": 0.98}, {"valence": 0.92}]
    return [base_percepts[i % len(base_percepts)] for i in range(turns)]

async def test_params_across_seeds(params):
    """Tests a single combination of parameters across multiple seeds."""
    k_val, f_max, g_mean, friction = params
    coherences = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        field = Field(n=0, K=k_val, f_max=f_max, g_mean=g_mean)
        conversation = generate_positive_scripted_conversation(CONVERSATION_TURNS)

        try:
            for percept_data in conversation:
                inject(field, [percept_data])
                for _ in range(10):  # Mini-settle
                    field.step(dt=DT, friction=friction)
            
            for _ in range(SETTLE_STEPS):
                field.step(dt=DT, friction=friction)

            if field.n > 0:
                coherence = torch.mean(field.states).item()
                if np.isnan(coherence) or np.isinf(coherence):
                    coherences.append(-1.0) # Mark as unstable
                    break # One bad seed fails the whole set
                coherences.append(coherence)
            else:
                coherences.append(0.0)

        except RuntimeError as e: # Catch NaN trap from field
            coherences.append(-1.0)
            break 

    min_coherence = min(coherences) if coherences else -1.0
    return min_coherence >= COHERENCE_TARGET, min_coherence

async def run_sweep():
    """Runs the parameter sweep."""
    print("--- Starting Stability Parameter Sweep (Round 2) ---")
    param_combinations = list(itertools.product(K_RANGE, F_MAX_RANGE, G_MEAN_RANGE, FRICTION_RANGE))
    print(f"Testing {len(param_combinations)} combinations across {len(SEEDS)} seeds each...")

    for i, params in enumerate(param_combinations):
        k_val, f_max, g_mean, friction = params
        print(f"  ({i+1}/{len(param_combinations)}) Testing K={k_val:.2f}, f_max={f_max:.2f}, g_mean={g_mean:.2f}, friction={friction:.3f}...", end="", flush=True)
        
        success, min_coherence = await test_params_across_seeds(params)
        
        if success:
            print(f" [bold green]✔ SUCCESS[/bold green]")
            print(f"  [green]Min Coherence across {len(SEEDS)} seeds: {min_coherence:.4f}[/green]")
            print(f"  [bold]Winning Parameters:[/bold]")
            print(f"    K: {k_val}")
            print(f"    f_max: {f_max}")
            print(f"    g_mean: {g_mean}")
            print(f"    friction: {friction}")
            
            with open("docs/tuned_params.yml", "w") as f:
                f.write(f"K: {k_val}\n")
                f.write(f"f_max: {f_max}\n")
                f.write(f"g_mean: {g_mean}\n")
                f.write(f"friction: {friction}\n")
            print("Parameters saved to docs/tuned_params.yml")
            return

        else:
            print(f" [red]Failed. Min Coherence: {min_coherence:.4f}[/red]")

    print("\n[bold red]--- Sweep finished. No robust parameters found. ---[/bold red]")

if __name__ == "__main__":
    asyncio.run(run_sweep()) 