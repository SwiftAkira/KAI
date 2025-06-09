# Pressure-AGI

This project is an experimental implementation of an Artificial General Intelligence architecture based on the concept of "mental pressure."

## Core Concepts

The agent's state is represented by a `Field` of `Nodes`, each with a `pressure` and `state`. 
- **Pressure:** Represents the immediate "energy" or "urgency" of a node.
- **State:** Represents the node's activation or information content.

An `Agent` orchestrates the main components:
- **`Field`**: The core data structure holding the agent's mental state.
- **`Critic`**: Evaluates the `Field`'s state, calculating entropy and other metrics.
- **`Planner`**: Simulates future actions to achieve goals (e.g., reduce entropy).
- **`EpisodicMemory`**: Stores and retrieves past states, allowing the agent to learn from experience.
- **`Adapter`**: Translates the agent's internal decisions into actions for a specific environment (e.g., text for a REPL, commands for a game).

## Running the Agent

This project uses `poetry` for dependency management.

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the REPL:**
    The primary way to interact with the agent is through the command-line REPL.
    ```bash
    python -m pressure_agi.demos.repl
    ```
    You can now type sentences and see how the agent responds. The agent's memory (`memory_snapshot.pt`) will be saved in the root directory when you exit.

3.  **Run with Dashboard:**
    For a more detailed view of the agent's internal state, you can run the REPL with a dashboard:
    ```bash
    python -m pressure_agi.demos.repl --dashboard
    ```

## How it Works

1.  User input is decoded into `Percepts` with sentiment values.
2.  These `Percepts` are injected into the `Field`, changing node pressures.
3.  The `Field` settles over several simulation steps.
4.  The `EpisodicMemory` retrieves similar past experiences, applying "resonance" to the current field pressures.
5.  The `Critic` evaluates the `Field`'s entropy.
6.  The `decide` function makes a high-level decision (positive, negative, or neutral) based on the field's coherence.
7.  The `TextAdapter` translates this decision into a conversational response.
8.  The current state is saved as a snapshot in memory for future recall.

## Phase 0: Core Simulation & Stability

The first phase of this project focused on establishing a stable, observable, and performant core simulation. Key achievements include:

- **Core Data Structures:** Implementation of `Node` and `Field` classes to represent the fundamental components of the system.
- **GPU Acceleration:** The simulation was ported from a naive O(N²) loop to a vectorized PyTorch implementation, enabling massive speed-ups on Apple Silicon via the MPS backend.
- **CLI Oscilloscope:** A `rich`-powered CLI tool was developed to visualize the state of the simulation in real-time.
- **Metrics & Testing:** A "coherence" metric was established to quantify system stability, with corresponding `pytest` tests to ensure model integrity.
- **Decision Read-out:** A simple decision-making function was implemented to demonstrate how the system's state can be interpreted as an output.

## Phase 0 Review: Go/No-Go

At the conclusion of Phase 0, a go/no-go review was conducted to determine if the project was ready to proceed to the next stage. The review was based on two key metrics:

- **Coherence:** The system needed to achieve a coherence score of `≥ 0.85` on a 100-node simulation.
- **Decision Latency:** The decision read-out latency needed to be `≤ 250 ms` on the available hardware.

### Results

| Metric | Status | Details |
| :--- | :--- | :--- |
| **Coherence** | ✅ **Pass** | The `pytest` suite confirmed that the system reliably exceeds the 0.85 coherence threshold. |
| **Decision Latency** | ✅ **Pass** | The system's decision latency was measured to be well under the 250 ms limit. |
| | | **CPU:** ~0.0050 ms |
| | | **GPU (MPS):** ~0.5492 ms |

**Decision: Green-light for Phase 1.** The project successfully met all criteria and is now ready for the next phase of development.

## Getting Started

### 1. Environment Setup

This project is optimized for Apple Silicon (M1/M2/M3) and requires Python 3.11 or newer.

**Clone the repository:**
```bash
git clone https://github.com/SwiftAkira/KAI.git
cd KAI
```

**Create a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

### 2. Running the Demos

#### CLI Oscilloscope

To watch the simulation run in your terminal, use the `oscilloscope.py` demo. This will display a live table of node states and the current system decision.

**Run on CPU:**
```bash
python -m pressure_agi.demos.oscilloscope
```

**Run on GPU (Apple Silicon):**
```bash
python -m pressure_agi.demos.oscilloscope --gpu
```

You can also customize the number of nodes and simulation steps:
```bash
python -m pressure_agi.demos.oscilloscope --nodes 50 --steps 1000
```

### 3. Running Tests

To verify the stability and correctness of the simulation, run the test suite using `pytest`:
```bash
pytest
```

## Project Structure
```
├── docs/                 # Project documentation and reports
├── pressure_agi/         # Core source code
│   ├── demos/            # Demonstrations and visualizations
│   ├── engine/           # Core simulation engine
│   └── ...
├── tests/                # Test suite
├── setup.py              # Project installation script
└── requirements.txt      # Project dependencies
``` 