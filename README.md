# KAI: A Pressure-Based AGI

This project is an experimental implementation of a novel artificial general intelligence architecture based on a concept of "pressure" propagating through a field of interconnected nodes. The simulation is built with performance in mind, leveraging vectorized computations and GPU acceleration via PyTorch and the Metal Performance Shaders (MPS) backend on Apple Silicon.

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