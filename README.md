# KAI: A Pressure-Based AGI

This project is an experimental implementation of a novel artificial general intelligence architecture based on a concept of "pressure" propagating through a field of interconnected nodes. The simulation is built with performance in mind, leveraging vectorized computations and GPU acceleration via PyTorch and the Metal Performance Shaders (MPS) backend on Apple Silicon.

## Phase 0: Core Simulation & Stability

The first phase of this project focused on establishing a stable, observable, and performant core simulation. Key achievements include:

- **Core Data Structures:** Implementation of `Node` and `Field` classes to represent the fundamental components of the system.
- **GPU Acceleration:** The simulation was ported from a naive O(N²) loop to a vectorized PyTorch implementation, enabling massive speed-ups on Apple Silicon via the MPS backend.
- **CLI Oscilloscope:** A `rich`-powered CLI tool was developed to visualize the state of the simulation in real-time.
- **Metrics & Testing:** A "coherence" metric was established to quantify system stability, with corresponding `pytest` tests to ensure model integrity.
- **Decision Read-out:** A simple decision-making function was implemented to demonstrate how the system's state can be interpreted as an output.

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