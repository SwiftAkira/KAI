# Architecture

This document outlines the high-level architecture of the Pressure-AGI system.

The core of the system is a `Field` of nodes, representing the agent's mental state. External inputs are processed by an `Adapter` and `Injector`, which apply "pressure" to the field. The system then enters a planning loop:

1.  The `Planner` observes the `Field`'s state.
2.  It generates candidate goals and, for each one, runs a `Monte-Pressure Rollout`.
3.  The rollout simulates the future by cloning the `Field` and a model of the external `Environment`, injecting the goal, and letting the simulation `settle`.
4.  The `Critic` scores the resulting state of the simulated field using a composite score of entropy and energy. This score, combined with any reward from the simulated environment, determines the goal's value.
5.  The `Planner` selects the highest-value goal and injects its vector into the *live* `Field`.
6.  Separately, `EpisodicMemory` can induce a `Resonance` effect by retrieving similar past states and applying their pressure to the live field.
7.  The final state of the live field is passed to an `Adapter` to be converted into an action in the environment.

```mermaid
graph TD;
    subgraph "I/O & Environment";
        direction LR;
        UserInput[/"User Input<br/>(Text or Env State)"'/] --> AdapterIn(Adapter In);
        AdapterIn --> Injector;
        AdapterOut(Adapter Out) --> EnvAction(("Environment<br/>Action"));
    end

    subgraph "Pressure-AGI Core";
        direction TB;
        
        subgraph "Planning Loop";
            Planner -- "Evaluates<br/>Candidates" --> Rollout;
            Rollout -- "Clones Field & Env" --> Sim;
            Sim(Simulated<br/>Field & Env) --> Critic;
            Critic -- "Provides<br/>Composite Score" --> Rollout;
            Rollout -- "Returns ΔReward" --> Planner;
        end

        subgraph "Main Field";
            Injector -- "Injects Pressure" --> Field;
            Field -- "State" --> Planner;
            Field -- "State" --> Memory(EpisodicMemory);
            Memory -- "Retrieves<br/>Similar Memories" --> Resonance;
            Resonance -- "Applies Resonance<br/>Pressure" --> Field;
            Critic -- "Applies<br/>Energy Penalty" --> Field;
            Planner -- "Injects<br/>Selected Goal" --> Field;
        end

        Field -- "Final State" --> AdapterOut;
    end
    
    style Rollout fill:#f9f,stroke:#333,stroke-width:2px;
    style Sim fill:#ccf,stroke:#333,stroke-width:2px;
``` 