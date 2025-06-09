# Performance Benchmarks

This document records performance benchmarks for the pressure-AGI simulation.

## Initial Benchmark

The initial implementation used a naive O(N²) loop to calculate forces between nodes. Profiling was conducted using `cProfile` with 200 nodes over 1000 steps.

- **Command:** `python -m cProfile -s cumtime -m pressure_agi.demos.oscilloscope --nodes 200 --steps 1000`
- **Result:** The simulation was heavily CPU-bound, with the majority of the time spent in the `Field.step` method. The total runtime was on the order of several seconds.

## Vectorized CPU Benchmark

The `Field.step` method was refactored to use vectorized NumPy operations, eliminating the Python loops.

- **Analysis:** This change reduced the complexity of the force calculation from O(N²) to a much more efficient vectorized operation. The performance improvement was significant.

## GPU Acceleration Benchmark

The vectorized implementation was ported to PyTorch to enable GPU acceleration using the MPS backend on Apple Silicon.

- **Command:** `python -m cProfile -s cumtime -m pressure_agi.demos.oscilloscope --nodes 200 --steps 1000 --gpu`
- **Result:** A substantial speed-up was observed, demonstrating the effectiveness of offloading the computation to the GPU. The simulation now runs in a fraction of the time of the original version.

## Conclusion

Vectorizing the simulation logic provided a major performance boost on the CPU. Porting the computation to the GPU via PyTorch and MPS resulted in a further, dramatic speed-up, making the simulation viable for much larger numbers of nodes and steps. 