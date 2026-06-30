# Sparse Matrix-Vector Multiplication (SpMV)

A C++ implementation and performance analysis of **Sparse Matrix-Vector Multiplication (SpMV)** using the **Compressed Sparse Row (CSR)** format, with serial, **OpenMP** (shared-memory), and **MPI** (distributed-memory) versions.

This project was built for a Parallel Processing course and includes full benchmarking, scaling analysis (Strong/Weak), and cross-platform performance comparison.

---

## Overview

Sparse matrices appear everywhere in scientific computing — numerical simulation, large-scale linear systems, finite element analysis, machine learning, and graph processing. Most of their entries are zero, so storing and computing with them naively wastes memory and time.

This project stores matrices in **CSR format** (only non-zero entries are kept) and implements `y = A·x` in three versions:

- **Serial** — baseline, single-threaded implementation
- **OpenMP** — shared-memory parallelism across CPU cores
- **MPI** — distributed-memory parallelism across processes (and potentially multiple machines)

All versions are benchmarked using GFLOPS, Speedup, Parallel Efficiency, Communication Time, Strong Scaling, and Weak Scaling.

---

## Why CSR?

The CSR format stores a sparse matrix using three arrays:

| Array | Description | Size |
|---|---|---|
| `values` | Non-zero values, row-major order | `nnz` |
| `col_idx` | Column index of each non-zero value | `nnz` |
| `row_ptr` | Start offset of each row in `values`/`col_idx` | `rows + 1` |

For row `i`, its non-zero entries live in the range `[row_ptr[i], row_ptr[i+1])`, giving:

```
y[i] = Σ values[k] * x[col_idx[k]]   for k in [row_ptr[i], row_ptr[i+1])
```

This reduces memory from **O(rows × cols)** to **O(nnz + rows)**, and computation time from **O(n²)** to **O(nnz)** — a major win for genuinely sparse matrices.

---

## Project Structure

```
.
├── include/              # Header files (declarations)
│   ├── csr_matrix.hpp    # CSRMatrix / VectorData structs
│   ├── generator.hpp     # Random sparse matrix/vector generation
│   ├── io.hpp             # Text & binary save/load utilities
│   ├── partition.hpp     # nnz-based row partitioning (for MPI)
│   ├── spmv.hpp           # SpMV kernel declarations (serial & OpenMP)
│   └── timer.hpp          # Simple chrono-based timer
│
├── src/                   # Implementation files
│   ├── generator.cpp
│   ├── io.cpp
│   ├── partition.cpp
│   ├── spmv_serial.cpp
│   └── spmv_omp.cpp
│
├── apps/                  # Executables
│   ├── gen_matrix.cpp     # Generates a random sparse matrix + vector
│   ├── run_serial.cpp     # Runs & benchmarks the serial version
│   ├── run_omp.cpp        # Runs & benchmarks the OpenMP version
│   ├── run_mpi.cpp        # Runs & benchmarks the MPI version
│   └── benchmark_help.cpp # Prints the benchmarking plan
│
├── plot_results.py        # Generates Speedup / GFLOPS / Scaling / Efficiency plots
├── Makefile
└── results/                # CSV output from benchmark runs
```

---

## Build

Requires a C++17 compiler, OpenMP, and an MPI implementation (OpenMPI / MPICH).

```bash
make            # builds all executables
make gen_matrix
make run_serial
make run_omp
make run_mpi
```

---

## Usage

### 1. Generate a random sparse matrix and vector

```bash
./gen_matrix <rows> <cols> <nnz> <matrix.txt> <matrix.bin> <vector_prefix>
```

Example:

```bash
./gen_matrix 20000 20000 1000000 data/A.txt data/A.bin data/x
```

### 2. Run the serial version

```bash
./run_serial bin data/A.bin data/x.bin data/y_serial.bin results/results.csv
```

### 3. Run the OpenMP version

```bash
./run_omp bin data/A.bin data/x.bin <threads> data/y_omp.bin results/results.csv
```

Example with 8 threads:

```bash
./run_omp bin data/A.bin data/x.bin 8 data/y_omp.bin results/results.csv
```

### 4. Run the MPI version

```bash
mpirun -np <processes> ./run_mpi bin data/A.bin data/x.bin data/y_mpi.bin results/results.csv
```

### 5. Plot results

```bash
python3 plot_results.py results/results.csv --output-dir plots
```

Generates: `speedup_*.png`, `gflops_*.png`, `strong_scaling_*.png`, `weak_scaling_*.png`, `efficiency_*.png`, `mpi_times_*.png`.

---

## Parallelization Strategy

### OpenMP

Each row of `y` is computed independently of all others, so rows are distributed across threads:

```cpp
#pragma omp parallel for schedule(dynamic, 64)
for (int i = 0; i < A.rows; ++i) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
        sum += A.values[j] * x[A.col_idx[j]];
    }
    y[i] = sum;
}
```

`schedule(dynamic, 64)` is used instead of `static` because the number of non-zeros per row is rarely uniform — dynamic scheduling rebalances load across threads as they finish their chunks.

### MPI

Since the number of non-zeros per row is uneven, **row-count partitioning is not enough** — instead, rows are partitioned so that each process gets roughly the same number of **non-zero elements** (`partition_by_nnz`, via binary search over `row_ptr`).

Execution flow:
1. Rank 0 reads the matrix and vector from disk.
2. Matrix metadata is broadcast to all ranks.
3. The full vector `x` is broadcast (every row may reference any column).
4. The matrix is partitioned by `nnz` and distributed to each rank.
5. Each rank computes its local SpMV.
6. Results are gathered back to rank 0 with `MPI_Gatherv`.
7. Rank 0 verifies correctness against the serial result and saves the output.

Communication time, compute time, and total time are measured separately to isolate communication overhead.

---

## Benchmarking Metrics

| Metric | Formula | Meaning |
|---|---|---|
| GFLOPS | `(2 × nnz) / (T × 1e9)` | Floating-point throughput |
| Speedup | `S = T_serial / T_parallel` | Gain from parallelization |
| Efficiency | `E = S / p` | How well each worker is utilized |
| Strong Scaling | Fixed problem size, varying workers | How runtime drops with more resources |
| Weak Scaling | Problem size grows with workers, density fixed | Whether per-worker load stays balanced |

---

## Key Findings

- **Correctness**: `max_error = 0` across all runs — OpenMP and MPI results exactly match the serial baseline.
- **OpenMP scales well up to the physical core count**, then plateaus — SpMV is fundamentally **memory-bandwidth-bound**, not compute-bound, so adding threads beyond available memory bandwidth yields little benefit.
- **MPI underperforms on a single machine** for matrices of this size — communication overhead (process startup, data copying over simulated sockets) dominates total runtime, often making MPI slower than the serial baseline. MPI is expected to outperform OpenMP only when the matrix is too large to fit in a single machine's memory and a real multi-node cluster is used.
- **Cross-platform comparison** (Windows/AMD vs. macOS Apple Silicon M1) shows that scaling behavior is highly architecture-dependent — heterogeneous P/E cores and unified memory bandwidth limits affect OpenMP scaling differently across systems.

---

## Future Improvements

- Hybrid MPI+OpenMP model (multiple nodes, multiple threads per node)
- GPU implementation (CUDA) for comparison
- Alternative sparse formats (ELL, Hybrid, CSR5) for irregular matrices
- Testing on real-world matrices (e.g. SuiteSparse Matrix Collection) instead of only synthetic random matrices
- Testing on an actual multi-node cluster instead of a single virtualized machine

---

