# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLU v3.0 is a CUDA-based sparse LU factorization solver for circuit simulation. It combines CPU symbolic analysis with GPU numeric factorization. The main binary is `lu_cmd`.

## Build Commands

```bash
cd src/

make MAIN          # Build lu_cmd (default target)
make clean         # Remove GLU objects and executable (preserves NICSLU)
make distclean     # Remove everything including NICSLU libraries
```

**Requirements:** NVIDIA CUDA Toolkit (`nvcc`), GCC/G++, pthreads, libm, librt.

## Running

```bash
./lu_cmd -i matrix.mtx       # Solve with static pivoting (GESP)
./lu_cmd -i matrix.mtx -p    # Solve with diagonal perturbation
```

Output is written to `x.dat` (one solution value per line). The RHS is implicitly `b = [1, 1, ..., 1]`.

## Architecture

The solver pipeline has three phases:

### 1. Preprocessing (CPU) — `src/preprocess/preprocess.c`
Wraps NICSLU to read the Matrix Market (`.mtx`) file, apply AMD reordering for sparsity, and row/column scaling. Outputs the matrix in CSC format (`ax`, `ai`, `ap`) with permutation arrays.

### 2. Symbolic Analysis (CPU) — `src/symbolic.cc`, `include/symbolic.h`
Four sequential steps in `Symbolic_Matrix`:
- **`fill_in()`** — Computes the LU sparsity pattern via symbolic factorization. Guarantees a structural diagonal in every column.
- **`csr()`** — Transposes CSC → CSR for row-wise dependency tracking; computes diagonal pointer positions.
- **`predictLU()`** — Fills the sparsity pattern with actual matrix values (zeros where fill-in was introduced).
- **`leveling()`** — Level scheduling: assigns columns to dependency levels. Columns in the same level are independent and can be factorized in parallel on GPU.
- **`solve()`** — Forward/backward substitution on CPU with NICSLU permutations and scaling applied to the solution.

### 3. Numeric Factorization (GPU) — `src/numeric.cu`, `include/numeric.h`
`LUonDevice()` manages GPU memory, streams, and kernel dispatch. Uses 16 CUDA streams for asynchronous column processing.

**Kernel variants:**
- `RL` / `RL_perturb` — Right-looking batch kernels; process multiple columns per launch (one CUDA block per column). Used for large levels.
- `RL_onecol_factorizeCurrentCol`, `RL_onecol_updateSubmat`, `RL_onecol_cleartmpMem` — Single-column kernels used for small levels (≤16 columns), one stream per column.

**Level dispatch thresholds** (in `LUonDevice`):
- `level_size > 896` → 2 warps/block
- `level_size > 448` → 4 warps/block
- `level_size > 16`  → 32 warps/block
- `level_size ≤ 16`  → single-column path with 1024 threads each

**tmpMem** is a GPU scratch buffer (`n × TMPMEMNUM` floats) sized dynamically at runtime: reserves 4 GB if available, else 50% of free memory.

## Key Data Structures

All in `Symbolic_Matrix` (defined in `include/symbolic.h`):
- `sym_c_ptr`, `sym_r_idx` — CSC sparsity structure of LU (post fill-in)
- `csr_r_ptr`, `csr_c_idx`, `csr_diag_ptr` — CSR transpose with diagonal positions
- `l_col_ptr` — Diagonal element positions in CSC (start of U part per column)
- `val` — Matrix values, updated in-place by GPU to become LU factors
- `level_ptr`, `level_idx`, `num_lev` — Level scheduling output

## NICSLU (Third-Party Library)

`src/nicslu/` contains the NICSLU library from Tsinghua University (LGPL 2.1). It provides matrix I/O, AMD reordering, scaling, and CPU factorization. It is built as static libraries `nicslu.a` and `nicslu_util.a` automatically by the main Makefile.

NICSLU demo programs can be built and run for standalone testing:
```bash
cd src/nicslu/demo/
make
./test -i ../../../matrix.mtx
```

## Debugging

Compile with `-DGLU_DEBUG=1` to enable ABFT (Algorithm-Based Fault Tolerance) checks:
- `ABFTCalculateCCA()` — column checksum verification
- `ABFTCheckResult()` — LU correctness check
- `PrintLevel()` — prints level structure

The flag is set in the Makefile; search for `GLU_DEBUG` to locate it.

## Precision

`REAL` is defined as `float` in `include/type.h`. Changing to `double` requires updating this typedef and recompiling.
