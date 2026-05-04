# Item 4 CPU Baseline

This repository is now focused on item 4 of the minimal prototype plan:

- direct transcription on CPU
- `acados` multiple shooting
- hard control bounds
- trivial dynamics `LQR` case

The trivial dynamics `LQR` problem uses:

- `dx/dt = A x + B u`
- `l(x, u) = x^T Q x + u^T R u`
- box-bounded control inputs

After discretization, the baseline solves a finite-horizon constrained optimal
control problem with `acados` on CPU.

## Problem Being Solved

The function
[`build_trivial_lqr_ocp()`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/acados_cpu.py)
sets up the following trivial finite-horizon LQR control problem with hard
input constraints:

- state dimension: `x in R^2`
- control dimension: `u in R`
- horizon length: `N = 20`
- final time: `T = 2.0`
- initial state: `x0 = [1.5, 0.0]`
- hard control bounds: `-0.75 <= u_k <= 0.75`

The continuous-time dynamics are linear:

- `dx/dt = A x + B u`

with

- `A = [[0.0, 1.0], [-0.25, -0.1]]`
- `B = [[0.0], [1.0]]`

The stage cost is quadratic:

- `l(x, u) = x^T Q x + u^T R u`

with

- `Q = diag([1.0, 0.2])`
- `R = [[0.05]]`

and the terminal cost is:

- `phi(x_N) = x_N^T Q_f x_N`

with

- `Q_f = diag([8.0, 1.0])`

So the solver tries to drive the initial state toward the origin over the
horizon while minimizing quadratic state/control cost and enforcing the hard
control bounds at every shooting interval.

## Code Layout

- item 4 solver: [`src/optimal_control_prototype_testing/acados_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/acados_cpu.py)
- startup script: [`scripts/run_item4_acados.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/run_item4_acados.sh)
- setup helper: [`scripts/setup_acados_macos.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/setup_acados_macos.sh)

## How Item 4 Solves the Problem

[`acados_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/acados_cpu.py)
implements the trivial-case CPU baseline in four steps:

1. It defines the continuous-time trivial `LQR` model:
   linear dynamics, quadratic running cost, quadratic terminal cost, and hard
   control bounds.
2. It builds an `AcadosOcp` problem with:
   multiple shooting, `ERK` integration, `SQP`, and the `HPIPM` QP solver.
3. It asks `acados` to generate and compile a problem-specific solver for that
   OCP, using temporary generated files under `/tmp`.
4. It solves the finite-horizon problem and prints the resulting solver status,
   SQP iteration count, objective value, control-bound violation, first control,
   final state, full state trajectory, and full control trajectory.

So the Python file is the high-level problem description and runner, while
`acados` provides the generated low-level solver that actually performs the CPU
optimization.

## One-Time Setup

On macOS, the baseline expects:

- Homebrew `cmake`
- Homebrew `gcc` / `g++` with OpenMP support
- a local `acados` clone and build
- the `t_renderer` executable in `<acados_root>/bin/t_renderer`

The helper script documents the working setup path on this machine:

```bash
./scripts/setup_acados_macos.sh /Users/jitongding/Documents/GitHub/acados
```

The successful build on this Mac used:

- `gcc-15` / `g++-15`
- `-DACADOS_WITH_OPENMP=ON`
- `-DBUILD_SHARED_LIBS=ON`
- `-DACADOS_WITH_QPOASES=OFF`

`qpOASES` was disabled because it failed to build under `gcc-15`, while the
baseline runs correctly with the default `HPIPM` QP solver.

## Run

After setup, start item 4 with:

```bash
./scripts/run_item4_acados.sh
```

This script sets:

- `ACADOS_SOURCE_DIR`
- `DYLD_LIBRARY_PATH`
- `MPLCONFIGDIR`
- `PYTHONPATH`

and then runs the trivial-case item 4 baseline.

## Current Output

The current trivial-case item 4 run completes successfully and reports:

- solver `status`
- `sqp_iterations`
- `objective`
- `max_control_violation`
- first control value
- final state
- full `state_trajectory`
- full `control_trajectory`
