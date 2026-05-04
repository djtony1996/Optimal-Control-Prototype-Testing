# Item 5 CPU Baseline

This README documents item 5 of the minimal prototype plan:

- `DDP/iLQR` on CPU
- `Crocoddyl`
- trivial dynamics `LQR` case

## Problem Being Solved

The item 5 baseline solves the same trivial finite-horizon `LQR` problem shape
as item 4, but with a mature `DDP/iLQR` solver from `Crocoddyl`.

- state dimension: `x in R^2`
- control dimension: `u in R`
- horizon length: `N = 20`
- final time: `T = 2.0`
- initial state: `x0 = [1.5, 0.0]`
- hard control bounds: `-0.75 <= u_k <= 0.75`

The continuous-time dynamics are:

- `dx/dt = A x + B u`

with

- `A = [[0.0, 1.0], [-0.25, -0.1]]`
- `B = [[0.0], [1.0]]`

The running cost is quadratic:

- `l(x, u) = x^T Q x + u^T R u`

with

- `Q = diag([1.0, 0.2])`
- `R = [[0.05]]`

and the terminal cost uses:

- `Q_f = diag([8.0, 1.0])`

The implementation discretizes the linear dynamics with zero-order hold and
then solves the bounded finite-horizon problem with Crocoddyl's box-constrained
DDP solver.

## Code Layout

- item 5 solver: [`src/optimal_control_prototype_testing/crocoddyl_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/crocoddyl_cpu.py)
- startup script: [`scripts/run_item5_crocoddyl.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/run_item5_crocoddyl.sh)
- setup helper: [`scripts/setup_crocoddyl_macos.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/setup_crocoddyl_macos.sh)

## How Item 5 Solves the Problem

[`crocoddyl_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/crocoddyl_cpu.py)
implements the trivial-case CPU baseline in four steps:

1. It defines the continuous-time trivial `LQR` model and discretizes it with
   zero-order hold.
2. It builds Crocoddyl `ActionModelLQR` stages for the running and terminal
   costs.
3. It applies hard control bounds on the running stages and creates a
   `ShootingProblem`.
4. It solves the horizon with `SolverBoxDDP` and prints the resulting summary
   metrics plus the full state and control trajectories.

## One-Time Setup

Install Crocoddyl and its wheel dependencies into the project virtual
environment with:

```bash
bash ./scripts/setup_crocoddyl_macos.sh
```

This Mac uses the published `cp312 macOS arm64` Crocoddyl wheel together with
its `cmeel`-based dependency stack.

## Run

After setup, run item 5 with:

```bash
bash ./scripts/run_item5_crocoddyl.sh
```

## Current Output

The current trivial-case item 5 run completes successfully and reports:

- solver convergence flag
- iteration count
- `objective`
- `max_control_violation`
- first control value
- final state
- full `state_trajectory`
- full `control_trajectory`
