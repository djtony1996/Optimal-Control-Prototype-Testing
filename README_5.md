# Item 5 CPU Baseline

This README documents item 5 of the minimal prototype plan:

- `DDP/iLQR` on CPU
- `Crocoddyl`
- both the trivial `LQR` case and the nonlinear pendulum case

## Problem Being Solved

Item 5 now supports two benchmark problems from the same Crocoddyl CPU path.

## Trivial LQR

The trivial benchmark solves the same finite-horizon `LQR` problem shape as
item 4.

- state dimension: `x in R^2`
- control dimension: `u in R`
- time step: `dt = 0.1`
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

## Nonlinear Pendulum

The nonlinear benchmark uses the shared pendulum swing-up problem from page 6
of the project PDF.

- state dimension: `x = (theta, theta_dot) in R^2`
- control dimension: `u in R`
- time step: `dt = 0.1`
- final time: `T = 4.0`
- initial state: `x0 = [0.0, 0.0]`
- target state: `x_goal = [pi, 0.0]`
- state bounds: `theta in [-2 pi, 2 pi]`, `theta_dot in [-8.0, 8.0]`
- nominal torque bounds: `-2.5 <= u_k <= 2.5`

The continuous-time dynamics are:

- `theta_dot = omega`
- `omega_dot = -(g / L) sin(theta) - (B / (M L^2)) omega + u / (M L^2)`

The Crocoddyl pendulum runner supports both:

- `hard` mode: `SolverBoxDDP` with hard control bounds and a large quadratic
  penalty on state-limit violation
- `soft` mode: `SolverDDP` with smooth quadratic state/control penalties and no
  hard solver-side control box

Important note:
- Crocoddyl handles the hard torque bounds directly in `hard` mode.
- The state limits are not implemented as exact solver constraints here; they
  are enforced through a very large penalty in `hard` mode and a softer
  quadratic penalty in `soft` mode.

The soft penalties used in the current nonlinear item 5 implementation are:

- state violation:
  `v_x(x) = max(x_min - x, 0) + max(x - x_max, 0)`
- control violation:
  `v_u(u) = max(u_min - u, 0) + max(u - u_max, 0)`

The nonlinear running cost in soft mode becomes:

- `l_soft(x, u) = dt * (e(x)^T Q e(x) + u^T R u) + w_x ||v_x(x)||^2 + w_u ||v_u(u)||^2`

where `e(x)` is the pendulum state error relative to the upright target, with
the angle component wrapped onto `[-pi, pi]`.

The nonlinear terminal cost in soft mode becomes:

- `phi_soft(x_N) = e(x_N)^T Q_f e(x_N) + w_x ||v_x(x_N)||^2`

## Code Layout

- item 5 solver: [`src/optimal_control_prototype_testing/crocoddyl_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/crocoddyl_cpu.py)
- shared nonlinear problem: [`src/optimal_control_prototype_testing/nonlinear_pendulum.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/nonlinear_pendulum.py)
- startup script: [`scripts/run_item5_crocoddyl.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/run_item5_crocoddyl.sh)
- setup helper: [`scripts/setup_crocoddyl_macos.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/setup_crocoddyl_macos.sh)

## How Item 5 Solves the Problem

[`crocoddyl_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/crocoddyl_cpu.py)
now supports both benchmarks:

1. For the trivial case, it discretizes the linear dynamics with zero-order
   hold and builds Crocoddyl `ActionModelLQR` stages.
2. For the nonlinear pendulum, it builds a custom discrete Crocoddyl action
   model using `RK4` integration of the shared pendulum dynamics.
3. It uses Crocoddyl numerical differentiation on the nonlinear action model.
4. It solves with:
   `SolverBoxDDP` for hard-mode pendulum and trivial LQR, or `SolverDDP` for
   soft-mode pendulum.
5. It prints summary metrics plus the full state and control trajectories.

## One-Time Setup

Install Crocoddyl and its wheel dependencies into the project virtual
environment with:

```bash
bash ./scripts/setup_crocoddyl_macos.sh
```

This Mac uses the published `cp312 macOS arm64` Crocoddyl wheel together with
its `cmeel`-based dependency stack.

## Run

After setup, the existing trivial baseline still runs with:

```bash
bash ./scripts/run_item5_crocoddyl.sh
```

To run the nonlinear pendulum in hard mode:

```bash
bash ./scripts/run_item5_crocoddyl.sh --problem nonlinear --constraint-mode hard
```

To run the nonlinear pendulum in soft mode:

```bash
bash ./scripts/run_item5_crocoddyl.sh --problem nonlinear --constraint-mode soft
```

To run both nonlinear modes in one command:

```bash
bash ./scripts/run_item5_crocoddyl.sh --problem nonlinear --constraint-mode both
```

## CLI Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--problem` | `trivial` \| `nonlinear` | `trivial` | Which benchmark to run |
| `--constraint-mode` | `hard` \| `soft` \| `both` | `both` | Constraint handling (nonlinear only) |
| `--dt` | float | `0.1` | Time step size |
| `--final-time` | float | `2.0` (trivial) / `4.0` (nonlinear) | Total horizon time |

To use a finer time step for the nonlinear pendulum:

```bash
bash ./scripts/run_item5_crocoddyl.sh --problem nonlinear --constraint-mode hard --dt 0.05
```

To run with a longer total time (useful when the solver needs more time to reach the goal):

```bash
bash ./scripts/run_item5_crocoddyl.sh --problem nonlinear --constraint-mode hard --final-time 8.0
```

Both flags can be combined:

```bash
bash ./scripts/run_item5_crocoddyl.sh --problem nonlinear --constraint-mode hard --dt 0.05 --final-time 6.0
```

## Current Output

The current item 5 runner prints:

- `problem`
- `constraint_mode`
- solver convergence flag
- iteration count
- `objective`
- `max_control_violation`
- `max_state_violation`
- first control value
- final state
- full `state_trajectory`
- full `control_trajectory`
