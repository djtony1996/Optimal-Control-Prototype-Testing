# Item 4 CPU Baseline

This README documents item 4 of the minimal prototype plan:

- direct transcription on CPU
- `acados` multiple shooting
- both the trivial `LQR` case and the nonlinear pendulum case

## Problem Being Solved

Item 4 now supports two benchmark problems from the same `acados` CPU path.

## Trivial LQR

The trivial benchmark solves the same finite-horizon `LQR` problem shape as
items 1, 2, 3, and 5.

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

## Nonlinear Pendulum

The nonlinear benchmark uses the shared pendulum swing-up problem from page 6
of the project PDF.

- state dimension: `x = (theta, theta_dot) in R^2`
- control dimension: `u in R`
- horizon length: `N = 40`
- final time: `T = 4.0`
- initial state: `x0 = [0.0, 0.0]`
- target state: `x_goal = [pi, 0.0]`
- state bounds: `theta in [-2 pi, 2 pi]`, `theta_dot in [-8.0, 8.0]`
- nominal torque bounds: `-2.5 <= u_k <= 2.5`

The continuous-time dynamics are:

- `theta_dot = omega`
- `omega_dot = -(g / L) sin(theta) - (B / (M L^2)) omega + u / (M L^2)`

The nonlinear acados runner supports both:

- `hard` mode: explicit `acados` box constraints on both torque and state
- `soft` mode: explicit `acados` box constraints on torque, with smooth
  quadratic state/control penalties added to the external cost

The soft penalties used in the current nonlinear item 4 implementation are:

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

- item 4 solver: [`src/optimal_control_prototype_testing/acados_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/acados_cpu.py)
- shared nonlinear problem: [`src/optimal_control_prototype_testing/nonlinear_pendulum.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/nonlinear_pendulum.py)
- startup script: [`scripts/run_item4_acados.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/run_item4_acados.sh)
- setup helper: [`scripts/setup_acados_macos.sh`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/scripts/setup_acados_macos.sh)

## How Item 4 Solves the Problem

[`acados_cpu.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/acados_cpu.py)
now supports both benchmarks:

1. For the trivial case, it builds a linear-quadratic `AcadosOcp` with
   multiple shooting, `ERK`, and `SQP`.
2. For the nonlinear pendulum, it builds a nonlinear `AcadosOcp` with
   pendulum dynamics, wrapped-angle swing-up cost, and either hard or soft
   constraint handling.
3. It asks `acados` to generate and compile a problem-specific solver for the
   selected OCP under `/tmp`.
4. It solves the horizon with `PARTIAL_CONDENSING_HPIPM` and reports the
   resulting trajectories and solver statistics.

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

After setup, the existing trivial baseline still runs with:

```bash
./scripts/run_item4_acados.sh
```

To run the nonlinear pendulum in hard mode:

```bash
./scripts/run_item4_acados.sh --problem nonlinear --constraint-mode hard
```

To run the nonlinear pendulum in soft mode:

```bash
./scripts/run_item4_acados.sh --problem nonlinear --constraint-mode soft
```

To run both nonlinear modes in one command:

```bash
./scripts/run_item4_acados.sh --problem nonlinear --constraint-mode both
```

## Current Output

The current item 4 runner prints:

- `problem`
- `constraint_mode`
- solver `status`
- `sqp_iterations`
- `objective`
- `max_control_violation`
- `max_state_violation`
- first control value
- final state
- full `state_trajectory`
- full `control_trajectory`
