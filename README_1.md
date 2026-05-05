# Item 1 GPU Prototype

This README documents item 1 of the minimal prototype plan:

- direct transcription on GPU
- `JAX + Diffrax`
- both the trivial `LQR` case and the nonlinear pendulum case
- multiple shooting first

## Problem Being Solved

Item 1 now supports two benchmark problems from the same multiple-shooting SQP path.

## Trivial LQR

The trivial benchmark uses the same finite-horizon `LQR` setup as items 2, 3,
4, and 5.

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

The nonlinear item 1 runner supports both:

- `hard` mode: bounded control parameterization plus a large quadratic penalty
  on state-limit violation
- `soft` mode: bounded control parameterization plus smooth quadratic
  state/control penalties

The soft penalties used in the current nonlinear item 1 implementation are:

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

## Current Status

The current item 1 implementation includes:

- both trivial and nonlinear benchmark paths
- a `JAX` multiple-shooting baseline
- a prototype JAX-native SQP loop
- bounded control parameterization
- `Diffrax`-based one-step reference checks

The current item 1 implementation does **not** yet include:

- direct collocation
- a dedicated Colab notebook wrapper

The current code is written to run in either:

- `float32` mode by default, which is the safer starting point for Colab GPU
- optional `float64` mode by setting `ITEM1_ENABLE_X64=1` before launch

The current multiple-shooting prototype has been verified on:

- local CPU for both benchmarks
- Google Colab GPU for the trivial LQR case

## Code Layout

- shared problem: [`src/optimal_control_prototype_testing/item1_jax/problem.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item1_jax/problem.py)
- multiple shooting solver: [`src/optimal_control_prototype_testing/item1_jax/multiple_shooting.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item1_jax/multiple_shooting.py)
- shared nonlinear problem: [`src/optimal_control_prototype_testing/nonlinear_pendulum.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/nonlinear_pendulum.py)
- runner: [`src/optimal_control_prototype_testing/item1_jax/run_item1.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item1_jax/run_item1.py)

## How Item 1 Solves the Problem

The current item 1 multiple-shooting prototype works in these steps:

1. It selects either the trivial `LQR` benchmark or the nonlinear pendulum benchmark.
2. It transcribes the horizon with multiple shooting:
   the optimization variables are the full state trajectory and a bounded
   parameterization of the control trajectory.
3. It enforces the initial condition and dynamics as equality constraints.
4. It solves the resulting equality-constrained direct-transcription problem
   with a small JAX-native SQP loop.
5. It uses a merit-based line search on each SQP update.

For the trivial case, the dynamics constraints use the exact zero-order-hold
discrete map. For the nonlinear pendulum, the dynamics constraints use an `RK4`
step map, and the runner reports a one-step consistency check against `Diffrax`.

## Run

In the current local environment, item 1 still runs the trivial benchmark on CPU by default:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1
```

To run the nonlinear pendulum in hard mode:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1 --problem nonlinear --constraint-mode hard
```

To run the nonlinear pendulum in soft mode:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1 --problem nonlinear --constraint-mode soft
```

If you want to force the current higher-precision local mode, use:

```bash
ITEM1_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1
```

## Google Colab

To run item 1 on GPU in Google Colab:

1. Open a new notebook in Google Colab.
2. Change the runtime to `GPU`.
3. Make the repository available in the notebook, for example by cloning a
   public repository URL or uploading a zip if the repository is private.
4. Install the dependencies.
5. Verify that `JAX` sees the GPU.
6. Run the item 1 module.

Example notebook cells:

```python
!pip install -U "jax[cuda12]" diffrax numpy
```

```python
!git clone <your-repo-url>
%cd Optimal-Control-Prototype-Testing
```

```python
import jax
print(jax.__version__)
print(jax.default_backend())
print(jax.devices())
```

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item1_jax.run_item1
```

To run the nonlinear pendulum instead:

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item1_jax.run_item1 --problem nonlinear --constraint-mode hard
```

Expected signs of a successful GPU run:

- `backend: gpu`
- `devices: ('cuda:0',)` or another CUDA device
- `converged: True`

If Colab reports a JAX or CUDA plugin version mismatch, reinstall `JAX` and
restart the runtime before rerunning the cells above.

If the repository is private, clone with a token or upload a zip archive
instead of using a public `git clone` command.

## Current Output

The current runner prints:

- detected `JAX` backend and devices
- whether `x64` is enabled
- the default solver dtype
- horizon and time-step information
- `problem`
- `constraint_mode`
- convergence flag
- iteration count
- `objective`
- `constraint_norm`
- `step_norm`
- `max_control_violation`
- `max_state_violation`
- `diffrax_vs_reference_step_error`
- full `state_trajectory`
- full `control_trajectory`
