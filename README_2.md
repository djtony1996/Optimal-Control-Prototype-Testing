# Item 2 GPU Prototype

This README documents item 2 of the minimal prototype plan:

- `DDP/iLQR` on GPU
- `JAX + Diffrax`
- both the trivial `LQR` case and the nonlinear pendulum case
- `iLQR` first

## Problem Being Solved

Item 2 now supports two benchmark problems from the same iLQR code path.

## Trivial LQR

The trivial benchmark uses the same finite-horizon `LQR` setup as items 1, 3,
4, and 5.

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

The nonlinear iLQR runner supports both:

- `hard` mode: log-barrier treatment for torque plus a large quadratic penalty
  for state-limit violation
- `soft` mode: log-barrier treatment for torque plus smooth quadratic
  state/control penalties

The soft penalties used in the current nonlinear item 2 implementation are:

- state violation:
  `v_x(x) = max(x_min - x, 0) + max(x - x_max, 0)`
- control violation:
  `v_u(u) = max(u_min - u, 0) + max(u - u_max, 0)`

The nonlinear running cost in soft mode becomes:

- `l_soft(x, u) = dt * (e(x)^T Q e(x) + u^T R u) + barrier(u) + w_x ||v_x(x)||^2 + w_u ||v_u(u)||^2`

where `e(x)` is the pendulum state error relative to the upright target, with
the angle component wrapped onto `[-pi, pi]`.

The nonlinear terminal cost in soft mode becomes:

- `phi_soft(x_N) = e(x_N)^T Q_f e(x_N) + w_x ||v_x(x_N)||^2`

## Current Status

The current item 2 implementation includes:

- a separate `item2_jax` module
- a pure `JAX` iLQR solver
- both trivial and nonlinear benchmark paths
- a JIT-compiled fixed-iteration solve loop
- autodiff-based dynamics and stage-cost derivatives
- `Diffrax`-based nonlinear rollout integration
- a log-barrier treatment for control bounds

The current item 2 implementation does **not** yet include:

- a separate DDP variant
- an explicit warm-start batching harness
- a dedicated Colab notebook wrapper

## Code Layout

- solver: [`src/optimal_control_prototype_testing/item2_jax/ilqr.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item2_jax/ilqr.py)
- runner: [`src/optimal_control_prototype_testing/item2_jax/run_item2.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item2_jax/run_item2.py)
- shared nonlinear problem: [`src/optimal_control_prototype_testing/nonlinear_pendulum.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/nonlinear_pendulum.py)

## How Item 2 Solves the Problem

The current item 2 prototype works in these steps:

1. It selects either the trivial `LQR` benchmark or the nonlinear pendulum benchmark.
2. It rolls out a nominal control sequence to get a state trajectory.
3. It computes dynamics and cost derivatives with `jax.jacfwd`, `jax.grad`,
   and `jax.hessian`.
4. It runs the iLQR backward Riccati sweep to compute feedforward and feedback gains.
5. It runs a forward rollout with line-search candidates to update the control trajectory.
6. It repeats until the control update is small or the cost improvement stalls.

For the trivial case, the rollout uses the zero-order-hold discrete dynamics.
For the nonlinear pendulum, the rollout uses `Diffrax` integration and reports a
one-step consistency check against an `RK4` reference step.

## Run

In the current local environment, item 2 still runs the trivial benchmark on CPU by default:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2
```

To run the nonlinear pendulum in hard mode:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard
```

To run the nonlinear pendulum in soft mode:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode soft
```

## CLI Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--problem` | `trivial` \| `nonlinear` | `trivial` | Which benchmark to run |
| `--constraint-mode` | `hard` \| `soft` \| `both` | `both` | Constraint handling (nonlinear only) |
| `--dt` | float | `0.1` | Time step size |
| `--final-time` | float | `2.0` (trivial) / `4.0` (nonlinear) | Total horizon time |

To use a finer time step:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard --dt 0.05
```

To run with a longer total time (useful when the solver needs more time to reach the goal):

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard --final-time 8.0
```

Both flags can be combined:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard --dt 0.05 --final-time 6.0
```

## Google Colab

To run item 2 on GPU in Google Colab:

1. Open a new notebook in Google Colab.
2. Change the runtime to `GPU`.
3. Make the repository available in the notebook, for example by cloning a
   public repository URL or uploading a zip if the repository is private.
4. Install the dependencies.
5. Verify that `JAX` sees the GPU.
6. Run the item 2 module.

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
!PYTHONPATH=src python -m optimal_control_prototype_testing.item2_jax.run_item2
```

To run the nonlinear pendulum instead:

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard
```

To run with a finer time step:

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard --dt 0.05
```

To run with a longer total time:

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item2_jax.run_item2 --problem nonlinear --constraint-mode hard --final-time 8.0
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
- `runtime_seconds`
- `control_update_norm`
- `max_control_violation`
- `max_state_violation`
- `diffrax_vs_reference_step_error`
- full `state_trajectory`
- full `control_trajectory`
