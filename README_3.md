# Item 3 GPU Prototype

This README documents item 3 of the minimal prototype plan:

- `MPPI/CEM` on GPU
- `JAX + Diffrax`
- both the trivial `LQR` case and the nonlinear pendulum case
- parallel rollouts under `vmap`

## Problem Being Solved

Item 3 now supports two benchmark problems from the same code path.

## Trivial LQR

The trivial benchmark uses the same finite-horizon `LQR` setup as items 1, 2,
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
- hard control bounds: `-2.5 <= u_k <= 2.5`

The continuous-time dynamics are:

- `theta_dot = omega`
- `omega_dot = -(g / L) sin(theta) - (B / (M L^2)) omega + u / (M L^2)`

The nonlinear runner supports both:

- `hard` mode: clipped torque bounds plus a large penalty for state-limit violation
- `soft` mode: clipped torque bounds plus smooth quadratic state/control penalties

In this item 3 implementation, `soft` constraint mode means the solver is not
trying to enforce state limits as exact must-satisfy conditions inside the
sampling update. Instead, if a sampled trajectory pushes the pendulum state or
control outside the preferred bounds, the rollout cost is increased by smooth
quadratic penalties. That makes violating the limits expensive rather than
strictly forbidden. By contrast, the current `hard` mode still clips torque to
the admissible box and adds a much larger penalty to trajectories that violate
state limits, so infeasible state excursions are discouraged much more strongly.

The soft penalties used in the current code are:

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

The current item 3 implementation includes:

- a separate `item3_jax` module
- a pure `JAX` MPPI solver
- a pure `JAX` CEM solver
- both trivial and nonlinear benchmark paths
- vectorised rollout evaluation under `vmap`
- a JIT-compiled fixed-iteration sampling loop
- `Diffrax`-based nonlinear rollout integration
- clipped torque bounds for all runs

## Code Layout

- solver: [`src/optimal_control_prototype_testing/item3_jax/sampling.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item3_jax/sampling.py)
- runner: [`src/optimal_control_prototype_testing/item3_jax/run_item3.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item3_jax/run_item3.py)
- shared nonlinear problem: [`src/optimal_control_prototype_testing/nonlinear_pendulum.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/nonlinear_pendulum.py)

## How Item 3 Solves the Problem

The current item 3 prototype works in these steps:

1. It selects either the trivial `LQR` benchmark or the nonlinear pendulum benchmark.
2. It samples a batch of control perturbations in parallel.
3. It rolls out all sampled trajectories under `vmap`.
4. For the trivial case, it uses the linear zero-order-hold step map and checks
   one step against the exact discrete dynamics.
5. For the nonlinear pendulum, it integrates the dynamics with `Diffrax`.
6. It updates the control sequence with either:
   the softmax-weighted MPPI rule, or the elite-set refit CEM rule.
7. It clips controls back into the torque box after each update.

## Run

In the current local environment, item 3 runs the nonlinear pendulum benchmark on CPU by default:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3
```

To run the previous trivial case instead:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem trivial
```

To run only one nonlinear constraint mode:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem nonlinear --constraint-mode hard
```

or

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem nonlinear --constraint-mode soft
```

For horizon-scaling tests, override the default horizon directly:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem nonlinear --constraint-mode hard --horizon 20
```

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem nonlinear --constraint-mode hard --horizon 200
```

## Google Colab

To run item 3 on GPU in Google Colab:

1. Open a new notebook in Google Colab.
2. Change the runtime to `GPU`.
3. Make the repository available in the notebook, for example by cloning a
   public repository URL or uploading a zip if the repository is private.
4. Install the dependencies.
5. Verify that `JAX` sees the GPU.
6. Run the item 3 module.

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
!PYTHONPATH=src python -m optimal_control_prototype_testing.item3_jax.run_item3
```

To run the trivial benchmark in Colab instead:

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem trivial
```

To collect the testing metrics at the two planned horizons:

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem nonlinear --constraint-mode hard --horizon 20
```

```python
!PYTHONPATH=src python -m optimal_control_prototype_testing.item3_jax.run_item3 --problem nonlinear --constraint-mode hard --horizon 200
```

Expected signs of a successful GPU run:

- `backend: gpu`
- `devices: ('cuda:0',)` or another CUDA device

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
- separate results for the selected problem and mode
- each result's `problem`
- each result's `constraint_mode`
- each method's `iterations`
- each method's `objective`
- each method's `runtime_seconds`
- each method's `max_control_violation`
- each method's `max_state_violation`
- each method's `diffrax_vs_exact_step_error`
- full `state_trajectory`
- full `control_trajectory`
