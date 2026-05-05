# Item 1 GPU Prototype

This README documents item 1 of the minimal prototype plan:

- direct transcription on GPU
- `JAX + Diffrax`
- trivial dynamics `LQR` case
- multiple shooting first

## Problem Being Solved

The item 1 baseline uses the same trivial finite-horizon `LQR` setup as items
4 and 5.

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

## Current Status

The current item 1 implementation includes:

- a shared trivial LQR problem definition
- a `JAX` multiple-shooting baseline
- a `Diffrax` vs exact discretization consistency check
- a prototype JAX-native SQP loop

The current item 1 implementation does **not** yet include:

- direct collocation
- a Colab notebook wrapper

The current code is now written to run in either:

- `float32` mode by default, which is the safer starting point for Colab GPU
- optional `float64` mode by setting `ITEM1_ENABLE_X64=1` before launch

The current multiple-shooting prototype has been verified on:

- local CPU
- Google Colab GPU for the trivial LQR case

## Code Layout

- shared problem: [`src/optimal_control_prototype_testing/item1_jax/problem.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item1_jax/problem.py)
- multiple shooting solver: [`src/optimal_control_prototype_testing/item1_jax/multiple_shooting.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item1_jax/multiple_shooting.py)
- runner: [`src/optimal_control_prototype_testing/item1_jax/run_item1.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item1_jax/run_item1.py)

## How Item 1 Solves the Problem

The current item 1 multiple-shooting prototype works in these steps:

1. It defines the continuous-time trivial `LQR` model.
2. It discretizes the linear dynamics with zero-order hold and also checks the
   one-step dynamics against a `Diffrax` integration result.
3. It transcribes the horizon with multiple shooting:
   the optimization variables are the full state trajectory and a bounded
   parameterization of the control trajectory.
4. It solves the resulting equality-constrained direct-transcription problem
   with a small JAX-native SQP loop.

Hard control bounds are enforced through a bounded control parameterization, so
the reported control trajectory remains feasible throughout the solve.

## Run

In the current local environment, item 1 runs on CPU with:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1
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
!git clone <your-repo-url>
%cd Optimal-Control-Prototype-Testing
```

```python
!pip install -U "jax[cuda12]" diffrax numpy
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
- convergence flag
- iteration count
- `objective`
- `constraint_norm`
- `step_norm`
- `max_control_violation`
- `diffrax_vs_exact_step_error`
- full `state_trajectory`
- full `control_trajectory`
