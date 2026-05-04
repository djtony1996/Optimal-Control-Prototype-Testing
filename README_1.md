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
- verified GPU execution on a cloud runtime

The current code is now written to run in either:

- `float32` mode by default, which is the safer starting point for Colab GPU
- optional `float64` mode by setting `ITEM1_ENABLE_X64=1` before launch

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

For a future Colab GPU run, the intended pattern is:

```bash
git clone <your-repo-url>
cd Optimal-Control-Prototype-Testing
pip install jax jaxlib diffrax numpy
PYTHONPATH=src python -m optimal_control_prototype_testing.item1_jax.run_item1
```

In Colab, the expected workflow is:

1. Open a new notebook and switch the runtime to `GPU`.
2. Clone this repository in a notebook cell.
3. Install `jax`, `jaxlib`, `diffrax`, and `numpy`.
4. Run the item 1 module from the notebook shell.
5. Check the printed `backend` and `devices` lines. A successful GPU run should
   report a GPU backend instead of `cpu`.

If you want to force the current higher-precision local mode, use:

```bash
ITEM1_ENABLE_X64=1 PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1
```

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

## GPU Note

The item 1 design is intended for GPU execution, but this Mac currently runs
the JAX backend on CPU. A cloud runtime such as Google Colab is the most likely
path for actual GPU validation later. The code now defaults to `float32`, which
is a better starting point for Colab GPU than forcing `float64`.
