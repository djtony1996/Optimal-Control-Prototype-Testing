# Item 3 GPU Prototype

This README documents item 3 of the minimal prototype plan:

- `MPPI/CEM` on GPU
- `JAX + Diffrax`
- trivial dynamics `LQR` case
- parallel rollouts under `vmap`

## Problem Being Solved

The item 3 baseline uses the same trivial finite-horizon `LQR` setup as items
1, 2, 4, and 5.

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

The current item 3 implementation includes:

- a separate `item3_jax` module
- a pure `JAX` MPPI solver
- a pure `JAX` CEM solver
- vectorised rollout evaluation under `vmap`
- a JIT-compiled fixed-iteration sampling loop
- a `Diffrax` vs exact discretization consistency check
- hard control bounds enforced by clipping

## Code Layout

- solver: [`src/optimal_control_prototype_testing/item3_jax/sampling.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item3_jax/sampling.py)
- runner: [`src/optimal_control_prototype_testing/item3_jax/run_item3.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item3_jax/run_item3.py)

## How Item 3 Solves the Problem

The current item 3 prototype works in these steps:

1. It reuses the same trivial continuous-time `LQR` setup as the other items.
2. It discretizes the linear dynamics with zero-order hold and checks one step
   against `Diffrax`.
3. It samples a batch of control perturbations in parallel.
4. It rolls out all sampled trajectories under `vmap`.
5. It updates the control sequence with either:
   the softmax-weighted MPPI rule, or the elite-set refit CEM rule.
6. It clips controls back into the torque box after each update.

## Run

In the current local environment, item 3 runs on CPU with:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3
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
!PYTHONPATH=src python -m optimal_control_prototype_testing.item3_jax.run_item3
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
- separate `mppi` and `cem` results
- each method's `objective`
- each method's `max_control_violation`
- each method's `diffrax_vs_exact_step_error`
- full `state_trajectory`
- full `control_trajectory`
