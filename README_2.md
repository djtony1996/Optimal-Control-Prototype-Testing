# Item 2 GPU Prototype

This README documents item 2 of the minimal prototype plan:

- `DDP/iLQR` on GPU
- `JAX + Diffrax`
- trivial dynamics `LQR` case
- `iLQR` first

## Problem Being Solved

The item 2 baseline uses the same trivial finite-horizon `LQR` setup as items
1, 4, and 5.

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

The current item 2 implementation includes:

- a separate `item2_jax` module
- a pure `JAX` iLQR solver
- a JIT-compiled fixed-iteration solve loop
- autodiff-based dynamics and stage-cost derivatives
- a `Diffrax` vs exact discretization consistency check
- a log-barrier treatment for the hard control bound

The current item 2 implementation does **not** yet include:

- a separate DDP variant
- an explicit warm-start batching harness
- a dedicated Colab notebook wrapper

## Code Layout

- solver: [`src/optimal_control_prototype_testing/item2_jax/ilqr.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item2_jax/ilqr.py)
- runner: [`src/optimal_control_prototype_testing/item2_jax/run_item2.py`](/Users/jitongding/Documents/GitHub/Optimal-Control-Prototype-Testing/src/optimal_control_prototype_testing/item2_jax/run_item2.py)

## How Item 2 Solves the Problem

The current item 2 prototype works in these steps:

1. It reuses the same trivial continuous-time `LQR` setup as the other items.
2. It discretizes the linear dynamics with zero-order hold and checks one step
   against `Diffrax`.
3. It rolls out a nominal control sequence to get the state trajectory.
4. It computes dynamics and cost derivatives with `jax.jacfwd`, `jax.grad`,
   and `jax.hessian`.
5. It runs the iLQR backward Riccati sweep to compute feedforward and feedback
   gains.
6. It runs a forward rollout with line-search candidates to update the control
   trajectory.
7. It repeats until the control update is small or the cost improvement stalls.

The hard control bound is handled with a small log barrier in the running cost,
and the forward rollout clips trial controls slightly inside the feasible box so
the barrier remains well-defined.

## Run

In the current local environment, item 2 runs on CPU with:

```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2
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
- `control_update_norm`
- `max_control_violation`
- `diffrax_vs_exact_step_error`
- full `state_trajectory`
- full `control_trajectory`
