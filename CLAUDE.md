# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmark suite comparing five optimal control solvers on standardized problems (trivial LQR and nonlinear pendulum swing-up). Items 1–3 use JAX (GPU-capable), Items 4–5 use CPU-based external solvers (acados, Crocoddyl).

## Running Solvers

**Items 1–3 (JAX):**
```bash
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item1_jax.run_item1
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item2_jax.run_item2
PYTHONPATH=src .venv/bin/python -m optimal_control_prototype_testing.item3_jax.run_item3
```

Common flags: `--problem {trivial|nonlinear}`, `--constraint-mode {hard|soft|both}`, `--horizon N`

**Items 4–5 (CPU):**
```bash
./scripts/run_item4_acados.sh
./scripts/run_item5_crocoddyl.sh
```

**Environment setup** (first time):
```bash
uv sync
./scripts/setup_acados_macos.sh    # Item 4 only
./scripts/setup_crocoddyl_macos.sh # Item 5 only
```

## Architecture

**Shared problem definitions:**
- [src/optimal_control_prototype_testing/nonlinear_pendulum.py](src/optimal_control_prototype_testing/nonlinear_pendulum.py) — pendulum dynamics used by all five items

**JAX solver modules** (Items 1–3), each structured identically:
- `run_item*.py` — entry point: parses args, detects JAX backend/devices, selects problem, runs solver, reports metrics
- solver file (`multiple_shooting.py`, `ilqr.py`, `sampling.py`) — algorithm implementation

**CPU solver modules** (Items 4–5):
- [src/optimal_control_prototype_testing/acados_cpu.py](src/optimal_control_prototype_testing/acados_cpu.py) — acados OCP wrapper
- [src/optimal_control_prototype_testing/crocoddyl_cpu.py](src/optimal_control_prototype_testing/crocoddyl_cpu.py) — Crocoddyl wrapper

**Shell scripts** in [scripts/](scripts/) set `PYTHONPATH` and any external solver env vars before invoking Python.

## Benchmark Problems

| Problem | Horizon | State dim | Control dim |
|---------|---------|-----------|-------------|
| Trivial LQR | N=20 | 2 | 1 |
| Nonlinear pendulum | N=40 | 2 | 1 |

Constraint modes: `hard` (bound constraints enforced exactly) or `soft` (penalty-based). Horizon scaling tests use N=20 and N=200.

## Metrics

Every solver run reports: convergence flag, iteration count, final objective, wall-clock time, max control-bound violation, max state-bound violation, full state/control trajectories, and one-step numerical consistency (Diffrax vs RK4 reference).

## Key Dependencies

- Python 3.12 (`uv` for package management, `.venv/` virtual environment)
- Items 1–3: `jax`, `diffrax`, `numpy`
- Item 4: `acados` (requires local compilation via setup script)
- Item 5: `crocoddyl` (via cmeel wheel stack)
