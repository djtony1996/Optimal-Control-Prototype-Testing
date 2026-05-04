from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class CrocoddylBaselineResult:
    converged: bool
    iterations: int
    objective_value: float
    max_control_violation: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray


def _missing_dependency_message(exc: Exception) -> str:
    return (
        "Item 5 requires a local Crocoddyl installation.\n"
        f"Import failure: {exc}\n\n"
        "Expected setup:\n"
        "1. Install crocoddyl and its dependencies for the Python environment.\n"
        "2. Confirm the Python import works:\n"
        "   python -c \"import crocoddyl\"\n\n"
        "See README_5.md for the intended item 5 workflow."
    )


def _default_cmeel_site_packages() -> str:
    return os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "cmeel.prefix",
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
    )


def _ensure_crocoddyl_python_path() -> None:
    candidate_paths = []
    extra_path = os.environ.get("CROCODDYL_PYTHON_PATH")
    if extra_path:
        candidate_paths.append(extra_path)
    candidate_paths.append(_default_cmeel_site_packages())

    for path in candidate_paths:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def zero_order_hold_discretization(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    nx, nu = A.shape[0], B.shape[1]
    block = np.zeros((nx + nu, nx + nu), dtype=float)
    block[:nx, :nx] = A
    block[:nx, nx:] = B
    transition = expm(block * dt)
    return transition[:nx, :nx], transition[:nx, nx:]


def build_trivial_lqr_problem():
    _ensure_crocoddyl_python_path()
    try:
        import crocoddyl
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(_missing_dependency_message(exc)) from exc

    nx = 2
    nu = 1
    horizon = 20
    final_time = 2.0
    dt = final_time / horizon

    A = np.array([[0.0, 1.0], [-0.25, -0.1]], dtype=float)
    B = np.array([[0.0], [1.0]], dtype=float)
    Q = np.diag([1.0, 0.2])
    R = np.array([[0.05]], dtype=float)
    Qf = np.diag([8.0, 1.0])
    N = np.zeros((nx, nu), dtype=float)
    x0 = np.array([1.5, 0.0], dtype=float)
    u_lb = np.array([-0.75], dtype=float)
    u_ub = np.array([0.75], dtype=float)

    Ad, Bd = zero_order_hold_discretization(A, B, dt)

    G = np.zeros((0, nx + nu), dtype=float)
    H = np.zeros((0, nx + nu), dtype=float)
    g = np.zeros(0, dtype=float)
    h = np.zeros(0, dtype=float)
    f = np.zeros(nx, dtype=float)
    q = np.zeros(nx, dtype=float)
    r = np.zeros(nu, dtype=float)

    running_model = crocoddyl.ActionModelLQR(nx, nu, False)
    running_model.setLQR(Ad, Bd, Q, R, N, G, H, f, q, r, g, h)
    running_model.u_lb = u_lb
    running_model.u_ub = u_ub

    terminal_model = crocoddyl.ActionModelLQR(nx, 0, False)
    terminal_model.setLQR(
        np.eye(nx),
        np.zeros((nx, 0), dtype=float),
        Qf,
        np.zeros((0, 0), dtype=float),
        np.zeros((nx, 0), dtype=float),
        np.zeros((0, nx), dtype=float),
        np.zeros((0, nx), dtype=float),
        np.zeros(nx, dtype=float),
        np.zeros(nx, dtype=float),
        np.zeros(0, dtype=float),
        np.zeros(0, dtype=float),
        np.zeros(0, dtype=float),
    )

    problem = crocoddyl.ShootingProblem(x0, [running_model] * horizon, terminal_model)
    solver = crocoddyl.SolverBoxDDP(problem)

    return solver, x0


def solve_trivial_lqr_with_crocoddyl() -> CrocoddylBaselineResult:
    solver, x0 = build_trivial_lqr_problem()
    horizon = solver.problem.T

    xs_init = [x0.copy() for _ in range(horizon + 1)]
    us_init = [np.zeros(solver.problem.runningModels[0].nu) for _ in range(horizon)]

    converged = bool(solver.solve(xs_init, us_init, 100, False))
    xs = np.asarray(solver.xs)
    us = np.asarray(solver.us)
    u_lb = solver.problem.runningModels[0].u_lb
    u_ub = solver.problem.runningModels[0].u_ub
    violation = np.maximum(us - u_ub, 0.0) + np.maximum(u_lb - us, 0.0)

    return CrocoddylBaselineResult(
        converged=converged,
        iterations=int(solver.iter),
        objective_value=float(solver.cost),
        max_control_violation=float(np.max(np.abs(violation))) if len(us) else 0.0,
        state_trajectory=xs,
        control_trajectory=us,
    )


def format_result(result: CrocoddylBaselineResult) -> str:
    final_state = np.array2string(result.state_trajectory[-1], precision=4)
    first_control = np.array2string(result.control_trajectory[0], precision=4)
    state_trajectory = np.array2string(result.state_trajectory, precision=4)
    control_trajectory = np.array2string(result.control_trajectory, precision=4)
    return (
        "item5_crocoddyl_cpu\n"
        f"  converged: {result.converged}\n"
        f"  iterations: {result.iterations}\n"
        f"  objective: {result.objective_value}\n"
        f"  max_control_violation: {result.max_control_violation:.3e}\n"
        f"  first_control: {first_control}\n"
        f"  final_state: {final_state}\n"
        f"  state_trajectory:\n{state_trajectory}\n"
        f"  control_trajectory:\n{control_trajectory}"
    )


def main() -> None:
    result = solve_trivial_lqr_with_crocoddyl()
    print(format_result(result))


if __name__ == "__main__":
    main()
