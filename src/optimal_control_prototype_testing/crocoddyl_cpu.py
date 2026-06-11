from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm

from optimal_control_prototype_testing.item1_jax.problem import (
    build_trivial_lqr_problem as _build_lqr_problem,
    pure_tracking_cost as _lqr_pure_cost,
)
from optimal_control_prototype_testing.nonlinear_pendulum import (
    NonlinearPendulumProblem,
    build_nonlinear_pendulum_problem,
    pure_tracking_cost as _nl_pure_cost,
    wrap_angle,
)


@dataclass(frozen=True)
class CrocoddylBaselineResult:
    problem_name: str
    constraint_mode: str
    horizon: int
    converged: bool
    iterations: int
    objective_value: float
    runtime_seconds: float
    max_control_violation: float
    max_state_violation: float
    final_position_error: float
    final_velocity_error: float
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


def _import_crocoddyl():
    _ensure_crocoddyl_python_path()
    try:
        import crocoddyl
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(_missing_dependency_message(exc)) from exc
    return crocoddyl


def zero_order_hold_discretization(A: np.ndarray, B: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    nx, nu = A.shape[0], B.shape[1]
    block = np.zeros((nx + nu, nx + nu), dtype=float)
    block[:nx, :nx] = A
    block[:nx, nx:] = B
    transition = expm(block * dt)
    return transition[:nx, :nx], transition[:nx, nx:]


class PendulumActionModel:
    def __init__(
        self,
        crocoddyl,
        problem: NonlinearPendulumProblem,
        *,
        soft_constraints: bool,
        terminal: bool,
    ) -> None:
        self._crocoddyl = crocoddyl
        self.problem = problem
        self.soft_constraints = soft_constraints
        self.terminal = terminal
        state = crocoddyl.StateVector(problem.nx)
        nu = 0 if terminal else problem.nu
        nr = 1
        crocoddyl.ActionModelAbstract.__init__(self, state, nu, nr)

    def createData(self):
        return self._crocoddyl.ActionDataAbstract(self)

    def _rk4_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        dt = self.problem.dt
        f = self.problem.continuous_dynamics
        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _hard_state_penalty(self, x: np.ndarray) -> float:
        violation = self.problem.hard_state_residual(x)
        return float(1e4 * violation @ violation)

    def calc(self, data, x, u=None):
        x = np.asarray(x, dtype=float)
        if self.terminal:
            u = np.zeros(0, dtype=float)
            data.xnext = x.copy()
        else:
            u = np.asarray(u, dtype=float)
            data.xnext = self._rk4_step(x, u)

        if self.terminal:
            if self.soft_constraints:
                cost = self.problem.terminal_cost(x, soft_constraints=True)
            else:
                cost = self.problem.terminal_cost(x, soft_constraints=False) + self._hard_state_penalty(x)
        else:
            if self.soft_constraints:
                cost = self.problem.running_cost(x, u, soft_constraints=True)
            else:
                cost = self.problem.running_cost(x, u, soft_constraints=False) + self._hard_state_penalty(x)
        data.cost = cost
        data.r = np.array([0.0], dtype=float)

    def calcDiff(self, data, x, u=None):
        raise NotImplementedError("Use ActionModelNumDiff for the nonlinear pendulum model.")


def build_trivial_lqr_problem(*, dt: float = 0.1):
    crocoddyl = _import_crocoddyl()

    nx = 2
    nu = 1
    final_time = 2.0
    horizon = int(round(final_time / dt))

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


def build_nonlinear_pendulum_solver(
    *,
    soft_constraints: bool,
    dt: float | None = None,
    final_time: float | None = None,
) -> tuple[object, NonlinearPendulumProblem]:
    crocoddyl = _import_crocoddyl()
    problem = build_nonlinear_pendulum_problem()
    if dt is not None:
        problem = replace(problem, dt=dt)
    if final_time is not None:
        problem = replace(problem, final_time=final_time)

    class RunningPendulumModel(PendulumActionModel, crocoddyl.ActionModelAbstract):
        pass

    class TerminalPendulumModel(PendulumActionModel, crocoddyl.ActionModelAbstract):
        pass

    running_base = RunningPendulumModel(crocoddyl, problem, soft_constraints=soft_constraints, terminal=False)
    terminal_base = TerminalPendulumModel(crocoddyl, problem, soft_constraints=soft_constraints, terminal=True)
    running_model = crocoddyl.ActionModelNumDiff(running_base)
    terminal_model = crocoddyl.ActionModelNumDiff(terminal_base)

    if not soft_constraints:
        running_model.u_lb = problem.u_min.copy()
        running_model.u_ub = problem.u_max.copy()

    shooting = crocoddyl.ShootingProblem(problem.x0.copy(), [running_model] * problem.horizon, terminal_model)
    solver = crocoddyl.SolverDDP(shooting) if soft_constraints else crocoddyl.SolverBoxDDP(shooting)
    return solver, problem


def _state_violation(problem: NonlinearPendulumProblem, xs: np.ndarray) -> float:
    upper = np.maximum(xs - problem.x_max[None, :], 0.0)
    lower = np.maximum(problem.x_min[None, :] - xs, 0.0)
    return float(np.max(np.abs(lower + upper)))


def solve_trivial_lqr_with_crocoddyl(*, dt: float = 0.1) -> CrocoddylBaselineResult:
    solver, x0 = build_trivial_lqr_problem(dt=dt)
    horizon = solver.problem.T
    xs_init = [x0.copy() for _ in range(horizon + 1)]
    us_init = [np.zeros(solver.problem.runningModels[0].nu) for _ in range(horizon)]
    start = time.perf_counter()
    converged = bool(solver.solve(xs_init, us_init, 100, False))
    runtime_seconds = time.perf_counter() - start
    xs = np.asarray(solver.xs)
    us = np.asarray(solver.us)
    u_lb = solver.problem.runningModels[0].u_lb
    u_ub = solver.problem.runningModels[0].u_ub
    violation = np.maximum(us - u_ub, 0.0) + np.maximum(u_lb - us, 0.0)
    lqr_problem = replace(_build_lqr_problem(), dt=dt)
    final_state_lqr = xs[-1]
    return CrocoddylBaselineResult(
        problem_name="trivial_lqr",
        constraint_mode="hard",
        horizon=horizon,
        converged=converged,
        iterations=int(solver.iter),
        objective_value=_lqr_pure_cost(lqr_problem, xs, us),
        runtime_seconds=runtime_seconds,
        max_control_violation=float(np.max(np.abs(violation))) if len(us) else 0.0,
        max_state_violation=0.0,
        final_position_error=float(final_state_lqr[0]),
        final_velocity_error=float(final_state_lqr[1]),
        state_trajectory=xs,
        control_trajectory=us,
    )


def solve_nonlinear_pendulum_with_crocoddyl(
    *,
    soft_constraints: bool,
    dt: float | None = None,
    final_time: float | None = None,
) -> CrocoddylBaselineResult:
    solver, problem = build_nonlinear_pendulum_solver(soft_constraints=soft_constraints, dt=dt, final_time=final_time)
    horizon = solver.problem.T
    xs_init = [problem.x0.copy() for _ in range(horizon + 1)]
    us_init = [np.zeros(solver.problem.runningModels[0].nu) for _ in range(horizon)]
    start = time.perf_counter()
    converged = bool(solver.solve(xs_init, us_init, 200, False))
    runtime_seconds = time.perf_counter() - start
    xs = np.asarray(solver.xs)
    us = np.asarray(solver.us)
    if soft_constraints:
        control_violation = np.maximum(us - problem.u_max[None, :], 0.0) + np.maximum(
            problem.u_min[None, :] - us,
            0.0,
        )
    else:
        u_lb = solver.problem.runningModels[0].u_lb
        u_ub = solver.problem.runningModels[0].u_ub
        control_violation = np.maximum(us - u_ub, 0.0) + np.maximum(u_lb - us, 0.0)
    final_error_nl = problem.state_error(xs[-1])
    return CrocoddylBaselineResult(
        problem_name="nonlinear_pendulum",
        constraint_mode="soft" if soft_constraints else "hard",
        horizon=horizon,
        converged=converged,
        iterations=int(solver.iter),
        objective_value=_nl_pure_cost(problem, xs, us),
        runtime_seconds=runtime_seconds,
        max_control_violation=float(np.max(np.abs(control_violation))) if len(us) else 0.0,
        max_state_violation=_state_violation(problem, xs),
        final_position_error=float(final_error_nl[0]),
        final_velocity_error=float(final_error_nl[1]),
        state_trajectory=xs,
        control_trajectory=us,
    )


def format_result(result: CrocoddylBaselineResult) -> str:
    final_state = np.array2string(result.state_trajectory[-1], precision=4)
    first_control = (
        np.array2string(result.control_trajectory[0], precision=4)
        if len(result.control_trajectory)
        else "[]"
    )
    state_trajectory = np.array2string(result.state_trajectory, precision=4)
    control_trajectory = np.array2string(result.control_trajectory, precision=4)
    return (
        "item5_crocoddyl_cpu\n"
        f"  problem: {result.problem_name}\n"
        f"  constraint_mode: {result.constraint_mode}\n"
        f"  horizon: {result.horizon}\n"
        f"  converged: {result.converged}\n"
        f"  iterations: {result.iterations}\n"
        f"  objective: {result.objective_value}\n"
        f"  runtime_seconds: {result.runtime_seconds:.6f}\n"
        f"  max_control_violation: {result.max_control_violation:.3e}\n"
        f"  max_state_violation: {result.max_state_violation:.3e}\n"
        f"  final_position_error: {result.final_position_error:.6f}\n"
        f"  final_velocity_error: {result.final_velocity_error:.6f}\n"
        f"  first_control: {first_control}\n"
        f"  final_state: {final_state}\n"
        f"  state_trajectory:\n{state_trajectory}\n"
        f"  control_trajectory:\n{control_trajectory}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run item 5 Crocoddyl CPU baselines.")
    parser.add_argument(
        "--problem",
        choices=("trivial", "nonlinear"),
        default="trivial",
        help="Select which benchmark problem to run.",
    )
    parser.add_argument(
        "--constraint-mode",
        choices=("hard", "soft", "both"),
        default="both",
        help="Constraint handling mode for the nonlinear pendulum benchmark.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override time step dt.",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=None,
        help="Override total time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.problem == "trivial":
        kwargs = {}
        if args.dt is not None:
            kwargs["dt"] = args.dt
        print(format_result(solve_trivial_lqr_with_crocoddyl(**kwargs)))
        return

    results = []
    if args.constraint_mode in ("hard", "both"):
        results.append(
            solve_nonlinear_pendulum_with_crocoddyl(
                soft_constraints=False,
                dt=args.dt,
                final_time=args.final_time,
            )
        )
    if args.constraint_mode in ("soft", "both"):
        results.append(
            solve_nonlinear_pendulum_with_crocoddyl(
                soft_constraints=True,
                dt=args.dt,
                final_time=args.final_time,
            )
        )
    for index, result in enumerate(results):
        if index:
            print()
        print(format_result(result))


if __name__ == "__main__":
    main()
