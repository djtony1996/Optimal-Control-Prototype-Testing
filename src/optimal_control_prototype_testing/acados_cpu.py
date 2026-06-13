from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace

import numpy as np

from optimal_control_prototype_testing.item1_jax.problem import (
    build_trivial_lqr_problem as _build_lqr_problem,
    pure_tracking_cost as _lqr_pure_cost,
)
from optimal_control_prototype_testing.nonlinear_pendulum import (
    build_nonlinear_pendulum_problem,
    pure_tracking_cost as _nl_pure_cost,
)


@dataclass(frozen=True)
class AcadosBaselineResult:
    problem_name: str
    constraint_mode: str
    horizon: int
    status: int
    sqp_iterations: int | None
    objective_value: float | None
    runtime_seconds: float
    max_control_violation: float
    max_state_violation: float
    final_position_error: float
    final_velocity_error: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray


def _missing_dependency_message(exc: Exception) -> str:
    acados_root = os.environ.get("ACADOS_SOURCE_DIR", "<acados_root>")
    return (
        "Item 4 requires a local acados installation.\n"
        f"Import failure: {exc}\n\n"
        "Expected setup:\n"
        "1. Install macOS build tools for acados: cmake and Homebrew gcc.\n"
        "2. Clone and build acados with shared libs and OpenMP enabled.\n"
        "3. Export ACADOS_SOURCE_DIR and DYLD_LIBRARY_PATH.\n"
        "4. Install the Python interface from:\n"
        f"   {acados_root}/interfaces/acados_template\n\n"
        "See scripts/setup_acados_macos.sh for the intended workflow."
    )


def _ensure_acados_python_interface_on_path() -> None:
    acados_root = os.environ.get("ACADOS_SOURCE_DIR")
    if not acados_root:
        return
    python_interface_root = os.path.join(acados_root, "interfaces", "acados_template")
    if python_interface_root not in sys.path:
        sys.path.append(python_interface_root)


def _imports():
    _ensure_acados_python_interface_on_path()
    try:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(_missing_dependency_message(exc)) from exc
    return ca, AcadosModel, AcadosOcp, AcadosOcpSolver


def build_trivial_lqr_ocp(*, dt: float = 0.1):
    ca, AcadosModel, AcadosOcp, _ = _imports()

    nx = 2
    nu = 1
    final_time = 2.0
    horizon = int(round(final_time / dt))

    A = np.array([[0.0, 1.0], [-0.25, -0.1]])
    B = np.array([[0.0], [1.0]])
    Q = np.diag([1.0, 0.2])
    R = np.diag([0.05])
    Qf = np.diag([8.0, 1.0])
    x_init = np.array([1.5, 0.0])
    u_min = np.array([-0.75])
    u_max = np.array([0.75])

    model = AcadosModel()
    model.name = "trivial_lqr_acados"
    model.x = ca.SX.sym("x", nx)
    model.u = ca.SX.sym("u", nu)
    model.xdot = ca.SX.sym("xdot", nx)
    model.f_expl_expr = ca.mtimes(ca.DM(A), model.x) + ca.mtimes(ca.DM(B), model.u)
    model.f_impl_expr = model.xdot - model.f_expl_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.code_gen_opts.code_export_directory = "/tmp/acados_item4_generated_code_trivial"
    ocp.code_gen_opts.json_file = "/tmp/trivial_lqr_acados_ocp.json"

    ocp.solver_options.N_horizon = horizon
    ocp.solver_options.tf = final_time

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
    ocp.cost.W_e = Qf
    ocp.cost.Vx = np.vstack([np.eye(nx), np.zeros((nu, nx))])
    ocp.cost.Vu = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
    ocp.cost.Vx_e = np.eye(nx)
    ocp.cost.yref = np.zeros(nx + nu)
    ocp.cost.yref_e = np.zeros(nx)

    ocp.constraints.x0 = x_init
    ocp.constraints.lbu = u_min
    ocp.constraints.ubu = u_max
    ocp.constraints.idxbu = np.array([0])

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"

    return ocp


def build_nonlinear_pendulum_ocp(
    *,
    soft_constraints: bool,
    dt: float | None = None,
    final_time: float | None = None,
    u_max: float | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    Qf: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
):
    ca, AcadosModel, AcadosOcp, _ = _imports()
    problem = build_nonlinear_pendulum_problem()
    if dt is not None:
        problem = replace(problem, dt=dt)
    if final_time is not None:
        problem = replace(problem, final_time=final_time)
    if u_max is not None:
        problem = replace(problem, u_max=np.array([u_max]), u_min=np.array([-u_max]))
    if Q is not None:
        problem = replace(problem, Q=Q)
    if R is not None:
        problem = replace(problem, R=R)
    if Qf is not None:
        problem = replace(problem, Qf=Qf)
    nx = problem.nx
    nu = problem.nu

    model = AcadosModel()
    model.name = f"nonlinear_pendulum_acados_{'soft' if soft_constraints else 'hard'}"
    model.x = ca.SX.sym("x", nx)
    model.u = ca.SX.sym("u", nu)
    model.xdot = ca.SX.sym("xdot", nx)
    theta = model.x[0]
    omega = model.x[1]
    torque = model.u[0]
    inertia = problem.inertia
    f_expl = ca.vertcat(
        omega,
        -(problem.gravity / problem.length) * ca.sin(theta)
        - (problem.damping / inertia) * omega
        + torque / inertia,
    )
    model.f_expl_expr = f_expl
    model.f_impl_expr = model.xdot - f_expl

    x_goal = ca.DM(problem.x_goal)
    Q = ca.DM(problem.Q)
    R = ca.DM(problem.R)
    Qf = ca.DM(problem.Qf)
    angle_error = ca.atan2(ca.sin(theta - x_goal[0]), ca.cos(theta - x_goal[0]))
    error = ca.vertcat(angle_error, omega - x_goal[1])
    running_cost = problem.dt * (
        ca.mtimes([error.T, Q, error]) + ca.mtimes([model.u.T, R, model.u])
    )
    terminal_cost = ca.mtimes([error.T, Qf, error])

    x_min = ca.DM(problem.x_min)
    x_max = ca.DM(problem.x_max)
    u_min = ca.DM(problem.u_min)
    u_max = ca.DM(problem.u_max)
    state_violation = ca.fmax(x_min - model.x, 0) + ca.fmax(model.x - x_max, 0)
    control_violation = ca.fmax(u_min - model.u, 0) + ca.fmax(model.u - u_max, 0)
    soft_penalty = (
        problem.state_soft_weight * ca.dot(state_violation, state_violation)
        + problem.control_soft_weight * ca.dot(control_violation, control_violation)
    )
    terminal_state_penalty = problem.state_soft_weight * ca.dot(state_violation, state_violation)
    hard_state_penalty = 1e4 * ca.dot(state_violation, state_violation)

    ocp = AcadosOcp()
    ocp.model = model
    mode_name = "soft" if soft_constraints else "hard"
    ocp.code_gen_opts.code_export_directory = f"/tmp/acados_item4_generated_code_pendulum_{mode_name}"
    ocp.code_gen_opts.json_file = f"/tmp/nonlinear_pendulum_{mode_name}_acados_ocp.json"

    ocp.solver_options.N_horizon = problem.horizon
    ocp.solver_options.tf = problem.final_time

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    if soft_constraints:
        ocp.model.cost_expr_ext_cost = running_cost + soft_penalty
        ocp.model.cost_expr_ext_cost_e = terminal_cost + terminal_state_penalty
    else:
        ocp.model.cost_expr_ext_cost = running_cost + hard_state_penalty
        ocp.model.cost_expr_ext_cost_e = terminal_cost + hard_state_penalty

    ocp.constraints.x0 = problem.x0.copy()
    ocp.constraints.lbu = problem.u_min.copy()
    ocp.constraints.ubu = problem.u_max.copy()
    ocp.constraints.idxbu = np.arange(nu)
    if not soft_constraints:
        ocp.constraints.lbx = problem.x_min.copy()
        ocp.constraints.ubx = problem.x_max.copy()
        ocp.constraints.idxbx = np.arange(nx)
        ocp.constraints.lbx_e = problem.x_min.copy()
        ocp.constraints.ubx_e = problem.x_max.copy()
        ocp.constraints.idxbx_e = np.arange(nx)

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = max_iter
    ocp.solver_options.nlp_solver_tol_stat = tol
    ocp.solver_options.nlp_solver_tol_eq = tol
    ocp.solver_options.nlp_solver_tol_ineq = tol
    ocp.solver_options.nlp_solver_tol_comp = tol

    return ocp, problem


def _extract_result(solver, ocp, *, problem_name: str, constraint_mode: str, x_min=None, x_max=None):
    start = time.perf_counter()
    status = solver.solve()
    runtime_seconds = time.perf_counter() - start
    horizon = ocp.solver_options.N_horizon
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    sim_x = np.zeros((horizon + 1, nx))
    sim_u = np.zeros((horizon, nu))
    for stage in range(horizon):
        sim_x[stage, :] = solver.get(stage, "x")
        sim_u[stage, :] = solver.get(stage, "u")
    sim_x[horizon, :] = solver.get(horizon, "x")

    sqp_iterations = None
    objective_value = None
    if status == 0:
        try:
            sqp_iterations = int(solver.get_stats("sqp_iter"))
        except Exception:
            pass
        try:
            objective_value = float(solver.get_cost())
        except Exception:
            pass

    control_violation = np.maximum(sim_u - ocp.constraints.ubu, 0.0) + np.maximum(
        ocp.constraints.lbu - sim_u, 0.0
    )
    if x_min is None or x_max is None:
        max_state_violation = 0.0
    else:
        state_violation = np.maximum(sim_x - x_max[None, :], 0.0) + np.maximum(
            x_min[None, :] - sim_x, 0.0
        )
        max_state_violation = float(np.max(np.abs(state_violation)))

    return AcadosBaselineResult(
        problem_name=problem_name,
        constraint_mode=constraint_mode,
        horizon=horizon,
        status=status,
        sqp_iterations=sqp_iterations,
        objective_value=objective_value,
        runtime_seconds=runtime_seconds,
        max_control_violation=float(np.max(np.abs(control_violation))) if len(sim_u) else 0.0,
        max_state_violation=max_state_violation,
        final_position_error=0.0,
        final_velocity_error=0.0,
        state_trajectory=sim_x,
        control_trajectory=sim_u,
    )


def solve_trivial_lqr_with_acados(*, dt: float = 0.1) -> AcadosBaselineResult:
    _, _, _, AcadosOcpSolver = _imports()
    ocp = build_trivial_lqr_ocp(dt=dt)
    solver = AcadosOcpSolver(ocp, json_file=ocp.code_gen_opts.json_file)
    result = _extract_result(solver, ocp, problem_name="trivial_lqr", constraint_mode="hard")
    lqr_problem = replace(_build_lqr_problem(), dt=dt)
    final_state = result.state_trajectory[-1]
    updates = dict(
        final_position_error=float(final_state[0]),
        final_velocity_error=float(final_state[1]),
    )
    if result.objective_value is not None:
        updates["objective_value"] = _lqr_pure_cost(lqr_problem, result.state_trajectory, result.control_trajectory)
    return replace(result, **updates)


def solve_nonlinear_pendulum_with_acados(
    *,
    soft_constraints: bool,
    dt: float | None = None,
    final_time: float | None = None,
    u_max: float | None = None,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    Qf: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> AcadosBaselineResult:
    _, _, _, AcadosOcpSolver = _imports()
    ocp, problem = build_nonlinear_pendulum_ocp(
        soft_constraints=soft_constraints,
        dt=dt,
        final_time=final_time,
        u_max=u_max,
        Q=Q,
        R=R,
        Qf=Qf,
        max_iter=max_iter,
        tol=tol,
    )
    solver = AcadosOcpSolver(ocp, json_file=ocp.code_gen_opts.json_file)
    result = _extract_result(
        solver,
        ocp,
        problem_name="nonlinear_pendulum",
        constraint_mode="soft" if soft_constraints else "hard",
        x_min=problem.x_min,
        x_max=problem.x_max,
    )
    final_error = problem.state_error(result.state_trajectory[-1])
    updates = dict(
        final_position_error=float(final_error[0]),
        final_velocity_error=float(final_error[1]),
    )
    if result.objective_value is not None:
        updates["objective_value"] = _nl_pure_cost(problem, result.state_trajectory, result.control_trajectory)
    return replace(result, **updates)


def format_result(result: AcadosBaselineResult) -> str:
    final_state = np.array2string(result.state_trajectory[-1], precision=4)
    first_control = (
        np.array2string(result.control_trajectory[0], precision=4)
        if len(result.control_trajectory)
        else "[]"
    )
    state_trajectory = np.array2string(result.state_trajectory, precision=4)
    control_trajectory = np.array2string(result.control_trajectory, precision=4)
    return (
        "item4_acados_cpu\n"
        f"  problem: {result.problem_name}\n"
        f"  constraint_mode: {result.constraint_mode}\n"
        f"  horizon: {result.horizon}\n"
        f"  status: {result.status}\n"
        f"  sqp_iterations: {result.sqp_iterations}\n"
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
    parser = argparse.ArgumentParser(description="Run item 4 acados CPU baselines.")
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
        print(format_result(solve_trivial_lqr_with_acados(**kwargs)))
        return

    results = []
    if args.constraint_mode in ("hard", "both"):
        results.append(solve_nonlinear_pendulum_with_acados(soft_constraints=False, dt=args.dt, final_time=args.final_time))
    if args.constraint_mode in ("soft", "both"):
        results.append(solve_nonlinear_pendulum_with_acados(soft_constraints=True, dt=args.dt, final_time=args.final_time))
    for index, result in enumerate(results):
        if index:
            print()
        print(format_result(result))


if __name__ == "__main__":
    main()
