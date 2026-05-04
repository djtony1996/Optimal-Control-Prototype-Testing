from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcadosBaselineResult:
    status: int
    sqp_iterations: int | None
    objective_value: float | None
    max_control_violation: float
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


def build_trivial_lqr_ocp():
    _ensure_acados_python_interface_on_path()
    try:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(_missing_dependency_message(exc)) from exc

    nx = 2
    nu = 1
    horizon = 20
    final_time = 2.0

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
    ocp.code_gen_opts.code_export_directory = "/tmp/acados_item4_generated_code"
    ocp.code_gen_opts.json_file = "/tmp/trivial_lqr_acados_ocp.json"

    ocp.solver_options.N_horizon = horizon
    ocp.solver_options.tf = final_time

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = np.block(
        [
            [Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), R],
        ]
    )
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


def solve_trivial_lqr_with_acados() -> AcadosBaselineResult:
    _ensure_acados_python_interface_on_path()
    try:
        from acados_template import AcadosOcpSolver
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(_missing_dependency_message(exc)) from exc

    ocp = build_trivial_lqr_ocp()
    solver = AcadosOcpSolver(ocp, json_file=ocp.code_gen_opts.json_file)
    status = solver.solve()

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
            sqp_iterations = None
        try:
            objective_value = float(solver.get_cost())
        except Exception:
            objective_value = None

    violation = np.maximum(sim_u - ocp.constraints.ubu, 0.0) + np.maximum(
        ocp.constraints.lbu - sim_u, 0.0
    )

    return AcadosBaselineResult(
        status=status,
        sqp_iterations=sqp_iterations,
        objective_value=objective_value,
        max_control_violation=float(np.max(np.abs(violation))),
        state_trajectory=sim_x,
        control_trajectory=sim_u,
    )


def format_result(result: AcadosBaselineResult) -> str:
    final_state = np.array2string(result.state_trajectory[-1], precision=4)
    first_control = np.array2string(result.control_trajectory[0], precision=4)
    state_trajectory = np.array2string(result.state_trajectory, precision=4)
    control_trajectory = np.array2string(result.control_trajectory, precision=4)
    return (
        "item4_acados_cpu\n"
        f"  status: {result.status}\n"
        f"  sqp_iterations: {result.sqp_iterations}\n"
        f"  objective: {result.objective_value}\n"
        f"  max_control_violation: {result.max_control_violation:.3e}\n"
        f"  first_control: {first_control}\n"
        f"  final_state: {final_state}\n"
        f"  state_trajectory:\n{state_trajectory}\n"
        f"  control_trajectory:\n{control_trajectory}"
    )


def main() -> None:
    result = solve_trivial_lqr_with_acados()
    print(format_result(result))


if __name__ == "__main__":
    main()
