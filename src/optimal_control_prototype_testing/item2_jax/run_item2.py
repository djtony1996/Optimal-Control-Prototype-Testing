from __future__ import annotations

import argparse
from dataclasses import replace
import numpy as np

from .ilqr import (
    detect_jax_environment,
    solve_nonlinear_pendulum_with_ilqr,
    solve_trivial_lqr_with_ilqr,
)
from optimal_control_prototype_testing.item1_jax.problem import build_trivial_lqr_problem
from optimal_control_prototype_testing.nonlinear_pendulum import build_nonlinear_pendulum_problem


def print_result(result) -> None:
    print(f"  problem: {result.problem_name}")
    print(f"  constraint_mode: {result.constraint_mode}")
    print(f"  converged: {result.converged}")
    print(f"  iterations: {result.iterations}")
    print(f"  objective: {result.objective_value}")
    print(f"  runtime_seconds: {result.runtime_seconds:.6f}")
    print(f"  control_update_norm: {result.control_update_norm:.3e}")
    print(f"  max_control_violation: {result.max_control_violation:.3e}")
    print(f"  max_state_violation: {result.max_state_violation:.3e}")
    print(f"  final_position_error: {result.final_position_error:.6f}")
    print(f"  final_velocity_error: {result.final_velocity_error:.6f}")
    print(f"  diffrax_vs_reference_step_error: {result.diffrax_vs_reference_step_error:.3e}")
    print(
        "  state_trajectory:\n"
        f"{np.array2string(result.state_trajectory, precision=4)}"
    )
    print(
        "  control_trajectory:\n"
        f"{np.array2string(result.control_trajectory, precision=4)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run item 2 JAX iLQR baselines.")
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
        "--horizon",
        type=int,
        default=None,
        help="Override the default horizon length for testing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    environment = detect_jax_environment()
    if args.problem == "trivial":
        problem = build_trivial_lqr_problem()
        if args.horizon is not None:
            problem = replace(problem, horizon=args.horizon)
        results = [solve_trivial_lqr_with_ilqr(problem)]
    else:
        problem = build_nonlinear_pendulum_problem()
        if args.horizon is not None:
            problem = replace(problem, horizon=args.horizon)
        results = []
        if args.constraint_mode in ("hard", "both"):
            results.append(solve_nonlinear_pendulum_with_ilqr(problem, soft_constraints=False))
        if args.constraint_mode in ("soft", "both"):
            results.append(solve_nonlinear_pendulum_with_ilqr(problem, soft_constraints=True))

    print("item2_jax_ilqr")
    print(f"  backend: {environment.backend}")
    print(f"  devices: {environment.devices}")
    print(f"  x64_enabled: {environment.x64_enabled}")
    print(f"  default_dtype: {environment.dtype}")
    print(f"  horizon: {problem.horizon}")
    print(f"  final_time: {problem.final_time}")
    print(f"  dt: {problem.dt}")
    for result in results:
        print_result(result)


if __name__ == "__main__":
    main()
