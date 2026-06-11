from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from .sampling import (
    detect_jax_environment,
    solve_nonlinear_pendulum_with_cem,
    solve_nonlinear_pendulum_with_mppi,
    solve_trivial_lqr_with_cem,
    solve_trivial_lqr_with_mppi,
)
from optimal_control_prototype_testing.item1_jax.problem import build_trivial_lqr_problem
from optimal_control_prototype_testing.nonlinear_pendulum import build_nonlinear_pendulum_problem


def print_result(label: str, result) -> None:
    print(label)
    print(f"  problem: {result.problem_name}")
    print(f"  constraint_mode: {result.constraint_mode}")
    print(f"  iterations: {result.iterations}")
    print(f"  objective: {result.objective_value}")
    print(f"  runtime_seconds: {result.runtime_seconds:.6f}")
    print(f"  max_control_violation: {result.max_control_violation:.3e}")
    print(f"  max_state_violation: {result.max_state_violation:.3e}")
    print(f"  final_position_error: {result.final_position_error:.6f}")
    print(f"  final_velocity_error: {result.final_velocity_error:.6f}")
    print(f"  diffrax_vs_exact_step_error: {result.diffrax_vs_exact_step_error:.3e}")
    print(
        "  state_trajectory:\n"
        f"{np.array2string(result.state_trajectory, precision=4)}"
    )
    print(
        "  control_trajectory:\n"
        f"{np.array2string(result.control_trajectory, precision=4)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run item 3 JAX sampling baselines.")
    parser.add_argument(
        "--problem",
        choices=("trivial", "nonlinear"),
        default="nonlinear",
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
    environment = detect_jax_environment()

    if args.problem == "trivial":
        problem = build_trivial_lqr_problem()
        if args.dt is not None:
            problem = replace(problem, dt=args.dt)
        if args.final_time is not None:
            problem = replace(problem, final_time=args.final_time)
        labels_and_results = [
            ("  mppi", solve_trivial_lqr_with_mppi(problem)),
            ("  cem", solve_trivial_lqr_with_cem(problem)),
        ]
    else:
        problem = build_nonlinear_pendulum_problem()
        if args.dt is not None:
            problem = replace(problem, dt=args.dt)
        if args.final_time is not None:
            problem = replace(problem, final_time=args.final_time)
        labels_and_results = []
        if args.constraint_mode in ("hard", "both"):
            labels_and_results.extend(
                [
                    ("  mppi_hard", solve_nonlinear_pendulum_with_mppi(problem, soft_constraints=False)),
                    ("  cem_hard", solve_nonlinear_pendulum_with_cem(problem, soft_constraints=False)),
                ]
            )
        if args.constraint_mode in ("soft", "both"):
            labels_and_results.extend(
                [
                    ("  mppi_soft", solve_nonlinear_pendulum_with_mppi(problem, soft_constraints=True)),
                    ("  cem_soft", solve_nonlinear_pendulum_with_cem(problem, soft_constraints=True)),
                ]
            )

    print("item3_jax_sampling")
    print(f"  backend: {environment.backend}")
    print(f"  devices: {environment.devices}")
    print(f"  x64_enabled: {environment.x64_enabled}")
    print(f"  default_dtype: {environment.dtype}")
    print(f"  horizon: {problem.horizon}")
    print(f"  final_time: {problem.final_time}")
    print(f"  dt: {problem.dt}")
    for label, result in labels_and_results:
        print_result(label, result)


if __name__ == "__main__":
    main()
