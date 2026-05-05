from __future__ import annotations

import numpy as np

from .sampling import (
    detect_jax_environment,
    solve_trivial_lqr_with_cem,
    solve_trivial_lqr_with_mppi,
)
from optimal_control_prototype_testing.item1_jax.problem import build_trivial_lqr_problem


def print_result(label: str, result) -> None:
    print(label)
    print(f"  objective: {result.objective_value}")
    print(f"  max_control_violation: {result.max_control_violation:.3e}")
    print(f"  diffrax_vs_exact_step_error: {result.diffrax_vs_exact_step_error:.3e}")
    print(
        "  state_trajectory:\n"
        f"{np.array2string(result.state_trajectory, precision=4)}"
    )
    print(
        "  control_trajectory:\n"
        f"{np.array2string(result.control_trajectory, precision=4)}"
    )


def main() -> None:
    environment = detect_jax_environment()
    problem = build_trivial_lqr_problem()
    mppi_result = solve_trivial_lqr_with_mppi(problem)
    cem_result = solve_trivial_lqr_with_cem(problem)

    print("item3_jax_sampling")
    print(f"  backend: {environment.backend}")
    print(f"  devices: {environment.devices}")
    print(f"  x64_enabled: {environment.x64_enabled}")
    print(f"  default_dtype: {environment.dtype}")
    print(f"  horizon: {problem.horizon}")
    print(f"  final_time: {problem.final_time}")
    print(f"  dt: {problem.dt}")
    print_result("  mppi", mppi_result)
    print_result("  cem", cem_result)


if __name__ == "__main__":
    main()
