from __future__ import annotations

import numpy as np

from .ilqr import detect_jax_environment, solve_trivial_lqr_with_ilqr
from optimal_control_prototype_testing.item1_jax.problem import build_trivial_lqr_problem


def main() -> None:
    environment = detect_jax_environment()
    problem = build_trivial_lqr_problem()
    result = solve_trivial_lqr_with_ilqr(problem)

    print("item2_jax_ilqr")
    print(f"  backend: {environment.backend}")
    print(f"  devices: {environment.devices}")
    print(f"  x64_enabled: {environment.x64_enabled}")
    print(f"  default_dtype: {environment.dtype}")
    print(f"  horizon: {problem.horizon}")
    print(f"  final_time: {problem.final_time}")
    print(f"  dt: {problem.dt}")
    print(f"  converged: {result.converged}")
    print(f"  iterations: {result.iterations}")
    print(f"  objective: {result.objective_value}")
    print(f"  control_update_norm: {result.control_update_norm:.3e}")
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


if __name__ == "__main__":
    main()
