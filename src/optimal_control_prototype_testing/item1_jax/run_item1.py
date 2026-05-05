from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .multiple_shooting import (
    solve_nonlinear_pendulum_with_multiple_shooting,
    solve_trivial_lqr_with_multiple_shooting,
)
from .problem import build_trivial_lqr_problem
from optimal_control_prototype_testing.nonlinear_pendulum import build_nonlinear_pendulum_problem


@dataclass(frozen=True)
class Item1Environment:
    backend: str
    devices: tuple[str, ...]
    x64_enabled: bool
    dtype: str


def detect_jax_environment() -> Item1Environment:
    try:
        import jax
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError(
            "Item 1 requires JAX. Install the item 1 dependencies in the target "
            "environment, for example a Colab GPU runtime."
        ) from exc

    return Item1Environment(
        backend=jax.default_backend(),
        devices=tuple(str(device) for device in jax.devices()),
        x64_enabled=bool(jax.config.jax_enable_x64),
        dtype="float64" if jax.config.jax_enable_x64 else "float32",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run item 1 JAX multiple-shooting baselines.")
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
    return parser.parse_args()


def print_result(result) -> None:
    print(f"  problem: {result.problem_name}")
    print(f"  constraint_mode: {result.constraint_mode}")
    print(f"  converged: {result.converged}")
    print(f"  iterations: {result.iterations}")
    print(f"  objective: {result.objective_value}")
    print(f"  constraint_norm: {result.constraint_norm:.3e}")
    print(f"  step_norm: {result.step_norm:.3e}")
    print(f"  max_control_violation: {result.max_control_violation:.3e}")
    print(f"  max_state_violation: {result.max_state_violation:.3e}")
    print(f"  diffrax_vs_reference_step_error: {result.diffrax_vs_reference_step_error:.3e}")
    print(
        "  state_trajectory:\n"
        f"{np.array2string(result.state_trajectory, precision=4)}"
    )
    print(
        "  control_trajectory:\n"
        f"{np.array2string(result.control_trajectory, precision=4)}"
    )


def main() -> None:
    args = parse_args()
    environment = detect_jax_environment()
    if args.problem == "trivial":
        problem = build_trivial_lqr_problem()
        results = [solve_trivial_lqr_with_multiple_shooting(problem)]
    else:
        problem = build_nonlinear_pendulum_problem()
        results = []
        if args.constraint_mode in ("hard", "both"):
            results.append(solve_nonlinear_pendulum_with_multiple_shooting(problem, soft_constraints=False))
        if args.constraint_mode in ("soft", "both"):
            results.append(solve_nonlinear_pendulum_with_multiple_shooting(problem, soft_constraints=True))

    print("item1_jax_multiple_shooting")
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
