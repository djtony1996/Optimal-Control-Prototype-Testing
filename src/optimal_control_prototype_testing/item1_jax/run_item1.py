from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multiple_shooting import solve_trivial_lqr_with_multiple_shooting
from .problem import build_trivial_lqr_problem


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


def main() -> None:
    environment = detect_jax_environment()
    problem = build_trivial_lqr_problem()
    result = solve_trivial_lqr_with_multiple_shooting(problem)

    print("item1_jax_multiple_shooting")
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
    print(f"  constraint_norm: {result.constraint_norm:.3e}")
    print(f"  step_norm: {result.step_norm:.3e}")
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
