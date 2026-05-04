from __future__ import annotations

import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from jax.scipy.linalg import expm

from .problem import TrivialLQRProblem, build_trivial_lqr_problem


def _x64_enabled() -> bool:
    value = os.environ.get("ITEM1_ENABLE_X64", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


if _x64_enabled():
    jax.config.update("jax_enable_x64", True)


Array = jax.Array


def default_dtype() -> jnp.dtype:
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


@dataclass(frozen=True)
class SQPOptions:
    max_iterations: int = 20
    constraint_tolerance: float = 1e-3
    step_tolerance: float = 1e-3
    regularization: float = 1e-5
    merit_weight: float = 1000.0
    line_search_shrink: float = 0.5
    min_step_size: float = 1e-4


def default_sqp_options() -> SQPOptions:
    if jax.config.jax_enable_x64:
        return SQPOptions()
    return SQPOptions(
        max_iterations=30,
        step_tolerance=3e-3,
        regularization=5e-5,
    )


@dataclass(frozen=True)
class MultipleShootingResult:
    converged: bool
    iterations: int
    objective_value: float
    constraint_norm: float
    step_norm: float
    max_control_violation: float
    diffrax_vs_exact_step_error: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray


def controls_from_raw(raw_controls: Array, problem: TrivialLQRProblem) -> Array:
    dtype = raw_controls.dtype
    midpoint = 0.5 * (
        jnp.asarray(problem.u_max, dtype=dtype) + jnp.asarray(problem.u_min, dtype=dtype)
    )
    radius = 0.5 * (
        jnp.asarray(problem.u_max, dtype=dtype) - jnp.asarray(problem.u_min, dtype=dtype)
    )
    return midpoint + radius * jnp.tanh(raw_controls)


def zero_order_hold_discretization(problem: TrivialLQRProblem) -> tuple[Array, Array]:
    dtype = default_dtype()
    A = jnp.asarray(problem.A, dtype=dtype)
    B = jnp.asarray(problem.B, dtype=dtype)
    nx, nu = A.shape[0], B.shape[1]
    block = jnp.zeros((nx + nu, nx + nu), dtype=A.dtype)
    block = block.at[:nx, :nx].set(A)
    block = block.at[:nx, nx:].set(B)
    transition = expm(block * problem.dt)
    return transition[:nx, :nx], transition[:nx, nx:]


def diffrax_step(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
    dtype = x.dtype
    A = jnp.asarray(problem.A, dtype=dtype)
    B = jnp.asarray(problem.B, dtype=dtype)
    term = ODETerm(lambda _t, y, args: A @ y + B @ args)
    solution = diffeqsolve(
        term,
        Dopri5(),
        t0=0.0,
        t1=problem.dt,
        dt0=problem.dt / 8.0,
        y0=x,
        args=u,
        saveat=SaveAt(t1=True),
        max_steps=32,
    )
    return solution.ys[0]


def pack_variables(states: Array, raw_controls: Array) -> Array:
    return jnp.concatenate([states.reshape(-1), raw_controls.reshape(-1)])


def unpack_variables(z: Array, problem: TrivialLQRProblem) -> tuple[Array, Array]:
    state_size = (problem.horizon + 1) * problem.nx
    states = z[:state_size].reshape(problem.horizon + 1, problem.nx)
    raw_controls = z[state_size:].reshape(problem.horizon, problem.nu)
    return states, raw_controls


def objective_from_variables(z: Array, problem: TrivialLQRProblem) -> Array:
    states, raw_controls = unpack_variables(z, problem)
    controls = controls_from_raw(raw_controls, problem)
    dtype = z.dtype
    Q = jnp.asarray(problem.Q, dtype=dtype)
    R = jnp.asarray(problem.R, dtype=dtype)
    Qf = jnp.asarray(problem.Qf, dtype=dtype)

    stage_cost = jnp.asarray(0.0, dtype=dtype)
    for k in range(problem.horizon):
        xk = states[k]
        uk = controls[k]
        stage_cost = stage_cost + problem.dt * (xk @ Q @ xk + uk @ R @ uk)
    terminal = states[-1] @ Qf @ states[-1]
    return stage_cost + terminal


def multiple_shooting_constraints(z: Array, problem: TrivialLQRProblem) -> Array:
    Ad, Bd = zero_order_hold_discretization(problem)
    states, raw_controls = unpack_variables(z, problem)
    controls = controls_from_raw(raw_controls, problem)
    residuals = [states[0] - jnp.asarray(problem.x0, dtype=states.dtype)]
    for k in range(problem.horizon):
        predicted = Ad @ states[k] + Bd @ controls[k]
        residuals.append(states[k + 1] - predicted)
    return jnp.concatenate(residuals)


def merit_value(z: Array, problem: TrivialLQRProblem, options: SQPOptions) -> Array:
    constraint_residual = multiple_shooting_constraints(z, problem)
    return objective_from_variables(z, problem) + options.merit_weight * jnp.linalg.norm(
        constraint_residual, ord=1
    )


def solve_sqp(
    z0: Array,
    problem: TrivialLQRProblem,
    options: SQPOptions,
) -> tuple[Array, int, bool, float]:
    z = z0
    iterations = 0
    converged = False
    last_step_norm = float("inf")

    objective_fn = lambda zz: objective_from_variables(zz, problem)
    constraints_fn = lambda zz: multiple_shooting_constraints(zz, problem)

    for iteration in range(1, options.max_iterations + 1):
        objective_grad = jax.grad(objective_fn)(z)
        constraint_residual = constraints_fn(z)
        constraint_jacobian = jax.jacfwd(constraints_fn)(z)
        objective_hessian = jax.hessian(objective_fn)(z)

        primal_size = z.shape[0]
        dual_size = constraint_residual.shape[0]
        regularized_hessian = objective_hessian + options.regularization * jnp.eye(
            primal_size, dtype=z.dtype
        )

        top = jnp.concatenate([regularized_hessian, constraint_jacobian.T], axis=1)
        bottom = jnp.concatenate(
            [constraint_jacobian, jnp.zeros((dual_size, dual_size), dtype=z.dtype)],
            axis=1,
        )
        kkt_matrix = jnp.concatenate([top, bottom], axis=0)
        rhs = -jnp.concatenate([objective_grad, constraint_residual])
        solution = jnp.linalg.solve(kkt_matrix, rhs)
        step = solution[:primal_size]
        constraint_norm = float(jnp.linalg.norm(constraint_residual))

        current_merit = merit_value(z, problem, options)
        step_size = 1.0
        current_states, current_raw_controls = unpack_variables(z, problem)
        current_controls = controls_from_raw(current_raw_controls, problem)
        while step_size >= options.min_step_size:
            candidate = z + step_size * step
            if merit_value(candidate, problem, options) < current_merit:
                candidate_states, candidate_raw_controls = unpack_variables(candidate, problem)
                candidate_controls = controls_from_raw(candidate_raw_controls, problem)
                state_step_norm = jnp.linalg.norm(candidate_states - current_states)
                control_step_norm = jnp.linalg.norm(candidate_controls - current_controls)
                last_step_norm = float(
                    jnp.sqrt(state_step_norm**2 + control_step_norm**2)
                )
                z = candidate
                break
            step_size *= options.line_search_shrink
        else:
            candidate = z + options.min_step_size * step
            candidate_states, candidate_raw_controls = unpack_variables(candidate, problem)
            candidate_controls = controls_from_raw(candidate_raw_controls, problem)
            state_step_norm = jnp.linalg.norm(candidate_states - current_states)
            control_step_norm = jnp.linalg.norm(candidate_controls - current_controls)
            last_step_norm = float(jnp.sqrt(state_step_norm**2 + control_step_norm**2))
            z = candidate

        updated_constraint_norm = float(jnp.linalg.norm(constraints_fn(z)))
        if (
            last_step_norm <= options.step_tolerance
            and updated_constraint_norm <= options.constraint_tolerance
        ):
            converged = True
            iterations = iteration
            break

        iterations = iteration

    return z, iterations, converged, last_step_norm


def initial_guess(problem: TrivialLQRProblem) -> Array:
    Ad, Bd = zero_order_hold_discretization(problem)
    Q = np.asarray(problem.Q, dtype=float)
    R = np.asarray(problem.R, dtype=float)
    Qf = np.asarray(problem.Qf, dtype=float)
    u_min = np.asarray(problem.u_min, dtype=float)
    u_max = np.asarray(problem.u_max, dtype=float)

    P = Qf.copy()
    gains: list[np.ndarray] = []
    Ad_np = np.asarray(Ad)
    Bd_np = np.asarray(Bd)
    for _ in range(problem.horizon):
        S = R + Bd_np.T @ P @ Bd_np
        K = -np.linalg.solve(S, Bd_np.T @ P @ Ad_np)
        gains.append(K)
        P = Q + Ad_np.T @ P @ Ad_np + Ad_np.T @ P @ Bd_np @ K
    gains.reverse()

    x_np = np.asarray(problem.x0, dtype=float)
    controls_np: list[np.ndarray] = []
    for K in gains:
        u = K @ x_np
        u = np.clip(u, u_min, u_max)
        controls_np.append(u)
        x_np = Ad_np @ x_np + Bd_np @ u

    dtype = default_dtype()
    controls = jnp.asarray(np.stack(controls_np), dtype=dtype)
    states = [jnp.asarray(problem.x0, dtype=dtype)]
    xk = states[0]
    for uk in controls:
        xk = Ad @ xk + Bd @ uk
        states.append(xk)
    midpoint = 0.5 * (
        jnp.asarray(problem.u_max, dtype=dtype) + jnp.asarray(problem.u_min, dtype=dtype)
    )
    radius = 0.5 * (
        jnp.asarray(problem.u_max, dtype=dtype) - jnp.asarray(problem.u_min, dtype=dtype)
    )
    normalized = jnp.clip((controls - midpoint) / radius, -0.999999, 0.999999)
    raw_controls = jnp.arctanh(normalized)
    return pack_variables(jnp.stack(states), raw_controls)


def solve_trivial_lqr_with_multiple_shooting(
    problem: TrivialLQRProblem | None = None,
    options: SQPOptions | None = None,
) -> MultipleShootingResult:
    problem = problem or build_trivial_lqr_problem()
    options = options or default_sqp_options()

    z0 = initial_guess(problem)
    z_star, iterations, converged, step_norm = solve_sqp(z0, problem, options)
    states, raw_controls = unpack_variables(z_star, problem)
    controls = controls_from_raw(raw_controls, problem)
    constraints = multiple_shooting_constraints(z_star, problem)

    reference_step = diffrax_step(
        problem,
        jnp.asarray(problem.x0, dtype=default_dtype()),
        jnp.zeros(problem.nu, dtype=default_dtype()),
    )
    Ad, Bd = zero_order_hold_discretization(problem)
    exact_step = Ad @ jnp.asarray(problem.x0, dtype=default_dtype()) + Bd @ jnp.zeros(
        problem.nu, dtype=default_dtype()
    )

    violation = jnp.maximum(
        controls - jnp.asarray(problem.u_max, dtype=controls.dtype), 0.0
    ) + jnp.maximum(
        jnp.asarray(problem.u_min, dtype=controls.dtype) - controls, 0.0
    )

    return MultipleShootingResult(
        converged=converged,
        iterations=iterations,
        objective_value=float(objective_from_variables(z_star, problem)),
        constraint_norm=float(jnp.linalg.norm(constraints)),
        step_norm=step_norm,
        max_control_violation=float(jnp.max(jnp.abs(violation))),
        diffrax_vs_exact_step_error=float(jnp.linalg.norm(reference_step - exact_step)),
        state_trajectory=np.asarray(states),
        control_trajectory=np.asarray(controls),
    )
