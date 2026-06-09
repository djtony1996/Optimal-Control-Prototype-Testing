from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from jax.scipy.linalg import expm

from optimal_control_prototype_testing.nonlinear_pendulum import (
    NonlinearPendulumProblem,
    build_nonlinear_pendulum_problem,
    pure_tracking_cost as _nl_pure_cost,
)

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
    state_penalty_weight: float = 1e4


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
    problem_name: str
    constraint_mode: str
    converged: bool
    iterations: int
    objective_value: float
    runtime_seconds: float
    constraint_norm: float
    step_norm: float
    max_control_violation: float
    max_state_violation: float
    diffrax_vs_reference_step_error: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray


def controls_from_raw(raw_controls: Array, u_min: Array, u_max: Array) -> Array:
    midpoint = 0.5 * (u_max + u_min)
    radius = 0.5 * (u_max - u_min)
    return midpoint + radius * jnp.tanh(raw_controls)


def pack_variables(states: Array, raw_controls: Array) -> Array:
    return jnp.concatenate([states.reshape(-1), raw_controls.reshape(-1)])


def unpack_variables(z: Array, horizon: int, nx: int, nu: int) -> tuple[Array, Array]:
    state_size = (horizon + 1) * nx
    states = z[:state_size].reshape(horizon + 1, nx)
    raw_controls = z[state_size:].reshape(horizon, nu)
    return states, raw_controls


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


def trivial_diffrax_step(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
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


def pendulum_diffrax_step(problem: NonlinearPendulumProblem, x: Array, u: Array) -> Array:
    dtype = x.dtype
    inertia = jnp.asarray(problem.inertia, dtype=dtype)
    gravity = jnp.asarray(problem.gravity, dtype=dtype)
    length = jnp.asarray(problem.length, dtype=dtype)
    damping = jnp.asarray(problem.damping, dtype=dtype)

    def dynamics(_t, y, args):
        theta = y[0]
        omega = y[1]
        torque = args[0]
        omega_dot = -(gravity / length) * jnp.sin(theta) - (damping / inertia) * omega + torque / inertia
        return jnp.array([omega, omega_dot], dtype=dtype)

    solution = diffeqsolve(
        ODETerm(dynamics),
        Dopri5(),
        t0=0.0,
        t1=problem.dt,
        dt0=problem.dt / 8.0,
        y0=x,
        args=u,
        saveat=SaveAt(t1=True),
        max_steps=64,
    )
    return solution.ys[0]


def pendulum_rk4_step(problem: NonlinearPendulumProblem, x: Array, u: Array) -> Array:
    dtype = x.dtype
    dt = jnp.asarray(problem.dt, dtype=dtype)
    inertia = jnp.asarray(problem.inertia, dtype=dtype)
    gravity = jnp.asarray(problem.gravity, dtype=dtype)
    length = jnp.asarray(problem.length, dtype=dtype)
    damping = jnp.asarray(problem.damping, dtype=dtype)

    def f(y: Array) -> Array:
        theta = y[0]
        omega = y[1]
        torque = u[0]
        omega_dot = -(gravity / length) * jnp.sin(theta) - (damping / inertia) * omega + torque / inertia
        return jnp.array([omega, omega_dot], dtype=dtype)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_sqp(
    z0: Array,
    objective_fn,
    constraints_fn,
    unpack_fn,
    controls_fn,
    options: SQPOptions,
) -> tuple[Array, int, bool, float]:
    z = z0
    iterations = 0
    converged = False
    last_step_norm = float("inf")

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

        current_merit = objective_fn(z) + options.merit_weight * jnp.linalg.norm(constraint_residual, ord=1)
        step_size = 1.0
        current_states, current_raw_controls = unpack_fn(z)
        current_controls = controls_fn(current_raw_controls)
        while step_size >= options.min_step_size:
            candidate = z + step_size * step
            candidate_constraints = constraints_fn(candidate)
            candidate_merit = objective_fn(candidate) + options.merit_weight * jnp.linalg.norm(
                candidate_constraints, ord=1
            )
            if candidate_merit < current_merit:
                candidate_states, candidate_raw_controls = unpack_fn(candidate)
                candidate_controls = controls_fn(candidate_raw_controls)
                state_step_norm = jnp.linalg.norm(candidate_states - current_states)
                control_step_norm = jnp.linalg.norm(candidate_controls - current_controls)
                last_step_norm = float(jnp.sqrt(state_step_norm**2 + control_step_norm**2))
                z = candidate
                break
            step_size *= options.line_search_shrink
        else:
            candidate = z + options.min_step_size * step
            candidate_states, candidate_raw_controls = unpack_fn(candidate)
            candidate_controls = controls_fn(candidate_raw_controls)
            state_step_norm = jnp.linalg.norm(candidate_states - current_states)
            control_step_norm = jnp.linalg.norm(candidate_controls - current_controls)
            last_step_norm = float(jnp.sqrt(state_step_norm**2 + control_step_norm**2))
            z = candidate

        updated_constraint_norm = float(jnp.linalg.norm(constraints_fn(z)))
        if last_step_norm <= options.step_tolerance and updated_constraint_norm <= options.constraint_tolerance:
            converged = True
            iterations = iteration
            break
        iterations = iteration

    return z, iterations, converged, last_step_norm


def solve_trivial_lqr_with_multiple_shooting(
    problem: TrivialLQRProblem | None = None,
    options: SQPOptions | None = None,
) -> MultipleShootingResult:
    problem = problem or build_trivial_lqr_problem()
    options = options or default_sqp_options()
    dtype = default_dtype()
    Ad, Bd = zero_order_hold_discretization(problem)
    u_min = jnp.asarray(problem.u_min, dtype=dtype)
    u_max = jnp.asarray(problem.u_max, dtype=dtype)

    def controls_fn(raw_controls: Array) -> Array:
        return controls_from_raw(raw_controls, u_min, u_max)

    def unpack_fn(z: Array) -> tuple[Array, Array]:
        return unpack_variables(z, problem.horizon, problem.nx, problem.nu)

    def objective_fn(z: Array) -> Array:
        states, raw_controls = unpack_fn(z)
        controls = controls_fn(raw_controls)
        Q = jnp.asarray(problem.Q, dtype=dtype)
        R = jnp.asarray(problem.R, dtype=dtype)
        Qf = jnp.asarray(problem.Qf, dtype=dtype)
        stage_cost = jnp.asarray(0.0, dtype=dtype)
        for k in range(problem.horizon):
            stage_cost = stage_cost + problem.dt * (
                states[k] @ Q @ states[k] + controls[k] @ R @ controls[k]
            )
        return stage_cost + states[-1] @ Qf @ states[-1]

    def constraints_fn(z: Array) -> Array:
        states, raw_controls = unpack_fn(z)
        controls = controls_fn(raw_controls)
        residuals = [states[0] - jnp.asarray(problem.x0, dtype=dtype)]
        for k in range(problem.horizon):
            predicted = Ad @ states[k] + Bd @ controls[k]
            residuals.append(states[k + 1] - predicted)
        return jnp.concatenate(residuals)

    Q = np.asarray(problem.Q, dtype=float)
    R = np.asarray(problem.R, dtype=float)
    Qf = np.asarray(problem.Qf, dtype=float)
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
        u = np.clip(K @ x_np, np.asarray(problem.u_min), np.asarray(problem.u_max))
        controls_np.append(u)
        x_np = Ad_np @ x_np + Bd_np @ u
    controls = jnp.asarray(np.stack(controls_np), dtype=dtype)
    states = [jnp.asarray(problem.x0, dtype=dtype)]
    xk = states[0]
    for uk in controls:
        xk = Ad @ xk + Bd @ uk
        states.append(xk)
    normalized = jnp.clip((controls - 0.5 * (u_max + u_min)) / (0.5 * (u_max - u_min)), -0.999999, 0.999999)
    raw_controls = jnp.arctanh(normalized)
    z0 = pack_variables(jnp.stack(states), raw_controls)

    start_time = time.perf_counter()
    z_star, iterations, converged, step_norm = solve_sqp(
        z0, objective_fn, constraints_fn, unpack_fn, controls_fn, options
    )
    jax.block_until_ready(z_star)
    runtime_seconds = time.perf_counter() - start_time
    states_star, raw_controls_star = unpack_fn(z_star)
    controls_star = controls_fn(raw_controls_star)
    constraints = constraints_fn(z_star)
    reference_step = trivial_diffrax_step(problem, jnp.asarray(problem.x0, dtype=dtype), jnp.zeros(problem.nu, dtype=dtype))
    exact_step = Ad @ jnp.asarray(problem.x0, dtype=dtype) + Bd @ jnp.zeros(problem.nu, dtype=dtype)
    control_violation = jnp.maximum(controls_star - u_max, 0.0) + jnp.maximum(u_min - controls_star, 0.0)
    return MultipleShootingResult(
        problem_name="trivial_lqr",
        constraint_mode="hard",
        converged=converged,
        iterations=iterations,
        objective_value=float(objective_fn(z_star)),
        runtime_seconds=runtime_seconds,
        constraint_norm=float(jnp.linalg.norm(constraints)),
        step_norm=step_norm,
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=0.0,
        diffrax_vs_reference_step_error=float(jnp.linalg.norm(reference_step - exact_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )


def solve_nonlinear_pendulum_with_multiple_shooting(
    problem: NonlinearPendulumProblem | None = None,
    options: SQPOptions | None = None,
    *,
    soft_constraints: bool,
) -> MultipleShootingResult:
    problem = problem or build_nonlinear_pendulum_problem()
    options = options or default_sqp_options()
    dtype = default_dtype()
    u_min = jnp.asarray(problem.u_min, dtype=dtype)
    u_max = jnp.asarray(problem.u_max, dtype=dtype)
    x_min = jnp.asarray(problem.x_min, dtype=dtype)
    x_max = jnp.asarray(problem.x_max, dtype=dtype)
    x_goal = jnp.asarray(problem.x_goal, dtype=dtype)
    Q = jnp.asarray(problem.Q, dtype=dtype)
    R = jnp.asarray(problem.R, dtype=dtype)
    Qf = jnp.asarray(problem.Qf, dtype=dtype)

    def controls_fn(raw_controls: Array) -> Array:
        return controls_from_raw(raw_controls, u_min, u_max)

    def unpack_fn(z: Array) -> tuple[Array, Array]:
        return unpack_variables(z, problem.horizon, problem.nx, problem.nu)

    def state_error(x: Array) -> Array:
        error = x - x_goal
        return error.at[0].set(jnp.arctan2(jnp.sin(error[0]), jnp.cos(error[0])))

    def state_violation(x: Array) -> Array:
        return jnp.maximum(x_min - x, 0.0) + jnp.maximum(x - x_max, 0.0)

    def control_violation_fn(u: Array) -> Array:
        return jnp.maximum(u_min - u, 0.0) + jnp.maximum(u - u_max, 0.0)

    def objective_fn(z: Array) -> Array:
        states, raw_controls = unpack_fn(z)
        controls = controls_fn(raw_controls)
        cost = jnp.asarray(0.0, dtype=dtype)
        for k in range(problem.horizon):
            err = state_error(states[k])
            uk = controls[k]
            stage = problem.dt * (err @ Q @ err + uk @ R @ uk)
            s_violation = state_violation(states[k])
            if soft_constraints:
                stage = stage + problem.state_soft_weight * (s_violation @ s_violation)
                c_violation = control_violation_fn(uk)
                stage = stage + problem.control_soft_weight * (c_violation @ c_violation)
            else:
                stage = stage + options.state_penalty_weight * (s_violation @ s_violation)
            cost = cost + stage
        terminal_err = state_error(states[-1])
        terminal = terminal_err @ Qf @ terminal_err
        terminal_violation = state_violation(states[-1])
        if soft_constraints:
            terminal = terminal + problem.state_soft_weight * (terminal_violation @ terminal_violation)
        else:
            terminal = terminal + options.state_penalty_weight * (terminal_violation @ terminal_violation)
        return cost + terminal

    def constraints_fn(z: Array) -> Array:
        states, raw_controls = unpack_fn(z)
        controls = controls_fn(raw_controls)
        residuals = [states[0] - jnp.asarray(problem.x0, dtype=dtype)]
        for k in range(problem.horizon):
            predicted = pendulum_rk4_step(problem, states[k], controls[k])
            residuals.append(states[k + 1] - predicted)
        return jnp.concatenate(residuals)

    controls0 = np.zeros((problem.horizon, problem.nu), dtype=float)
    quarter = problem.horizon // 4
    controls0[: quarter + 2, 0] = problem.u_min[0]
    controls0[quarter + 2 : 2 * quarter, 0] = problem.u_max[0]
    controls0[2 * quarter : 3 * quarter, 0] = problem.u_min[0]
    controls0[3 * quarter :, 0] = problem.u_max[0]
    controls = jnp.asarray(controls0, dtype=dtype)
    states = [jnp.asarray(problem.x0, dtype=dtype)]
    xk = states[0]
    for uk in controls:
        xk = pendulum_rk4_step(problem, xk, uk)
        states.append(xk)
    normalized = jnp.clip((controls - 0.5 * (u_max + u_min)) / (0.5 * (u_max - u_min)), -0.999999, 0.999999)
    raw_controls = jnp.arctanh(normalized)
    z0 = pack_variables(jnp.stack(states), raw_controls)

    start_time = time.perf_counter()
    z_star, iterations, converged, step_norm = solve_sqp(
        z0, objective_fn, constraints_fn, unpack_fn, controls_fn, options
    )
    jax.block_until_ready(z_star)
    runtime_seconds = time.perf_counter() - start_time
    states_star, raw_controls_star = unpack_fn(z_star)
    controls_star = controls_fn(raw_controls_star)
    constraints = constraints_fn(z_star)
    reference_step = pendulum_diffrax_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    rk4_step = pendulum_rk4_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    control_violation = jnp.maximum(controls_star - u_max, 0.0) + jnp.maximum(u_min - controls_star, 0.0)
    state_violation_vals = jax.vmap(state_violation)(states_star)
    return MultipleShootingResult(
        problem_name="nonlinear_pendulum",
        constraint_mode="soft" if soft_constraints else "hard",
        converged=converged,
        iterations=iterations,
        objective_value=_nl_pure_cost(problem, np.asarray(states_star), np.asarray(controls_star)),
        runtime_seconds=runtime_seconds,
        constraint_norm=float(jnp.linalg.norm(constraints)),
        step_norm=step_norm,
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=float(jnp.max(jnp.abs(state_violation_vals))),
        diffrax_vs_reference_step_error=float(jnp.linalg.norm(reference_step - rk4_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )
