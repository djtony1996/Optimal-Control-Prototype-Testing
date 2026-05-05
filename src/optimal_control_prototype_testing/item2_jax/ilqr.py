from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from jax.scipy.linalg import expm

from optimal_control_prototype_testing.item1_jax.problem import (
    TrivialLQRProblem,
    build_trivial_lqr_problem,
)


Array = jax.Array


@dataclass(frozen=True)
class ILQROptions:
    max_iterations: int = 25
    regularization: float = 1e-5
    barrier_weight: float = 1e-3
    barrier_margin: float = 1e-4
    cost_tolerance: float = 1e-6
    control_tolerance: float = 1e-4


@dataclass(frozen=True)
class ILQRResult:
    converged: bool
    iterations: int
    objective_value: float
    control_update_norm: float
    max_control_violation: float
    diffrax_vs_exact_step_error: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray


@dataclass(frozen=True)
class Item2Environment:
    backend: str
    devices: tuple[str, ...]
    x64_enabled: bool
    dtype: str


def detect_jax_environment() -> Item2Environment:
    return Item2Environment(
        backend=jax.default_backend(),
        devices=tuple(str(device) for device in jax.devices()),
        x64_enabled=bool(jax.config.jax_enable_x64),
        dtype="float64" if jax.config.jax_enable_x64 else "float32",
    )


def default_dtype() -> jnp.dtype:
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def zero_order_hold_discretization(problem: TrivialLQRProblem) -> tuple[Array, Array]:
    dtype = default_dtype()
    A = jnp.asarray(problem.A, dtype=dtype)
    B = jnp.asarray(problem.B, dtype=dtype)
    nx, nu = A.shape[0], B.shape[1]
    block = jnp.zeros((nx + nu, nx + nu), dtype=dtype)
    block = block.at[:nx, :nx].set(A)
    block = block.at[:nx, nx:].set(B)
    transition = expm(block * problem.dt)
    return transition[:nx, :nx], transition[:nx, nx:]


def diffrax_step(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
    A = jnp.asarray(problem.A, dtype=x.dtype)
    B = jnp.asarray(problem.B, dtype=x.dtype)
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


def clamp_controls(u: Array, problem: TrivialLQRProblem, margin: Array) -> Array:
    dtype = u.dtype
    lower = jnp.asarray(problem.u_min, dtype=dtype) + margin
    upper = jnp.asarray(problem.u_max, dtype=dtype) - margin
    return jnp.clip(u, lower, upper)


def discrete_dynamics(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
    Ad, Bd = zero_order_hold_discretization(problem)
    return Ad @ x + Bd @ u


def stage_cost(problem: TrivialLQRProblem, x: Array, u: Array, options: ILQROptions) -> Array:
    dtype = x.dtype
    Q = jnp.asarray(problem.Q, dtype=dtype)
    R = jnp.asarray(problem.R, dtype=dtype)
    lower = jnp.asarray(problem.u_min, dtype=dtype)
    upper = jnp.asarray(problem.u_max, dtype=dtype)
    slack_low = jnp.maximum(u - lower, options.barrier_margin)
    slack_high = jnp.maximum(upper - u, options.barrier_margin)
    barrier = -jnp.sum(jnp.log(slack_low) + jnp.log(slack_high))
    return problem.dt * (x @ Q @ x + u @ R @ u) + options.barrier_weight * barrier


def terminal_cost(problem: TrivialLQRProblem, x: Array) -> Array:
    Qf = jnp.asarray(problem.Qf, dtype=x.dtype)
    return x @ Qf @ x


def rollout_controls(
    problem: TrivialLQRProblem,
    controls: Array,
    options: ILQROptions,
) -> tuple[Array, Array]:
    x0 = jnp.asarray(problem.x0, dtype=controls.dtype)

    def step(xk: Array, uk: Array) -> tuple[Array, Array]:
        next_x = discrete_dynamics(problem, xk, uk)
        return next_x, xk

    x_last, x_prefix = jax.lax.scan(step, x0, controls)
    states = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    running_cost = jax.vmap(lambda xk, uk: stage_cost(problem, xk, uk, options))(
        states[:-1], controls
    ).sum()
    total_cost = running_cost + terminal_cost(problem, states[-1])
    return states, total_cost


def linearize_trajectory(
    problem: TrivialLQRProblem,
    states: Array,
    controls: Array,
    options: ILQROptions,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    dyn = lambda x, u: discrete_dynamics(problem, x, u)
    cost = lambda x, u: stage_cost(problem, x, u, options)

    f_x = jax.vmap(jax.jacfwd(dyn, argnums=0))(states[:-1], controls)
    f_u = jax.vmap(jax.jacfwd(dyn, argnums=1))(states[:-1], controls)
    l_x = jax.vmap(jax.grad(cost, argnums=0))(states[:-1], controls)
    l_u = jax.vmap(jax.grad(cost, argnums=1))(states[:-1], controls)
    l_xx = jax.vmap(jax.hessian(cost, argnums=0))(states[:-1], controls)
    l_uu = jax.vmap(jax.hessian(cost, argnums=1))(states[:-1], controls)
    l_ux = jax.vmap(lambda x, u: jax.jacfwd(jax.grad(cost, argnums=1), argnums=0)(x, u))(
        states[:-1], controls
    )
    return f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux


def backward_pass(
    problem: TrivialLQRProblem,
    states: Array,
    controls: Array,
    options: ILQROptions,
) -> tuple[Array, Array]:
    dtype = controls.dtype
    nx = problem.nx
    nu = problem.nu
    terminal_grad = jax.grad(lambda x: terminal_cost(problem, x))(states[-1])
    terminal_hess = jax.hessian(lambda x: terminal_cost(problem, x))(states[-1])
    f_x, f_u, l_x, l_u, l_xx, l_uu, l_ux = linearize_trajectory(
        problem, states, controls, options
    )

    def step(carry: tuple[Array, Array], data: tuple[Array, ...]) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        V_x, V_xx = carry
        A_k, B_k, lx_k, lu_k, lxx_k, luu_k, lux_k = data
        Q_x = lx_k + A_k.T @ V_x
        Q_u = lu_k + B_k.T @ V_x
        Q_xx = lxx_k + A_k.T @ V_xx @ A_k
        Q_uu = luu_k + B_k.T @ V_xx @ B_k
        Q_ux = lux_k + B_k.T @ V_xx @ A_k
        Q_uu = 0.5 * (Q_uu + Q_uu.T) + options.regularization * jnp.eye(nu, dtype=dtype)
        k_ff = -jnp.linalg.solve(Q_uu, Q_u)
        K_fb = -jnp.linalg.solve(Q_uu, Q_ux)
        V_x_new = (
            Q_x
            + K_fb.T @ Q_u
            + Q_ux.T @ k_ff
            + K_fb.T @ Q_uu @ k_ff
        )
        V_xx_new = (
            Q_xx
            + K_fb.T @ Q_ux
            + Q_ux.T @ K_fb
            + K_fb.T @ Q_uu @ K_fb
        )
        V_xx_new = 0.5 * (V_xx_new + V_xx_new.T)
        return (V_x_new, V_xx_new), (k_ff, K_fb)

    reversed_data = (
        jnp.flip(f_x, axis=0),
        jnp.flip(f_u, axis=0),
        jnp.flip(l_x, axis=0),
        jnp.flip(l_u, axis=0),
        jnp.flip(l_xx, axis=0),
        jnp.flip(l_uu, axis=0),
        jnp.flip(l_ux, axis=0),
    )
    (_, _), (k_rev, K_rev) = jax.lax.scan(step, (terminal_grad, terminal_hess), reversed_data)
    return jnp.flip(k_rev, axis=0), jnp.flip(K_rev, axis=0)


def rollout_with_gains(
    problem: TrivialLQRProblem,
    nominal_states: Array,
    nominal_controls: Array,
    k_feedforward: Array,
    K_feedback: Array,
    alpha: Array,
    options: ILQROptions,
) -> tuple[Array, Array, Array]:
    dtype = nominal_controls.dtype
    margin = jnp.asarray(options.barrier_margin, dtype=dtype)

    def step(xk: Array, data: tuple[Array, Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        x_nom, u_nom, k_ff, K_fb = data
        delta_x = xk - x_nom
        u_trial = u_nom + alpha * k_ff + K_fb @ delta_x
        u_trial = clamp_controls(u_trial, problem, margin)
        x_next = discrete_dynamics(problem, xk, u_trial)
        return x_next, (xk, u_trial)

    x0 = jnp.asarray(problem.x0, dtype=dtype)
    x_last, (x_prefix, us) = jax.lax.scan(
        step,
        x0,
        (nominal_states[:-1], nominal_controls, k_feedforward, K_feedback),
    )
    xs = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    cost = jax.vmap(lambda xk, uk: stage_cost(problem, xk, uk, options))(xs[:-1], us).sum()
    cost = cost + terminal_cost(problem, xs[-1])
    return xs, us, cost


def ilqr_iteration(
    problem: TrivialLQRProblem,
    controls: Array,
    options: ILQROptions,
) -> tuple[Array, Array, Array, Array, Array]:
    states, nominal_cost = rollout_controls(problem, controls, options)
    k_feedforward, K_feedback = backward_pass(problem, states, controls, options)
    alphas = jnp.asarray([1.0, 0.5, 0.25, 0.1, 0.05], dtype=controls.dtype)
    candidate_states, candidate_controls, candidate_costs = jax.vmap(
        lambda alpha: rollout_with_gains(
            problem,
            states,
            controls,
            k_feedforward,
            K_feedback,
            alpha,
            options,
        )
    )(alphas)
    best_index = jnp.argmin(candidate_costs)
    next_states = candidate_states[best_index]
    next_controls = candidate_controls[best_index]
    next_cost = candidate_costs[best_index]
    control_update_norm = jnp.linalg.norm(next_controls - controls)
    return next_states, next_controls, nominal_cost, next_cost, control_update_norm


def solve_trivial_lqr_with_ilqr(
    problem: TrivialLQRProblem | None = None,
    options: ILQROptions | None = None,
) -> ILQRResult:
    problem = problem or build_trivial_lqr_problem()
    options = options or ILQROptions()
    dtype = default_dtype()
    controls0 = jnp.zeros((problem.horizon, problem.nu), dtype=dtype)

    def solve_loop(initial_controls: Array) -> tuple[Array, Array, Array, Array, Array, Array]:
        def body(_i: int, carry: tuple[Array, Array, Array, Array, Array, Array]) -> tuple[Array, Array, Array, Array, Array, Array]:
            controls, states, cost, update_norm, converged_flag, iteration_count = carry
            next_states, next_controls, nominal_cost, next_cost, next_update_norm = ilqr_iteration(
                problem, controls, options
            )
            improvement = nominal_cost - next_cost
            converged_now = jnp.logical_or(
                improvement <= options.cost_tolerance,
                next_update_norm <= options.control_tolerance,
            )
            use_old = converged_flag
            controls_out = jnp.where(use_old, controls, next_controls)
            states_out = jnp.where(use_old, states, next_states)
            cost_out = jnp.where(use_old, cost, next_cost)
            update_out = jnp.where(use_old, update_norm, next_update_norm)
            converged_out = jnp.logical_or(converged_flag, converged_now)
            iteration_out = jnp.where(use_old, iteration_count, iteration_count + 1)
            return controls_out, states_out, cost_out, update_out, converged_out, iteration_out

        initial_states, initial_cost = rollout_controls(problem, initial_controls, options)
        init = (
            initial_controls,
            initial_states,
            initial_cost,
            jnp.asarray(jnp.inf, dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return jax.lax.fori_loop(0, options.max_iterations, body, init)

    compiled_solve_loop = jax.jit(solve_loop)
    controls_star, states_star, cost_star, update_norm, converged, iterations = compiled_solve_loop(
        controls0
    )
    reference_step = diffrax_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    Ad, Bd = zero_order_hold_discretization(problem)
    exact_step = Ad @ jnp.asarray(problem.x0, dtype=dtype) + Bd @ jnp.zeros(problem.nu, dtype=dtype)
    violation = jnp.maximum(
        controls_star - jnp.asarray(problem.u_max, dtype=dtype), 0.0
    ) + jnp.maximum(jnp.asarray(problem.u_min, dtype=dtype) - controls_star, 0.0)
    return ILQRResult(
        converged=bool(converged),
        iterations=int(iterations),
        objective_value=float(cost_star),
        control_update_norm=float(update_norm),
        max_control_violation=float(jnp.max(jnp.abs(violation))),
        diffrax_vs_exact_step_error=float(jnp.linalg.norm(reference_step - exact_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )
