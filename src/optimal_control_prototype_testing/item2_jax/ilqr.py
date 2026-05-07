from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from jax.scipy.linalg import expm

from optimal_control_prototype_testing.item1_jax.problem import (
    TrivialLQRProblem,
    build_trivial_lqr_problem,
)
from optimal_control_prototype_testing.nonlinear_pendulum import (
    NonlinearPendulumProblem,
    build_nonlinear_pendulum_problem,
)


Array = jax.Array


@dataclass(frozen=True)
class ILQROptions:
    max_iterations: int = 40
    regularization: float = 1e-5
    barrier_weight: float = 1e-3
    barrier_margin: float = 1e-4
    state_penalty_weight: float = 1e4
    cost_tolerance: float = 1e-6
    control_tolerance: float = 1e-4


@dataclass(frozen=True)
class ILQRResult:
    problem_name: str
    constraint_mode: str
    converged: bool
    iterations: int
    objective_value: float
    runtime_seconds: float
    control_update_norm: float
    max_control_violation: float
    max_state_violation: float
    diffrax_vs_reference_step_error: float
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


def linear_diffrax_step(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
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


def nonlinear_diffrax_step(problem: NonlinearPendulumProblem, x: Array, u: Array) -> Array:
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


def nonlinear_rk4_step(problem: NonlinearPendulumProblem, x: Array, u: Array) -> Array:
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


def clamp_controls(u: Array, u_min: Array, u_max: Array, margin: Array) -> Array:
    return jnp.clip(u, u_min + margin, u_max - margin)


def trivial_stage_cost(problem: TrivialLQRProblem, x: Array, u: Array, options: ILQROptions) -> Array:
    dtype = x.dtype
    Q = jnp.asarray(problem.Q, dtype=dtype)
    R = jnp.asarray(problem.R, dtype=dtype)
    lower = jnp.asarray(problem.u_min, dtype=dtype)
    upper = jnp.asarray(problem.u_max, dtype=dtype)
    slack_low = jnp.maximum(u - lower, options.barrier_margin)
    slack_high = jnp.maximum(upper - u, options.barrier_margin)
    barrier = -jnp.sum(jnp.log(slack_low) + jnp.log(slack_high))
    return problem.dt * (x @ Q @ x + u @ R @ u) + options.barrier_weight * barrier


def trivial_terminal_cost(problem: TrivialLQRProblem, x: Array) -> Array:
    Qf = jnp.asarray(problem.Qf, dtype=x.dtype)
    return x @ Qf @ x


def pendulum_state_error(problem: NonlinearPendulumProblem, x: Array) -> Array:
    goal = jnp.asarray(problem.x_goal, dtype=x.dtype)
    error = x - goal
    wrapped = jnp.arctan2(jnp.sin(error[0]), jnp.cos(error[0]))
    return error.at[0].set(wrapped)


def pendulum_state_violation(problem: NonlinearPendulumProblem, x: Array) -> Array:
    lower = jnp.asarray(problem.x_min, dtype=x.dtype)
    upper = jnp.asarray(problem.x_max, dtype=x.dtype)
    return jnp.maximum(lower - x, 0.0) + jnp.maximum(x - upper, 0.0)


def pendulum_control_violation(problem: NonlinearPendulumProblem, u: Array) -> Array:
    lower = jnp.asarray(problem.u_min, dtype=u.dtype)
    upper = jnp.asarray(problem.u_max, dtype=u.dtype)
    return jnp.maximum(lower - u, 0.0) + jnp.maximum(u - upper, 0.0)


def pendulum_stage_cost(
    problem: NonlinearPendulumProblem,
    x: Array,
    u: Array,
    options: ILQROptions,
    *,
    soft_constraints: bool,
) -> Array:
    dtype = x.dtype
    error = pendulum_state_error(problem, x)
    Q = jnp.asarray(problem.Q, dtype=dtype)
    R = jnp.asarray(problem.R, dtype=dtype)
    lower = jnp.asarray(problem.u_min, dtype=dtype)
    upper = jnp.asarray(problem.u_max, dtype=dtype)
    slack_low = jnp.maximum(u - lower, options.barrier_margin)
    slack_high = jnp.maximum(upper - u, options.barrier_margin)
    barrier = -jnp.sum(jnp.log(slack_low) + jnp.log(slack_high))
    base_cost = problem.dt * (error @ Q @ error + u @ R @ u) + options.barrier_weight * barrier
    state_violation = pendulum_state_violation(problem, x)
    control_violation = pendulum_control_violation(problem, u)
    if soft_constraints:
        return (
            base_cost
            + problem.state_soft_weight * (state_violation @ state_violation)
            + problem.control_soft_weight * (control_violation @ control_violation)
        )
    return base_cost + options.state_penalty_weight * (state_violation @ state_violation)


def pendulum_terminal_cost(
    problem: NonlinearPendulumProblem,
    x: Array,
    options: ILQROptions,
    *,
    soft_constraints: bool,
) -> Array:
    dtype = x.dtype
    error = pendulum_state_error(problem, x)
    Qf = jnp.asarray(problem.Qf, dtype=dtype)
    cost = error @ Qf @ error
    state_violation = pendulum_state_violation(problem, x)
    if soft_constraints:
        return cost + problem.state_soft_weight * (state_violation @ state_violation)
    return cost + options.state_penalty_weight * (state_violation @ state_violation)


def rollout_trivial_controls(
    problem: TrivialLQRProblem,
    controls: Array,
    options: ILQROptions,
) -> tuple[Array, Array]:
    x0 = jnp.asarray(problem.x0, dtype=controls.dtype)
    Ad, Bd = zero_order_hold_discretization(problem)

    def step(xk: Array, uk: Array) -> tuple[Array, Array]:
        next_x = Ad @ xk + Bd @ uk
        return next_x, xk

    x_last, x_prefix = jax.lax.scan(step, x0, controls)
    states = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    running_cost = jax.vmap(lambda xk, uk: trivial_stage_cost(problem, xk, uk, options))(states[:-1], controls).sum()
    return states, running_cost + trivial_terminal_cost(problem, states[-1])


def rollout_pendulum_controls(
    problem: NonlinearPendulumProblem,
    controls: Array,
    options: ILQROptions,
    *,
    soft_constraints: bool,
) -> tuple[Array, Array]:
    x0 = jnp.asarray(problem.x0, dtype=controls.dtype)

    def step(xk: Array, uk: Array) -> tuple[Array, Array]:
        next_x = nonlinear_diffrax_step(problem, xk, uk)
        return next_x, xk

    x_last, x_prefix = jax.lax.scan(step, x0, controls)
    states = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    running_cost = jax.vmap(
        lambda xk, uk: pendulum_stage_cost(problem, xk, uk, options, soft_constraints=soft_constraints)
    )(states[:-1], controls).sum()
    return states, running_cost + pendulum_terminal_cost(problem, states[-1], options, soft_constraints=soft_constraints)


def backward_pass(
    states: Array,
    controls: Array,
    dynamics_fn,
    stage_cost_fn,
    terminal_cost_fn,
    regularization: float,
    *,
    use_reverse_dynamics_jacobians: bool = False,
) -> tuple[Array, Array]:
    dtype = controls.dtype
    nu = controls.shape[1]
    terminal_grad = jax.grad(terminal_cost_fn)(states[-1])
    terminal_hess = jax.hessian(terminal_cost_fn)(states[-1])

    if use_reverse_dynamics_jacobians:
        dyn_x_jac = jax.jacrev(dynamics_fn, argnums=0)
        dyn_u_jac = jax.jacrev(dynamics_fn, argnums=1)
    else:
        dyn_x_jac = jax.jacfwd(dynamics_fn, argnums=0)
        dyn_u_jac = jax.jacfwd(dynamics_fn, argnums=1)
    f_x = jax.vmap(dyn_x_jac)(states[:-1], controls)
    f_u = jax.vmap(dyn_u_jac)(states[:-1], controls)
    l_x = jax.vmap(jax.grad(stage_cost_fn, argnums=0))(states[:-1], controls)
    l_u = jax.vmap(jax.grad(stage_cost_fn, argnums=1))(states[:-1], controls)
    l_xx = jax.vmap(jax.hessian(stage_cost_fn, argnums=0))(states[:-1], controls)
    l_uu = jax.vmap(jax.hessian(stage_cost_fn, argnums=1))(states[:-1], controls)
    l_ux = jax.vmap(lambda x, u: jax.jacfwd(jax.grad(stage_cost_fn, argnums=1), argnums=0)(x, u))(
        states[:-1], controls
    )

    def step(carry: tuple[Array, Array], data: tuple[Array, ...]) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
        V_x, V_xx = carry
        A_k, B_k, lx_k, lu_k, lxx_k, luu_k, lux_k = data
        Q_x = lx_k + A_k.T @ V_x
        Q_u = lu_k + B_k.T @ V_x
        Q_xx = lxx_k + A_k.T @ V_xx @ A_k
        Q_uu = luu_k + B_k.T @ V_xx @ B_k
        Q_ux = lux_k + B_k.T @ V_xx @ A_k
        Q_uu = 0.5 * (Q_uu + Q_uu.T) + regularization * jnp.eye(nu, dtype=dtype)
        k_ff = -jnp.linalg.solve(Q_uu, Q_u)
        K_fb = -jnp.linalg.solve(Q_uu, Q_ux)
        V_x_new = Q_x + K_fb.T @ Q_u + Q_ux.T @ k_ff + K_fb.T @ Q_uu @ k_ff
        V_xx_new = Q_xx + K_fb.T @ Q_ux + Q_ux.T @ K_fb + K_fb.T @ Q_uu @ K_fb
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
    initial_state: Array,
    nominal_states: Array,
    nominal_controls: Array,
    k_feedforward: Array,
    K_feedback: Array,
    alpha: Array,
    dynamics_fn,
    stage_cost_fn,
    terminal_cost_fn,
    u_min: Array,
    u_max: Array,
    margin: Array,
) -> tuple[Array, Array, Array]:
    def step(xk: Array, data: tuple[Array, Array, Array, Array]) -> tuple[Array, tuple[Array, Array]]:
        x_nom, u_nom, k_ff, K_fb = data
        delta_x = xk - x_nom
        u_trial = u_nom + alpha * k_ff + K_fb @ delta_x
        u_trial = clamp_controls(u_trial, u_min, u_max, margin)
        x_next = dynamics_fn(xk, u_trial)
        return x_next, (xk, u_trial)

    x_last, (x_prefix, us) = jax.lax.scan(
        step,
        initial_state,
        (nominal_states[:-1], nominal_controls, k_feedforward, K_feedback),
    )
    xs = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    cost = jax.vmap(stage_cost_fn)(xs[:-1], us).sum() + terminal_cost_fn(xs[-1])
    return xs, us, cost


def ilqr_solve(
    initial_state: Array,
    initial_controls: Array,
    dynamics_fn,
    rollout_fn,
    stage_cost_fn,
    terminal_cost_fn,
    u_min: Array,
    u_max: Array,
    options: ILQROptions,
    *,
    use_reverse_dynamics_jacobians: bool = False,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    margin = jnp.asarray(options.barrier_margin, dtype=initial_controls.dtype)

    def solve_loop(controls0: Array):
        def body(_i: int, carry):
            controls, states, cost, update_norm, converged_flag, iteration_count = carry
            k_feedforward, K_feedback = backward_pass(
                states,
                controls,
                dynamics_fn,
                stage_cost_fn,
                terminal_cost_fn,
                options.regularization,
                use_reverse_dynamics_jacobians=use_reverse_dynamics_jacobians,
            )
            alphas = jnp.asarray([1.0, 0.5, 0.25, 0.1, 0.05], dtype=controls.dtype)
            candidate_states, candidate_controls, candidate_costs = jax.vmap(
                lambda alpha: rollout_with_gains(
                    initial_state,
                    states,
                    controls,
                    k_feedforward,
                    K_feedback,
                    alpha,
                    dynamics_fn,
                    stage_cost_fn,
                    terminal_cost_fn,
                    u_min,
                    u_max,
                    margin,
                )
            )(alphas)
            best_index = jnp.argmin(candidate_costs)
            next_states = candidate_states[best_index]
            next_controls = candidate_controls[best_index]
            next_cost = candidate_costs[best_index]
            next_update_norm = jnp.linalg.norm(next_controls - controls)
            improvement = cost - next_cost
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

        initial_states, initial_cost = rollout_fn(controls0)
        init = (
            controls0,
            initial_states,
            initial_cost,
            jnp.asarray(jnp.inf, dtype=controls0.dtype),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return jax.lax.fori_loop(0, options.max_iterations, body, init)

    return jax.jit(solve_loop)(initial_controls)


def trivial_initial_controls(problem: TrivialLQRProblem) -> Array:
    return jnp.zeros((problem.horizon, problem.nu), dtype=default_dtype())


def pendulum_initial_controls(problem: NonlinearPendulumProblem) -> Array:
    dtype = default_dtype()
    controls = np.zeros((problem.horizon, problem.nu), dtype=float)
    quarter = problem.horizon // 4
    controls[: quarter + 2, 0] = problem.u_min[0]
    controls[quarter + 2 : 2 * quarter, 0] = problem.u_max[0]
    controls[2 * quarter : 3 * quarter, 0] = problem.u_min[0]
    controls[3 * quarter :, 0] = problem.u_max[0]
    return jnp.asarray(controls, dtype=dtype)


def solve_trivial_lqr_with_ilqr(
    problem: TrivialLQRProblem | None = None,
    options: ILQROptions | None = None,
) -> ILQRResult:
    problem = problem or build_trivial_lqr_problem()
    options = options or ILQROptions()
    dtype = default_dtype()
    initial_controls = trivial_initial_controls(problem)
    initial_state = jnp.asarray(problem.x0, dtype=dtype)
    u_min = jnp.asarray(problem.u_min, dtype=dtype)
    u_max = jnp.asarray(problem.u_max, dtype=dtype)
    Ad, Bd = zero_order_hold_discretization(problem)

    dynamics_fn = lambda x, u: Ad @ x + Bd @ u
    stage_cost_fn = lambda x, u: trivial_stage_cost(problem, x, u, options)
    terminal_cost_fn = lambda x: trivial_terminal_cost(problem, x)
    rollout_fn = lambda controls: rollout_trivial_controls(problem, controls, options)

    start_time = time.perf_counter()
    controls_star, states_star, cost_star, update_norm, converged, iterations = ilqr_solve(
        initial_state,
        initial_controls,
        dynamics_fn,
        rollout_fn,
        stage_cost_fn,
        terminal_cost_fn,
        u_min,
        u_max,
        options,
    )
    jax.block_until_ready(cost_star)
    runtime_seconds = time.perf_counter() - start_time
    reference_step = linear_diffrax_step(
        problem,
        initial_state,
        jnp.zeros(problem.nu, dtype=dtype),
    )
    exact_step = Ad @ initial_state + Bd @ jnp.zeros(problem.nu, dtype=dtype)
    violation = jnp.maximum(controls_star - u_max, 0.0) + jnp.maximum(u_min - controls_star, 0.0)
    return ILQRResult(
        problem_name="trivial_lqr",
        constraint_mode="hard",
        converged=bool(converged),
        iterations=int(iterations),
        objective_value=float(cost_star),
        runtime_seconds=runtime_seconds,
        control_update_norm=float(update_norm),
        max_control_violation=float(jnp.max(jnp.abs(violation))),
        max_state_violation=0.0,
        diffrax_vs_reference_step_error=float(jnp.linalg.norm(reference_step - exact_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )


def solve_nonlinear_pendulum_with_ilqr(
    problem: NonlinearPendulumProblem | None = None,
    options: ILQROptions | None = None,
    *,
    soft_constraints: bool,
) -> ILQRResult:
    problem = problem or build_nonlinear_pendulum_problem()
    options = options or ILQROptions()
    dtype = default_dtype()
    initial_controls = pendulum_initial_controls(problem)
    initial_state = jnp.asarray(problem.x0, dtype=dtype)
    u_min = jnp.asarray(problem.u_min, dtype=dtype)
    u_max = jnp.asarray(problem.u_max, dtype=dtype)

    dynamics_fn = lambda x, u: nonlinear_diffrax_step(problem, x, u)
    stage_cost_fn = lambda x, u: pendulum_stage_cost(
        problem, x, u, options, soft_constraints=soft_constraints
    )
    terminal_cost_fn = lambda x: pendulum_terminal_cost(
        problem, x, options, soft_constraints=soft_constraints
    )
    rollout_fn = lambda controls: rollout_pendulum_controls(
        problem, controls, options, soft_constraints=soft_constraints
    )

    start_time = time.perf_counter()
    controls_star, states_star, cost_star, update_norm, converged, iterations = ilqr_solve(
        initial_state,
        initial_controls,
        dynamics_fn,
        rollout_fn,
        stage_cost_fn,
        terminal_cost_fn,
        u_min,
        u_max,
        options,
        use_reverse_dynamics_jacobians=True,
    )
    jax.block_until_ready(cost_star)
    runtime_seconds = time.perf_counter() - start_time
    reference_step = nonlinear_diffrax_step(
        problem,
        initial_state,
        jnp.zeros(problem.nu, dtype=dtype),
    )
    rk4_step = nonlinear_rk4_step(problem, initial_state, jnp.zeros(problem.nu, dtype=dtype))
    control_violation = jnp.maximum(controls_star - u_max, 0.0) + jnp.maximum(u_min - controls_star, 0.0)
    lower_x = jnp.asarray(problem.x_min, dtype=dtype)
    upper_x = jnp.asarray(problem.x_max, dtype=dtype)
    state_violation = jax.vmap(
        lambda x: jnp.maximum(lower_x - x, 0.0) + jnp.maximum(x - upper_x, 0.0)
    )(states_star)
    return ILQRResult(
        problem_name="nonlinear_pendulum",
        constraint_mode="soft" if soft_constraints else "hard",
        converged=bool(converged),
        iterations=int(iterations),
        objective_value=float(cost_star),
        runtime_seconds=runtime_seconds,
        control_update_norm=float(update_norm),
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=float(jnp.max(jnp.abs(state_violation))),
        diffrax_vs_reference_step_error=float(jnp.linalg.norm(reference_step - rk4_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )
