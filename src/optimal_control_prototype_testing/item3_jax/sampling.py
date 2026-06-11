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
    pure_tracking_cost as _nl_pure_cost,
)


Array = jax.Array


@dataclass(frozen=True)
class SamplingOptions:
    num_rollouts: int = 512
    iterations: int = 20
    noise_std: float = 0.25
    temperature: float = 1.0
    elite_fraction: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class SamplingResult:
    method: str
    problem_name: str
    constraint_mode: str
    iterations: int
    objective_value: float
    runtime_seconds: float
    max_control_violation: float
    max_state_violation: float
    final_position_error: float
    final_velocity_error: float
    diffrax_vs_exact_step_error: float
    state_trajectory: np.ndarray
    control_trajectory: np.ndarray


@dataclass(frozen=True)
class Item3Environment:
    backend: str
    devices: tuple[str, ...]
    x64_enabled: bool
    dtype: str


def detect_jax_environment() -> Item3Environment:
    return Item3Environment(
        backend=jax.default_backend(),
        devices=tuple(str(device) for device in jax.devices()),
        x64_enabled=bool(jax.config.jax_enable_x64),
        dtype="float64" if jax.config.jax_enable_x64 else "float32",
    )


def default_dtype() -> jnp.dtype:
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


def clip_controls(u: Array, problem: TrivialLQRProblem | NonlinearPendulumProblem) -> Array:
    lower = jnp.asarray(problem.u_min, dtype=u.dtype)
    upper = jnp.asarray(problem.u_max, dtype=u.dtype)
    return jnp.clip(u, lower, upper)


def make_initial_controls(problem: TrivialLQRProblem | NonlinearPendulumProblem) -> Array:
    return jnp.zeros((problem.horizon, problem.nu), dtype=default_dtype())


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


def linear_discrete_dynamics(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
    Ad, Bd = zero_order_hold_discretization(problem)
    return Ad @ x + Bd @ u


def linear_stage_cost(problem: TrivialLQRProblem, x: Array, u: Array) -> Array:
    Q = jnp.asarray(problem.Q, dtype=x.dtype)
    R = jnp.asarray(problem.R, dtype=x.dtype)
    return problem.dt * (x @ Q @ x + u @ R @ u)


def linear_terminal_cost(problem: TrivialLQRProblem, x: Array) -> Array:
    Qf = jnp.asarray(problem.Qf, dtype=x.dtype)
    return x @ Qf @ x


def rollout_trivial_lqr(problem: TrivialLQRProblem, controls: Array) -> tuple[Array, Array]:
    x0 = jnp.asarray(problem.x0, dtype=controls.dtype)

    def step(xk: Array, uk: Array) -> tuple[Array, Array]:
        next_x = linear_discrete_dynamics(problem, xk, uk)
        return next_x, xk

    x_last, x_prefix = jax.lax.scan(step, x0, controls)
    states = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    running_cost = jax.vmap(lambda xk, uk: linear_stage_cost(problem, xk, uk))(
        states[:-1], controls
    ).sum()
    total_cost = running_cost + linear_terminal_cost(problem, states[-1])
    return states, total_cost


def sample_trivial_rollout_costs(
    problem: TrivialLQRProblem,
    nominal_controls: Array,
    noise: Array,
) -> tuple[Array, Array]:
    trial_controls = clip_controls(nominal_controls[None, :, :] + noise, problem)
    states, costs = jax.vmap(lambda controls: rollout_trivial_lqr(problem, controls))(trial_controls)
    return states, costs


def nonlinear_pendulum_dynamics(problem: NonlinearPendulumProblem, x: Array, u: Array) -> Array:
    dtype = x.dtype
    inertia = jnp.asarray(problem.inertia, dtype=dtype)
    gravity = jnp.asarray(problem.gravity, dtype=dtype)
    length = jnp.asarray(problem.length, dtype=dtype)
    damping = jnp.asarray(problem.damping, dtype=dtype)
    theta = x[0]
    omega = x[1]
    torque = u[0]
    omega_dot = -(gravity / length) * jnp.sin(theta) - (damping / inertia) * omega + torque / inertia
    return jnp.array([omega, omega_dot], dtype=dtype)


def nonlinear_pendulum_step(problem: NonlinearPendulumProblem, x: Array, u: Array) -> Array:
    term = ODETerm(lambda _t, y, args: nonlinear_pendulum_dynamics(problem, y, args))
    solution = diffeqsolve(
        term,
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


def nonlinear_state_error(problem: NonlinearPendulumProblem, x: Array) -> Array:
    goal = jnp.asarray(problem.x_goal, dtype=x.dtype)
    error = x - goal
    return error.at[0].set(jnp.arctan2(jnp.sin(error[0]), jnp.cos(error[0])))


def nonlinear_state_violation(problem: NonlinearPendulumProblem, x: Array) -> Array:
    lower = jnp.asarray(problem.x_min, dtype=x.dtype)
    upper = jnp.asarray(problem.x_max, dtype=x.dtype)
    return jnp.maximum(lower - x, 0.0) + jnp.maximum(x - upper, 0.0)


def nonlinear_control_violation(problem: NonlinearPendulumProblem, u: Array) -> Array:
    lower = jnp.asarray(problem.u_min, dtype=u.dtype)
    upper = jnp.asarray(problem.u_max, dtype=u.dtype)
    return jnp.maximum(lower - u, 0.0) + jnp.maximum(u - upper, 0.0)


def nonlinear_running_cost(
    problem: NonlinearPendulumProblem,
    x: Array,
    u: Array,
    soft_constraints: bool,
) -> Array:
    dtype = x.dtype
    error = nonlinear_state_error(problem, x)
    Q = jnp.asarray(problem.Q, dtype=dtype)
    R = jnp.asarray(problem.R, dtype=dtype)
    cost = problem.dt * (error @ Q @ error + u @ R @ u)
    state_violation = nonlinear_state_violation(problem, x)
    control_violation = nonlinear_control_violation(problem, u)
    if soft_constraints:
        cost = cost + problem.state_soft_weight * (state_violation @ state_violation)
        cost = cost + problem.control_soft_weight * (control_violation @ control_violation)
    else:
        hard_weight = jnp.asarray(1e4, dtype=dtype)
        cost = cost + hard_weight * (state_violation @ state_violation)
    return cost


def nonlinear_terminal_cost(
    problem: NonlinearPendulumProblem,
    x: Array,
    soft_constraints: bool,
) -> Array:
    dtype = x.dtype
    error = nonlinear_state_error(problem, x)
    Qf = jnp.asarray(problem.Qf, dtype=dtype)
    cost = error @ Qf @ error
    state_violation = nonlinear_state_violation(problem, x)
    if soft_constraints:
        cost = cost + problem.state_soft_weight * (state_violation @ state_violation)
    else:
        hard_weight = jnp.asarray(1e4, dtype=dtype)
        cost = cost + hard_weight * (state_violation @ state_violation)
    return cost


def rollout_nonlinear_pendulum(
    problem: NonlinearPendulumProblem,
    controls: Array,
    soft_constraints: bool,
) -> tuple[Array, Array]:
    x0 = jnp.asarray(problem.x0, dtype=controls.dtype)

    def step(xk: Array, uk: Array) -> tuple[Array, Array]:
        next_x = nonlinear_pendulum_step(problem, xk, uk)
        return next_x, xk

    x_last, x_prefix = jax.lax.scan(step, x0, controls)
    states = jnp.concatenate([x_prefix, x_last[None, :]], axis=0)
    running_cost = jax.vmap(
        lambda xk, uk: nonlinear_running_cost(problem, xk, uk, soft_constraints)
    )(states[:-1], controls).sum()
    total_cost = running_cost + nonlinear_terminal_cost(problem, states[-1], soft_constraints)
    return states, total_cost


def sample_nonlinear_rollout_costs(
    problem: NonlinearPendulumProblem,
    nominal_controls: Array,
    noise: Array,
    soft_constraints: bool,
) -> tuple[Array, Array]:
    trial_controls = clip_controls(nominal_controls[None, :, :] + noise, problem)
    states, costs = jax.vmap(
        lambda controls: rollout_nonlinear_pendulum(problem, controls, soft_constraints)
    )(trial_controls)
    return states, costs


def solve_trivial_lqr_with_mppi(
    problem: TrivialLQRProblem | None = None,
    options: SamplingOptions | None = None,
) -> SamplingResult:
    problem = problem or build_trivial_lqr_problem()
    options = options or SamplingOptions()
    dtype = default_dtype()
    key = jax.random.PRNGKey(options.seed)
    nominal_controls = make_initial_controls(problem)

    def body(carry: tuple[Array, Array], _i: Array) -> tuple[tuple[Array, Array], None]:
        controls, rng = carry
        rng, subkey = jax.random.split(rng)
        noise = options.noise_std * jax.random.normal(
            subkey,
            (options.num_rollouts, problem.horizon, problem.nu),
            dtype=dtype,
        )
        _, costs = sample_trivial_rollout_costs(problem, controls, noise)
        shifted = costs - jnp.min(costs)
        weights = jax.nn.softmax(-shifted / options.temperature)
        delta = jnp.tensordot(weights, noise, axes=(0, 0))
        next_controls = clip_controls(controls + delta, problem)
        return (next_controls, rng), None

    compiled = jax.jit(
        lambda controls, rng: jax.lax.scan(
            body,
            (controls, rng),
            xs=jnp.arange(options.iterations),
        )[0]
    )
    start_time = time.perf_counter()
    controls_star, _ = compiled(nominal_controls, key)
    states_star, cost_star = rollout_trivial_lqr(problem, controls_star)
    jax.block_until_ready(cost_star)
    runtime_seconds = time.perf_counter() - start_time
    reference_step = linear_diffrax_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    Ad, Bd = zero_order_hold_discretization(problem)
    exact_step = Ad @ jnp.asarray(problem.x0, dtype=dtype) + Bd @ jnp.zeros(problem.nu, dtype=dtype)
    control_violation = jnp.maximum(
        controls_star - jnp.asarray(problem.u_max, dtype=dtype), 0.0
    ) + jnp.maximum(jnp.asarray(problem.u_min, dtype=dtype) - controls_star, 0.0)
    final_state_mppi_lqr = np.asarray(states_star[-1])
    return SamplingResult(
        method="mppi",
        problem_name="trivial_lqr",
        constraint_mode="hard",
        iterations=options.iterations,
        objective_value=float(cost_star),
        runtime_seconds=runtime_seconds,
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=0.0,
        final_position_error=float(final_state_mppi_lqr[0]),
        final_velocity_error=float(final_state_mppi_lqr[1]),
        diffrax_vs_exact_step_error=float(jnp.linalg.norm(reference_step - exact_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )


def solve_trivial_lqr_with_cem(
    problem: TrivialLQRProblem | None = None,
    options: SamplingOptions | None = None,
) -> SamplingResult:
    problem = problem or build_trivial_lqr_problem()
    options = options or SamplingOptions()
    dtype = default_dtype()
    key = jax.random.PRNGKey(options.seed)
    mean_controls = make_initial_controls(problem)
    std_controls = jnp.full((problem.horizon, problem.nu), options.noise_std, dtype=dtype)
    elite_count = max(1, int(options.num_rollouts * options.elite_fraction))

    def body(carry: tuple[Array, Array, Array], _i: Array) -> tuple[tuple[Array, Array, Array], None]:
        mean, std, rng = carry
        rng, subkey = jax.random.split(rng)
        noise = std[None, :, :] * jax.random.normal(
            subkey,
            (options.num_rollouts, problem.horizon, problem.nu),
            dtype=dtype,
        )
        trial_controls = clip_controls(mean[None, :, :] + noise, problem)
        _, costs = jax.vmap(lambda controls: rollout_trivial_lqr(problem, controls))(trial_controls)
        elite_indices = jnp.argsort(costs)[:elite_count]
        elites = trial_controls[elite_indices]
        next_mean = jnp.mean(elites, axis=0)
        next_std = jnp.maximum(jnp.std(elites, axis=0), 1e-3)
        return (next_mean, next_std, rng), None

    compiled = jax.jit(
        lambda mean, std, rng: jax.lax.scan(
            body,
            (mean, std, rng),
            xs=jnp.arange(options.iterations),
        )[0]
    )
    start_time = time.perf_counter()
    controls_star, _, _ = compiled(mean_controls, std_controls, key)
    controls_star = clip_controls(controls_star, problem)
    states_star, cost_star = rollout_trivial_lqr(problem, controls_star)
    jax.block_until_ready(cost_star)
    runtime_seconds = time.perf_counter() - start_time
    reference_step = linear_diffrax_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    Ad, Bd = zero_order_hold_discretization(problem)
    exact_step = Ad @ jnp.asarray(problem.x0, dtype=dtype) + Bd @ jnp.zeros(problem.nu, dtype=dtype)
    control_violation = jnp.maximum(
        controls_star - jnp.asarray(problem.u_max, dtype=dtype), 0.0
    ) + jnp.maximum(jnp.asarray(problem.u_min, dtype=dtype) - controls_star, 0.0)
    final_state_cem_lqr = np.asarray(states_star[-1])
    return SamplingResult(
        method="cem",
        problem_name="trivial_lqr",
        constraint_mode="hard",
        iterations=options.iterations,
        objective_value=float(cost_star),
        runtime_seconds=runtime_seconds,
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=0.0,
        final_position_error=float(final_state_cem_lqr[0]),
        final_velocity_error=float(final_state_cem_lqr[1]),
        diffrax_vs_exact_step_error=float(jnp.linalg.norm(reference_step - exact_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )


def solve_nonlinear_pendulum_with_mppi(
    problem: NonlinearPendulumProblem | None = None,
    options: SamplingOptions | None = None,
    *,
    soft_constraints: bool = False,
) -> SamplingResult:
    problem = problem or build_nonlinear_pendulum_problem()
    options = options or SamplingOptions(num_rollouts=512, iterations=20, noise_std=0.6)
    dtype = default_dtype()
    key = jax.random.PRNGKey(options.seed)
    nominal_controls = make_initial_controls(problem)

    def body(carry: tuple[Array, Array], _i: Array) -> tuple[tuple[Array, Array], None]:
        controls, rng = carry
        rng, subkey = jax.random.split(rng)
        noise = options.noise_std * jax.random.normal(
            subkey,
            (options.num_rollouts, problem.horizon, problem.nu),
            dtype=dtype,
        )
        _, costs = sample_nonlinear_rollout_costs(problem, controls, noise, soft_constraints)
        shifted = costs - jnp.min(costs)
        weights = jax.nn.softmax(-shifted / options.temperature)
        delta = jnp.tensordot(weights, noise, axes=(0, 0))
        next_controls = clip_controls(controls + delta, problem)
        return (next_controls, rng), None

    compiled = jax.jit(
        lambda controls, rng: jax.lax.scan(
            body,
            (controls, rng),
            xs=jnp.arange(options.iterations),
        )[0]
    )
    start_time = time.perf_counter()
    controls_star, _ = compiled(nominal_controls, key)
    states_star, cost_star = rollout_nonlinear_pendulum(problem, controls_star, soft_constraints)
    jax.block_until_ready(cost_star)
    runtime_seconds = time.perf_counter() - start_time
    control_violation = jax.vmap(lambda uk: nonlinear_control_violation(problem, uk))(controls_star)
    state_violation = jax.vmap(lambda xk: nonlinear_state_violation(problem, xk))(states_star)
    reference_step = nonlinear_pendulum_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    final_error_mppi = problem.state_error(np.asarray(states_star[-1]))
    return SamplingResult(
        method="mppi",
        problem_name="nonlinear_pendulum",
        constraint_mode="soft" if soft_constraints else "hard",
        iterations=options.iterations,
        objective_value=_nl_pure_cost(problem, np.asarray(states_star), np.asarray(controls_star)),
        runtime_seconds=runtime_seconds,
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=float(jnp.max(jnp.abs(state_violation))),
        final_position_error=float(final_error_mppi[0]),
        final_velocity_error=float(final_error_mppi[1]),
        diffrax_vs_exact_step_error=0.0 * float(jnp.linalg.norm(reference_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )


def solve_nonlinear_pendulum_with_cem(
    problem: NonlinearPendulumProblem | None = None,
    options: SamplingOptions | None = None,
    *,
    soft_constraints: bool = False,
) -> SamplingResult:
    problem = problem or build_nonlinear_pendulum_problem()
    options = options or SamplingOptions(num_rollouts=512, iterations=20, noise_std=0.8, elite_fraction=0.1)
    dtype = default_dtype()
    key = jax.random.PRNGKey(options.seed)
    mean_controls = make_initial_controls(problem)
    std_controls = jnp.full((problem.horizon, problem.nu), options.noise_std, dtype=dtype)
    elite_count = max(1, int(options.num_rollouts * options.elite_fraction))

    def body(carry: tuple[Array, Array, Array], _i: Array) -> tuple[tuple[Array, Array, Array], None]:
        mean, std, rng = carry
        rng, subkey = jax.random.split(rng)
        noise = std[None, :, :] * jax.random.normal(
            subkey,
            (options.num_rollouts, problem.horizon, problem.nu),
            dtype=dtype,
        )
        trial_controls = clip_controls(mean[None, :, :] + noise, problem)
        _, costs = jax.vmap(
            lambda controls: rollout_nonlinear_pendulum(problem, controls, soft_constraints)
        )(trial_controls)
        elite_indices = jnp.argsort(costs)[:elite_count]
        elites = trial_controls[elite_indices]
        next_mean = jnp.mean(elites, axis=0)
        next_std = jnp.maximum(jnp.std(elites, axis=0), 1e-3)
        return (next_mean, next_std, rng), None

    compiled = jax.jit(
        lambda mean, std, rng: jax.lax.scan(
            body,
            (mean, std, rng),
            xs=jnp.arange(options.iterations),
        )[0]
    )
    start_time = time.perf_counter()
    controls_star, _, _ = compiled(mean_controls, std_controls, key)
    controls_star = clip_controls(controls_star, problem)
    states_star, cost_star = rollout_nonlinear_pendulum(problem, controls_star, soft_constraints)
    jax.block_until_ready(cost_star)
    runtime_seconds = time.perf_counter() - start_time
    control_violation = jax.vmap(lambda uk: nonlinear_control_violation(problem, uk))(controls_star)
    state_violation = jax.vmap(lambda xk: nonlinear_state_violation(problem, xk))(states_star)
    reference_step = nonlinear_pendulum_step(
        problem,
        jnp.asarray(problem.x0, dtype=dtype),
        jnp.zeros(problem.nu, dtype=dtype),
    )
    final_error_cem = problem.state_error(np.asarray(states_star[-1]))
    return SamplingResult(
        method="cem",
        problem_name="nonlinear_pendulum",
        constraint_mode="soft" if soft_constraints else "hard",
        iterations=options.iterations,
        objective_value=_nl_pure_cost(problem, np.asarray(states_star), np.asarray(controls_star)),
        runtime_seconds=runtime_seconds,
        max_control_violation=float(jnp.max(jnp.abs(control_violation))),
        max_state_violation=float(jnp.max(jnp.abs(state_violation))),
        final_position_error=float(final_error_cem[0]),
        final_velocity_error=float(final_error_cem[1]),
        diffrax_vs_exact_step_error=0.0 * float(jnp.linalg.norm(reference_step)),
        state_trajectory=np.asarray(states_star),
        control_trajectory=np.asarray(controls_star),
    )
