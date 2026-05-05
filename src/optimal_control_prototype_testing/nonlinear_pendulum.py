from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


@dataclass(frozen=True)
class NonlinearPendulumProblem:
    mass: float
    length: float
    damping: float
    gravity: float
    x0: np.ndarray
    x_goal: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    state_soft_weight: float
    control_soft_weight: float
    horizon: int
    final_time: float

    @property
    def nx(self) -> int:
        return int(self.x0.shape[0])

    @property
    def nu(self) -> int:
        return int(self.u_min.shape[0])

    @property
    def dt(self) -> float:
        return self.final_time / self.horizon

    @property
    def inertia(self) -> float:
        return self.mass * self.length**2

    def state_error(self, x: np.ndarray) -> np.ndarray:
        error = np.asarray(x, dtype=float) - self.x_goal
        error = error.copy()
        error[0] = float(wrap_angle(error[0]))
        return error

    def continuous_dynamics(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        theta = float(x[0])
        omega = float(x[1])
        torque = float(u[0])
        omega_dot = (
            -(self.gravity / self.length) * np.sin(theta)
            - (self.damping / self.inertia) * omega
            + torque / self.inertia
        )
        return np.array([omega, omega_dot], dtype=float)

    def hard_state_residual(self, x: np.ndarray) -> np.ndarray:
        upper = np.maximum(np.asarray(x, dtype=float) - self.x_max, 0.0)
        lower = np.maximum(self.x_min - np.asarray(x, dtype=float), 0.0)
        return lower + upper

    def hard_control_residual(self, u: np.ndarray) -> np.ndarray:
        upper = np.maximum(np.asarray(u, dtype=float) - self.u_max, 0.0)
        lower = np.maximum(self.u_min - np.asarray(u, dtype=float), 0.0)
        return lower + upper

    def soft_state_penalty(self, x: np.ndarray) -> float:
        violation = self.hard_state_residual(x)
        return float(self.state_soft_weight * violation @ violation)

    def soft_control_penalty(self, u: np.ndarray) -> float:
        violation = self.hard_control_residual(u)
        return float(self.control_soft_weight * violation @ violation)

    def running_cost(self, x: np.ndarray, u: np.ndarray, soft_constraints: bool = False) -> float:
        error = self.state_error(np.asarray(x, dtype=float))
        control = np.asarray(u, dtype=float)
        cost = self.dt * float(error @ self.Q @ error + control @ self.R @ control)
        if soft_constraints:
            cost += self.soft_state_penalty(error + self.x_goal)
            cost += self.soft_control_penalty(control)
        return cost

    def terminal_cost(self, x: np.ndarray, soft_constraints: bool = False) -> float:
        error = self.state_error(np.asarray(x, dtype=float))
        cost = float(error @ self.Qf @ error)
        if soft_constraints:
            cost += self.soft_state_penalty(error + self.x_goal)
        return cost


def build_nonlinear_pendulum_problem() -> NonlinearPendulumProblem:
    """Shared single-pendulum swing-up problem from page 6 of the project PDF.

    Assumptions not fixed by the document are chosen to give a compact benchmark:
    unit mass/length, moderate damping, a four-second horizon, bounded torque, and
    bounded angle/velocity states. The target is upright at rest, measured as
    x_goal = (pi, 0) when angle zero is the hanging equilibrium.
    """

    return NonlinearPendulumProblem(
        mass=1.0,
        length=1.0,
        damping=0.1,
        gravity=9.81,
        x0=np.array([0.0, 0.0], dtype=float),
        x_goal=np.array([np.pi, 0.0], dtype=float),
        x_min=np.array([-2.0 * np.pi, -8.0], dtype=float),
        x_max=np.array([2.0 * np.pi, 8.0], dtype=float),
        u_min=np.array([-2.5], dtype=float),
        u_max=np.array([2.5], dtype=float),
        Q=np.diag([2.0, 0.2]),
        R=np.array([[0.02]], dtype=float),
        Qf=np.diag([25.0, 2.0]),
        state_soft_weight=50.0,
        control_soft_weight=50.0,
        horizon=40,
        final_time=4.0,
    )
