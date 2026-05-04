from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrivialLQRProblem:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    x0: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    horizon: int
    final_time: float

    @property
    def nx(self) -> int:
        return int(self.A.shape[0])

    @property
    def nu(self) -> int:
        return int(self.B.shape[1])

    @property
    def dt(self) -> float:
        return self.final_time / self.horizon


def build_trivial_lqr_problem() -> TrivialLQRProblem:
    return TrivialLQRProblem(
        A=np.array([[0.0, 1.0], [-0.25, -0.1]], dtype=float),
        B=np.array([[0.0], [1.0]], dtype=float),
        Q=np.diag([1.0, 0.2]),
        R=np.array([[0.05]], dtype=float),
        Qf=np.diag([8.0, 1.0]),
        x0=np.array([1.5, 0.0], dtype=float),
        u_min=np.array([-0.75], dtype=float),
        u_max=np.array([0.75], dtype=float),
        horizon=20,
        final_time=2.0,
    )
