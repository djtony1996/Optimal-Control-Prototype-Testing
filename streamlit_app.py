"""Streamlit frontend for the nonlinear pendulum optimal control benchmark (item 4 — acados)."""
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── path & acados environment (mirrors scripts/run_item4_acados.sh) ──────────
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_repo_root, "src"))

_acados_root = os.environ.get("ACADOS_SOURCE_DIR", "/Users/jitongding/Documents/GitHub/acados")
os.environ.setdefault("ACADOS_SOURCE_DIR", _acados_root)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from optimal_control_prototype_testing.acados_cpu import (  # noqa: E402
    AcadosBaselineResult,
    solve_nonlinear_pendulum_with_acados,
)

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Nonlinear Pendulum — Item 4", layout="wide")
st.markdown(
    """
    <style>
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Nonlinear Pendulum Optimal Control — Item 4 (acados)")

# ── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Problem Parameters")

    dt = st.number_input(
        "Time step  dt  (s)",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        format="%.3f",
    )
    final_time = st.number_input(
        "Total time  T  (s)",
        min_value=1.0,
        value=4.0,
        step=0.5,
        format="%.1f",
    )
    constraint_mode = st.selectbox(
        "Constraint mode",
        options=["hard", "soft", "both"],
        index=0,
    )

    st.divider()
    st.subheader("Solver settings")
    max_iter = st.number_input(
        "Max SQP iterations",
        min_value=1,
        value=100,
        step=50,
        format="%d",
    )
    tol = st.number_input(
        "Convergence tolerance",
        min_value=1e-12,
        value=1e-6,
        format="%.2e",
    )

    st.divider()
    st.subheader("Control bounds")
    u_max_mag = st.number_input(
        "Max torque  |u|  (Nm)",
        min_value=0.1,
        value=2.5,
        step=0.1,
        format="%.2f",
    )

    st.divider()
    st.subheader("Cost matrices")
    st.caption("Q — running state cost (diagonal)")
    q1 = st.number_input("Q[θ, θ]", min_value=0.0, value=2.0, step=0.1, format="%.3f")
    q2 = st.number_input("Q[θ̇, θ̇]", min_value=0.0, value=0.2, step=0.01, format="%.3f")
    st.caption("R — running control cost")
    r1 = st.number_input("R[u, u]", min_value=0.0, value=0.02, step=0.001, format="%.4f")
    st.caption("Qf — terminal state cost (diagonal)")
    qf1 = st.number_input("Qf[θ, θ]", min_value=0.0, value=25.0, step=1.0, format="%.2f")
    qf2 = st.number_input("Qf[θ̇, θ̇]", min_value=0.0, value=2.0, step=0.1, format="%.2f")

    solve_clicked = st.button("Solve", type="primary", use_container_width=True)

# ── solver ───────────────────────────────────────────────────────────────────
def run_solver(
    dt: float,
    final_time: float,
    mode: str,
    u_max: float,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
    max_iter: int,
    tol: float,
) -> list[AcadosBaselineResult]:
    kwargs = dict(dt=dt, final_time=final_time, u_max=u_max, Q=Q, R=R, Qf=Qf, max_iter=max_iter, tol=tol)
    results = []
    if mode in ("hard", "both"):
        results.append(solve_nonlinear_pendulum_with_acados(soft_constraints=False, **kwargs))
    if mode in ("soft", "both"):
        results.append(solve_nonlinear_pendulum_with_acados(soft_constraints=True, **kwargs))
    return results


# ── display helpers ──────────────────────────────────────────────────────────
def _time_axes(result: AcadosBaselineResult, final_time: float) -> tuple[np.ndarray, np.ndarray]:
    N = len(result.control_trajectory)
    t_state = np.linspace(0.0, final_time, N + 1)
    t_ctrl = np.linspace(0.0, final_time - final_time / N, N)
    return t_state, t_ctrl


def show_metrics(result: AcadosBaselineResult) -> None:
    cols = st.columns(5)
    cols[0].metric("Converged", "Yes" if result.status == 0 else "No")
    cols[1].metric("SQP iterations", str(result.sqp_iterations))
    obj = f"{result.objective_value:.4f}" if result.objective_value is not None else "N/A"
    cols[2].metric("Final cost", obj)
    cols[3].metric("Runtime (s)", f"{result.runtime_seconds:.3f}")
    cols[4].metric("Max ctrl violation", f"{result.max_control_violation:.2e}")

    cols2 = st.columns(4)
    final_state = result.state_trajectory[-1]
    cols2[0].metric("Final θ (rad)", f"{final_state[0]:.4f}")
    cols2[1].metric("Final θ̇ (rad/s)", f"{final_state[1]:.4f}")
    cols2[2].metric("Position error (rad)", f"{result.final_position_error:.4f}")
    cols2[3].metric("Velocity error (rad/s)", f"{result.final_velocity_error:.4f}")


def show_plots(result: AcadosBaselineResult, final_time: float, u_max: float = 2.5) -> None:
    t_state, t_ctrl = _time_axes(result, final_time)
    states = result.state_trajectory
    controls = result.control_trajectory

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Trajectories — {result.constraint_mode} constraints", fontsize=13)

    # theta
    axes[0].plot(t_state, states[:, 0], color="steelblue", linewidth=2)
    axes[0].axhline(np.pi, color="tomato", linestyle="--", linewidth=1, label="goal (π)")
    axes[0].axhline(2 * np.pi, color="gray", linestyle=":", linewidth=1, label="bound")
    axes[0].axhline(-2 * np.pi, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("θ (rad)")
    axes[0].set_title("Angle")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # theta_dot
    axes[1].plot(t_state, states[:, 1], color="darkorange", linewidth=2)
    axes[1].axhline(0.0, color="tomato", linestyle="--", linewidth=1, label="goal (0)")
    axes[1].axhline(8.0, color="gray", linestyle=":", linewidth=1, label="bound")
    axes[1].axhline(-8.0, color="gray", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("θ̇ (rad/s)")
    axes[1].set_title("Angular velocity")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # torque
    axes[2].step(t_ctrl, controls[:, 0], color="seagreen", linewidth=2, where="post")
    axes[2].axhline(u_max, color="gray", linestyle=":", linewidth=1, label=f"bound (±{u_max})")
    axes[2].axhline(-u_max, color="gray", linestyle=":", linewidth=1)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Torque (Nm)")
    axes[2].set_title("Control input")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_tables(result: AcadosBaselineResult, final_time: float) -> None:
    t_state, t_ctrl = _time_axes(result, final_time)
    states = result.state_trajectory
    controls = result.control_trajectory

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**State trajectory**")
        df_state = pd.DataFrame(
            {
                "Time (s)": np.round(t_state, 6),
                "θ (rad)": np.round(states[:, 0], 6),
                "θ̇ (rad/s)": np.round(states[:, 1], 6),
            }
        )
        st.dataframe(df_state, use_container_width=True, height=300)

    with col_right:
        st.markdown("**Control trajectory**")
        df_ctrl = pd.DataFrame(
            {
                "Time (s)": np.round(t_ctrl, 6),
                "Torque (Nm)": np.round(controls[:, 0], 6),
            }
        )
        st.dataframe(df_ctrl, use_container_width=True, height=300)


def show_result(result: AcadosBaselineResult, final_time: float, u_max: float = 2.5) -> None:
    st.subheader(f"Constraint mode: {result.constraint_mode}")
    if result.status != 0:
        st.warning(
            f"Solver did not converge (status {result.status}). "
            "Metrics and trajectories below reflect the last iterate, not the optimum. "
            "Try a smaller total time, finer dt, or larger |u|."
        )
    show_metrics(result)
    st.divider()
    show_plots(result, final_time, u_max)
    st.divider()
    show_tables(result, final_time)


# ── main ─────────────────────────────────────────────────────────────────────
if solve_clicked:
    Q_mat = np.diag([q1, q2])
    R_mat = np.array([[r1]])
    Qf_mat = np.diag([qf1, qf2])
    with st.spinner("Running acados solver…"):
        try:
            results = run_solver(dt, final_time, constraint_mode, u_max_mag, Q_mat, R_mat, Qf_mat, int(max_iter), float(tol))
        except Exception as exc:
            st.error(f"Solver error: {exc}")
            st.stop()

    if constraint_mode == "both":
        tabs = st.tabs(["Hard constraints", "Soft constraints"])
        for tab, result in zip(tabs, results):
            with tab:
                show_result(result, final_time, u_max_mag)
    else:
        show_result(results[0], final_time, u_max_mag)
else:
    st.info("Set the parameters in the sidebar and press **Solve**.")
