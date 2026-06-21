"""
Simulacao em malha fechada: junta robotica + PID com anti-windup.

Suporta setpoint e disturbio variaveis no tempo (perfil arbitrario).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import compute_performance_metrics
from model import MotorJointParams, print_model_summary
from pid_anti_windup import AntiWindupMethod, PIDAntiWindup, PIDGains
from plant import JointPlant
from tuning import CLOSED_LOOP, WINDUP_DEMO

SignalFn = Callable[[float], float]


@dataclass
class SimulationConfig:
    t_end: float = 30.0
    dt: float = 0.001
    initial_theta: float = 0.0
    setpoint_fn: SignalFn | None = None
    disturbance_fn: SignalFn | None = None
    setpoint: float = field(default_factory=lambda: np.deg2rad(60.0))
    load_disturbance: float = 0.0
    load_step_time: float = 12.0

    def resolve_setpoint(self, t: float) -> float:
        if self.setpoint_fn is not None:
            return float(self.setpoint_fn(t))
        return float(self.setpoint)

    def resolve_disturbance(self, t: float) -> float:
        if self.disturbance_fn is not None:
            return float(self.disturbance_fn(t))
        return float(self.load_disturbance if t >= self.load_step_time else 0.0)


def simulate_closed_loop(
    params: MotorJointParams,
    gains: PIDGains,
    config: SimulationConfig,
    anti_windup: AntiWindupMethod = AntiWindupMethod.BACK_CALCULATION,
) -> dict[str, np.ndarray]:
    """Integracao numerica do sistema nao linear com saturacao."""
    n_steps = int(config.t_end / config.dt) + 1
    t = np.linspace(0.0, config.t_end, n_steps)

    theta = np.zeros(n_steps)
    omega = np.zeros(n_steps)
    u_pid = np.zeros(n_steps)
    u_sat = np.zeros(n_steps)
    td = np.zeros(n_steps)
    error = np.zeros(n_steps)
    setpoint = np.zeros(n_steps)

    plant = JointPlant(params, config.dt)
    plant.reset(theta=config.initial_theta)

    controller = PIDAntiWindup(
        gains=gains,
        v_max=params.V_max,
        dt=config.dt,
        method=anti_windup,
    )

    for k in range(n_steps):
        sp = config.resolve_setpoint(t[k])
        dist = config.resolve_disturbance(t[k])
        setpoint[k] = sp
        td[k] = dist
        error[k] = sp - plant.state.theta

        # Malha fechada: erro e = r - theta (realimentacao da posicao medida)
        V = controller.step(sp, plant.state.theta)
        u_pid[k] = controller.u_pid
        u_sat[k] = V
        plant.step(V, dist)
        theta[k] = plant.state.theta
        omega[k] = plant.state.omega

    return {
        "t": t,
        "theta": theta,
        "omega": omega,
        "error": error,
        "u_pid": u_pid,
        "u_sat": u_sat,
        "td": td,
        "setpoint": setpoint,
    }


def step_setpoint_profile(
    value_rad: float, step_time: float = 0.0
) -> SignalFn:
    return lambda t: value_rad if t >= step_time else 0.0


def disturbance_step_profile(
    magnitude: float, step_time: float
) -> SignalFn:
    return lambda t: magnitude if t >= step_time else 0.0


def plot_comparison(
    results: dict[str, dict[str, np.ndarray]],
    save_path: str | None = None,
    title: str = "Malha Fechada — PID com Saturacao (Comparacao Anti-Windup)",
    ylim_deg: tuple[float, float] = (-20.0, 130.0),
) -> None:
    """Compara respostas; eixo Y fixo para referencia e rastreamento ficarem visiveis."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    colors = {"none": "#e74c3c", "clamping": "#3498db", "back_calculation": "#2ecc71"}
    labels = {
        "none": "PID sem Anti-Windup",
        "clamping": "PID + Clamping",
        "back_calculation": "PID + Back-Calculation",
    }

    first = results[list(results.keys())[0]]
    sp_deg = np.rad2deg(first["setpoint"])
    axes[0].plot(first["t"], sp_deg, "k--", lw=2.0, label="Referencia r(t)")

    for name, data in results.items():
        c = colors.get(name, "gray")
        lbl = labels.get(name, name)
        axes[0].plot(data["t"], np.rad2deg(data["theta"]), color=c, label=lbl, lw=1.5)
        axes[1].plot(data["t"], np.rad2deg(data["error"]), color=c, lw=1.5)
        axes[2].plot(data["t"], data["u_sat"], color=c, label=lbl, lw=1.5)
        axes[3].plot(data["t"], data["u_pid"], color=c, ls="--", alpha=0.7, lw=1.0)

    axes[0].set_ylabel("Posicao [deg]")
    axes[0].set_ylim(ylim_deg)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].set_ylabel("Erro e=r-theta [deg]")
    axes[1].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Tensao aplicada [V]")
    axes[2].grid(True, alpha=0.3)
    axes[3].set_ylabel("Comando PID [V]")
    axes[3].set_xlabel("Tempo [s]")
    axes[3].grid(True, alpha=0.3)

    fig.text(
        0.99, 0.01,
        "Malha fechada: u = PID(r - theta)  |  curva vermelha diverge por windup, nao por malha aberta",
        ha="right", fontsize=7, color="gray",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvo em: {save_path}")
    plt.close(fig)


def plot_single_response(
    data: dict[str, np.ndarray],
    save_path: str,
    title: str,
    v_max: float,
) -> None:
    """Grafico de malha fechada: referencia, posicao, erro e atuador."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    t = data["t"]

    axes[0].plot(t, np.rad2deg(data["setpoint"]), "k--", lw=2.0, label="Referencia r(t)")
    axes[0].plot(t, np.rad2deg(data["theta"]), "#2ecc71", lw=1.8, label="Saida theta(t)")
    axes[0].set_ylabel("Posicao [deg]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)

    axes[1].plot(t, np.rad2deg(data["error"]), "#9b59b6", lw=1.5)
    axes[1].axhline(0, color="gray", ls=":", alpha=0.5)
    axes[1].set_ylabel("Erro e=r-theta [deg]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, data["u_sat"], "#3498db", lw=1.5, label="u = PID(e)")
    axes[2].axhline(v_max, color="gray", ls=":", alpha=0.6)
    axes[2].axhline(-v_max, color="gray", ls=":", alpha=0.6)
    axes[2].set_ylabel("Tensao [V]")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(t, data["td"], "#e67e22", lw=1.5)
    axes[3].set_ylabel("Disturbio Td [N*m]")
    axes[3].set_xlabel("Tempo [s]")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico salvo em: {save_path}")


def default_gains() -> PIDGains:
    return CLOSED_LOOP.gains


def windup_demo_gains() -> PIDGains:
    return WINDUP_DEMO.gains


def default_config_step() -> SimulationConfig:
    return SimulationConfig(
        t_end=30.0,
        dt=0.001,
        setpoint=np.deg2rad(60.0),
        load_disturbance=0.04,
        load_step_time=12.0,
    )


def default_config_disturbance_only() -> SimulationConfig:
    """Referencia fixa; disturbio aplicado em t=5 s."""
    return SimulationConfig(
        t_end=25.0,
        dt=0.001,
        setpoint=np.deg2rad(45.0),
        load_disturbance=0.06,
        load_step_time=5.0,
    )


def main() -> None:
    params = MotorJointParams()
    config = default_config_step()

    print_model_summary(params)
    print("\n" + "=" * 60)
    print("MALHA FECHADA — PID + Anti-Windup")
    print("  u = PID(r - theta),  realimentacao: theta medido")
    print("=" * 60)

    # 1) Resposta tipica malha fechada (sintonia estavel)
    cl_data = simulate_closed_loop(
        params, default_gains(), config, AntiWindupMethod.BACK_CALCULATION,
    )
    plot_single_response(
        cl_data,
        "../docs/figuras/fig_malha_fechada_degrau.png",
        "Malha Fechada — Resposta ao Degrau (60 deg)",
        params.V_max,
    )

    # 2) Comparacao anti-windup (ganhos agressivos, eixo Y fixo)
    results: dict[str, dict[str, np.ndarray]] = {}
    for method in AntiWindupMethod:
        data = simulate_closed_loop(params, windup_demo_gains(), config, method)
        results[method.value] = data
        m = compute_performance_metrics(
            data["t"], data["theta"], data["setpoint"], data["u_sat"], params.V_max
        )
        print(f"\n  [demo windup — {method.value}]")
        print(f"    Overshoot:          {m['overshoot_pct']:.2f} %")
        print(f"    Tempo assentamento: {m['settling_time_s']:.3f} s")

    plot_comparison(results, save_path="../docs/figuras/fig_anti_windup_comparacao.png")


if __name__ == "__main__":
    main()
