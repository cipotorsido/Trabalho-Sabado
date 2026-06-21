"""
Analise de estabilidade e geracao de figuras para o relatorio.

- Sintonia analitica (SIMC / IMC)
- Diagrama de Bode e margens (MG, MF)
- Lugar das raizes
- Resposta ao degrau e rejeicao de disturbio
- Diagrama de blocos
- Tabela de metricas (Mesa Redonda)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import control as ctrl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from metrics import compute_performance_metrics
from model import (
    MotorJointParams,
    dc_gain_position,
    mechanical_time_constant,
    transfer_function_voltage_to_position,
)
from pid_anti_windup import AntiWindupMethod
from simulate import (
    default_config_disturbance_only,
    default_config_step,
    default_gains,
    plot_single_response,
    simulate_closed_loop,
    windup_demo_gains,
)
from tuning import CLOSED_LOOP

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figuras"


@dataclass
class AnalyticalTuning:
    """Resultado da sintonia analitica SIMC."""
    lambda_cl: float
    Kp: float
    Ki: float
    Kd: float
    Ti: float
    Td: float


def simc_pid_tuning(params: MotorJointParams, lambda_cl: float = 1.0) -> AnalyticalTuning:
    """
    Sintonia SIMC (Skogestad) para G(s) = K / [s(tau_m s + 1)].

    Regra para processo integrador com atraso de 1a ordem (serie PID):
        Kp = (tau_m + tau_d/2) / (K * lambda)
        Ti = tau_m + tau_d/2
        Td = tau_m * tau_d / (2*tau_m + tau_d)   com tau_d = 0 => Td = 0

    Para tau_d = 0 (modelo sem atraso explicito):
        Kp = tau_m / (K * lambda)
        Ti = tau_m
        Td = 0
    """
    K = dc_gain_position(params)
    tau_m = mechanical_time_constant(params)
    tau_d = 0.0

    Ti = tau_m + tau_d / 2.0
    Kp = (tau_m + tau_d / 2.0) / (K * lambda_cl)
    Ki = Kp / Ti
    Td = tau_m * tau_d / (2.0 * tau_m + tau_d) if (2.0 * tau_m + tau_d) > 0 else 0.0
    Kd = Kp * Td

    return AnalyticalTuning(
        lambda_cl=lambda_cl,
        Kp=Kp,
        Ki=Ki,
        Kd=Kd,
        Ti=Ti,
        Td=Td,
    )


def pid_transfer_function(Kp: float, Ki: float, Kd: float) -> ctrl.TransferFunction:
    return ctrl.tf([Kd, Kp, Ki], [1.0, 0.0])


def linear_closed_loop(
    params: MotorJointParams, Kp: float, Ki: float, Kd: float
) -> ctrl.TransferFunction:
    G = transfer_function_voltage_to_position(params)
    C = pid_transfer_function(Kp, Ki, Kd)
    return ctrl.feedback(C * G, 1)


def compute_stability_margins(
    params: MotorJointParams, Kp: float, Ki: float, Kd: float
) -> dict[str, float]:
    G = transfer_function_voltage_to_position(params)
    C = pid_transfer_function(Kp, Ki, Kd)
    L = C * G
    gm, pm, wcg, wcp = ctrl.margin(L)
    return {
        "gain_margin_db": float(20 * np.log10(gm)) if gm > 0 else float("inf"),
        "phase_margin_deg": float(pm),
        "omega_cg_rad_s": float(wcg),
        "omega_cp_rad_s": float(wcp),
    }


def plot_block_diagram(save_path: Path) -> None:
    """Diagrama de blocos — fluxo da esquerda para direita, realimentacao embaixo."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Diagrama de Blocos - Malha Fechada com Anti-Windup", fontsize=12, pad=10)

    bw, bh = 1.35, 0.95
    y_main = 2.8

    boxes = [
        (0.4, y_main, "r(t)\nReferencia"),
        (2.0, y_main, "Sumador\n(+/-)"),
        (3.6, y_main, "PID\nKp, Ki, Kd"),
        (5.2, y_main, "Saturacao\n|V| <= Vmax"),
        (6.8, y_main, "Planta G(s)\nMotor+Junta"),
        (8.4, y_main, "theta(t)\nSaida"),
    ]

    def box_center(x: float, y: float) -> tuple[float, float]:
        return x + bw / 2, y + bh / 2

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str = "#34495e") -> None:
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="->", mutation_scale=12, color=color, lw=1.4,
        ))

    for x, y, txt in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), bw, bh, boxstyle="round,pad=0.05",
            facecolor="#ecf0f1", edgecolor="#2c3e50", lw=1.5,
        ))
        ax.text(x + bw / 2, y + bh / 2, txt, ha="center", va="center", fontsize=8)

    # Anti-windup alinhado abaixo de PID + Saturacao (lado esquerdo da malha)
    aw_x, aw_y, aw_w, aw_h = 3.5, 0.55, 3.1, 0.85
    ax.add_patch(FancyBboxPatch(
        (aw_x, aw_y), aw_w, aw_h, boxstyle="round,pad=0.05",
        facecolor="#d5f5e3", edgecolor="#27ae60", lw=1.5,
    ))
    ax.text(aw_x + aw_w / 2, aw_y + aw_h / 2,
            "Anti-Windup\n(clamping / back-calc)", ha="center", va="center", fontsize=8)

    # Caminho direto (esquerda -> direita)
    cx = [box_center(x, y)[0] for x, y, _ in boxes]
    for i in range(len(cx) - 1):
        arrow((cx[i] + bw / 2 - 0.05, y_main + bh / 2), (cx[i + 1] - bw / 2 + 0.05, y_main + bh / 2))

    # Realimentacao: saida desce, percorre trilho inferior da esquerda, sobe no sumador
    y_fb = 0.15
    sum_x = boxes[1][0]
    out_x = boxes[-1][0]
    arrow(box_center(out_x, y_main), (box_center(out_x, y_main)[0], y_fb + 0.05))
    ax.plot([box_center(out_x, y_main)[0], box_center(sum_x, y_main)[0]], [y_fb, y_fb], color="#34495e", lw=1.4)
    arrow((box_center(sum_x, y_main)[0], y_fb + 0.05), box_center(sum_x, y_main))
    ax.text(box_center(sum_x, y_main)[0] - 0.15, y_main + bh / 2 + 0.1, "-", fontsize=11, ha="center")

    # Anti-windup: recebe u_sat e u_pid (entre PID e Saturacao)
    pid_x, sat_x = boxes[2][0], boxes[3][0]
    arrow((box_center(pid_x, y_main)[0], y_main), (box_center(pid_x, y_main)[0], aw_y + aw_h))
    arrow((box_center(sat_x, y_main)[0], y_main), (box_center(sat_x, y_main)[0], aw_y + aw_h))
    arrow((aw_x + aw_w / 2, aw_y + aw_h), (box_center(pid_x, y_main)[0], aw_y + aw_h + 0.05))

    # Disturbio entra na planta por cima
    plant_cx = box_center(boxes[4][0], y_main)[0]
    ax.text(plant_cx, y_main + bh + 0.55, "Td(t) — disturbio de carga", ha="center", fontsize=9, color="#e67e22")
    arrow((plant_cx, y_main + bh + 0.35), (plant_cx, y_main + bh + 0.02), color="#e67e22")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Diagrama salvo em: {save_path}")


def plot_bode(params: MotorJointParams, tuning: AnalyticalTuning, save_path: Path) -> dict:
    G = transfer_function_voltage_to_position(params)
    C = pid_transfer_function(tuning.Kp, tuning.Ki, tuning.Kd)
    L = C * G

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ctrl.bode_plot(L, dB=True, Hz=False, omega_limits=[0.01, 100], ax=ax)
    ax[0].set_title(f"Diagrama de Bode - L(s)=C(s)G(s)  (lambda={tuning.lambda_cl}s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    margins = compute_stability_margins(params, tuning.Kp, tuning.Ki, tuning.Kd)
    print(f"Bode salvo em: {save_path}")
    return margins


def plot_root_locus(params: MotorJointParams, save_path: Path) -> None:
    G = transfer_function_voltage_to_position(params)
    C = pid_transfer_function(1.0, 0.5, 0.05)
    L = C * G

    fig, ax = plt.subplots(figsize=(8, 6))
    ctrl.root_locus(L, ax=ax, grid=True)
    ax.set_title("Lugar das Raizes - L(s) = C(s)G(s) (ganhos normalizados)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Lugar das raizes salvo em: {save_path}")


def plot_closed_loop_poles(
    params: MotorJointParams, tuning: AnalyticalTuning, save_path: Path
) -> None:
    T = linear_closed_loop(params, tuning.Kp, tuning.Ki, tuning.Kd)
    poles = ctrl.poles(T)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.scatter(np.real(poles), np.imag(poles), s=80, c="#e74c3c", zorder=5, label="Polos MF")
    ax.set_xlabel("Re(s)")
    ax.set_ylabel("Im(s)")
    ax.grid(True, alpha=0.3)
    ax.set_title("Polos da Malha Fechada T(s) - Sintonia SIMC")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Polos MF salvos em: {save_path}")


def generate_simulation_figures() -> None:
    params = MotorJointParams()
    gains = default_gains()
    method = AntiWindupMethod.BACK_CALCULATION

    step_cfg = default_config_step()
    step_cfg.load_disturbance = 0.0
    step_data = simulate_closed_loop(params, gains, step_cfg, method)
    plot_single_response(
        step_data,
        FIG_DIR / "fig_resposta_degrau.png",
        "Malha Fechada — Resposta ao Degrau (60 deg) | u = PID(r - theta)",
        params.V_max,
    )

    dist_cfg = default_config_disturbance_only()
    dist_data = simulate_closed_loop(params, gains, dist_cfg, method)
    plot_single_response(
        dist_data,
        FIG_DIR / "fig_rejeicao_disturbio.png",
        "Malha Fechada — Rejeicao de Disturbio | referencia 45 deg, Td em t=5s",
        params.V_max,
    )

    results = {}
    for m in AntiWindupMethod:
        results[m.value] = simulate_closed_loop(
            params, windup_demo_gains(), default_config_step(), m,
        )
    from simulate import plot_comparison
    plot_comparison(
        results,
        save_path=str(FIG_DIR / "fig_anti_windup_comparacao.png"),
    )


def export_metrics_table() -> Path:
    """Gera CSV para Mesa Redonda comparativa."""
    params = MotorJointParams()
    gains = windup_demo_gains()
    cfg = default_config_step()

    rows = []
    complexity = {
        "none": "Baixa (~15 linhas)",
        "clamping": "Media (~25 linhas)",
        "back_calculation": "Media (~20 linhas)",
    }
    labels = {
        "none": "PID sem Anti-Windup",
        "clamping": "PID + Clamping",
        "back_calculation": "PID + Back-Calculation",
    }

    for method in AntiWindupMethod:
        data = simulate_closed_loop(params, gains, cfg, method)
        m = compute_performance_metrics(
            data["t"], data["theta"], data["setpoint"], data["u_sat"], params.V_max
        )
        rows.append({
            "Controlador": labels[method.value],
            "Tempo_Assentamento_s": m["settling_time_s"],
            "Sobresinal_pct": m["overshoot_pct"],
            "Tensao_RMS_V": m["actuator_rms_v"],
            "Tensao_Pico_V": m["actuator_peak_v"],
            "Tempo_Saturado_pct": m["saturation_time_pct"],
            "Erro_Estacionario_rad": m["steady_state_error_rad"],
            "Complexidade_Codigo": complexity[method.value],
        })

    csv_path = FIG_DIR / "tabela_mesa_redonda.csv"
    header = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(";".join(header) + "\n")
        for row in rows:
            f.write(";".join(str(row[k]) for k in header) + "\n")
    print(f"Tabela salva em: {csv_path}")
    return csv_path


def print_analytical_report(params: MotorJointParams) -> tuple[AnalyticalTuning, AnalyticalTuning]:
    K = dc_gain_position(params)
    tau_m = mechanical_time_constant(params)
    b_eff = params.b + params.Kt * params.Ke / params.R

    print("=" * 65)
    print("SINTONIA ANALITICA (SIMC)")
    print("=" * 65)
    print(f"  b_eff = b + Kt*Ke/R = {b_eff:.6f} N*m*s/rad")
    print(f"  tau_m = J/b_eff     = {tau_m:.4f} s")
    print(f"  K     = Kt/(R*b_eff)= {K:.4f} rad/(V*s)")
    print(f"  G(s)  = {K:.4f} / [s({tau_m:.4f}s + 1)]")
    print("-" * 65)

    tuning = simc_pid_tuning(params, lambda_cl=1.0)
    print(f"  lambda (constante de tempo desejada) = {tuning.lambda_cl} s")
    print(f"  Kp = tau_m / (K*lambda) = {tuning.Kp:.4f}")
    print(f"  Ti = tau_m              = {tuning.Ti:.4f} s")
    print(f"  Ki = Kp/Ti              = {tuning.Ki:.4f}")
    print(f"  Kd = 0 (sem atraso dominante no modelo)")
    print("-" * 65)
    print("  Ganhos MALHA FECHADA (rastreamento estavel — entregaveis):")
    print(f"  Kp={CLOSED_LOOP.gains.Kp}, Ki={CLOSED_LOOP.gains.Ki}, Kd={CLOSED_LOOP.gains.Kd}")
    print("  Ganhos demo anti-windup (comparacao de windup apenas):")
    wd = windup_demo_gains()
    print(f"  Kp={wd.Kp}, Ki={wd.Ki}, Kd={wd.Kd}, Kaw={wd.Kaw}")
    print("-" * 65)
    print("  Sintonia refinada (LR / alocacao de polos, malha linear estavel):")
    refined = AnalyticalTuning(
        lambda_cl=0.0,
        Kp=1.5,
        Ki=0.5,
        Kd=0.3,
        Ti=3.0,
        Td=0.2,
    )
    print(f"  Kp={refined.Kp}, Ki={refined.Ki}, Kd={refined.Kd}")
    print("=" * 65)
    return tuning, refined


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    params = MotorJointParams()

    tuning, refined = print_analytical_report(params)

    margins_simc = compute_stability_margins(params, tuning.Kp, tuning.Ki, tuning.Kd)
    margins_ref = compute_stability_margins(params, refined.Kp, refined.Ki, refined.Kd)

    print("\nMargens — Sintonia SIMC (PI, linear):")
    print(f"  MF = {margins_simc['phase_margin_deg']:.2f} deg  (integrador puro => MF critica)")
    print("\nMargens — Sintonia refinada (PID, linear):")
    print(f"  Margem de Ganho:  {margins_ref['gain_margin_db']:.2f} dB")
    print(f"  Margem de Fase:   {margins_ref['phase_margin_deg']:.2f} deg")
    print(f"  omega_cp:         {margins_ref['omega_cp_rad_s']:.4f} rad/s")

    plot_block_diagram(FIG_DIR / "fig_diagrama_blocos.png")
    plot_bode(params, refined, FIG_DIR / "fig_bode.png")
    plot_root_locus(params, FIG_DIR / "fig_lugar_raizes.png")
    plot_closed_loop_poles(params, refined, FIG_DIR / "fig_polos_mf.png")
    generate_simulation_figures()
    export_metrics_table()

    margins_path = FIG_DIR / "margens_estabilidade.txt"
    with open(margins_path, "w", encoding="utf-8") as f:
        f.write("Margens de Estabilidade\n\n")
        f.write("[SIMC — PI]\n")
        for k, v in margins_simc.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n[Refinada — PID Kp=1.5 Ki=0.5 Kd=0.3]\n")
        for k, v in margins_ref.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nRelatorio numerico salvo em: {margins_path}")


if __name__ == "__main__":
    main()
