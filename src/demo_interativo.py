"""
Demonstracao interativa ao vivo: malha fechada com setpoint e disturbio variaveis.

Layout: controles fixos na coluna ESQUERDA, graficos a direita.

Executar:  python demo_interativo.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, RadioButtons, Slider

from model import MotorJointParams
from pid_anti_windup import AntiWindupMethod, PIDAntiWindup
from plant import JointPlant

from tuning import CLOSED_LOOP

DT = 0.02
HISTORY = 600
PARAMS = MotorJointParams()
GAINS = CLOSED_LOOP.gains

# Layout normalizado da figura
LEFT = 0.22   # inicio dos graficos (controles ocupam 0-22%)
PLOT_R = 0.96


class LiveDemo:
    def __init__(self) -> None:
        self.setpoint_deg = 60.0
        self.disturbance = 0.0
        self.method = AntiWindupMethod.BACK_CALCULATION
        self.plant = JointPlant(PARAMS, DT)
        self.controller = self._make_controller()
        self.plant.reset()
        self.t_hist = np.zeros(HISTORY)
        self.sp_hist = np.zeros(HISTORY)
        self.th_hist = np.zeros(HISTORY)
        self.u_hist = np.zeros(HISTORY)
        self.td_hist = np.zeros(HISTORY)
        self._idx = 0
        self._time = 0.0

    def _make_controller(self) -> PIDAntiWindup:
        return PIDAntiWindup(
            gains=GAINS, v_max=PARAMS.V_max, dt=DT, method=self.method,
        )

    def reset(self) -> None:
        self.plant.reset()
        self.controller.reset()
        self.t_hist[:] = 0
        self.sp_hist[:] = 0
        self.th_hist[:] = 0
        self.u_hist[:] = 0
        self.td_hist[:] = 0
        self._idx = 0
        self._time = 0.0

    def step(self) -> None:
        sp = np.deg2rad(self.setpoint_deg)
        V = self.controller.step(sp, self.plant.state.theta)
        self.plant.step(V, self.disturbance)
        i = self._idx % HISTORY
        self.t_hist[i] = self._time
        self.sp_hist[i] = sp
        self.th_hist[i] = self.plant.state.theta
        self.u_hist[i] = V
        self.td_hist[i] = self.disturbance
        self._idx += 1
        self._time += DT

    def ordered_history(self) -> tuple[np.ndarray, ...]:
        if self._idx <= HISTORY:
            sl = slice(0, self._idx)
        else:
            start = self._idx % HISTORY
            sl = np.r_[start:HISTORY, 0:start]
        return (
            self.t_hist[sl],
            np.rad2deg(self.sp_hist[sl]),
            np.rad2deg(self.th_hist[sl]),
            self.u_hist[sl],
            self.td_hist[sl],
        )


def main() -> None:
    demo = LiveDemo()

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title("Demo PID Anti-Windup - Junta Robotica")
    fig.subplots_adjust(left=LEFT, right=PLOT_R, bottom=0.10, top=0.93, hspace=0.35)

    ax_pos = fig.add_subplot(4, 1, 1)
    ax_err = fig.add_subplot(4, 1, 2)
    ax_u = fig.add_subplot(4, 1, 3)
    ax_td = fig.add_subplot(4, 1, 4)

    (line_sp,) = ax_pos.plot([], [], "k--", lw=1.2, label="Referencia r")
    (line_th,) = ax_pos.plot([], [], "#2ecc71", lw=2.0, label="Saida theta")
    (line_err,) = ax_err.plot([], [], "#9b59b6", lw=1.5)
    (line_u,) = ax_u.plot([], [], "#3498db", lw=1.5)
    (line_td,) = ax_td.plot([], [], "#e67e22", lw=1.5)

    ax_pos.set_ylabel("Posicao [deg]")
    ax_pos.set_ylim(-20, 130)
    ax_pos.legend(loc="upper left")
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_title("Malha Fechada ao Vivo — u = PID(r - theta) + Anti-Windup")

    ax_err.set_ylabel("Erro e [deg]")
    ax_err.axhline(0, color="gray", ls=":", alpha=0.5)
    ax_err.grid(True, alpha=0.3)

    ax_u.set_ylabel("Tensao [V]")
    ax_u.set_ylim(-PARAMS.V_max * 1.2, PARAMS.V_max * 1.2)
    ax_u.axhline(PARAMS.V_max, color="gray", ls=":", alpha=0.5)
    ax_u.axhline(-PARAMS.V_max, color="gray", ls=":", alpha=0.5)
    ax_u.grid(True, alpha=0.3)

    ax_td.set_ylabel("Td [N*m]")
    ax_td.set_xlabel("Tempo [s]")
    ax_td.set_ylim(0, 0.16)
    ax_td.grid(True, alpha=0.3)

    # --- Painel de controles FIXO na esquerda ---
    fig.text(0.02, 0.97, "Controles", fontsize=10, fontweight="bold", va="top")

    ax_sp = fig.add_axes([0.02, 0.78, 0.16, 0.03])
    ax_dist = fig.add_axes([0.02, 0.72, 0.16, 0.03])
    ax_radio = fig.add_axes([0.02, 0.42, 0.16, 0.22])
    ax_reset = fig.add_axes([0.02, 0.32, 0.16, 0.06])

    slider_sp = Slider(ax_sp, "Setpoint", 0.0, 120.0, valinit=60.0, valstep=1.0)
    slider_dist = Slider(ax_dist, "Disturbio", 0.0, 0.15, valinit=0.0, valstep=0.005)
    btn_reset = Button(ax_reset, "Reset")
    radio = RadioButtons(ax_radio, ("back_calc", "clamping", "none"), active=0)

    fig.text(0.02, 0.68, "Anti-Windup:", fontsize=8, va="top")

    slider_sp.label.set_fontsize(8)
    slider_dist.label.set_fontsize(8)

    def on_sp(val: float) -> None:
        demo.setpoint_deg = val

    def on_dist(val: float) -> None:
        demo.disturbance = val

    def on_reset(_event) -> None:
        demo.reset()

    def on_method(label: str) -> None:
        demo.method = {
            "none": AntiWindupMethod.NONE,
            "clamping": AntiWindupMethod.CLAMPING,
            "back_calc": AntiWindupMethod.BACK_CALCULATION,
        }[label]
        demo.controller = demo._make_controller()

    slider_sp.on_changed(on_sp)
    slider_dist.on_changed(on_dist)
    btn_reset.on_clicked(on_reset)
    radio.on_clicked(on_method)

    def update(_frame: int):
        demo.step()
        t, sp, th, u, td = demo.ordered_history()
        line_sp.set_data(t, sp)
        line_th.set_data(t, th)
        line_err.set_data(t, sp - th)
        line_u.set_data(t, u)
        line_td.set_data(t, td)
        if len(t) > 1:
            t_min, t_max = t[0], t[-1]
            for ax in (ax_pos, ax_err, ax_u, ax_td):
                ax.set_xlim(t_min, max(t_max, 1.0))
        return line_sp, line_th, line_err, line_u, line_td

    FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
