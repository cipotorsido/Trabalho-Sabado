"""
Controle PID com tratamento de saturação (Anti-Windup).

Controlador 10: simula limite físico de tensão do motor e evita
acúmulo integral durante saturação via back-calculation ou clamping.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AntiWindupMethod(str, Enum):
    NONE = "none"
    CLAMPING = "clamping"
    BACK_CALCULATION = "back_calculation"


@dataclass
class PIDGains:
    Kp: float
    Ki: float
    Kd: float
    Kaw: float = 1.0  # Ganho anti-windup (back-calculation)


class PIDAntiWindup:
    """
    PID discreto com anti-windup.

    Estado integral I armazena diretamente a contribuicao Ki*integral(e).

    u_pid = Kp*e + I + Kd*de/dt
    u_sat = sat(u_pid, -V_max, +V_max)

    Back-calculation:
        dI/dt = Ki*e + Kaw*(u_sat - u_pid)

    Clamping:
        Congela I se u_pid != u_sat e e*(u_pid - u_sat) > 0.
    """

    def __init__(
        self,
        gains: PIDGains,
        v_max: float,
        dt: float,
        method: AntiWindupMethod = AntiWindupMethod.BACK_CALCULATION,
    ) -> None:
        self.gains = gains
        self.v_max = abs(v_max)
        self.dt = dt
        self.method = method

        self._i_state = 0.0
        self._prev_measurement = 0.0

        self.u_pid = 0.0
        self.u_sat = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def reset(self) -> None:
        self._i_state = 0.0
        self._prev_measurement = 0.0
        self.u_pid = 0.0
        self.u_sat = 0.0

    @staticmethod
    def _saturate(u: float, u_min: float, u_max: float) -> float:
        return max(u_min, min(u, u_max))

    def step(self, setpoint: float, measurement: float) -> float:
        error = setpoint - measurement

        self.p_term = self.gains.Kp * error
        self.d_term = -self.gains.Kd * (measurement - self._prev_measurement) / self.dt
        self.i_term = self._i_state

        self.u_pid = self.p_term + self.i_term + self.d_term
        self.u_sat = self._saturate(self.u_pid, -self.v_max, self.v_max)

        if self.method == AntiWindupMethod.NONE:
            self._i_state += self.gains.Ki * error * self.dt
        elif self.method == AntiWindupMethod.CLAMPING:
            saturated = abs(self.u_pid - self.u_sat) > 1e-12
            pushing_further = error * (self.u_pid - self.u_sat) > 0
            if not (saturated and pushing_further):
                self._i_state += self.gains.Ki * error * self.dt
        elif self.method == AntiWindupMethod.BACK_CALCULATION:
            self._i_state += (
                self.gains.Ki * error + self.gains.Kaw * (self.u_sat - self.u_pid)
            ) * self.dt

        self.i_term = self._i_state
        self._prev_measurement = measurement

        return self.u_sat
