"""
Dinamica da planta: motor DC acoplado a junta robotica.

Integracao numerica (Euler) com entrada de tensao e disturbio de torque.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import MotorJointParams


@dataclass
class PlantState:
    theta: float = 0.0
    omega: float = 0.0


class JointPlant:
    """Simulador da junta em tempo continuo discretizado."""

    def __init__(self, params: MotorJointParams, dt: float) -> None:
        self.params = params
        self.dt = dt
        b_eff = params.b + params.Kt * params.Ke / params.R
        self._b_eff = b_eff
        self._b2 = params.Kt / (params.R * params.J)
        self._td_gain = -1.0 / params.J
        self.state = PlantState()

    def reset(self, theta: float = 0.0, omega: float = 0.0) -> None:
        self.state = PlantState(theta=theta, omega=omega)

    def step(self, voltage: float, disturbance_torque: float = 0.0) -> PlantState:
        theta, omega = self.state.theta, self.state.omega
        dtheta = omega
        domega = (
            -self._b_eff / self.params.J * omega
            + self._b2 * voltage
            + self._td_gain * disturbance_torque
        )
        self.state = PlantState(
            theta=theta + self.dt * dtheta,
            omega=omega + self.dt * domega,
        )
        return self.state
