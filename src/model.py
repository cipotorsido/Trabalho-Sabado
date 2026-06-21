"""
Modelagem matemática: motor DC acoplado à junta de braço robótico.

Equações em tempo contínuo
--------------------------
Elétrica (indutância desprezada, L ≈ 0):
    V(t) = R·i(t) + Ke·ω(t)

Mecânica:
    J·dω/dt + b·ω = Kt·i(t) - Td(t)

Cinemática:
    dθ/dt = ω

Substituindo i = (V - Ke·ω)/R e eliminando variáveis intermediárias:

    (J·s + b + Kt·Ke/R)·Ω(s) = (Kt/R)·V(s) - (1/J_eff)·Td(s)

Com Td = 0 (sem distúrbio de carga na FT principal):

    G(s) = Θ(s)/V(s) = (Kt/R) / [s·(J·s + b + Kt·Ke/R)]

Forma padrão de 2ª ordem (posição/tensão):
    G(s) = K / [s·(τ_m·s + 1)]

onde:
    K  = Kt / (R·(b + Kt·Ke/R))   [rad/(V·s)]
    τ_m = J / (b + Kt·Ke/R)        [s]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from control import tf, step_response, TransferFunction


@dataclass(frozen=True)
class MotorJointParams:
    """Parâmetros físicos do motor DC + junta robótica."""

    J: float = 0.01       # Inércia total [kg·m²]
    b: float = 0.005      # Atrito viscoso [N·m·s/rad]
    Kt: float = 0.05      # Constante de torque [N·m/A]
    Ke: float = 0.05      # Constante de back-EMF [V·s/rad] (Ke ≈ Kt em SI)
    R: float = 2.0        # Resistência elétrica [Ω]
    V_max: float = 4.5   # Tensao de saturacao [V] — evidencia windup com Ki alto


def mechanical_time_constant(p: MotorJointParams) -> float:
    """Constante de tempo mecânica τ_m = J / (b + Kt·Ke/R)."""
    return p.J / (p.b + p.Kt * p.Ke / p.R)


def dc_gain_position(p: MotorJointParams) -> float:
    """Ganho DC posição/tensão K [rad/(V·s)]."""
    denom = p.b + p.Kt * p.Ke / p.R
    return p.Kt / (p.R * denom)


def transfer_function_voltage_to_position(p: MotorJointParams) -> TransferFunction:
    """
    G(s) = Θ(s)/V(s) = (Kt/R) / [s·(J·s + b + Kt·Ke/R)].

    Retorna objeto python-control para análise clássica.
    """
    num = [p.Kt / p.R]
    den = [p.J, p.b + p.Kt * p.Ke / p.R, 0.0]
    return tf(num, den)


def state_space_matrices(p: MotorJointParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Representação em espaço de estados (forma controlável).

    x = [θ, ω]ᵀ,  u = V,  y = θ

        dx/dt = A·x + B·u + E·Td
        y     = C·x

    Sem distúrbio: Td = 0.
    """
    b_eff = p.b + p.Kt * p.Ke / p.R
    A = np.array([[0.0, 1.0], [0.0, -b_eff / p.J]])
    B = np.array([[0.0], [p.Kt / (p.R * p.J)]])
    C = np.array([[1.0, 0.0]])
    return A, B, C


def print_model_summary(p: MotorJointParams) -> None:
    """Imprime resumo da modelagem no console."""
    tau_m = mechanical_time_constant(p)
    K = dc_gain_position(p)
    G = transfer_function_voltage_to_position(p)

    print("=" * 60)
    print("ETAPA 1 - Modelagem Matematica do Sistema")
    print("=" * 60)
    print(f"  J  (inercia)           = {p.J:.4f} kg*m^2")
    print(f"  b  (atrito viscoso)    = {p.b:.4f} N*m*s/rad")
    print(f"  Kt (const. torque)     = {p.Kt:.4f} N*m/A")
    print(f"  Ke (back-EMF)          = {p.Ke:.4f} V*s/rad")
    print(f"  R  (resistencia)       = {p.R:.4f} Ohm")
    print(f"  V_max (saturacao)      = {p.V_max:.1f} V")
    print("-" * 60)
    print(f"  tau_m (const. tempo)   = {tau_m:.4f} s")
    print(f"  K   (ganho DC theta/V) = {K:.4f} rad/(V*s)")
    print("-" * 60)
    print("  Funcao de Transferencia (Td = 0):")
    print(f"  G(s) = Theta(s)/V(s) = {G}")
    print("=" * 60)
