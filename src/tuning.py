"""
Perfis de sintonia PID para malha fechada.

CLOSED_LOOP  — rastreamento estavel da referencia (entregaveis / demo / relatorio)
WINDUP_DEMO  — Ki alto + saturacao (comparacao anti-windup apenas)
"""

from __future__ import annotations

from dataclasses import dataclass

from pid_anti_windup import PIDGains


@dataclass(frozen=True)
class TuningProfile:
    name: str
    gains: PIDGains
    description: str


# Sintonia refinada via LR / margens (MF ~ 30 deg) — malha fechada estavel
CLOSED_LOOP = TuningProfile(
    name="malha_fechada",
    gains=PIDGains(Kp=1.5, Ki=0.5, Kd=0.3, Kaw=1.0),
    description="PID malha fechada — referencia rastreada com realimentacao de theta",
)

# Ganhos agressivos para evidenciar windup integral (somente figura comparativa)
WINDUP_DEMO = TuningProfile(
    name="demo_windup",
    gains=PIDGains(Kp=4.0, Ki=6.0, Kd=0.15, Kaw=10.0),
    description="Ki alto com V_max baixo — destaca anti-windup",
)
