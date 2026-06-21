"""
Metricas de desempenho para a Mesa Redonda comparativa.
"""

from __future__ import annotations

import numpy as np


def compute_performance_metrics(
    t: np.ndarray,
    theta: np.ndarray,
    setpoint: np.ndarray,
    u_sat: np.ndarray,
    v_max: float,
) -> dict[str, float]:
    """
    Calcula indicadores para tabela comparativa da turma.

    - Tempo de assentamento (2%)
    - Sobresinal (%)
    - Esforco do atuador (RMS, pico, % tempo saturado)
    - Erro estacionario medio
    """
    error = setpoint - theta
    sp_final = float(setpoint[-1])
    sp_ref = abs(sp_final) if abs(sp_final) > 1e-9 else 1.0

    overshoot = 0.0
    if sp_final > 0:
        peak = float(np.max(theta))
        overshoot = max(0.0, (peak - sp_final) / sp_ref * 100.0)

    tol = 0.02 * sp_ref
    inside = np.abs(error) <= tol
    settling_time = float("nan")
    if np.any(inside):
        idx_enter = int(np.where(inside)[0][0])
        after = np.where(~inside[idx_enter:])[0]
        if len(after) == 0:
            settling_time = float(t[idx_enter])
        else:
            last_exit = idx_enter + int(after[-1])
            if last_exit < len(t) - 1:
                rest = inside[last_exit + 1 :]
                if np.all(rest):
                    settling_time = float(t[last_exit + 1])
            else:
                settling_time = float(t[idx_enter])

    steady_mask = t >= 0.85 * t[-1]
    sse = float(np.mean(np.abs(error[steady_mask])))

    u_rms = float(np.sqrt(np.mean(u_sat**2)))
    u_peak = float(np.max(np.abs(u_sat)))
    sat_mask = np.abs(u_sat) >= 0.98 * abs(v_max)
    sat_pct = float(np.mean(sat_mask) * 100.0)

    return {
        "settling_time_s": settling_time,
        "overshoot_pct": overshoot,
        "steady_state_error_rad": sse,
        "actuator_rms_v": u_rms,
        "actuator_peak_v": u_peak,
        "saturation_time_pct": sat_pct,
    }
