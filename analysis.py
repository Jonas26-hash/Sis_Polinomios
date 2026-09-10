"""Orquestación compartida del análisis numérico."""

from __future__ import annotations

from muller import MullerError, muller, solve_all_roots
from polynomial import lagrange_bound
from theory import coefficients_at_negative_x, sign_changes


COEFFICIENTS = [8, -6, -3, 3, -1]
DEFAULT_POINTS = (0.0, 0.5, 1.0)
DEFAULT_TOLERANCE = 1e-5


def run_analysis(
    z0: float = DEFAULT_POINTS[0],
    z1: float = DEFAULT_POINTS[1],
    z2: float = DEFAULT_POINTS[2],
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = 50,
) -> dict[str, object]:
    """Ejecuta Müller, deflación, verificación teórica y estabilidad."""
    first = muller(COEFFICIENTS, z0, z1, z2, tolerance, max_iterations)
    if not first.converged:
        raise MullerError("Müller alcanzó el máximo de iteraciones sin converger.")
    roots, stages = solve_all_roots(COEFFICIENTS, first, tolerance, max_iterations)
    bound, ratios = lagrange_bound(COEFFICIENTS)
    negative_coefficients = coefficients_at_negative_x(COEFFICIENTS)
    positive_changes = sign_changes(COEFFICIENTS)
    negative_changes = sign_changes(negative_coefficients)
    return {
        "first": first,
        "roots": roots,
        "stages": stages,
        "bound": bound,
        "ratios": ratios,
        "negative_coefficients": negative_coefficients,
        "positive_changes": positive_changes,
        "negative_changes": negative_changes,
        "stable": all(abs(root) < 1 for root in roots),
    }
