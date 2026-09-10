"""Herramientas para el análisis teórico de polinomios."""

from __future__ import annotations

from typing import Sequence


def coefficient_signs(coefficients: Sequence[complex], tolerance: float = 1e-14) -> list[int]:
    """Devuelve signos (+1/-1), ignorando ceros, para coeficientes reales."""
    signs: list[int] = []
    for coefficient in coefficients:
        value = complex(coefficient)
        if abs(value.imag) > tolerance:
            raise ValueError("El criterio de Descartes requiere coeficientes reales.")
        if abs(value.real) <= tolerance:
            continue
        signs.append(1 if value.real > 0 else -1)
    return signs


def sign_changes(coefficients: Sequence[complex]) -> int:
    signs = coefficient_signs(coefficients)
    return sum(left != right for left, right in zip(signs, signs[1:]))


def coefficients_at_negative_x(coefficients: Sequence[complex]) -> list[complex]:
    """Construye P(-z), cambiando términos de grado impar."""
    degree = len(coefficients) - 1
    return [complex(value) * (-1 if (degree - index) % 2 else 1) for index, value in enumerate(coefficients)]


def descartes_possibilities(changes: int) -> list[int]:
    if changes < 0:
        raise ValueError("El número de cambios no puede ser negativo.")
    return list(range(changes, -1, -2))
