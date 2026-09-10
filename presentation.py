"""Formateo y preparación de datos para la capa Streamlit."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import pandas as pd

from polynomial import evaluate_polynomial


def format_complex(value: complex, decimals: int = 6) -> str:
    value = complex(value)
    threshold = 0.5 * 10 ** (-decimals)
    real = 0.0 if abs(value.real) < threshold else value.real
    imag = 0.0 if abs(value.imag) < threshold else value.imag
    if imag == 0:
        return f"{real:.{decimals}f}"
    sign = "+" if imag >= 0 else "−"
    return f"{real:.{decimals}f} {sign} {abs(imag):.{decimals}f}j"


def format_number(value: float, decimals: int = 6) -> str:
    if math.isinf(value):
        return "∞"
    return f"{value:.{decimals}g}"


def polynomial_text(coefficients: Sequence[complex], variable: str = "z", decimals: int = 6) -> str:
    degree = len(coefficients) - 1
    terms: list[str] = []
    for index, raw in enumerate(coefficients):
        coefficient = complex(raw)
        power = degree - index
        if abs(coefficient) < 1e-14:
            continue
        if abs(coefficient.imag) < 1e-12:
            number = f"{abs(coefficient.real):.{decimals}g}"
            sign = "−" if coefficient.real < 0 else "+"
        else:
            number = f"({format_complex(coefficient, decimals)})"
            sign = "+"
        factor = "" if number == "1" and power > 0 else number
        term = factor if power == 0 else f"{factor}{variable}" + (f"^{power}" if power > 1 else "")
        if not terms:
            terms.append(("−" if sign == "−" else "") + term)
        else:
            terms.append(f" {sign} {term}")
    return "".join(terms) or "0"


def iterations_dataframe(rows: list[dict[str, object]], decimals: int) -> pd.DataFrame:
    labels = {
        "iteration": "Iteración", "z0": "z0", "z1": "z1", "z2": "z2",
        "f_z2": "f(z2)", "a": "a", "b": "b", "c": "c",
        "discriminant": "Discriminante", "dx": "Δz", "z_new": "z nuevo", "ea": "Ea",
    }
    records = []
    for row in rows:
        record: dict[str, object] = {}
        for key, label in labels.items():
            value = row[key]
            if key == "iteration":
                record[label] = int(value)  # type: ignore[arg-type]
            elif key == "ea":
                record[label] = format_number(float(value), decimals)
            else:
                record[label] = format_complex(complex(value), decimals)
        records.append(record)
    return pd.DataFrame(records)


def iterations_csv(rows: list[dict[str, object]]) -> str:
    records = []
    keys = ["iteration", "z0", "z1", "z2", "f_z2", "a", "b", "c", "discriminant", "dx", "z_new", "ea", "residual"]
    for row in rows:
        records.append({key: _precise(row[key]) for key in keys})
    return pd.DataFrame(records).to_csv(index=False)


def _precise(value: object) -> object:
    if isinstance(value, complex):
        return f"{value.real:.16g}{value.imag:+.16g}j"
    if isinstance(value, float):
        return f"{value:.16g}"
    return value


def roots_dataframe(roots: Iterable[complex], original: Sequence[complex], decimals: int) -> pd.DataFrame:
    records = []
    for index, root in enumerate(roots, start=1):
        residual = abs(evaluate_polynomial(original, root))
        records.append({
            "Raíz": f"z{index}", "Valor": format_complex(root, decimals),
            "Parte real": round(root.real, decimals), "Parte imaginaria": round(root.imag, decimals),
            "|D(raíz)|": f"{residual:.3e}", "|z|": round(abs(root), decimals),
            "|z| < 1": "Sí" if abs(root) < 1 else "No",
        })
    return pd.DataFrame(records)


def roots_csv(roots: Iterable[complex], original: Sequence[complex]) -> str:
    rows = []
    for index, root in enumerate(roots, start=1):
        rows.append({"root": f"z{index}", "real": f"{root.real:.16g}", "imag": f"{root.imag:.16g}",
                     "residual": f"{abs(evaluate_polynomial(original, root)):.16g}",
                     "modulus": f"{abs(root):.16g}", "inside_unit_circle": abs(root) < 1})
    return pd.DataFrame(rows).to_csv(index=False)


def build_summary(
    coefficients: Sequence[complex], positive_changes: int, positive_options: list[int],
    negative_changes: int, negative_options: list[int], bound: float, first_root: complex,
    iteration_count: int, tolerance: float, roots: Sequence[complex], stable: bool,
) -> str:
    root_lines = "\n".join(f"  z{i} = {root.real:.12g}{root.imag:+.12g}j; |z{i}| = {abs(root):.12g}"
                           for i, root in enumerate(roots, start=1))
    return f"""NUMÉRICA — Raíces de Polinomios
Polinomio: {polynomial_text(coefficients)}
Descartes D(z): {positive_changes} cambios; raíces positivas posibles: {positive_options}
Descartes D(-z): {negative_changes} cambios; raíces negativas posibles: {negative_options}
Cota Global de Lagrange: |z_i| ≤ {bound:.12g}
Primera raíz por Müller: {first_root.real:.12g}{first_root.imag:+.12g}j
Iteraciones: {iteration_count}
Tolerancia: {tolerance:.12g} (error relativo adimensional)
Raíces finales:
{root_lines}
Estabilidad: {'ESTABLE' if stable else 'INESTABLE'}
"""
