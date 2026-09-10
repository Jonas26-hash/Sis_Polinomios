"""Aplicación educativa NUMÉRICA: raíces de polinomios y estabilidad IIR."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from charts import convergence_chart, z_plane_chart
from muller import MullerError, muller, solve_all_roots
from polynomial import (
    evaluate_polynomial,
    lagrange_bound,
    synthetic_division_steps,
)
from presentation import (
    build_summary,
    format_complex,
    format_number,
    iterations_csv,
    iterations_dataframe,
    polynomial_text,
    roots_csv,
    roots_dataframe,
)
from theory import coefficient_signs, coefficients_at_negative_x, descartes_possibilities, sign_changes


COEFFICIENTS = [8, -6, -3, 3, -1]
DEFAULT_POINTS = (0.0, 0.5, 1.0)
DEFAULT_TOLERANCE = 1e-5


def run_analysis(z0: float, z1: float, z2: float, tolerance: float, max_iterations: int) -> dict[str, object]:
    """Ejecuta todo el flujo; se mantiene separada para facilitar auditoría y pruebas."""
    first = muller(COEFFICIENTS, z0, z1, z2, tolerance, max_iterations)
    if not first.converged:
        raise MullerError("Müller alcanzó el máximo de iteraciones sin converger.")
    roots, stages = solve_all_roots(COEFFICIENTS, first, tolerance, max_iterations)
    bound, ratios = lagrange_bound(COEFFICIENTS)
    negative_coefficients = coefficients_at_negative_x(COEFFICIENTS)
    positive_changes = sign_changes(COEFFICIENTS)
    negative_changes = sign_changes(negative_coefficients)
    stable = all(abs(root) < 1 for root in roots)
    return {
        "first": first, "roots": roots, "stages": stages, "bound": bound, "ratios": ratios,
        "negative_coefficients": negative_coefficients, "positive_changes": positive_changes,
        "negative_changes": negative_changes, "stable": stable,
    }


def _load_css() -> None:
    st.markdown(f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _sign_text(coefficients: list[complex | int]) -> str:
    return " &nbsp; ".join("+" if sign > 0 else "−" for sign in coefficient_signs(coefficients))


def _metric(label: str, value: str) -> None:
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)


def _theory_panel(data: dict[str, object], decimals: int) -> None:
    negative = data["negative_coefficients"]
    positive_changes = int(data["positive_changes"])
    negative_changes = int(data["negative_changes"])
    left, right = st.columns(2)
    with left:
        st.subheader("Criterio de Descartes sobre D(z)")
        st.latex(r"D(z)=8z^4-6z^3-3z^2+3z-1")
        st.markdown(f"**Coeficientes:** `{COEFFICIENTS}`")
        st.markdown(f"**Signos:** <span class='formula-box'>{_sign_text(COEFFICIENTS)}</span>", unsafe_allow_html=True)
        st.metric("Cambios de signo", positive_changes)
        st.info(f"Raíces reales positivas posibles: **{', '.join(map(str, descartes_possibilities(positive_changes)))}**.")
    with right:
        st.subheader("Criterio de Descartes sobre D(−z)")
        st.latex(r"D(-z)=8z^4+6z^3-3z^2-3z-1")
        shown_negative = [int(complex(value).real) for value in negative]
        st.markdown(f"**Coeficientes:** `{shown_negative}`")
        st.markdown(f"**Signos:** <span class='formula-box'>{_sign_text(negative)}</span>", unsafe_allow_html=True)
        st.metric("Cambios de signo", negative_changes)
        st.info(f"Raíces reales negativas posibles: **{', '.join(map(str, descartes_possibilities(negative_changes)))}**.")
    st.caption("Descartes da posibilidades: el número de raíces positivas o negativas puede disminuir en un número par; no localiza ni determina por sí solo todas las raíces.")
    st.divider()
    st.subheader("Cota Global de Lagrange")
    ratios = data["ratios"]
    st.latex(r"R=1+\max_{0\leq k<n}\left|\frac{a_k}{a_n}\right|")
    st.write(f"Coeficiente principal: **aₙ = {COEFFICIENTS[0]}**")
    st.write(f"Coeficientes restantes: **{COEFFICIENTS[1:]}**")
    ratio_text = ", ".join(format_number(float(value), decimals) for value in ratios)
    st.markdown(fr"$R = 1 + \max({ratio_text}) = \mathbf{{{float(data['bound']):.{decimals}g}}}$")
    st.success(f"Todas las raíces satisfacen la cota **|zᵢ| ≤ {float(data['bound']):.{decimals}g}**. Es una región garantizada, no el módulo de cada raíz.")


def _muller_steps(rows: list[dict[str, object]], decimals: int) -> None:
    st.write("Las primeras tres iteraciones se construyen directamente con el registro del algoritmo.")
    for row in rows[:3]:
        with st.expander(f"Iteración {row['iteration']}", expanded=int(row["iteration"]) == 1):
            pairs = [
                ("z0", row["z0"]), ("z1", row["z1"]), ("z2", row["z2"]),
                ("h0 = z1 − z0", row["h0"]), ("h1 = z2 − z1", row["h1"]),
                ("δ0", row["delta0"]), ("δ1", row["delta1"]), ("a", row["a"]),
                ("b", row["b"]), ("c = f(z2)", row["c"]),
                ("√(b² − 4ac)", row["discriminant"]), ("denominador elegido", row["denominator"]),
                ("Δz", row["dx"]), ("z nuevo", row["z_new"]),
            ]
            st.dataframe(pd.DataFrame({"Magnitud": [name for name, _ in pairs],
                                       "Valor": [format_complex(complex(value), decimals) for _, value in pairs]}),
                         hide_index=True, width="stretch")
            st.markdown(f"**Ea = {format_number(float(row['ea']), decimals)}** (error relativo estimado adimensional)")


def _deflation_panel(data: dict[str, object], decimals: int) -> None:
    stages = data["stages"]
    st.subheader("Müller + deflación sucesiva")
    st.markdown("Cada raíz calculada reduce el grado exactamente en uno. Al llegar a grado 2 se aplica la fórmula general con `cmath.sqrt`.")
    for index, stage in enumerate(stages, start=1):
        degree = int(stage["degree"])
        with st.expander(f"Etapa {index} · grado {degree}", expanded=index == 1):
            st.write(f"Polinomio: **{polynomial_text(stage['coefficients'], decimals=decimals)}**")
            if "root" in stage:
                st.write(f"Raíz usada para deflactar: **{format_complex(stage['root'], decimals)}**")
                st.write(f"Cociente: **{polynomial_text(stage['quotient'], decimals=decimals)}**")
                st.write(f"Residuo de la división: **{format_complex(stage['remainder'], decimals)}**")
            else:
                st.write(f"Método final: **{stage['method']}**")
                st.write("Raíces: " + ", ".join(format_complex(root, decimals) for root in stage["roots"]))
    first_root = data["first"].root
    steps, quotient, remainder = synthetic_division_steps(COEFFICIENTS, first_root)
    st.subheader("Esquema de Horner — primera deflación")
    st.write("Coeficientes originales:", COEFFICIENTS)
    st.dataframe(pd.DataFrame([{
        "Paso": step["paso"], "Coeficiente que baja": format_complex(step["coeficiente"], decimals),
        "Producto anterior × r": format_complex(step["producto"], decimals),
        "Suma": format_complex(step["resultado"], decimals),
    } for step in steps]), hide_index=True, width="stretch")
    st.write("Coeficientes del cociente:", [format_complex(value, decimals) for value in quotient])
    st.write("Residuo:", format_complex(remainder, decimals))


def main() -> None:
    st.set_page_config(page_title="NUMÉRICA — Raíces de Polinomios", layout="wide")
    _load_css()
    with st.sidebar:
        st.markdown('<div class="brand">NUMÉRICA</div>', unsafe_allow_html=True)
        st.caption("Laboratorio de ingeniería")
        st.divider()
        st.markdown("**Polinomio**")
        st.latex(r"8z^4-6z^3-3z^2+3z-1")
        st.markdown("**Puntos iniciales**")
        st.info("Valores indicados por la actividad: 0, 0.5 y 1.0")
        z0 = st.number_input("z0", value=DEFAULT_POINTS[0], step=0.1, format="%.6f")
        z1 = st.number_input("z1", value=DEFAULT_POINTS[1], step=0.1, format="%.6f")
        z2 = st.number_input("z2", value=DEFAULT_POINTS[2], step=0.1, format="%.6f")
        tolerance = st.number_input("Tolerancia", value=DEFAULT_TOLERANCE, min_value=1e-12, max_value=0.1, format="%.8f")
        max_iterations = st.number_input("Máximo de iteraciones", value=50, min_value=1, max_value=500, step=1)
        decimals = st.selectbox("Decimales visibles", [4, 6, 8, 10], index=1)
        execute = st.button("▶ Ejecutar análisis", type="primary", width="stretch")

    st.markdown('<div class="hero"><div class="brand">NUMÉRICA · LABORATORIO DE INGENIERÍA</div><h1>Métodos Numéricos — Raíces de Polinomios</h1><p>Método de Müller · Deflación · Estabilidad de un filtro IIR</p></div>', unsafe_allow_html=True)
    if execute:
        try:
            st.session_state.analysis = run_analysis(z0, z1, z2, tolerance, int(max_iterations))
            st.session_state.parameters = (z0, z1, z2, tolerance, int(max_iterations))
        except (ValueError, ZeroDivisionError, MullerError) as error:
            st.error(f"No fue posible completar el análisis: {error}")
            return
    if "analysis" not in st.session_state:
        st.info("Configura los parámetros en la barra lateral y pulsa **▶ Ejecutar análisis**. Los valores de la actividad ya están cargados.")
        return

    data = st.session_state.analysis
    used_tolerance = st.session_state.parameters[3]
    first = data["first"]
    roots = data["roots"]
    stable = bool(data["stable"])
    columns = st.columns(6)
    metric_values = [("MÉTODO", "Müller"), ("TOLERANCIA", f"{used_tolerance:.0e}"),
                     ("ITERACIONES", str(len(first.iterations))), ("PRIMERA RAÍZ", format_complex(first.root, decimals)),
                     ("RAÍCES TOTALES", str(len(roots))), ("ESTABILIDAD", "ESTABLE" if stable else "INESTABLE")]
    for column, (label, value) in zip(columns, metric_values):
        with column:
            _metric(label, value)

    tabs = st.tabs(["📘 Problema", "📐 Análisis teórico", "🧮 Müller", "📊 Iteraciones", "🔻 Deflación", "⭕ Plano Z", "💡 Interpretación"])
    with tabs[0]:
        st.subheader("Estabilidad de un filtro digital IIR")
        st.write("El denominador característico es:")
        st.latex(r"D(z)=8z^4-6z^3-3z^2+3z-1=0")
        st.markdown("El filtro es estable si y solo si **todas** sus raíces (polos) cumplen $|z_i|<1$.")
        st.warning("Una sola raíz sobre o fuera del círculo unitario implica inestabilidad.")
    with tabs[1]:
        _theory_panel(data, decimals)
    with tabs[2]:
        st.subheader("Método de Müller desde cero")
        st.latex(r"h_0=x_1-x_0,\quad h_1=x_2-x_1,\quad \delta_k=\frac{f(x_{k+1})-f(x_k)}{h_k}")
        st.latex(r"a=\frac{\delta_1-\delta_0}{h_1+h_0},\quad b=ah_1+\delta_1,\quad \Delta x=\frac{-2c}{b\pm\sqrt{b^2-4ac}}")
        st.caption("Se elige el denominador de mayor módulo para reducir cancelación numérica. La raíz compleja se calcula con cmath.sqrt.")
        _muller_steps(first.iterations, decimals)
        st.success(f"Primera raíz: **{format_complex(first.root, decimals)}** · parada por **{first.stop_reason}** · {len(first.iterations)} iteraciones.")
    with tabs[3]:
        st.subheader("Tabla completa de iteraciones")
        st.caption("Ea es un error relativo estimado adimensional: |(z_nuevo − z_anterior) / z_nuevo|. No está expresado en porcentaje.")
        st.dataframe(iterations_dataframe(first.iterations, decimals), hide_index=True, width="stretch")
        st.plotly_chart(convergence_chart(first.iterations, used_tolerance), width="stretch")
        st.download_button("⬇ Descargar iteraciones CSV", iterations_csv(first.iterations), "iteraciones_muller.csv", "text/csv")
    with tabs[4]:
        _deflation_panel(data, decimals)
    with tabs[5]:
        st.subheader("Raíces en el plano complejo Z")
        st.plotly_chart(z_plane_chart(roots, float(data["bound"])), width="stretch")
        st.dataframe(roots_dataframe(roots, COEFFICIENTS, decimals), hide_index=True, width="stretch")
        st.download_button("⬇ Descargar raíces CSV", roots_csv(roots, COEFFICIENTS), "raices_polinomio.csv", "text/csv")
    with tabs[6]:
        st.subheader("Conclusión de ingeniería")
        st.dataframe(roots_dataframe(roots, COEFFICIENTS, decimals)[["Raíz", "Valor", "|z|", "|z| < 1"]], hide_index=True, width="stretch")
        css_class = "result-ok" if stable else "result-bad"
        icon = "✅" if stable else "❌"
        st.markdown(f'<div class="{css_class}">{icon} FILTRO {"ESTABLE" if stable else "INESTABLE"}</div>', unsafe_allow_html=True)
        qualifier = "todas" if stable else "no todas"
        st.write(f"Se obtuvieron cuatro raíces del polinomio característico. Como **{qualifier}** cumplen $|z|<1$, el filtro es **{'estable' if stable else 'inestable'}**.")
        st.caption("El filtro IIR es estable únicamente si todos sus polos se encuentran estrictamente dentro del círculo unitario.")
        summary = build_summary(COEFFICIENTS, int(data["positive_changes"]), descartes_possibilities(int(data["positive_changes"])),
                                int(data["negative_changes"]), descartes_possibilities(int(data["negative_changes"])),
                                float(data["bound"]), first.root, len(first.iterations), used_tolerance, roots, stable)
        if st.button("📋 Copiar resumen"):
            st.code(summary, language=None)
            st.caption("Usa el icono de copia del bloque para llevar el resumen al portapapeles.")


if __name__ == "__main__":
    main()
