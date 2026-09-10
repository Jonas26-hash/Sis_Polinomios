# NUMÉRICA — Raíces de Polinomios

Aplicación educativa y profesional de la **Sesión 4 de Métodos Numéricos**. Analiza la estabilidad de un filtro IIR a partir de las raíces del polinomio característico:

\[
D(z)=8z^4-6z^3-3z^2+3z-1.
\]

El recorrido incluye el criterio de Descartes, la Cota Global de Lagrange, el método de Müller, Horner y deflación sucesiva, verificación por sustitución y análisis del círculo unitario.

## Instalación

Requiere Python 3.11 o posterior.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Fundamento

- **Descartes:** cuenta programáticamente los cambios de signo de `D(z)` y `D(-z)` para establecer las cantidades posibles de raíces reales positivas y negativas. No presenta esas posibilidades como resultados exactos.
- **Lagrange:** usa `R = 1 + max(|a_k/a_n|)` para producir una cota global `|z_i| ≤ R`.
- **Müller:** aproxima la primera raíz con `z0=0`, `z1=0.5`, `z2=1`, `epsilon=1e-5` y aritmética compleja. El denominador de mayor módulo reduce la cancelación numérica.
- **Horner y deflación:** evalúa el polinomio y divide sucesivamente por factores `(z-r)`. Müller obtiene raíces adicionales hasta llegar a un cuadrático, que se resuelve mediante la fórmula general implementada con `cmath.sqrt`.
- **Estabilidad IIR:** el filtro es estable únicamente cuando todas las raíces cumplen estrictamente `|z_i| < 1`.

Las raíces **no se obtienen mediante `numpy.roots` ni solucionadores externos**. NumPy está permitido únicamente como dependencia auxiliar del entorno; toda la lógica principal está implementada en el proyecto.

## Pruebas

```powershell
pytest -q
```

La suite audita Descartes, Lagrange, Horner, Müller, deflación, residuos, raíces y la coherencia de la conclusión de estabilidad.
