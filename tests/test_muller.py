import math

import pytest

from muller import MullerError, muller, solve_all_roots
from polynomial import evaluate_polynomial, multiply_by_linear


COEFFICIENTS = [8, -6, -3, 3, -1]


def test_first_iteration_is_computed_correctly():
    result = muller(COEFFICIENTS, 0, 0.5, 1, max_iterations=1, residual_tolerance=0)
    row = result.iterations[0]
    assert row["a"] == pytest.approx(2)
    assert row["b"] == pytest.approx(4)
    assert row["c"] == pytest.approx(1)
    assert row["discriminant"] == pytest.approx(math.sqrt(8))
    assert row["dx"] == pytest.approx(-0.2928932188134525)
    assert row["z_new"] == pytest.approx(0.7071067811865475)
    assert row["ea"] == pytest.approx(0.4142135623730951)


def test_denominator_with_larger_modulus_is_selected():
    row = muller(COEFFICIENTS, 0, 0.5, 1, max_iterations=1, residual_tolerance=0).iterations[0]
    assert abs(row["denominator"]) == max(abs(row["den1"]), abs(row["den2"]))


def test_complex_number_support():
    result = muller([1, 0, 1], 0, 1, 2, tolerance=1e-8)
    assert result.converged
    assert abs(result.root.imag) > 0.9
    assert abs(evaluate_polynomial([1, 0, 1], result.root)) < 1e-8


def test_tolerance_stops_iterations():
    result = muller(COEFFICIENTS, 0, 0.5, 1, tolerance=1e-5, residual_tolerance=0)
    assert result.converged
    assert result.stop_reason == "tolerancia"
    assert result.iterations[-1]["ea"] <= 1e-5


def test_maximum_iterations_is_respected():
    result = muller(COEFFICIENTS, 0, 0.5, 1, tolerance=1e-30, max_iterations=1, residual_tolerance=0)
    assert not result.converged
    assert result.stop_reason == "máximo de iteraciones"
    assert len(result.iterations) == 1


def test_zero_division_and_stagnation_are_controlled():
    with pytest.raises(MullerError, match="División por cero"):
        muller(COEFFICIENTS, 0, 0, 1)
    with pytest.raises(MullerError):
        muller([1], 0, 0.5, 1)


def test_successive_deflation_and_all_root_residuals():
    first = muller(COEFFICIENTS, 0, 0.5, 1, tolerance=1e-5)
    roots, stages = solve_all_roots(COEFFICIENTS, first, tolerance=1e-5)
    assert len(roots) == 4
    deflation_stages = [stage for stage in stages if "quotient" in stage]
    for stage in deflation_stages:
        assert len(stage["quotient"]) == len(stage["coefficients"]) - 1
        reconstructed = multiply_by_linear(stage["quotient"], stage["root"])
        reconstructed[-1] += stage["remainder"]
        assert reconstructed == pytest.approx(stage["coefficients"], abs=1e-9)
        assert abs(stage["remainder"]) < 1e-8
    assert all(abs(evaluate_polynomial(COEFFICIENTS, root)) < 1e-8 for root in roots)
