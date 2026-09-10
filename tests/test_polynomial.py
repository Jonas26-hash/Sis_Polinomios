import cmath

import pytest

from polynomial import (
    evaluate_polynomial,
    lagrange_bound,
    multiply_by_linear,
    quadratic_roots,
    synthetic_division,
)
from theory import coefficients_at_negative_x, descartes_possibilities, sign_changes


COEFFICIENTS = [8, -6, -3, 3, -1]


def test_descartes_for_dz_and_negative_z():
    negative = coefficients_at_negative_x(COEFFICIENTS)
    assert sign_changes(COEFFICIENTS) == 3
    assert sign_changes(negative) == 1
    assert descartes_possibilities(3) == [3, 1]
    assert descartes_possibilities(1) == [1]


def test_lagrange_bound_and_ratios():
    bound, ratios = lagrange_bound(COEFFICIENTS)
    assert ratios == pytest.approx([0.75, 0.375, 0.375, 0.125])
    assert bound == pytest.approx(1.75)


def test_horner_evaluation():
    assert evaluate_polynomial(COEFFICIENTS, 0) == pytest.approx(-1)
    assert evaluate_polynomial(COEFFICIENTS, 1) == pytest.approx(1)
    assert evaluate_polynomial([1, 0, 1], 1j) == pytest.approx(0j)


def test_synthetic_division_remainder_and_degree():
    root = 0.8753608379733562
    quotient, remainder = synthetic_division(COEFFICIENTS, root)
    assert len(quotient) == len(COEFFICIENTS) - 1
    assert remainder == pytest.approx(evaluate_polynomial(COEFFICIENTS, root))
    reconstructed = multiply_by_linear(quotient, root)
    reconstructed[-1] += remainder
    assert reconstructed == pytest.approx(COEFFICIENTS, abs=1e-11)


def test_quadratic_formula_supports_complex_roots():
    roots = quadratic_roots([1, 0, 1])
    assert sorted(root.imag for root in roots) == pytest.approx([-1, 1])
    assert all(abs(evaluate_polynomial([1, 0, 1], root)) < 1e-12 for root in roots)
