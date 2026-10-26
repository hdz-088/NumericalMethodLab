"""
test_solvers.py - Comprehensive Unit & Mathematical Correctness Test Suite
==========================================================================
Verifies all four numerical methods, error analysis routines, expression parsing,
and edge case diagnostics according to Section 6 of the project specifications:

1. f(x) = x^3 - x - 2 = 0 (root ~ 1.5213797)
2. f(x) = x^2 - 4 = 0 (root = 2.0)
3. f(x) = cos(x) - x = 0 (root ~ 0.7390851)
4. f(x) = exp(-x) - x = 0 (root ~ 0.5671433)
5. Non-bracketing interval rejection (f(a)*f(b) > 0)
6. Newton-Raphson zero derivative handling (f'(x0) == 0)
7. Secant method equal initial function values (f(x0) == f(x1))
8. Safe parser syntax validation & disallowed token rejection
9. Maximum iteration count boundary handling
10. Undefined/non-finite function domain errors
"""

import pytest
import math
import numpy as np

from solvers import (
    bisection_method,
    regula_falsi_method,
    newton_raphson_method,
    secant_method,
    SolverResult,
)
from function_parser import (
    parse_math_expression,
    build_numerical_function,
    build_vectorized_function,
    validate_function_domain,
    ExpressionParserError,
    EvaluationDomainError,
)
from error_analysis import (
    compute_approximate_errors,
    compute_residual,
    compute_true_errors,
    evaluate_stopping_criterion,
)


# ==============================================================================
# 1. Benchmark Problem 1: f(x) = x^3 - x - 2 = 0 (root ~ 1.5213797)
# ==============================================================================
def test_case_1_cubic_polynomial():
    """Verify all four methods solve x^3 - x - 2 = 0 to high precision."""
    expr, x, d_expr, _ = parse_math_expression("x**3 - x - 2")
    f = build_numerical_function(expr, x)
    f_prime = build_numerical_function(d_expr, x)
    expected_root = 1.5213797068
    tol = 1e-5

    # 1. Bisection
    res_bis = bisection_method(f, 1.0, 2.0, tol=tol, max_iter=60)
    assert res_bis.converged is True
    assert res_bis.final_root is not None
    assert abs(res_bis.final_root - expected_root) < 1e-4
    assert res_bis.absolute_residual < 1e-3
    assert res_bis.iteration_count > 0
    assert len(res_bis.iteration_history) == res_bis.iteration_count

    # 2. Regula-Falsi
    res_rf = regula_falsi_method(f, 1.0, 2.0, tol=tol, max_iter=60)
    assert res_rf.converged is True
    assert res_rf.final_root is not None
    assert abs(res_rf.final_root - expected_root) < 1e-4

    # 3. Newton-Raphson
    res_nr = newton_raphson_method(f, f_prime, x0=1.5, tol=tol, max_iter=30)
    assert res_nr.converged is True
    assert res_nr.final_root is not None
    assert abs(res_nr.final_root - expected_root) < 1e-5
    assert res_nr.absolute_residual < 1e-5
    # Newton should converge in very few iterations (< 8)
    assert res_nr.iteration_count <= 8

    # 4. Secant
    res_sec = secant_method(f, x0=1.0, x1=2.0, tol=tol, max_iter=30)
    assert res_sec.converged is True
    assert res_sec.final_root is not None
    assert abs(res_sec.final_root - expected_root) < 1e-4


# ==============================================================================
# 2. Benchmark Problem 2: f(x) = x^2 - 4 = 0 (root = 2.0)
# ==============================================================================
def test_case_2_quadratic():
    """Verify all four methods solve x^2 - 4 = 0."""
    expr, x, d_expr, _ = parse_math_expression("x**2 - 4")
    f = build_numerical_function(expr, x)
    f_prime = build_numerical_function(d_expr, x)
    tol = 1e-6

    res_bis = bisection_method(f, 1.0, 3.0, tol=tol)
    assert res_bis.converged is True
    assert abs(res_bis.final_root - 2.0) < 1e-5

    res_rf = regula_falsi_method(f, 1.0, 3.0, tol=tol)
    assert res_rf.converged is True
    assert abs(res_rf.final_root - 2.0) < 1e-5

    res_nr = newton_raphson_method(f, f_prime, x0=3.0, tol=tol)
    assert res_nr.converged is True
    assert abs(res_nr.final_root - 2.0) < 1e-6

    res_sec = secant_method(f, x0=1.0, x1=3.0, tol=tol)
    assert res_sec.converged is True
    assert abs(res_sec.final_root - 2.0) < 1e-5


# ==============================================================================
# 3. Benchmark Problem 3: f(x) = cos(x) - x = 0 (Dottie number ~ 0.73908513)
# ==============================================================================
def test_case_3_trigonometric():
    """Verify transcendental equation cos(x) - x = 0 with radian angle domain."""
    expr, x, d_expr, _ = parse_math_expression("cos(x) - x")
    f = build_numerical_function(expr, x)
    f_prime = build_numerical_function(d_expr, x)
    expected_root = 0.7390851332
    tol = 1e-6

    res_bis = bisection_method(f, 0.0, 1.0, tol=tol)
    assert res_bis.converged is True
    assert abs(res_bis.final_root - expected_root) < 1e-5

    res_rf = regula_falsi_method(f, 0.0, 1.0, tol=tol)
    assert res_rf.converged is True
    assert abs(res_rf.final_root - expected_root) < 1e-5

    res_nr = newton_raphson_method(f, f_prime, x0=0.5, tol=tol)
    assert res_nr.converged is True
    assert abs(res_nr.final_root - expected_root) < 1e-6

    res_sec = secant_method(f, x0=0.0, x1=1.0, tol=tol)
    assert res_sec.converged is True
    assert abs(res_sec.final_root - expected_root) < 1e-6


# ==============================================================================
# 4. Benchmark Problem 4: f(x) = exp(-x) - x = 0 (root ~ 0.56714329)
# ==============================================================================
def test_case_4_exponential():
    """Verify exponential transcendental equation exp(-x) - x = 0."""
    expr, x, d_expr, _ = parse_math_expression("exp(-x) - x")
    f = build_numerical_function(expr, x)
    f_prime = build_numerical_function(d_expr, x)
    expected_root = 0.5671432904
    tol = 1e-6

    res_bis = bisection_method(f, 0.0, 1.0, tol=tol)
    assert res_bis.converged is True
    assert abs(res_bis.final_root - expected_root) < 1e-5

    res_rf = regula_falsi_method(f, 0.0, 1.0, tol=tol)
    assert res_rf.converged is True
    assert abs(res_rf.final_root - expected_root) < 1e-5

    res_nr = newton_raphson_method(f, f_prime, x0=0.5, tol=tol)
    assert res_nr.converged is True
    assert abs(res_nr.final_root - expected_root) < 1e-6

    res_sec = secant_method(f, x0=0.0, x1=1.0, tol=tol)
    assert res_sec.converged is True
    assert abs(res_sec.final_root - expected_root) < 1e-6


# ==============================================================================
# 5. Non-Bracketing Interval (f(a) * f(b) > 0)
# ==============================================================================
def test_case_5_non_bracketing_interval():
    """Verify Bisection and Regula-Falsi reject intervals that do not bracket a root."""
    expr, x, _, _ = parse_math_expression("x**2 - 4")
    f = build_numerical_function(expr, x)

    # In [3.0, 5.0], f(3) = 5 > 0 and f(5) = 21 > 0
    res_bis = bisection_method(f, 3.0, 5.0)
    assert res_bis.converged is False
    assert res_bis.final_root is None
    assert "Bracketing condition violated" in res_bis.warning_message
    assert res_bis.iteration_count == 0

    res_rf = regula_falsi_method(f, 3.0, 5.0)
    assert res_rf.converged is False
    assert res_rf.final_root is None
    assert "Bracketing condition violated" in res_rf.warning_message


# ==============================================================================
# 6. Newton-Raphson Zero Derivative (f'(x) == 0)
# ==============================================================================
def test_case_6_newton_zero_derivative():
    """Verify Newton-Raphson safely detects f'(x0) = 0 and does not crash."""
    expr, x, d_expr, _ = parse_math_expression("x**2 - 4")
    f = build_numerical_function(expr, x)
    f_prime = build_numerical_function(d_expr, x)

    # At x0 = 0.0, f'(0) = 2*(0) = 0
    res = newton_raphson_method(f, f_prime, x0=0.0)
    assert res.converged is False
    assert "Zero or near-zero derivative encountered" in res.stopping_reason
    assert "tangent line is horizontal" in res.warning_message.lower()
    assert res.iteration_count == 1


# ==============================================================================
# 7. Secant Equal Initial Function Values (f(x0) == f(x1))
# ==============================================================================
def test_case_7_secant_equal_initial_values():
    """Verify Secant Method detects equal function values at x0 and x1."""
    expr, x, _, _ = parse_math_expression("x**2 - 4")
    f = build_numerical_function(expr, x)

    # For f(x) = x^2 - 4, f(-2) = 0, but let's test f(-3) = 5 and f(3) = 5
    res = secant_method(f, x0=-3.0, x1=3.0)
    assert res.converged is False
    assert "Equal initial function values" in res.stopping_reason
    assert "Secant line is horizontal" in res.warning_message
    assert res.iteration_count == 0


# ==============================================================================
# 8. Expression Parsing: Syntax Validation & Disallowed Tokens
# ==============================================================================
def test_case_8_invalid_equation_syntax():
    """Verify parser rejects malformed equations and security risks."""
    # Syntax error
    with pytest.raises(ExpressionParserError):
        parse_math_expression("x** + ")

    with pytest.raises(ExpressionParserError):
        parse_math_expression("x +* 2")

    # Empty string
    with pytest.raises(ExpressionParserError):
        parse_math_expression("   ")

    # Multiple variables
    with pytest.raises(ExpressionParserError) as exc_info:
        parse_math_expression("x**2 + y - 4")
    assert "Disallowed free variable 'y'" in str(exc_info.value)

    # Disallowed system commands
    with pytest.raises(ExpressionParserError):
        parse_math_expression("__import__('os').system('dir')")


# ==============================================================================
# 9. Maximum Iteration Limit Reached Before Convergence
# ==============================================================================
def test_case_9_max_iterations_reached():
    """Verify solver flags non-convergence when max iterations is reached."""
    expr, x, _, _ = parse_math_expression("x**3 - x - 2")
    f = build_numerical_function(expr, x)

    # Request impossible tolerance in 2 iterations
    res = bisection_method(f, 1.0, 2.0, tol=1e-15, max_iter=2)
    assert res.converged is False
    assert res.iteration_count == 2
    assert "Maximum iterations reached" in res.stopping_reason


# ==============================================================================
# 10. Undefined / Non-Finite Function Domain Handling
# ==============================================================================
def test_case_10_undefined_domain_handling():
    """Verify graceful handling when function produces math domain errors or complex numbers."""
    expr, x, _, _ = parse_math_expression("log(x)")
    f = build_numerical_function(expr, x)

    # log(-1.0) is not real
    with pytest.raises(EvaluationDomainError):
        f(-1.0)

    # Solver should catch domain error and return structured failure result
    res = bisection_method(f, -2.0, -0.5)
    assert res.converged is False
    assert "Endpoint evaluation failure" in res.stopping_reason
    assert res.final_root is None


# ==============================================================================
# 11. Additional Quality Tests: Telemetry & DataFrame Conversion
# ==============================================================================
def test_iteration_history_telemetry_structure():
    """Verify that iteration history contains all required non-null fields and converts to DataFrame."""
    expr, x, _, _ = parse_math_expression("x**3 - 2*x - 5")
    f = build_numerical_function(expr, x)
    res = bisection_method(f, 2.0, 3.0, tol=1e-5)

    df = res.to_dataframe()
    assert not df.empty
    assert "Iteration" in df.columns
    assert "Root Estimate (x_k)" in df.columns
    assert "f(x_k)" in df.columns
    assert "Approx Abs Error (|x_k - x_{k-1}|)" in df.columns
    assert len(df) == res.iteration_count


def test_error_metrics_and_stopping_criteria():
    """Verify mathematical error computations and stopping conditions."""
    # Absolute error
    e_abs, e_rel, e_pct = compute_approximate_errors(2.0, 2.1)
    assert pytest.approx(e_abs, 1e-9) == 0.1
    assert pytest.approx(e_rel, 1e-9) == 0.1 / 2.0
    assert pytest.approx(e_pct, 1e-9) == 5.0

    # Residual
    assert compute_residual(-0.005) == 0.005

    # Stopping evaluation: residual
    stop, reason = evaluate_stopping_criterion("residual", 1e-4, e_abs=0.01, e_rel=0.01, residual=1e-5)
    assert stop is True
    assert "Residual" in reason

    # Stopping evaluation: interval width
    stop, reason = evaluate_stopping_criterion(
        "interval_width", 1e-3, e_abs=0.01, e_rel=0.01, residual=0.1, interval_width=5e-4
    )
    assert stop is True
    assert "Interval width" in reason
