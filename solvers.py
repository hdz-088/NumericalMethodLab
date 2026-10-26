"""
solvers.py - Manual Implementations of the Four Fundamental Root-Finding Solvers
================================================================================
Implements:
1. Bisection Method (Bracketing)
2. Regula-Falsi Method (False Position, Bracketing)
3. Newton-Raphson Method (Open, Derivative-Based)
4. Secant Method (Open, Quasi-Newton)

Every solver manually executes the mathematical formula, captures full iteration
telemetry without relying on black-box solvers, and returns a uniform SolverResult.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any
import time
import math
import pandas as pd

from error_analysis import (
    compute_approximate_errors,
    compute_residual,
    evaluate_stopping_criterion,
)


@dataclass
class IterationRecord:
    """
    Standardized iteration telemetry record.
    Unused or inapplicable fields are strictly None (no fabricated values).
    """
    iteration: int
    current_estimate: float
    previous_estimate: Optional[float] = None
    function_value: float = 0.0
    derivative_value: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    approx_abs_error: Optional[float] = None
    approx_rel_error: Optional[float] = None
    approx_percent_error: Optional[float] = None
    interval_width: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for display and tabular export."""
        return {
            "Iteration": self.iteration,
            "Root Estimate (x_k)": self.current_estimate,
            "f(x_k)": self.function_value,
            "f'(x_k)": self.derivative_value,
            "Lower Bound (a)": self.lower_bound,
            "Upper Bound (b)": self.upper_bound,
            "Interval Width (b-a)": self.interval_width,
            "Approx Abs Error (|x_k - x_{k-1}|)": self.approx_abs_error,
            "Approx Rel Error": self.approx_rel_error,
            "Approx % Error": self.approx_percent_error,
            "Notes": self.notes or "",
        }


@dataclass
class SolverResult:
    """
    Standardized result structure returned by all four numerical solvers.
    """
    method_name: str
    equation_str: str
    final_root: Optional[float]
    final_function_value: Optional[float]
    absolute_residual: Optional[float]
    iteration_count: int
    function_evaluations: int
    converged: bool
    stopping_reason: str
    iteration_history: List[IterationRecord] = field(default_factory=list)
    stopping_criterion: str = "approx_abs_error"
    tolerance: float = 1e-6
    max_iterations: int = 100
    warning_message: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Export iteration history to a clean pandas DataFrame."""
        if not self.iteration_history:
            return pd.DataFrame()
        records = [rec.to_dict() for rec in self.iteration_history]
        df = pd.DataFrame(records)
        # Drop columns that are completely None/empty across all iterations
        return df.dropna(axis=1, how="all")


def bisection_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    stopping_criterion: str = "approx_abs_error",
    equation_str: str = "f(x) = 0",
) -> SolverResult:
    """
    Solve f(x) = 0 using the Bisection Method.

    Formula:
        c = (a + b) / 2

    Mathematical Properties:
    - Type: Bracketing Method
    - Order of convergence: Linear (p = 1), asymptotic error factor = 0.5
    - Guaranteed convergence if f is continuous and f(a)*f(b) < 0 (Intermediate Value Theorem)
    """
    start_time = time.perf_counter()
    history: List[IterationRecord] = []
    fev = 0

    # Interval normalization: ensure a < b
    if a > b:
        a, b = b, a

    # Evaluate endpoints
    try:
        fa = f(a)
        fb = f(b)
        fev += 2
    except Exception as err:
        return SolverResult(
            method_name="Bisection Method",
            equation_str=equation_str,
            final_root=None,
            final_function_value=None,
            absolute_residual=None,
            iteration_count=0,
            function_evaluations=fev,
            converged=False,
            stopping_reason="Endpoint evaluation failure",
            warning_message=f"Error evaluating endpoints [{a}, {b}]: {str(err)}",
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    # Immediate root check at endpoints
    if abs(fa) < 1e-15:
        history.append(
            IterationRecord(
                iteration=0,
                current_estimate=a,
                function_value=fa,
                lower_bound=a,
                upper_bound=b,
                interval_width=b - a,
                notes="Endpoint 'a' is an exact root.",
            )
        )
        return SolverResult(
            method_name="Bisection Method",
            equation_str=equation_str,
            final_root=a,
            final_function_value=fa,
            absolute_residual=abs(fa),
            iteration_count=0,
            function_evaluations=fev,
            converged=True,
            stopping_reason="Lower endpoint 'a' is an exact root.",
            iteration_history=history,
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    if abs(fb) < 1e-15:
        history.append(
            IterationRecord(
                iteration=0,
                current_estimate=b,
                function_value=fb,
                lower_bound=a,
                upper_bound=b,
                interval_width=b - a,
                notes="Endpoint 'b' is an exact root.",
            )
        )
        return SolverResult(
            method_name="Bisection Method",
            equation_str=equation_str,
            final_root=b,
            final_function_value=fb,
            absolute_residual=abs(fb),
            iteration_count=0,
            function_evaluations=fev,
            converged=True,
            stopping_reason="Upper endpoint 'b' is an exact root.",
            iteration_history=history,
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    # Bracketing check: f(a) * f(b) must be negative
    if fa * fb > 0:
        return SolverResult(
            method_name="Bisection Method",
            equation_str=equation_str,
            final_root=None,
            final_function_value=None,
            absolute_residual=None,
            iteration_count=0,
            function_evaluations=fev,
            converged=False,
            stopping_reason="Interval does not bracket a root",
            warning_message=(
                f"Bracketing condition violated: f({a}) = {fa:.6e} and f({b}) = {fb:.6e} "
                f"have the same sign. Bisection requires f(a)*f(b) < 0."
            ),
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    c_old: Optional[float] = None
    converged = False
    stopping_reason = "Maximum iterations reached before convergence."

    for k in range(1, max_iter + 1):
        c = (a + b) / 2.0
        width = b - a

        try:
            fc = f(c)
            fev += 1
        except Exception as err:
            return SolverResult(
                method_name="Bisection Method",
                equation_str=equation_str,
                final_root=c,
                final_function_value=None,
                absolute_residual=None,
                iteration_count=k - 1,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Function evaluation error at midpoint",
                warning_message=f"Evaluation error at c = {c:.6f} on iteration {k}: {str(err)}",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        e_abs, e_rel, e_pct = compute_approximate_errors(c, c_old)
        # Note: on iteration 1, e_abs can be estimated as width / 2
        if e_abs is None:
            e_abs = width / 2.0
            if abs(c) > 1e-15:
                e_rel = e_abs / abs(c)
                e_pct = e_rel * 100.0

        residual = compute_residual(fc)

        rec = IterationRecord(
            iteration=k,
            current_estimate=c,
            previous_estimate=c_old,
            function_value=fc,
            lower_bound=a,
            upper_bound=b,
            interval_width=width,
            approx_abs_error=e_abs,
            approx_rel_error=e_rel,
            approx_percent_error=e_pct,
        )
        history.append(rec)

        # Check stopping criteria
        should_stop, reason = evaluate_stopping_criterion(
            stopping_criterion, tol, e_abs, e_rel, residual, interval_width=width
        )

        if abs(fc) == 0.0:
            converged = True
            stopping_reason = f"Exact root found at c = {c:.10f} with f(c) = 0."
            rec.notes = "Exact root hit."
            break

        if should_stop:
            converged = True
            stopping_reason = reason
            rec.notes = f"Tolerance met: {reason}"
            break

        # Sub-interval selection preserving the bracket
        if fa * fc < 0:
            b = c
            fb = fc
            rec.notes = "Root in left sub-interval [a, c]; updated b = c."
        else:
            a = c
            fa = fc
            rec.notes = "Root in right sub-interval [c, b]; updated a = c."

        c_old = c

    last_rec = history[-1]
    elapsed = (time.perf_counter() - start_time) * 1000

    return SolverResult(
        method_name="Bisection Method",
        equation_str=equation_str,
        final_root=last_rec.current_estimate,
        final_function_value=last_rec.function_value,
        absolute_residual=abs(last_rec.function_value),
        iteration_count=len(history),
        function_evaluations=fev,
        converged=converged,
        stopping_reason=stopping_reason,
        iteration_history=history,
        stopping_criterion=stopping_criterion,
        tolerance=tol,
        max_iterations=max_iter,
        execution_time_ms=elapsed,
    )


def regula_falsi_method(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    stopping_criterion: str = "approx_abs_error",
    equation_str: str = "f(x) = 0",
) -> SolverResult:
    """
    Solve f(x) = 0 using the Regula-Falsi (False Position) Method.

    Formula:
        c = (a*f(b) - b*f(a)) / (f(b) - f(a))
          = a - f(a) * (b - a) / (f(b) - f(a))

    Mathematical Properties:
    - Type: Bracketing Method
    - Order of convergence: Typically linear (p = 1) due to endpoint stagnation.
    - Preserves sign change while using linear interpolation chord.
    """
    start_time = time.perf_counter()
    history: List[IterationRecord] = []
    fev = 0

    if a > b:
        a, b = b, a

    try:
        fa = f(a)
        fb = f(b)
        fev += 2
    except Exception as err:
        return SolverResult(
            method_name="Regula-Falsi Method",
            equation_str=equation_str,
            final_root=None,
            final_function_value=None,
            absolute_residual=None,
            iteration_count=0,
            function_evaluations=fev,
            converged=False,
            stopping_reason="Endpoint evaluation failure",
            warning_message=f"Error evaluating endpoints [{a}, {b}]: {str(err)}",
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    # Immediate root checks
    if abs(fa) < 1e-15:
        history.append(
            IterationRecord(
                iteration=0,
                current_estimate=a,
                function_value=fa,
                lower_bound=a,
                upper_bound=b,
                interval_width=b - a,
                notes="Endpoint 'a' is an exact root.",
            )
        )
        return SolverResult(
            method_name="Regula-Falsi Method",
            equation_str=equation_str,
            final_root=a,
            final_function_value=fa,
            absolute_residual=abs(fa),
            iteration_count=0,
            function_evaluations=fev,
            converged=True,
            stopping_reason="Lower endpoint 'a' is an exact root.",
            iteration_history=history,
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    if abs(fb) < 1e-15:
        history.append(
            IterationRecord(
                iteration=0,
                current_estimate=b,
                function_value=fb,
                lower_bound=a,
                upper_bound=b,
                interval_width=b - a,
                notes="Endpoint 'b' is an exact root.",
            )
        )
        return SolverResult(
            method_name="Regula-Falsi Method",
            equation_str=equation_str,
            final_root=b,
            final_function_value=fb,
            absolute_residual=abs(fb),
            iteration_count=0,
            function_evaluations=fev,
            converged=True,
            stopping_reason="Upper endpoint 'b' is an exact root.",
            iteration_history=history,
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    # Bracketing check
    if fa * fb > 0:
        return SolverResult(
            method_name="Regula-Falsi Method",
            equation_str=equation_str,
            final_root=None,
            final_function_value=None,
            absolute_residual=None,
            iteration_count=0,
            function_evaluations=fev,
            converged=False,
            stopping_reason="Interval does not bracket a root",
            warning_message=(
                f"Bracketing condition violated: f({a}) = {fa:.6e} and f({b}) = {fb:.6e} "
                f"have the same sign. Regula-Falsi requires f(a)*f(b) < 0."
            ),
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    c_old: Optional[float] = None
    converged = False
    stopping_reason = "Maximum iterations reached before convergence."
    fixed_a_count = 0
    fixed_b_count = 0
    stagnation_warning: Optional[str] = None

    for k in range(1, max_iter + 1):
        denom = fb - fa
        if abs(denom) < 1e-15:
            return SolverResult(
                method_name="Regula-Falsi Method",
                equation_str=equation_str,
                final_root=c_old,
                final_function_value=None,
                absolute_residual=None,
                iteration_count=k - 1,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Zero or near-zero denominator encountered",
                warning_message=f"Denominator (f(b) - f(a)) = {denom:.2e} ~ 0. Numerical instability avoided.",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Regula-Falsi formula
        c = (a * fb - b * fa) / denom
        width = b - a

        try:
            fc = f(c)
            fev += 1
        except Exception as err:
            return SolverResult(
                method_name="Regula-Falsi Method",
                equation_str=equation_str,
                final_root=c,
                final_function_value=None,
                absolute_residual=None,
                iteration_count=k - 1,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Function evaluation error at false position estimate",
                warning_message=f"Evaluation error at c = {c:.6f} on iteration {k}: {str(err)}",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        e_abs, e_rel, e_pct = compute_approximate_errors(c, c_old)
        residual = compute_residual(fc)

        notes_parts = []

        # Check stopping criteria
        should_stop, reason = evaluate_stopping_criterion(
            stopping_criterion, tol, e_abs, e_rel, residual, interval_width=width
        )

        if abs(fc) == 0.0:
            converged = True
            stopping_reason = f"Exact root found at c = {c:.10f} with f(c) = 0."
            notes_parts.append("Exact root hit.")
        elif should_stop:
            converged = True
            stopping_reason = reason
            notes_parts.append(f"Tolerance met: {reason}")

        # Update interval
        if fa * fc < 0:
            b = c
            fb = fc
            fixed_a_count += 1
            fixed_b_count = 0
            notes_parts.append("Root in [a, c]; updated b = c.")
            if fixed_a_count >= 3:
                notes_parts.append(f"Endpoint 'a' stagnated ({fixed_a_count} iterations).")
        else:
            a = c
            fa = fc
            fixed_b_count += 1
            fixed_a_count = 0
            notes_parts.append("Root in [c, b]; updated a = c.")
            if fixed_b_count >= 3:
                notes_parts.append(f"Endpoint 'b' stagnated ({fixed_b_count} iterations).")

        rec = IterationRecord(
            iteration=k,
            current_estimate=c,
            previous_estimate=c_old,
            function_value=fc,
            lower_bound=a,
            upper_bound=b,
            interval_width=width,
            approx_abs_error=e_abs,
            approx_rel_error=e_rel,
            approx_percent_error=e_pct,
            notes=" ".join(notes_parts),
        )
        history.append(rec)

        if converged:
            break

        c_old = c

    if fixed_a_count >= 4 or fixed_b_count >= 4:
        stagnation_warning = (
            "Noticeable endpoint stagnation occurred: one endpoint remained stationary while "
            "the other advanced slowly. This is a classic characteristic of standard Regula-Falsi "
            "on convex or concave functions."
        )

    last_rec = history[-1]
    elapsed = (time.perf_counter() - start_time) * 1000

    return SolverResult(
        method_name="Regula-Falsi Method",
        equation_str=equation_str,
        final_root=last_rec.current_estimate,
        final_function_value=last_rec.function_value,
        absolute_residual=abs(last_rec.function_value),
        iteration_count=len(history),
        function_evaluations=fev,
        converged=converged,
        stopping_reason=stopping_reason,
        iteration_history=history,
        stopping_criterion=stopping_criterion,
        tolerance=tol,
        max_iterations=max_iter,
        warning_message=stagnation_warning,
        execution_time_ms=elapsed,
    )


def newton_raphson_method(
    f: Callable[[float], float],
    f_prime: Callable[[float], float],
    x0: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    stopping_criterion: str = "approx_abs_error",
    equation_str: str = "f(x) = 0",
) -> SolverResult:
    """
    Solve f(x) = 0 using the Newton-Raphson Method.

    Formula:
        x_{next} = x_current - f(x_current) / f'(x_current)

    Mathematical Properties:
    - Type: Open Method (single initial estimate)
    - Order of convergence: Quadratic (p = 2) for simple roots (f'(x*) != 0)
    - Requires continuous first derivative and x0 sufficiently close to the root
    - Does NOT guarantee convergence (can oscillate, diverge, or divide by zero)
    """
    start_time = time.perf_counter()
    history: List[IterationRecord] = []
    fev = 0

    x_curr = float(x0)
    x_prev: Optional[float] = None
    converged = False
    stopping_reason = "Maximum iterations reached before convergence."
    warning_msg: Optional[str] = None

    for k in range(1, max_iter + 1):
        try:
            fx = f(x_curr)
            fpx = f_prime(x_curr)
            fev += 2
        except Exception as err:
            return SolverResult(
                method_name="Newton-Raphson Method",
                equation_str=equation_str,
                final_root=x_curr,
                final_function_value=None,
                absolute_residual=None,
                iteration_count=k - 1,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Function/Derivative evaluation failure",
                warning_message=f"Evaluation error at x = {x_curr:.6f} on iteration {k}: {str(err)}",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check for zero or near-zero derivative
        if abs(fpx) < 1e-14:
            history.append(
                IterationRecord(
                    iteration=k,
                    current_estimate=x_curr,
                    previous_estimate=x_prev,
                    function_value=fx,
                    derivative_value=fpx,
                    notes=f"Zero derivative: f'({x_curr:.4f}) = {fpx:.2e}. Tangent line horizontal.",
                )
            )
            return SolverResult(
                method_name="Newton-Raphson Method",
                equation_str=equation_str,
                final_root=x_curr,
                final_function_value=fx,
                absolute_residual=abs(fx),
                iteration_count=k,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Zero or near-zero derivative encountered",
                warning_message=(
                    f"At x = {x_curr:.6f}, the derivative f'(x) is {fpx:.2e} (~ 0). "
                    "The tangent line is horizontal, making division by zero impossible. "
                    "Convergence failed; select a different initial estimate x0."
                ),
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Newton step
        delta = fx / fpx
        x_next = x_curr - delta

        # Check for non-finite values or runaway divergence
        if not math.isfinite(x_next) or abs(x_next) > 1e12:
            history.append(
                IterationRecord(
                    iteration=k,
                    current_estimate=x_curr,
                    previous_estimate=x_prev,
                    function_value=fx,
                    derivative_value=fpx,
                    notes="Estimate exploded or diverged to non-finite value.",
                )
            )
            return SolverResult(
                method_name="Newton-Raphson Method",
                equation_str=equation_str,
                final_root=x_curr,
                final_function_value=fx,
                absolute_residual=abs(fx),
                iteration_count=k,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Divergence detected",
                warning_message=f"Divergence detected: x_next = {x_next}. Initial estimate x0 may be too far from root.",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        e_abs, e_rel, e_pct = compute_approximate_errors(x_next, x_curr)
        residual = compute_residual(fx)

        rec = IterationRecord(
            iteration=k,
            current_estimate=x_curr,
            previous_estimate=x_prev,
            function_value=fx,
            derivative_value=fpx,
            approx_abs_error=e_abs,
            approx_rel_error=e_rel,
            approx_percent_error=e_pct,
        )
        history.append(rec)

        # Check stopping criteria
        should_stop, reason = evaluate_stopping_criterion(
            stopping_criterion, tol, e_abs, e_rel, residual
        )

        if abs(fx) == 0.0:
            converged = True
            stopping_reason = f"Exact root found at x = {x_curr:.10f} with f(x) = 0."
            rec.notes = "Exact root hit."
            break

        if should_stop:
            converged = True
            stopping_reason = reason
            rec.notes = f"Tolerance met: {reason}"
            # Record final converged estimate
            x_curr = x_next
            break

        x_prev = x_curr
        x_curr = x_next

    # Evaluate residual at final point
    try:
        final_fx = f(x_curr)
        fev += 1
    except Exception:
        final_fx = history[-1].function_value

    elapsed = (time.perf_counter() - start_time) * 1000

    return SolverResult(
        method_name="Newton-Raphson Method",
        equation_str=equation_str,
        final_root=x_curr,
        final_function_value=final_fx,
        absolute_residual=abs(final_fx),
        iteration_count=len(history),
        function_evaluations=fev,
        converged=converged,
        stopping_reason=stopping_reason,
        iteration_history=history,
        stopping_criterion=stopping_criterion,
        tolerance=tol,
        max_iterations=max_iter,
        warning_message=warning_msg,
        execution_time_ms=elapsed,
    )


def secant_method(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-6,
    max_iter: int = 100,
    stopping_criterion: str = "approx_abs_error",
    equation_str: str = "f(x) = 0",
) -> SolverResult:
    """
    Solve f(x) = 0 using the Secant Method.

    Formula:
        x_{next} = x_1 - f(x_1) * (x_1 - x_0) / (f(x_1) - f(x_0))

    Mathematical Properties:
    - Type: Open Method (two starting estimates, no bracketing required)
    - Order of convergence: Superlinear (p = (1 + sqrt(5))/2 ~ 1.618 - Golden Ratio)
    - Does NOT require analytical derivative f'(x)
    - Does NOT guarantee convergence (can diverge or divide by zero if f(x0) == f(x1))
    """
    start_time = time.perf_counter()
    history: List[IterationRecord] = []
    fev = 0

    x_prev = float(x0)
    x_curr = float(x1)

    try:
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        fev += 2
    except Exception as err:
        return SolverResult(
            method_name="Secant Method",
            equation_str=equation_str,
            final_root=None,
            final_function_value=None,
            absolute_residual=None,
            iteration_count=0,
            function_evaluations=fev,
            converged=False,
            stopping_reason="Initial estimate evaluation failure",
            warning_message=f"Error evaluating initial estimates [x0={x0}, x1={x1}]: {str(err)}",
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    # Immediate root checks
    if abs(f_prev) < 1e-15:
        history.append(
            IterationRecord(
                iteration=0,
                current_estimate=x_prev,
                function_value=f_prev,
                notes="Initial estimate x0 is an exact root.",
            )
        )
        return SolverResult(
            method_name="Secant Method",
            equation_str=equation_str,
            final_root=x_prev,
            final_function_value=f_prev,
            absolute_residual=abs(f_prev),
            iteration_count=0,
            function_evaluations=fev,
            converged=True,
            stopping_reason="Initial estimate x0 is an exact root.",
            iteration_history=history,
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    if abs(f_curr) < 1e-15:
        history.append(
            IterationRecord(
                iteration=0,
                current_estimate=x_curr,
                function_value=f_curr,
                notes="Initial estimate x1 is an exact root.",
            )
        )
        return SolverResult(
            method_name="Secant Method",
            equation_str=equation_str,
            final_root=x_curr,
            final_function_value=f_curr,
            absolute_residual=abs(f_curr),
            iteration_count=0,
            function_evaluations=fev,
            converged=True,
            stopping_reason="Initial estimate x1 is an exact root.",
            iteration_history=history,
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    # Check for equal initial function values (horizontal secant line)
    if abs(f_curr - f_prev) < 1e-15:
        return SolverResult(
            method_name="Secant Method",
            equation_str=equation_str,
            final_root=None,
            final_function_value=None,
            absolute_residual=None,
            iteration_count=0,
            function_evaluations=fev,
            converged=False,
            stopping_reason="Equal initial function values",
            warning_message=(
                f"Secant line is horizontal: f(x0={x_prev}) = {f_prev:.6e} and "
                f"f(x1={x_curr}) = {f_curr:.6e} are equal. Denominator f(x1) - f(x0) = 0."
            ),
            stopping_criterion=stopping_criterion,
            tolerance=tol,
            max_iterations=max_iter,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    converged = False
    stopping_reason = "Maximum iterations reached before convergence."

    for k in range(1, max_iter + 1):
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            return SolverResult(
                method_name="Secant Method",
                equation_str=equation_str,
                final_root=x_curr,
                final_function_value=f_curr,
                absolute_residual=abs(f_curr),
                iteration_count=k - 1,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Zero or near-zero denominator encountered",
                warning_message=f"Denominator (f(x_curr) - f(x_prev)) = {denom:.2e} ~ 0. Division by zero avoided.",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Secant formula
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom

        if not math.isfinite(x_next) or abs(x_next) > 1e12:
            return SolverResult(
                method_name="Secant Method",
                equation_str=equation_str,
                final_root=x_curr,
                final_function_value=f_curr,
                absolute_residual=abs(f_curr),
                iteration_count=k - 1,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Divergence detected",
                warning_message=f"Estimate diverged: x_next = {x_next}. Method diverged.",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        e_abs, e_rel, e_pct = compute_approximate_errors(x_next, x_curr)
        residual = compute_residual(f_curr)

        rec = IterationRecord(
            iteration=k,
            current_estimate=x_curr,
            previous_estimate=x_prev,
            function_value=f_curr,
            approx_abs_error=e_abs,
            approx_rel_error=e_rel,
            approx_percent_error=e_pct,
        )
        history.append(rec)

        should_stop, reason = evaluate_stopping_criterion(
            stopping_criterion, tol, e_abs, e_rel, residual
        )

        if abs(f_curr) == 0.0:
            converged = True
            stopping_reason = f"Exact root found at x = {x_curr:.10f} with f(x) = 0."
            rec.notes = "Exact root hit."
            break

        if should_stop:
            converged = True
            stopping_reason = reason
            rec.notes = f"Tolerance met: {reason}"
            x_curr = x_next
            break

        # Move to next step
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_next

        try:
            f_curr = f(x_curr)
            fev += 1
        except Exception as err:
            return SolverResult(
                method_name="Secant Method",
                equation_str=equation_str,
                final_root=x_curr,
                final_function_value=None,
                absolute_residual=None,
                iteration_count=k,
                function_evaluations=fev,
                converged=False,
                stopping_reason="Function evaluation error at new estimate",
                warning_message=f"Evaluation error at x = {x_curr:.6f}: {str(err)}",
                iteration_history=history,
                stopping_criterion=stopping_criterion,
                tolerance=tol,
                max_iterations=max_iter,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

    try:
        final_fx = f(x_curr)
        fev += 1
    except Exception:
        final_fx = history[-1].function_value

    elapsed = (time.perf_counter() - start_time) * 1000

    return SolverResult(
        method_name="Secant Method",
        equation_str=equation_str,
        final_root=x_curr,
        final_function_value=final_fx,
        absolute_residual=abs(final_fx),
        iteration_count=len(history),
        function_evaluations=fev,
        converged=converged,
        stopping_reason=stopping_reason,
        iteration_history=history,
        stopping_criterion=stopping_criterion,
        tolerance=tol,
        max_iterations=max_iter,
        execution_time_ms=elapsed,
    )
