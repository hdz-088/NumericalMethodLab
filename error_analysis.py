"""
error_analysis.py - Mathematical Error Metrics & Stopping Criteria
==================================================================
Provides standardized error calculations for root-finding algorithms:
- Absolute approximate error: E_abs = |x_new - x_old|
- Relative approximate error: E_rel = |x_new - x_old| / |x_new| (x_new != 0)
- Percentage approximate error: E_percent = 100 * E_rel
- Residual: R = |f(x_new)|
- Interval width: |b - a| (for bracketing methods)
- True errors (when exact root is known)

Also includes stopping criteria evaluators and numerical sensitivity commentary.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class ErrorMetrics:
    """Standardized error metrics container for a single iteration step."""
    approx_abs_error: Optional[float] = None
    approx_rel_error: Optional[float] = None
    approx_percent_error: Optional[float] = None
    residual: float = 0.0
    interval_width: Optional[float] = None
    true_abs_error: Optional[float] = None
    true_rel_error: Optional[float] = None


def compute_approximate_errors(
    x_new: float,
    x_old: Optional[float]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute absolute, relative, and percentage approximate errors.

    Parameters
    ----------
    x_new : float
        Current root approximation.
    x_old : Optional[float]
        Previous root approximation. If None (first iteration), returns (None, None, None).

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[float]]
        (E_abs, E_rel, E_percent)
    """
    if x_old is None:
        return None, None, None

    if not math.isfinite(x_new) or not math.isfinite(x_old):
        return None, None, None

    e_abs = abs(x_new - x_old)

    # Avoid division by zero when x_new is zero or extremely close to zero
    if abs(x_new) > 1e-15:
        e_rel = e_abs / abs(x_new)
        e_percent = e_rel * 100.0
    else:
        # Fallback for root near 0: relative error is ill-conditioned
        e_rel = None
        e_percent = None

    return e_abs, e_rel, e_percent


def compute_residual(f_val: float) -> float:
    """
    Compute the absolute residual: R = |f(x)|.
    """
    if not math.isfinite(f_val):
        return float("inf")
    return abs(f_val)


def compute_true_errors(
    x_approx: float,
    x_true: Optional[float]
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute absolute and relative true errors when exact root is known.
    E_true_abs = |x_approx - x_true|
    E_true_rel = |x_approx - x_true| / |x_true| (when x_true != 0)
    """
    if x_true is None or not math.isfinite(x_approx) or not math.isfinite(x_true):
        return None, None

    e_true_abs = abs(x_approx - x_true)
    if abs(x_true) > 1e-15:
        e_true_rel = e_true_abs / abs(x_true)
    else:
        e_true_rel = None

    return e_true_abs, e_true_rel


def evaluate_stopping_criterion(
    criterion: str,
    tol: float,
    e_abs: Optional[float],
    e_rel: Optional[float],
    residual: float,
    interval_width: Optional[float] = None
) -> Tuple[bool, str]:
    """
    Evaluate whether the current iteration meets the chosen stopping criterion.

    Supported criteria:
    - 'approx_abs_error': |x_new - x_old| < tol
    - 'approx_rel_error': |x_new - x_old| / |x_new| < tol
    - 'residual': |f(x_new)| < tol
    - 'interval_width': |b - a| < tol (bracketing only)
    - 'combined': |x_new - x_old| < tol AND |f(x_new)| < tol

    Returns
    -------
    Tuple[bool, str]
        (should_stop, stopping_description)
    """
    if criterion == "residual":
        if residual < tol:
            return True, f"Residual |f(x)| = {residual:.2e} < tolerance {tol:.2e}"
        return False, ""

    if criterion == "approx_abs_error":
        if e_abs is not None and e_abs < tol:
            return True, f"Absolute approximate error E_abs = {e_abs:.2e} < tolerance {tol:.2e}"
        return False, ""

    if criterion == "approx_rel_error":
        if e_rel is not None and e_rel < tol:
            return True, f"Relative approximate error E_rel = {e_rel:.2e} < tolerance {tol:.2e}"
        elif e_abs is not None and e_abs < tol:
            # Fallback if x_new ~ 0
            return True, f"Absolute error E_abs = {e_abs:.2e} < tolerance {tol:.2e} (fallback for x ~ 0)"
        return False, ""

    if criterion == "interval_width":
        if interval_width is not None and interval_width < tol:
            return True, f"Interval width |b - a| = {interval_width:.2e} < tolerance {tol:.2e}"
        return False, ""

    if criterion == "combined":
        if e_abs is not None and e_abs < tol and residual < tol:
            return True, f"Combined criterion met: E_abs = {e_abs:.2e} < {tol:.2e} and |f(x)| = {residual:.2e} < {tol:.2e}"
        return False, ""

    # Default fallback to absolute approximate error
    if e_abs is not None and e_abs < tol:
        return True, f"Absolute error = {e_abs:.2e} < tolerance {tol:.2e}"

    return False, ""


def get_error_analysis_explanation() -> dict:
    """
    Returns an academic dictionary explaining error metrics, ill-conditioning,
    and why small residual != small error.
    """
    return {
        "approximate_vs_true": (
            "In practice, the exact analytical root x* is unknown (which is why numerical methods "
            "are needed). Therefore, true error |x_k - x*| cannot be computed directly. "
            "Instead, we monitor the approximate step change E_abs = |x_k - x_{k-1}| or "
            "relative change E_rel = |x_k - x_{k-1}| / |x_k|. Under contractive or superlinear "
            "convergence, successive difference is asymptotically proportional to true error."
        ),
        "residual_vs_error": (
            "A small residual |f(x)| < eps does NOT guarantee that the root error |x - x*| is small. "
            "By Taylor expansion: f(x) ~ f(x*) + f'(x*)(x - x*) = f'(x*)(x - x*). "
            "Thus, |x - x*| ~ |f(x)| / |f'(x*)|.\n\n"
            "- Flat Function (|f'(x*)| << 1, e.g., multiple roots like (x-1)^3): "
            "The residual |f(x)| can be minuscule (1e-6) while the root error is substantial (0.01).\n"
            "- Steep Function (|f'(x*)| >> 1, e.g., exp(10*x) - 1): "
            "The root estimate can be accurate to 10 decimal digits while |f(x)| remains 0.1."
        ),
        "condition_number": (
            "The absolute condition number for root finding is kappa = 1 / |f'(x*)|. "
            "When kappa is large (derivative near zero), the root is ill-conditioned, and numerical "
            "solvers experience slow convergence or loss of significant digits."
        )
    }
