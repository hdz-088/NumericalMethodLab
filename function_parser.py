"""
function_parser.py - Safe Mathematical Expression Parsing and Symbolic Derivation
==================================================================================
Safely parses user mathematical expressions f(x) = 0 using SymPy:
- No unrestricted eval() or exec()
- Whitelisted functions (sin, cos, tan, exp, log, sqrt, etc.)
- Whitelisted constants (pi, E)
- Converts '^' to '**' for user convenience
- Computes exact symbolic derivative f'(x)
- Creates fast, safe numerical callables using sympy.lambdify
- Validates real domain & catches discontinuities, complex numbers, and overflows
"""

from typing import Callable, Tuple, Optional, List, Dict, Any
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)
import numpy as np
import math


# Allowed safe mathematical namespace
ALLOWED_LOCAL_DICT: Dict[str, Any] = {
    # Trigonometric & Hyperbolic
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    # Exponential & Logarithmic
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda x: sp.log(x, 10),
    "log2": lambda x: sp.log(x, 2),
    # Powers and roots
    "sqrt": sp.sqrt,
    "cbrt": sp.cbrt,
    "abs": sp.Abs,
    # Constants
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
}

# Standard transformations for mathematical strings
TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

# Standard presets for students
BUILT_IN_EXAMPLES: List[Dict[str, Any]] = [
    {
        "name": "Polynomial: x³ - x - 2 = 0",
        "expr": "x**3 - x - 2",
        "latex": "x^3 - x - 2 = 0",
        "bracket": [1.0, 2.0],
        "x0": 1.5,
        "x1": 2.0,
        "approx_root": 1.5213797068,
        "description": "Standard cubic benchmark from syllabus with one real root and two complex conjugate roots.",
    },
    {
        "name": "Quadratic: x² - 4 = 0",
        "expr": "x**2 - 4",
        "latex": "x^2 - 4 = 0",
        "bracket": [1.0, 3.0],
        "x0": 3.0,
        "x1": 2.5,
        "approx_root": 2.0,
        "description": "Simple algebraic polynomial with symmetric real roots at x = ±2.",
    },
    {
        "name": "Transcendental: cos(x) - x = 0 (Dottie Number)",
        "expr": "cos(x) - x",
        "latex": "\\cos(x) - x = 0",
        "bracket": [0.0, 1.0],
        "x0": 0.5,
        "x1": 0.8,
        "approx_root": 0.7390851332,
        "description": "Famous transcendental fixed point equation where cos(x) intersects the line y = x (angles in radians).",
    },
    {
        "name": "Transcendental: exp(-x) - x = 0",
        "expr": "exp(-x) - x",
        "latex": "e^{-x} - x = 0",
        "bracket": [0.0, 1.0],
        "x0": 0.5,
        "x1": 1.0,
        "approx_root": 0.5671432904,
        "description": "Decaying exponential transcendental equation (related to the Lambert W function W(1)).",
    },
    {
        "name": "Polynomial: x³ - 2*x - 5 = 0 (Horner's Equation)",
        "expr": "x**3 - 2*x - 5",
        "latex": "x^3 - 2x - 5 = 0",
        "bracket": [2.0, 3.0],
        "x0": 2.0,
        "x1": 2.5,
        "approx_root": 2.0945514815,
        "description": "Classic numerical analysis example; root lies strictly in [2, 3].",
    },
    {
        "name": "Trigonometric: x*sin(x) - 1 = 0",
        "expr": "x*sin(x) - 1",
        "latex": "x\\sin(x) - 1 = 0",
        "bracket": [0.5, 2.0],
        "x0": 1.0,
        "x1": 1.5,
        "approx_root": 1.1141571408,
        "description": "Oscillatory transcendental equation with infinitely many roots.",
    },
]


class ExpressionParserError(ValueError):
    """Raised when an expression cannot be parsed safely or contains disallowed symbols."""
    pass


class EvaluationDomainError(ValueError):
    """Raised when an expression evaluation produces complex numbers, division by zero, or domain errors."""
    pass


import re
import warnings

# Disallowed dangerous tokens
BANNED_TOKENS = {
    "__",
    "import",
    "system",
    "eval",
    "exec",
    "os",
    "sys",
    "open",
    "subprocess",
    "builtin",
    "lambda",
    "globals",
    "locals",
}


def _check_security_tokens(expr_str: str) -> None:
    """Check for suspicious Python identifiers or dunder attributes."""
    lower_str = expr_str.lower()
    for token in BANNED_TOKENS:
        if re.search(r"\b" + re.escape(token) + r"\b", lower_str) or token in ("__",):
            if token in lower_str:
                raise ExpressionParserError(
                    f"Disallowed security token or keyword '{token}' detected in equation."
                )


def parse_math_expression(
    expr_str: str,
) -> Tuple[sp.Expr, sp.Symbol, sp.Expr, str]:
    """
    Safely parse an algebraic or transcendental string into a SymPy expression and its derivative.

    Parameters
    ----------
    expr_str : str
        Mathematical expression representing f(x) = 0.

    Returns
    -------
    Tuple[sp.Expr, sp.Symbol, sp.Expr, str]
        (sympy_expr, x_symbol, derivative_expr, latex_string)

    Raises
    ------
    ExpressionParserError
        If string contains syntax errors, disallowed function calls, or multiple variables.
    """
    if not expr_str or not expr_str.strip():
        raise ExpressionParserError("Equation expression cannot be empty.")

    cleaned = expr_str.strip()
    _check_security_tokens(cleaned)

    # Preprocessing: normalize power symbol ^ to **
    cleaned = cleaned.replace("^", "**")

    # If user provided "f(x) = ...", strip left-hand side
    if "=" in cleaned:
        parts = cleaned.split("=")
        if len(parts) == 2:
            left, right = parts[0].strip(), parts[1].strip()
            if left in ("0", "f(x)", "y"):
                cleaned = right
            elif right in ("0", "f(x)", "y"):
                cleaned = left
            else:
                # Expression f(x) = g(x) becomes f(x) - (g(x))
                cleaned = f"({left}) - ({right})"
        else:
            raise ExpressionParserError("Expression can contain at most one '=' sign.")

    x = sp.Symbol("x", real=True)
    local_dict = {**ALLOWED_LOCAL_DICT, "x": x}

    try:
        parsed_raw = parse_expr(
            cleaned,
            local_dict=local_dict,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise ExpressionParserError(
            f"Failed to parse mathematical expression '{expr_str}': {str(exc)}"
        ) from exc

    if not isinstance(parsed_raw, (sp.Expr, sp.Basic)):
        parsed_expr = sp.sympify(parsed_raw)
    else:
        parsed_expr = parsed_raw

    # Validate symbols: only 'x' is permitted
    free_symbols = parsed_expr.free_symbols
    for sym in free_symbols:
        if sym.name != "x":
            raise ExpressionParserError(
                f"Disallowed free variable '{sym.name}'. Only the single independent variable 'x' is allowed."
            )

    # Compute symbolic derivative f'(x)
    try:
        derivative_expr = sp.diff(parsed_expr, x)
    except Exception as exc:
        raise ExpressionParserError(
            f"Could not compute symbolic derivative for '{expr_str}': {str(exc)}"
        ) from exc

    latex_repr = sp.latex(parsed_expr)
    return parsed_expr, x, derivative_expr, latex_repr


def build_numerical_function(
    expr: sp.Expr,
    x_symbol: sp.Symbol,
) -> Callable[[float], float]:
    """
    Convert a SymPy expression into a robust numerical callable f(float) -> float.
    Safely rejects complex results, overflow, and math domain errors.
    """
    # Create lambdified function using numpy/math
    raw_callable = sp.lambdify(x_symbol, expr, modules=["numpy", "math"])

    def safe_f(x_val: float) -> float:
        try:
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                val = raw_callable(x_val)

            # Convert numpy 0-d array or scalar to python float
            if isinstance(val, (np.ndarray, np.generic)):
                if val.size == 1:
                    val = val.item()
                else:
                    raise EvaluationDomainError(f"Unexpected non-scalar output shape: {val.shape}")

            # Disallow complex numbers
            if isinstance(val, complex):
                if abs(val.imag) > 1e-12:
                    raise EvaluationDomainError(
                        f"Function evaluated to a complex number {val} at x = {x_val}. Only real roots are supported."
                    )
                val = val.real

            val_float = float(val)

            if not math.isfinite(val_float) or math.isnan(val_float):
                raise EvaluationDomainError(
                    f"Function evaluated to non-finite value ({val_float}) at x = {x_val}."
                )

            return val_float

        except (ZeroDivisionError, OverflowError, ValueError) as err:
            raise EvaluationDomainError(
                f"Numerical domain error at x = {x_val}: {str(err)}"
            ) from err
        except TypeError as err:
            raise EvaluationDomainError(
                f"Type error during function evaluation at x = {x_val}: {str(err)}"
            ) from err

    return safe_f


def build_vectorized_function(
    expr: sp.Expr,
    x_symbol: sp.Symbol,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Build a vectorized function for plotting over arrays.
    Returns np.nan on undefined or complex evaluations instead of crashing.
    """
    raw_callable = sp.lambdify(x_symbol, expr, modules=["numpy"])

    def vectorized_f(x_arr: np.ndarray) -> np.ndarray:
        try:
            # Ensure x_arr is a float numpy array
            x_arr = np.asarray(x_arr, dtype=np.float64)
            with np.errstate(all="ignore"):
                y_arr = raw_callable(x_arr)
                if not isinstance(y_arr, np.ndarray):
                    y_arr = np.full_like(x_arr, float(y_arr))

                # If output has complex dtype, mask complex entries with nan
                if np.iscomplexobj(y_arr):
                    real_part = np.real(y_arr)
                    imag_part = np.imag(y_arr)
                    y_arr = np.where(np.abs(imag_part) < 1e-12, real_part, np.nan)

                # Replace infinities with nan for clean plotting
                y_arr = np.where(np.isfinite(y_arr), y_arr, np.nan)
                return y_arr
        except Exception:
            # Fallback elementwise evaluation
            safe_scalar = build_numerical_function(expr, x_symbol)
            result = np.empty_like(x_arr, dtype=np.float64)
            for idx, val in enumerate(x_arr):
                try:
                    result[idx] = safe_scalar(float(val))
                except Exception:
                    result[idx] = np.nan
            return result

    return vectorized_f


def validate_function_domain(
    f: Callable[[float], float],
    domain: Tuple[float, float],
    num_samples: int = 15,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that f(x) can be evaluated across the domain [x_min, x_max].
    Checks for undefined points, complex outputs, or singularities.
    """
    x_min, x_max = domain
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    samples = np.linspace(x_min, x_max, num_samples)
    for x_pt in samples:
        try:
            val = f(float(x_pt))
            if not math.isfinite(val):
                return False, f"Function produced non-finite value ({val}) at x = {x_pt:.4f}."
        except EvaluationDomainError as err:
            return False, str(err)
        except Exception as err:
            return False, f"Evaluation failure at x = {x_pt:.4f}: {str(err)}"

    return True, None
