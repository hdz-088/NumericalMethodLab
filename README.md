# Numerical Methods Visual Lab

### _An Interactive Root-Finding and Convergence Analyzer_

**Syllabus Unit Mapping**: Unit 02 — Numerical Methods (Algebraic & Transcendental Equations)

---

## 1. Project Introduction

The **Numerical Methods Visual Lab** is an interactive educational and research tool designed for undergraduate and graduate Mathematics and Engineering curricula. It bridges the gap between pure numerical analysis theory and hands-on computer simulation.

Traditional numerical analysis coursework often presents iterative root-finding methods as rote mechanical calculations. Students frequently struggle to develop intuition regarding:

- The geometric significance of tangent lines, secant chords, and interval brackets.
- The fundamental distinction between **bracketing methods** (guaranteed global convergence under the Intermediate Value Theorem) and **open methods** (fast local convergence with potential divergence).
- Why different algorithms exhibit drastically different convergence rates (linear, quadratic, and superlinear).
- Why a small residual $|f(x)|$ does not necessarily indicate a small root error $|x - x^*|$ for ill-conditioned or flat functions.
- How pathological edge cases (such as zero derivatives or endpoint stagnation) manifest in practice.

This application provides manual, uncompromised implementations of the four core syllabus algorithms:

1. **Bisection Method**
2. **Regula-Falsi Method (False Position)**
3. **Newton-Raphson Method**
4. **Secant Method**

It incorporates safe symbolic expression parsing via **SymPy**, interactive geometric step-by-step visualizations using **Plotly**, comprehensive tabular telemetry exported via **Pandas**, and automated mathematical test suites via **Pytest**.

---

## 2. Problem Statement

In engineering and scientific computation, non-linear equations of the form:

$$f(x) = 0$$

arise ubiquitously in orbital mechanics, structural analysis, circuit simulation, thermodynamics, and chemical equilibrium. In general, closed-form analytical solutions (such as the quadratic formula) either do not exist or are intractable (e.g., fifth-degree polynomials by the Abel-Ruffini Theorem, or transcendental equations involving $\sin, \cos, e^x, \ln x$).

Consequently, engineers must resort to iterative approximation algorithms. However, choosing the appropriate algorithm requires understanding:

- How initial parameters (interval bounds $[a, b]$ vs. point estimates $x_0, x_1$) affect convergence.
- How computational cost (iterations vs. function evaluations) scales.
- How stopping criteria (approximate absolute error, relative error, residual) impact precision.

The goal of this project is to construct a software laboratory that allows students and examiners to solve arbitrary non-linear equations, inspect every intermediate iteration, visualize convergence curves, and compare all four algorithms under identical conditions.

---

## 3. Project Objectives

1. **Manual Algorithmic Implementations**: Implement all four algorithms from first principles in pure Python without relying on black-box solvers (e.g. `scipy.optimize`), ensuring that every mathematical formula and intermediate variable is transparent and accessible.
2. **Safe Mathematical Expression Parsing**: Develop a robust parsing pipeline that accepts algebraic and transcendental functions, enforces strict security sandboxing (preventing code injection), and computes exact symbolic derivatives $f'(x)$ using SymPy.
3. **Multi-Faceted Telemetry & Error Metrics**: Track and report absolute approximate error, relative approximate error, percentage error, residual, and interval width across every iteration step.
4. **Interactive Geometric Visualizations**: Create dynamic Plotly figures showing the function landscape, step-by-step tangent and secant line constructions, and semilogarithmic convergence graphs.
5. **Head-to-Head Comparison Dashboard**: Enable users to run all four methods on the same equation with customized starting values, displaying side-by-side performance tables and cost comparisons.
6. **Academic Export and Reporting**: Provide instant downloads of CSV iteration tables and structured Markdown experiment reports.
7. **Comprehensive Unit Testing**: Maintain a test suite with 100% pass rates across syllabus benchmark functions and critical mathematical edge cases.

---

## 4. Syllabus Mapping

- **Subject**: Numerical Methods / Computational Mathematics
- **Unit**: **Unit 02 — Numerical Methods: Algebraic and Transcendental Equations**
- **Topics Covered**:
  - Bisection Method: Theory, Interval Halving, Error Bounds, Iteration Count Formula.
  - Regula-Falsi Method: Interpolation Chord, Endpoint Stagnation, Stagnation Mitigation.
  - Newton-Raphson Method: First-order Taylor Derivation, Quadratic Convergence, Zero-Derivative Breakdown.
  - Secant Method: Finite Difference Approximation, Golden Ratio Convergence Order ($p \approx 1.618$), Efficiency Index.
  - Error Analysis: True vs. Approximate Error, Residuals, Condition Number $\kappa = 1/|f'(x^*)|$.

---

## 5. Mathematical Formulas & Algorithmic Details

### 5.1 Bisection Method

- **Classification**: Bracketing Method
- **Theoretical Basis**: Intermediate Value Theorem (IVT). If $f \in C[a, b]$ and $f(a) \cdot f(b) < 0$, a root $x^* \in (a, b)$ exists.
- **Formula**:
  $$c_k = \frac{a_k + b_k}{2}$$
- **Interval Update**:
  - If $f(a_k) \cdot f(c_k) < 0$: new interval is $[a_{k+1}, b_{k+1}] = [a_k, c_k]$.
  - If $f(a_k) \cdot f(c_k) > 0$: new interval is $[a_{k+1}, b_{k+1}] = [c_k, b_k]$.
  - If $f(c_k) = 0$: $c_k$ is the exact root.
- **Order of Convergence**: Linear ($p = 1$, error reduction factor $C = 0.5$).
- **Theoretical Error Bound**:
  $$|c_k - x^*| \le \frac{b_0 - a_0}{2^k}$$
- **Required Iterations for Tolerance $\epsilon$**:
  $$k \ge \frac{\ln(b_0 - a_0) - \ln(\epsilon)}{\ln(2)}$$

### 5.2 Regula-Falsi Method (False Position)

- **Classification**: Bracketing Method
- **Theoretical Basis**: Linear interpolation between $(a, f(a))$ and $(b, f(b))$.
- **Formula**:
  $$c_k = a_k - f(a_k) \frac{b_k - a_k}{f(b_k) - f(a_k)} = \frac{a_k f(b_k) - b_k f(a_k)}{f(b_k) - f(a_k)}$$
- **Order of Convergence**: Linear ($p = 1$).
- **Key Phenomenon**: **Endpoint Stagnation**. For convex or concave curves (e.g. $x^3 - x - 2$), one endpoint remains stationary while the other creeps slowly, causing the interval width $|b_k - a_k|$ to remain large even as $c_k$ approaches the root.

### 5.3 Newton-Raphson Method

- **Classification**: Open Method (Single initial estimate $x_0$)
- **Theoretical Basis**: Truncated Taylor series expansion: $f(x) \approx f(x_k) + f'(x_k)(x - x_k) = 0$.
- **Formula**:
  $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$
- **Order of Convergence**: **Quadratic ($p = 2$)** for simple roots ($f'(x^*) \neq 0$):
  $$e_{k+1} \approx \frac{f''(x^*)}{2 f'(x^*)} e_k^2$$
- **Primary Failure Modes**:
  1. Horizontal tangent ($f'(x_k) = 0$): Division by zero.
  2. Inflection point divergence: Estimate shoots to infinity.
  3. Limit cycles / Oscillation: Iterate alternates periodically without converging.

### 5.4 Secant Method

- **Classification**: Open Method (Two initial estimates $x_0, x_1$, no bracketing required)
- **Theoretical Basis**: Approximating $f'(x_k)$ by the finite difference quotient $\frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$.
- **Formula**:
  $$x_{k+1} = x_k - f(x_k) \frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}$$
- **Order of Convergence**: **Superlinear**:
  $$p = \frac{1 + \sqrt{5}}{2} \approx 1.618 \quad (\text{The Golden Ratio } \phi)$$
- **Computational Efficiency**: Requires only 1 new function evaluation per iteration step (no derivative evaluation). Ostrowski's Computational Efficiency Index:
  $$\text{Eff}_{\text{Secant}} = 1.618^{1/1} \approx 1.618 > \text{Eff}_{\text{Newton}} = 2^{1/2} \approx 1.414$$

---

## 6. Technology Stack

- **Python 3.11+ / 3.13**: Core programming language.
- **Streamlit**: Interactive web dashboard and user interface.
- **SymPy**: Safe mathematical parsing, abstract syntax tree validation, and symbolic derivation.
- **NumPy**: High-performance vectorized function evaluations and domain sampling.
- **Plotly**: Interactive scientific visualizations (function landscapes, step-by-step tangents/chords, semi-log convergence curves).
- **Pandas**: Structured iteration telemetry tables, data formatting, and CSV export.
- **Pytest**: Automated mathematical test suite covering benchmark cases and numerical edge cases.

---

## 7. Project File Structure

```
numerical-methods-lab/
│
├── app.py                  # Streamlit interface, navigation, tabs, and export handlers
├── solvers.py              # Manual implementations of Bisection, Regula-Falsi, Newton, Secant
├── function_parser.py      # Safe SymPy expression parser, sanitization, and lambda builders
├── error_analysis.py       # Standardized error calculations and stopping criteria evaluators
├── visualizations.py       # Plotly interactive graphs (landscape, step stepper, convergence)
├── requirements.txt        # Exact project dependencies
├── test_solvers.py         # 12 automated unit and mathematical correctness tests
└── README.md               # Complete project documentation, report outline, and viva guide
```

---

## 8. Installation & Setup Instructions

### Prerequisites

- Python 3.11 or newer installed on your machine.
- Git (optional, for version control).

### Step 1: Clone or Navigate to the Project Directory

### Step 2: Create and Activate a Virtual Environment

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies

```powershell
pip install -r requirements.txt
```

---

## 9. Launching the Application

To start the interactive Streamlit dashboard:

```powershell
streamlit run app.py
```

Once executed, Streamlit will display the local URL (typically `http://localhost:8501`). Open this URL in any modern web browser.

---

## 10. Syllabus Benchmark Equations & Recommended Inputs

The application includes pre-configured benchmark equations matching standard university exam questions:

| Problem Type              | Equation $f(x) = 0$  | Bracket $[a, b]$ | Initial $x_0, x_1$ | True Root $x^*$     | Key Academic Feature                                                       |
| :------------------------ | :------------------- | :--------------- | :----------------- | :------------------ | :------------------------------------------------------------------------- |
| **Cubic Polynomial**      | `x**3 - x - 2 = 0`   | `[1.0, 2.0]`     | $x_0=1.5, x_1=2.0$ | $\approx 1.5213797$ | Standard syllabus cubic; demonstrates endpoint stagnation in Regula-Falsi. |
| **Quadratic Algebraic**   | `x**2 - 4 = 0`       | `[1.0, 3.0]`     | $x_0=3.0, x_1=2.5$ | $2.0000000$         | Clean integer root; shows Newton's rapid quadratic convergence.            |
| **Transcendental (Trig)** | `cos(x) - x = 0`     | `[0.0, 1.0]`     | $x_0=0.5, x_1=0.8$ | $\approx 0.7390851$ | Dottie Number; verifies radian angle handling.                             |
| **Transcendental (Exp)**  | `exp(-x) - x = 0`    | `[0.0, 1.0]`     | $x_0=0.5, x_1=1.0$ | $\approx 0.5671433$ | Decaying exponential; related to the Lambert W function $W(1)$.            |
| **Horner's Polynomial**   | `x**3 - 2*x - 5 = 0` | `[2.0, 3.0]`     | $x_0=2.0, x_1=2.5$ | $\approx 2.0945515$ | Classic textbook benchmark; simple real root in $[2, 3]$.                  |

---

## 11. Automated Testing

The project includes an automated test suite implemented with `pytest` covering all 10 required syllabus test cases:

1. $f(x) = x^3 - x - 2 = 0$ (Cubic root verification).
2. $f(x) = x^2 - 4 = 0$ (Algebraic root verification).
3. $f(x) = \cos(x) - x = 0$ (Trigonometric radian verification).
4. $f(x) = e^{-x} - x = 0$ (Exponential transcendental verification).
5. Non-bracketing intervals ($f(a) \cdot f(b) > 0$) properly rejected with diagnostics.
6. Newton-Raphson zero derivative ($f'(x_0) = 0$) detected and handled safely.
7. Secant method equal initial function values ($f(x_0) == f(x_1)$) handled without division by zero.
8. Invalid mathematical syntax and disallowed system tokens safely rejected.
9. Maximum iteration boundaries handled without false convergence reports.
10. Undefined and non-finite domain evaluations handled cleanly.
11. Iteration history telemetry integrity and DataFrame export.
12. Error metrics and stopping criteria accuracy.

### Running the Tests:

```powershell
pytest -v test_solvers.py
```

### Verified Test Output:

```
============================= test session starts =============================
platform win32 -- Python 3.13.14, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\skipper\HDz\Developer\Numericalvisual
collected 12 items

test_solvers.py::test_case_1_cubic_polynomial PASSED                     [  8%]
test_solvers.py::test_case_2_quadratic PASSED                            [ 16%]
test_solvers.py::test_case_3_trigonometric PASSED                        [ 25%]
test_solvers.py::test_case_4_exponential PASSED                          [ 33%]
test_solvers.py::test_case_5_non_bracketing_interval PASSED              [ 41%]
test_solvers.py::test_case_6_newton_zero_derivative PASSED               [ 50%]
test_solvers.py::test_case_7_secant_equal_initial_values PASSED          [ 58%]
test_solvers.py::test_case_8_invalid_equation_syntax PASSED              [ 66%]
test_solvers.py::test_case_9_max_iterations_reached PASSED               [ 75%]
test_solvers.py::test_case_10_undefined_domain_handling PASSED           [ 83%]
test_solvers.py::test_iteration_history_telemetry_structure PASSED       [ 91%]
test_solvers.py::test_error_metrics_and_stopping_criteria PASSED         [100%]

============================= 12 passed in 2.00s ==============================
```

---

## 12. Known Limitations & Future Scope

### Limitations

1. **Real Roots Only**: The visual lab is currently focused on finding real roots $x \in \mathbb{R}$. Polynomials with only complex conjugate roots (e.g. $x^2 + 1 = 0$) cannot be solved using standard real-domain bracketing.
2. **Single Independent Variable**: Only scalar equations $f(x) = 0$ in the variable $x$ are supported; systems of non-linear equations ($F(\mathbf{x}) = \mathbf{0}$) are not covered in Unit 02.
3. **Discontinuous Functions**: Bracketing methods assume continuity via the Intermediate Value Theorem. For rational functions with vertical asymptotes (e.g. $f(x) = 1/(x - 1)$), sign changes across an asymptote do not indicate a root.
