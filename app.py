"""
app.py - Numerical Methods Visual Lab
=====================================
Interactive Root-Finding & Convergence Analyzer for Syllabus Unit 02: Numerical Methods.

Features:
1. Home / Project Overview & Syllabus Mapping
2. Root Solver (Interactive Equation Input, Method Selection, Iteration Table, CSV/Report Export)
3. Step-by-Step Geometric Iteration Visualizer
4. Multi-Method Comparison Dashboard
5. Error Analysis Deep Dive
6. Theory, Derivations & Viva Voce Q&A Reference
"""

import streamlit as st
import numpy as np
import pandas as pd
import sympy as sp
import time
import math
from typing import Optional

# Project module imports
from function_parser import (
    parse_math_expression,
    build_numerical_function,
    build_vectorized_function,
    validate_function_domain,
    BUILT_IN_EXAMPLES,
    ExpressionParserError,
    EvaluationDomainError,
)
from solvers import (
    bisection_method,
    regula_falsi_method,
    newton_raphson_method,
    secant_method,
    SolverResult,
)
from error_analysis import (
    get_error_analysis_explanation,
    compute_true_errors,
)
from visualizations import (
    create_function_plot,
    create_step_iteration_plot,
    create_convergence_plot,
    create_multi_method_convergence_comparison,
    create_cost_comparison_barchart,
)


# ==============================================================================
# Page Configuration & Academic Styling
# ==============================================================================
st.set_page_config(
    page_title="Numerical Methods Visual Lab",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished scientific styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-bracketing {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-open {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Session State Initialization
# ==============================================================================
if "current_equation" not in st.session_state:
    st.session_state["current_equation"] = "x**3 - x - 2"
if "last_solver_result" not in st.session_state:
    st.session_state["last_solver_result"] = None
if "parsed_data" not in st.session_state:
    st.session_state["parsed_data"] = None


# ==============================================================================
# Sidebar: Navigation & Project Meta
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=JKGkT60yzwgl&format=png&color=000000", width=64)
    st.markdown("### **Numerical Methods Lab**")
    #st.markdown("*Unit 02 — Root-Finding & Convergence*")
    st.markdown("---")

    nav_options = [
        "🏠 Home / Overview",
        "🔍 Root Solver",
        "🎬 Iteration Visualizer",
        "📊 Compare Methods",
        "📉 Error Analysis",
        "📚 Theory & Viva Prep",
    ]

    # nav_index tracks the selected page index independently of the widget key
    if "nav_index" not in st.session_state:
        st.session_state["nav_index"] = 0

    def _set_nav(idx: int):
        st.session_state["nav_index"] = idx

    nav_selection = st.radio(
        "Navigation",
        nav_options,
        index=st.session_state["nav_index"],
    )
    # Keep nav_index in sync when user clicks the sidebar radio directly
    st.session_state["nav_index"] = nav_options.index(nav_selection)

    st.markdown("---")
    st.markdown("#### **Project Specifications**")
    st.caption("• **Syllabus**: Unit 02 Numerical Methods")
    st.caption("• **Algorithms**: Bisection, Regula-Falsi, Newton-Raphson, Secant")
    st.caption("• **Angle System**: Radians for Trigonometry")
    st.caption("• **Symbolic Engine**: SymPy Exact Derivatives")


# Helper report generator
def generate_markdown_report(result: SolverResult, initial_inputs: dict) -> str:
    """Generate a clean, structured scientific report ready for download."""
    df = result.to_dataframe()
    table_md = df.to_markdown(index=False) if not df.empty else "No iterations recorded."
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Numerical Methods Visual Lab — Experiment Report
**Generated on**: {timestamp}  
**Syllabus Unit**: Unit 02 — Numerical Methods  

---

## 1. Problem Specification
- **Target Equation**: `{result.equation_str} = 0`
- **Method Applied**: {result.method_name}
- **Initial Parameters**: {initial_inputs}
- **Stopping Criterion**: {result.stopping_criterion}
- **Prescribed Tolerance**: {result.tolerance:.2e}
- **Maximum Allowed Iterations**: {result.max_iterations}

---

## 2. Solver Outcome
- **Convergence Status**: {"SUCCESS (Converged)" if result.converged else "FAILURE / DIVERGED"}
- **Computed Root (x*)**: {result.final_root if result.final_root is not None else "N/A"}
- **Function Value at Root f(x*)**: {result.final_function_value if result.final_function_value is not None else "N/A"}
- **Absolute Residual |f(x*)|**: {result.absolute_residual if result.absolute_residual is not None else "N/A"}
- **Total Iterations Required (k)**: {result.iteration_count}
- **Total Function Evaluations**: {result.function_evaluations}
- **Execution Time**: {result.execution_time_ms:.3f} ms
- **Stopping Reason**: {result.stopping_reason}
{"- **Warning/Diagnostic**: " + result.warning_message if result.warning_message else ""}

---

## 3. Iteration History Table
{table_md}

---

## 4. Mathematical Interpretation
- **Method Classification**: {"Bracketing (Guaranteed under IVT)" if "Bisection" in result.method_name or "Regula" in result.method_name else "Open Method (Local convergence)"}
- **Convergence Order**: {"Linear (p=1)" if "Bisection" in result.method_name or "Regula" in result.method_name else ("Quadratic (p=2)" if "Newton" in result.method_name else "Superlinear (p≈1.618)")}
- **Observation**: The algorithm terminated after {result.iteration_count} steps. The final residual is {result.absolute_residual:.2e}.
"""
    return report


# ==============================================================================
# TAB 1: HOME / OVERVIEW
# ==============================================================================
if nav_selection == "🏠 Home / Overview":
    st.markdown('<div class="main-title">Numerical Methods Visual Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">An Interactive Root-Finding and Convergence Analyzer for College Mathematics</div>', unsafe_allow_html=True)

    col_intro, col_syllabus = st.columns([3, 2])

    with col_intro:
        st.markdown(
            """
            Welcome to the **Numerical Methods Visual Lab**, an interactive laboratory designed to simulate, 
            visualize, and rigorously analyze algorithms for solving non-linear algebraic and transcendental equations:
            
            $$f(x) = 0$$
            
            Unlike standard calculators that yield only a single number, this visual lab exposes the **entire intermediate 
            mathematical trajectory**, geometric constructions (tangents, secants, and bracketing chords), 
            approximate error decay rates, and failure modes across four fundamental algorithms.
            """
        )

        st.info(
            "📐 **Note on Trigonometry**: All trigonometric calculations (`sin`, `cos`, `tan`) strictly utilize **radian measure**, "
            "conforming to standard calculus definitions."
        )

    with col_syllabus:
        st.markdown("### 🎓 Syllabus Alignment")
        st.markdown(
            """
            - **Course**: College Mathematics / Computational Engineering
            - **Unit**: **Unit 02 — Numerical Methods**
            - **Topic**: Solution of Algebraic & Transcendental Equations
            - **Implemented Methods**:
              1. **Bisection Method** (Bolzano's Interval Halving)
              2. **Regula-Falsi Method** (False Position Chord Method)
              3. **Newton-Raphson Method** (Tangent / Derivative Method)
              4. **Secant Method** (Finite-Difference Quasi-Newton Method)
            """
        )

    st.markdown("---")
    st.subheader("⚡ The Four Numerical Algorithms at a Glance")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown('<span class="badge-bracketing">Bracketing Method</span>', unsafe_allow_html=True)
        st.markdown("#### 1. Bisection")
        st.latex(r"c = \frac{a + b}{2}")
        st.markdown(
            r"""
            - **Order**: Linear ($p = 1$, factor $= 0.5$)
            - **Guarantee**: $100\%$ guaranteed if $f(a) \cdot f(b) < 0$.
            - **Advantage**: Extreme robustness; cannot diverge.
            - **Limitation**: Slow rate of convergence.
            """
        )

    with c2:
        st.markdown('<span class="badge-bracketing">Bracketing Method</span>', unsafe_allow_html=True)
        st.markdown("#### 2. Regula-Falsi")
        st.latex(r"c = \frac{a f(b) - b f(a)}{f(b) - f(a)}")
        st.markdown(
            """
            - **Order**: Linear ($p = 1$)
            - **Guarantee**: Guaranteed bracketing.
            - **Advantage**: Weights endpoints by function magnitude.
            - **Limitation**: Susceptible to **endpoint stagnation**.
            """
        )

    with c3:
        st.markdown('<span class="badge-open">Open Method</span>', unsafe_allow_html=True)
        st.markdown("#### 3. Newton-Raphson")
        st.latex(r"x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}")
        st.markdown(
            """
            - **Order**: Quadratic ($p = 2$)
            - **Guarantee**: None (local convergence only).
            - **Advantage**: Rapid convergence near simple roots.
            - **Limitation**: Requires analytical $f'(x) \neq 0$.
            """
        )

    with c4:
        st.markdown('<span class="badge-open">Open Method</span>', unsafe_allow_html=True)
        st.markdown("#### 4. Secant")
        st.latex(r"x_{k+1} = x_k - f(x_k)\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}")
        st.markdown(
            """
            - **Order**: Superlinear ($p \approx 1.618$)
            - **Guarantee**: None (open method).
            - **Advantage**: No derivative calculation needed.
            - **Limitation**: Can divide by zero if $f(x_k) = f(x_{k-1})$.
            """
        )

    st.markdown("---")
    col_cta1, col_cta2 = st.columns([1, 4])
    with col_cta1:
        st.button(
            "🚀 Open Root Solver Now",
            type="primary",
            use_container_width=True,
            on_click=_set_nav,
            args=(1,),
        )


# ==============================================================================
# TAB 2: ROOT SOLVER
# ==============================================================================
elif nav_selection == "🔍 Root Solver":
    st.markdown('<div class="main-title">Interactive Root Solver</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Configure equations, select algorithms, execute manual iterations, and inspect telemetry</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Section A: Equation Input & Preset Selection
    # --------------------------------------------------------------------------
    with st.expander("📝 Step 1: Input Non-Linear Equation f(x) = 0", expanded=True):
        col_eq_select, col_eq_input = st.columns([1, 1])

        with col_eq_select:
            preset_names = ["-- Select Built-In Preset --"] + [ex["name"] for ex in BUILT_IN_EXAMPLES]
            chosen_preset = st.selectbox("Choose a Syllabus Benchmark Preset:", preset_names)

        default_expr = "x**3 - x - 2"
        preset_info = None
        if chosen_preset != "-- Select Built-In Preset --":
            for ex in BUILT_IN_EXAMPLES:
                if ex["name"] == chosen_preset:
                    default_expr = ex["expr"]
                    preset_info = ex
                    break

        with col_eq_input:
            user_expr_str = st.text_input(
                "Enter Algebraic or Transcendental Expression f(x):",
                value=default_expr,
                help="Support sin(x), cos(x), exp(x), log(x), sqrt(x), x**3, x^2, etc. Function assumed equal to zero: f(x) = 0.",
            )

        if preset_info:
            st.caption(f"ℹ️ **Preset Notes**: {preset_info['description']} *(Approximate root: {preset_info['approx_root']})*")

        # Parse the mathematical expression safely
        try:
            parsed_expr, x_sym, deriv_expr, latex_eq = parse_math_expression(user_expr_str)
            f_scalar = build_numerical_function(parsed_expr, x_sym)
            f_prime_scalar = build_numerical_function(deriv_expr, x_sym)
            f_vec = build_vectorized_function(parsed_expr, x_sym)

            col_disp_eq, col_disp_deriv = st.columns(2)
            with col_disp_eq:
                st.markdown("**Parsed Function f(x):**")
                st.latex(f"f(x) = {latex_eq}")
            with col_disp_deriv:
                st.markdown("**Symbolic Derivative f'(x) (SymPy Computed):**")
                st.latex(f"f'(x) = {sp.latex(deriv_expr)}")

            st.session_state["parsed_data"] = {
                "expr_str": user_expr_str,
                "parsed_expr": parsed_expr,
                "deriv_expr": deriv_expr,
                "f_scalar": f_scalar,
                "f_prime_scalar": f_prime_scalar,
                "f_vec": f_vec,
            }

        except ExpressionParserError as parse_err:
            st.error(f"⚠️ Expression Error: {str(parse_err)}")
            st.stop()
        except Exception as general_err:
            st.error(f"⚠️ Unexpected error parsing equation: {str(general_err)}")
            st.stop()

    # --------------------------------------------------------------------------
    # Section B: Method & Parameter Configuration
    # --------------------------------------------------------------------------
    st.markdown("### ⚙️ Step 2: Method Selection & Solver Parameters")

    col_method, col_params, col_stopping = st.columns([1, 1, 1])

    with col_method:
        selected_method = st.selectbox(
            "Select Numerical Algorithm:",
            [
                "Bisection Method",
                "Regula-Falsi Method",
                "Newton-Raphson Method",
                "Secant Method",
            ],
            index=0,
        )

    # Defaults from preset if available
    init_a = preset_info["bracket"][0] if preset_info else 1.0
    init_b = preset_info["bracket"][1] if preset_info else 2.0
    init_x0 = preset_info["x0"] if preset_info else 1.5
    init_x1 = preset_info["x1"] if preset_info else 2.0

    with col_params:
        if selected_method in ["Bisection Method", "Regula-Falsi Method"]:
            c_a, c_b = st.columns(2)
            with c_a:
                param_a = st.number_input("Lower Bound (a):", value=float(init_a), step=0.5, format="%.4f")
            with c_b:
                param_b = st.number_input("Upper Bound (b):", value=float(init_b), step=0.5, format="%.4f")
        elif selected_method == "Newton-Raphson Method":
            param_x0 = st.number_input("Initial Estimate (x0):", value=float(init_x0), step=0.5, format="%.4f")
        elif selected_method == "Secant Method":
            c_x0, c_x1 = st.columns(2)
            with c_x0:
                param_x0 = st.number_input("First Estimate (x0):", value=float(init_x0), step=0.5, format="%.4f")
            with c_x1:
                param_x1 = st.number_input("Second Estimate (x1):", value=float(init_x1), step=0.5, format="%.4f")

    with col_stopping:
        col_tol, col_iter = st.columns(2)
        with col_tol:
            tol_choice = st.selectbox(
                "Tolerance (ε):",
                [1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
                index=1,
                format_func=lambda v: f"{v:.1e}",
            )
        with col_iter:
            max_iterations = st.number_input("Max Iterations:", min_value=2, max_value=200, value=50, step=5)

        stopping_criterion = st.selectbox(
            "Stopping Criterion:",
            [
                "approx_abs_error",
                "approx_rel_error",
                "residual",
                "interval_width",
                "combined",
            ],
            index=0,
            format_func=lambda c: {
                "approx_abs_error": "Approximate Absolute Error (|x_k - x_{k-1}| < ε)",
                "approx_rel_error": "Approximate Relative Error (E_rel < ε)",
                "residual": "Residual (|f(x_k)| < ε)",
                "interval_width": "Interval Width (|b - a| < ε) [Bracketing Only]",
                "combined": "Combined (|x_k - x_{k-1}| < ε AND |f(x_k)| < ε)",
            }[c],
        )

    # --------------------------------------------------------------------------
    # Section C: Execution & Telemetry Results
    # --------------------------------------------------------------------------
    run_btn = st.button("🚀 Execute Numerical Solver", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner(f"Computing root trajectory using {selected_method}..."):
            f_calc = st.session_state["parsed_data"]["f_scalar"]
            f_prime_calc = st.session_state["parsed_data"]["f_prime_scalar"]

            initial_inputs = {}
            if selected_method == "Bisection Method":
                initial_inputs = {"a": param_a, "b": param_b}
                res = bisection_method(
                    f_calc,
                    param_a,
                    param_b,
                    tol=tol_choice,
                    max_iter=int(max_iterations),
                    stopping_criterion=stopping_criterion,
                    equation_str=user_expr_str,
                )
            elif selected_method == "Regula-Falsi Method":
                initial_inputs = {"a": param_a, "b": param_b}
                res = regula_falsi_method(
                    f_calc,
                    param_a,
                    param_b,
                    tol=tol_choice,
                    max_iter=int(max_iterations),
                    stopping_criterion=stopping_criterion,
                    equation_str=user_expr_str,
                )
            elif selected_method == "Newton-Raphson Method":
                initial_inputs = {"x0": param_x0}
                res = newton_raphson_method(
                    f_calc,
                    f_prime_calc,
                    param_x0,
                    tol=tol_choice,
                    max_iter=int(max_iterations),
                    stopping_criterion=stopping_criterion,
                    equation_str=user_expr_str,
                )
            else:  # Secant Method
                initial_inputs = {"x0": param_x0, "x1": param_x1}
                res = secant_method(
                    f_calc,
                    param_x0,
                    param_x1,
                    tol=tol_choice,
                    max_iter=int(max_iterations),
                    stopping_criterion=stopping_criterion,
                    equation_str=user_expr_str,
                )

            st.session_state["last_solver_result"] = res
            st.session_state["last_inputs"] = initial_inputs

    # Display results if available
    result: Optional[SolverResult] = st.session_state.get("last_solver_result")
    if result:
        st.markdown("---")
        st.subheader(f"📊 Computation Outcome: {result.method_name}")

        # Status banners
        if result.converged:
            st.success(f"✅ **Convergence Achieved!** {result.stopping_reason}")
        else:
            st.error(f"❌ **Algorithm Failed or Terminated Without Full Convergence:** {result.stopping_reason}")

        if result.warning_message:
            st.warning(f"⚠️ **Diagnostic Notice**: {result.warning_message}")

        # Metrics bar
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.metric("Computed Root x*", f"{result.final_root:.8f}" if result.final_root is not None else "N/A")
        with m2:
            st.metric("Final Residual |f(x*)|", f"{result.absolute_residual:.2e}" if result.absolute_residual is not None else "N/A")
        with m3:
            st.metric("Iterations (k)", f"{result.iteration_count}")
        with m4:
            st.metric("Function Evals", f"{result.function_evaluations}")
        with m5:
            st.metric("Execution Time", f"{result.execution_time_ms:.2f} ms")
        with m6:
            st.metric("Status", "Converged" if result.converged else "Not Converged")

        # Function Plot
        f_v = st.session_state["parsed_data"]["f_vec"]
        root_val = result.final_root if result.final_root is not None else 1.0
        plot_window = (root_val - 3.0, root_val + 3.0)

        fig_func = create_function_plot(
            f_v,
            plot_window,
            final_root=result.final_root,
            equation_str=result.equation_str,
        )
        st.plotly_chart(fig_func, use_container_width=True)

        # Iteration Table
        st.markdown("### 📋 Iteration Telemetry History")
        df_iter = result.to_dataframe()

        if not df_iter.empty:
            # Format numeric columns for clean academic readability
            formatted_df = df_iter.copy()
            for col in formatted_df.columns:
                if col in ["Root Estimate (x_k)", "f(x_k)", "f'(x_k)", "Lower Bound (a)", "Upper Bound (b)", "Interval Width (b-a)"]:
                    formatted_df[col] = formatted_df[col].apply(lambda v: f"{v:.8f}" if pd.notnull(v) else "")
                elif col in ["Approx Abs Error (|x_k - x_{k-1}|)", "Approx Rel Error"]:
                    formatted_df[col] = formatted_df[col].apply(lambda v: f"{v:.4e}" if pd.notnull(v) else "")
                elif col in ["Approx % Error"]:
                    formatted_df[col] = formatted_df[col].apply(lambda v: f"{v:.4f}%" if pd.notnull(v) else "")

            st.dataframe(formatted_df, use_container_width=True)

            # Export Options
            col_csv, col_rep = st.columns(2)
            with col_csv:
                csv_bytes = df_iter.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Iteration History (CSV)",
                    data=csv_bytes,
                    file_name=f"iteration_history_{result.method_name.replace(' ', '_').lower()}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_rep:
                report_md = generate_markdown_report(result, st.session_state.get("last_inputs", {}))
                st.download_button(
                    label="📄 Download Experiment Report (Markdown)",
                    data=report_md.encode("utf-8"),
                    file_name=f"experiment_report_{result.method_name.replace(' ', '_').lower()}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
        else:
            st.info("No iteration steps were recorded for this run.")


# ==============================================================================
# TAB 3: STEP-BY-STEP ITERATION VISUALIZER
# ==============================================================================
elif nav_selection == "🎬 Iteration Visualizer":
    st.markdown('<div class="main-title">Step-by-Step Geometric Iteration Visualizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Inspect how chords, tangents, and intervals evolve step by step on the function curve</div>', unsafe_allow_html=True)

    result: Optional[SolverResult] = st.session_state.get("last_solver_result")

    if not result or not result.iteration_history:
        st.warning("⚠️ No solver execution found in session! Please go to **'🔍 Root Solver'** and run an algorithm first.")
        if st.button("Go to Root Solver"):
            st.session_state["nav_selection"] = "🔍 Root Solver"
            st.rerun()
    else:
        history = result.iteration_history
        max_step = len(history)

        col_slider, col_info = st.columns([2, 1])

        with col_slider:
            step_idx = st.slider(
                "Select Iteration Step to Inspect:",
                min_value=1,
                max_value=max_step,
                value=1,
                step=1,
                help="Slide to observe the step-by-step geometric construction.",
            )

        rec = history[step_idx - 1]

        with col_info:
            st.markdown(f"**Method**: `{result.method_name}`")
            st.markdown(f"**Viewing Step**: `{step_idx} of {max_step}`")
            st.markdown(f"**Estimate (x_k)**: `{rec.current_estimate:.8f}`")
            st.markdown(f"**f(x_k)**: `{rec.function_value:.2e}`")
            if rec.approx_abs_error is not None:
                st.markdown(f"**Approx Abs Error**: `{rec.approx_abs_error:.2e}`")

        # Step geometry plot
        f_v = st.session_state["parsed_data"]["f_vec"]
        root_val = result.final_root if result.final_root is not None else rec.current_estimate
        plot_window = (root_val - 2.5, root_val + 2.5)

        fig_step = create_step_iteration_plot(
            result,
            step_idx,
            f_v,
            plot_window,
        )
        st.plotly_chart(fig_step, use_container_width=True)

        # Step decision commentary
        st.info(f"💡 **Step #{step_idx} Mathematical Decision**: {rec.notes if rec.notes else 'Standard algorithmic update applied.'}")

        # Semi-log convergence plot for this run
        st.markdown("---")
        st.subheader("📉 Convergence Trajectory (Error Decay Curve)")
        use_log = st.checkbox("Logarithmic Error Scale (Semilogy)", value=True)
        fig_conv = create_convergence_plot(result, log_scale=use_log)
        st.plotly_chart(fig_conv, use_container_width=True)


# ==============================================================================
# TAB 4: COMPARE METHODS
# ==============================================================================
elif nav_selection == "📊 Compare Methods":
    st.markdown('<div class="main-title">Multi-Method Comparison Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Benchmark all four root-finding algorithms simultaneously on the same non-linear equation</div>', unsafe_allow_html=True)

    # Equation Input
    curr_eq = st.session_state.get("current_equation", "x**3 - x - 2")
    comp_eq = st.text_input("Equation f(x) = 0 for Comparison:", value=curr_eq)

    try:
        parsed_expr, x_sym, deriv_expr, latex_eq = parse_math_expression(comp_eq)
        f_calc = build_numerical_function(parsed_expr, x_sym)
        f_prime_calc = build_numerical_function(deriv_expr, x_sym)
        f_vec = build_vectorized_function(parsed_expr, x_sym)
    except Exception as err:
        st.error(f"Error parsing comparison equation: {str(err)}")
        st.stop()

    st.markdown(f"**Solving:** $f(x) = {latex_eq} = 0$")

    # Initial parameters per method
    st.markdown("### 🎛️ Starting Parameters for Each Algorithm")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)

    with col_p1:
        st.markdown("**1. Bisection**")
        b_a = st.number_input("Bisect a:", value=1.0, step=0.5, key="comp_b_a")
        b_b = st.number_input("Bisect b:", value=2.0, step=0.5, key="comp_b_b")

    with col_p2:
        st.markdown("**2. Regula-Falsi**")
        rf_a = st.number_input("RF a:", value=1.0, step=0.5, key="comp_rf_a")
        rf_b = st.number_input("RF b:", value=2.0, step=0.5, key="comp_rf_b")

    with col_p3:
        st.markdown("**3. Newton-Raphson**")
        nr_x0 = st.number_input("NR x0:", value=1.5, step=0.5, key="comp_nr_x0")

    with col_p4:
        st.markdown("**4. Secant**")
        sec_x0 = st.number_input("Secant x0:", value=1.0, step=0.5, key="comp_sec_x0")
        sec_x1 = st.number_input("Secant x1:", value=2.0, step=0.5, key="comp_sec_x1")

    col_bench_tol, col_bench_iter = st.columns(2)
    with col_bench_tol:
        comp_tol = st.selectbox(
            "Benchmark Tolerance:",
            [1e-4, 1e-6, 1e-8, 1e-10],
            index=1,
            format_func=lambda v: f"{v:.1e}",
            key="comp_tol",
        )
    with col_bench_iter:
        comp_max_iter = st.number_input("Max Iterations:", value=60, min_value=5, max_value=200, key="comp_max_iter")

    if st.button("🏁 Run Multi-Method Benchmark", type="primary", use_container_width=True):
        with st.spinner("Executing all four solvers..."):
            res_bis = bisection_method(f_calc, b_a, b_b, tol=comp_tol, max_iter=comp_max_iter, equation_str=comp_eq)
            res_rf = regula_falsi_method(f_calc, rf_a, rf_b, tol=comp_tol, max_iter=comp_max_iter, equation_str=comp_eq)
            res_nr = newton_raphson_method(f_calc, f_prime_calc, nr_x0, tol=comp_tol, max_iter=comp_max_iter, equation_str=comp_eq)
            res_sec = secant_method(f_calc, sec_x0, sec_x1, tol=comp_tol, max_iter=comp_max_iter, equation_str=comp_eq)

            all_results = [res_bis, res_rf, res_nr, res_sec]
            st.session_state["comparison_results"] = all_results

    comp_results = st.session_state.get("comparison_results")
    if comp_results:
        st.markdown("---")
        st.subheader("📋 Performance Comparison Table")

        table_data = []
        for r in comp_results:
            last_err = r.iteration_history[-1].approx_abs_error if r.iteration_history else None
            table_data.append(
                {
                    "Method": r.method_name,
                    "Computed Root": f"{r.final_root:.8f}" if r.final_root is not None else "N/A",
                    "Iterations (k)": r.iteration_count,
                    "Function Evaluations": r.function_evaluations,
                    "Residual |f(x*)|": f"{r.absolute_residual:.2e}" if r.absolute_residual is not None else "N/A",
                    "Final Approx Error": f"{last_err:.2e}" if last_err is not None else "N/A",
                    "Execution Time (ms)": f"{r.execution_time_ms:.2f}",
                    "Status": "✅ Converged" if r.converged else "❌ Failed",
                    "Stopping Reason": r.stopping_reason,
                }
            )

        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # Graphs: Multi-method convergence & Cost bar chart
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            fig_multi = create_multi_method_convergence_comparison(comp_results)
            st.plotly_chart(fig_multi, use_container_width=True)

        with c_chart2:
            fig_cost = create_cost_comparison_barchart(comp_results)
            st.plotly_chart(fig_cost, use_container_width=True)

        # Mathematical synthesis commentary
        st.markdown("### 🎓 Academic Synthesis & Performance Interpretation")

        # Analyze fast convergence vs cost
        converged_list = [r for r in comp_results if r.converged]
        if converged_list:
            fastest_iter = min(converged_list, key=lambda r: r.iteration_count)
            least_fev = min(converged_list, key=lambda r: r.function_evaluations)

            st.markdown(
                f"""
                - **Iteration Efficiency**: `{fastest_iter.method_name}` reached the tolerance in the fewest iterations (**{fastest_iter.iteration_count} steps**).
                - **Computational Cost**: `{least_fev.method_name}` incurred the fewest total function evaluations (**{least_fev.function_evaluations} evaluations**).
                - **Crucial Distinction**: Note that Newton-Raphson evaluates **both** $f(x)$ and $f'(x)$ on every iteration (2 evaluations/step), whereas Secant evaluates only 1 new function value per step after initialization. Thus, fewer iterations do not always guarantee lower computational overhead if derivative evaluations are computationally demanding!
                - **Reliability vs Speed Tradeoff**: Bisection guarantees convergence under the Intermediate Value Theorem regardless of starting proximity, while Newton and Secant converge much faster when placed close to the root but can diverge if initial estimates are poorly chosen.
                """
            )


# ==============================================================================
# TAB 5: ERROR ANALYSIS
# ==============================================================================
elif nav_selection == "📉 Error Analysis":
    st.markdown('<div class="main-title">Numerical Error Analysis & Condition Theory</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Understanding true errors, approximate errors, residuals, and numerical sensitivity</div>', unsafe_allow_html=True)

    expl = get_error_analysis_explanation()

    col_e1, col_e2 = st.columns(2)

    with col_e1:
        st.markdown("### 1. Approximate vs True Errors")
        st.markdown(
            r"""
            In practical root-finding, the **true analytical root** $x^*$ is unknown (which is why numerical methods are deployed).
            Hence, solvers rely on successive approximations:
            
            - **Absolute Approximate Error**:
              $$E_{\text{abs}} = |x_k - x_{k-1}|$$
            - **Relative Approximate Error**:
              $$E_{\text{rel}} = \frac{|x_k - x_{k-1}|}{|x_k|} \quad (x_k \neq 0)$$
            - **Percentage Error**:
              $$E_{\%} = 100 \times E_{\text{rel}}$$
            """
        )
        st.info(expl["approximate_vs_true"])

    with col_e2:
        st.markdown("### 2. Residual vs Root Error")
        st.markdown(
            r"""
            The **residual** is defined as:
            $$R = |f(x_k)|$$
            
            **Warning**: A small residual does **not** guarantee a small error in the root estimate!
            By Taylor's theorem:
            $$f(x_k) \approx f'(x^*) (x_k - x^*)$$
            $$|x_k - x^*| \approx \frac{|f(x_k)|}{|f'(x^*)|}$$
            """
        )
        st.warning(expl["residual_vs_error"])

    st.markdown("---")
    st.markdown("### 3. Condition Number for Root-Finding")
    st.markdown(
        r"""
        The condition number measures how sensitive the root is to perturbations in the function:
        $$\kappa = \frac{1}{|f'(x^*)|}$$
        - When $f'(x^*) \approx 0$ (such as multiple roots like $f(x) = (x-1)^3$), $\kappa \to \infty$. The problem is **ill-conditioned**, and standard Newton-Raphson loses its quadratic convergence rate, degenerating to linear convergence.
        - When $|f'(x^*)| \gg 1$, the problem is **well-conditioned**, but residuals can remain large even when the root estimate is extremely accurate.
        """
    )


# ==============================================================================
# TAB 6: THEORY & VIVA PREP
# ==============================================================================
elif nav_selection == "📚 Theory & Viva Prep":
    st.markdown('<div class="main-title">Theory, Derivations & College Viva Voce Examination Guide</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Complete academic reference for Unit 02 Numerical Methods syllabus</div>', unsafe_allow_html=True)

    tab_th, tab_viva = st.tabs(["📖 Mathematical Theory & Derivations", "🎯 Viva Voce Examination Questions (15+)"])

    with tab_th:
        st.markdown("### 1. Bisection Method (Interval Halving)")
        st.markdown(
            r"""
            - **Basis**: Intermediate Value Theorem (IVT). If $f \in C[a, b]$ and $f(a) \cdot f(b) < 0$, there exists at least one $c \in (a, b)$ such that $f(c) = 0$.
            - **Formula**: $c_k = \frac{a_k + b_k}{2}$
            - **Convergence Analysis**:
              $$\text{Width of interval at step } k: \quad \Delta x_k = \frac{b_0 - a_0}{2^k}$$
              $$\text{Maximum error bound}: \quad |c_k - x^*| \le \frac{b_0 - a_0}{2^k}$$
            - **Order of Convergence**: Linear ($p = 1$, rate factor $C = 0.5$).
            - **Number of iterations needed for tolerance $\epsilon$**:
              $$k \ge \frac{\ln(b_0 - a_0) - \ln(\epsilon)}{\ln(2)}$$
            """
        )

        st.markdown("---")
        st.markdown("### 2. Regula-Falsi Method (False Position)")
        st.markdown(
            r"""
            - **Basis**: Linear interpolation between $(a, f(a))$ and $(b, f(b))$.
            - **Formula**:
              $$\text{Equation of chord: } y - f(a) = \frac{f(b) - f(a)}{b - a}(x - a)$$
              $$\text{Setting } y = 0: \quad c = a - f(a)\frac{b - a}{f(b) - f(a)} = \frac{a f(b) - b f(a)}{f(b) - f(a)}$$
            - **Endpoint Stagnation**: On convex/concave curves, the chord always lies to one side of the curve, causing one endpoint to remain fixed while the other creeps slowly toward the root.
            """
        )

        st.markdown("---")
        st.markdown("### 3. Newton-Raphson Method")
        st.markdown(
            r"""
            - **Basis**: First-order Taylor series expansion about $x_k$:
              $$f(x) \approx f(x_k) + f'(x_k)(x - x_k) = 0 \implies x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$
            - **Convergence Order**: Quadratic ($p = 2$) for simple roots ($f'(x^*) \neq 0$):
              $$e_{k+1} \approx \frac{f''(x^*)}{2 f'(x^*)} e_k^2$$
            - **Failure Modes**:
              1. $f'(x_k) = 0$: Tangent is horizontal; division by zero.
              2. Cycle / Oscillation: e.g. bouncing symmetrically between two points.
              3. Runaway divergence: Inflection point throws iterate to infinity.
            """
        )

        st.markdown("---")
        st.markdown("### 4. Secant Method")
        st.markdown(
            r"""
            - **Basis**: Finite difference approximation to $f'(x_k)$:
              $$f'(x_k) \approx \frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}$$
            - **Formula**:
              $$x_{k+1} = x_k - f(x_k)\frac{x_k - x_{k-1}}{f(x_k) - f(x_{k-1})}$$
            - **Convergence Order**: Superlinear ($p = \frac{1 + \sqrt{5}}{2} \approx 1.618$ — Golden Ratio).
            - **Efficiency**: Requires only 1 function evaluation per step, making its computational efficiency index $1.618^1 \approx 1.618$, compared to Newton's $2^{1/2} \approx 1.414$.
            """
        )

    with tab_viva:
        st.markdown("### 🎯 15+ Viva Voce Questions & Model Answers for Examination")

        viva_qa = [
            (
                "Q1. What is the fundamental difference between bracketing and open methods?",
                "Bracketing methods (Bisection, Regula-Falsi) require two initial bounds [a, b] that bracket the root with opposite signs (f(a)*f(b) < 0). They are guaranteed to converge under continuity (Intermediate Value Theorem). Open methods (Newton-Raphson, Secant) require one or two initial estimates without bracketing requirements; they converge much faster locally but can diverge or fail.",
            ),
            (
                "Q2. State the order of convergence for each of the four methods.",
                "1. Bisection: Order p = 1 (Linear convergence with error reduction factor 0.5 per step).\n2. Regula-Falsi: Order p = 1 (Linear convergence, often slowed by stagnation).\n3. Newton-Raphson: Order p = 2 (Quadratic convergence for simple roots).\n4. Secant Method: Order p = (1 + sqrt(5))/2 ≈ 1.618 (Superlinear convergence).",
            ),
            (
                "Q3. Why does Newton-Raphson have quadratic convergence?",
                "By Taylor expansion of f(x*) = 0 around x_k: 0 = f(x_k) + f'(x_k)(x* - x_k) + 0.5*f''(ξ)(x* - x_k)^2. Dividing by f'(x_k) and subtracting from x_k gives e_{k+1} = (f''(ξ) / (2*f'(x_k))) * e_k^2. Thus, the error in step k+1 is proportional to the square of the error in step k.",
            ),
            (
                "Q4. What is endpoint stagnation in the Regula-Falsi method?",
                "Endpoint stagnation occurs when the curvature (convexity or concavity) of f(x) causes the chord to consistently intersect the x-axis on the same side of the root. As a result, one endpoint never updates, and the interval width does not shrink to zero. Modified algorithms like the Illinois method mitigate this by halving f at the stagnant endpoint.",
            ),
            (
                "Q5. When does the Newton-Raphson method fail?",
                "Newton-Raphson fails when:\n1. The derivative f'(x_k) is zero or very close to zero (horizontal tangent causes division by zero or infinite overshoot).\n2. The initial estimate x0 is too far from the root (enters an oscillatory cycle or diverges).\n3. The root has multiplicity m > 1 (convergence degrades to linear).",
            ),
            (
                "Q6. Why is the Secant method often preferred over Newton-Raphson in engineering?",
                "Because Newton-Raphson requires an explicit analytical expression for the derivative f'(x), which may be difficult, expensive, or impossible to evaluate (e.g. when f is the output of a black-box numerical simulation). The Secant method approximates the derivative using function differences and requires only 1 new function evaluation per step.",
            ),
            (
                "Q7. Explain why iterations count and function evaluations count are different.",
                "An iteration is a single algorithmic loop step. However, different algorithms evaluate different numbers of functions per step. For instance, Newton-Raphson evaluates both f(x) and f'(x) (2 evaluations per iteration), while Bisection and Secant evaluate only 1 new function value per iteration.",
            ),
            (
                "Q8. Can a numerical method achieve a very small residual |f(x)| but still have a large error in the root?",
                "Yes! If the function is flat near the root (f'(x*) ≈ 0), |f(x)| can be extremely small (e.g., 1e-6) while |x - x*| is relatively large (e.g., 0.01). This is governed by condition number kappa = 1 / |f'(x*)|.",
            ),
            (
                "Q9. What are the common stopping criteria used in root-finding?",
                "1. Absolute approximate error: |x_k - x_{k-1}| < tol\n2. Relative approximate error: |x_k - x_{k-1}| / |x_k| < tol\n3. Residual: |f(x_k)| < tol\n4. Interval width: |b_k - a_k| < tol (only applicable to bracketing methods)\n5. Maximum iterations limit to prevent infinite loops.",
            ),
            (
                "Q10. How many iterations does Bisection require to achieve tolerance ε on [a, b]?",
                "Since interval width after k steps is (b - a)/2^k, setting (b - a)/2^k < ε yields k > (ln(b - a) - ln(ε)) / ln(2). For example, finding a root in [1, 2] to 1e-6 requires ceil(ln(1e6)/ln(2)) = 20 iterations.",
            ),
            (
                "Q11. What is the Intermediate Value Theorem, and what is its role here?",
                "If f is continuous on [a, b] and f(a) and f(b) have opposite signs (f(a)*f(b) < 0), then there exists at least one real value c in (a, b) such that f(c) = 0. This provides the mathematical existence guarantee for Bisection and Regula-Falsi.",
            ),
            (
                "Q12. What happens if f(x) has a multiple root (e.g. (x - 2)^2 = 0)?",
                "At a multiple root of multiplicity m > 1, f(x*) = 0 and f'(x*) = 0. For Newton-Raphson, convergence becomes linear with rate factor (1 - 1/m). Modified Newton's method x_{k+1} = x_k - m * f(x_k)/f'(x_k) restores quadratic convergence if m is known.",
            ),
            (
                "Q13. How does our application handle division by zero in the Secant method?",
                "If f(x_k) == f(x_{k-1}), the denominator in the Secant formula becomes zero. Our solver detects when |f(x_k) - f(x_{k-1})| < 1e-15 before division, safely terminates, and returns a clear warning message rather than raising an unhandled exception.",
            ),
            (
                "Q14. Why is SymPy preferred over Python eval() for equation parsing?",
                "Python's raw eval() is unsafe because it executes arbitrary code, posing serious security vulnerabilities (code injection). SymPy's parse_expr parses the input into an Abstract Syntax Tree (AST), validates mathematical tokens against a strict whitelist, and allows symbolic derivation of f'(x).",
            ),
            (
                "Q15. What is the Computational Efficiency Index (Ostrowski index)?",
                "The Computational Efficiency Index is defined as Eff = p^(1/d), where p is the order of convergence and d is the number of function evaluations per step. For Newton-Raphson, p=2, d=2, so Eff = 2^(1/2) ≈ 1.414. For Secant, p≈1.618, d=1, so Eff = 1.618^1 = 1.618. Thus, Secant is asymptotically more efficient than Newton-Raphson per function evaluation!",
            ),
            (
                "Q16. What angle unit is used for trigonometric functions in numerical analysis?",
                "Radians are used exclusively. In calculus, standard derivative formulas like d/dx(sin x) = cos x are valid only when x is in radians. If degrees were used, a scale factor of π/180 would appear in every derivative.",
            ),
        ]

        for q, a in viva_qa:
            with st.expander(f"📌 {q}"):
                st.markdown(a)
