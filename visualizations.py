"""
visualizations.py - Interactive Plotly Visualizations for Root-Finding Algorithms
================================================================================
Generates high-quality, academic-grade interactive figures:
1. Function Landscape Plot: f(x) vs x with root locator, roots interval, and zero axis.
2. Iteration Step Visualizer:
   - Bisection: interval [a, b], midpoint c, function values, sign change shading
   - Regula-Falsi: interpolation chord between (a, f(a)) and (b, f(b)), x-intercept
   - Newton-Raphson: tangent line at (x_k, f(x_k)), x-intercept x_{k+1}, drop lines
   - Secant: chord connecting (x_{k-1}, f(x_{k-1})) and (x_k, f(x_k)), intercept
3. Convergence History Plot: log-scale error decay vs iteration count.
4. Multi-Method Convergence Comparison: overlay curves of all 4 methods.
5. Computational Cost Comparison: iterations vs function evaluations bar chart.
"""

from typing import Callable, Optional, List, Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from solvers import SolverResult, IterationRecord


# Color palette for academic charts
THEME_COLORS = {
    "curve": "#1f77b4",          # Deep Blue
    "zero_line": "#7f7f7f",      # Subtle gray
    "root": "#2ca02c",           # Vibrant green
    "tangent": "#d62728",        # Crimson red
    "secant": "#ff7f0e",         # Orange
    "interval": "rgba(31, 119, 180, 0.15)",  # Light blue shade
    "point_a": "#9467bd",        # Purple
    "point_b": "#8c564b",        # Brown
    "current_point": "#e377c2",  # Magenta
    "grid": "#eaeaea",
}


def create_function_plot(
    f_vec: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    final_root: Optional[float] = None,
    equation_str: str = "f(x) = 0",
    initial_points: Optional[List[float]] = None,
    title_suffix: str = "",
) -> go.Figure:
    """
    Generate an interactive 2D function curve plot with axis guidelines and root marker.
    """
    x_min, x_max = x_range
    if x_min >= x_max:
        x_min, x_max = x_min - 2.0, x_min + 2.0

    # High-resolution sampling
    x_pts = np.linspace(x_min, x_max, 600)
    y_pts = f_vec(x_pts)

    fig = go.Figure()

    # Base function curve
    fig.add_trace(
        go.Scatter(
            x=x_pts,
            y=y_pts,
            mode="lines",
            name=f"f(x) = {equation_str}",
            line=dict(color=THEME_COLORS["curve"], width=2.5),
            hovertemplate="x: %{x:.4f}<br>f(x): %{y:.4f}<extra></extra>",
        )
    )

    # Reference zero line y = 0
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[0, 0],
            mode="lines",
            name="y = 0 (x-axis)",
            line=dict(color=THEME_COLORS["zero_line"], width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    # Initial points marker (e.g., initial bracket or estimates)
    if initial_points:
        for idx, pt in enumerate(initial_points):
            y_val = f_vec(np.array([pt]))[0]
            fig.add_trace(
                go.Scatter(
                    x=[pt],
                    y=[y_val],
                    mode="markers+text",
                    name=f"Initial point {idx + 1} (x = {pt:.2f})",
                    marker=dict(size=10, symbol="diamond", color="#ff7f0e"),
                    text=[f"x_{idx}={pt:.2f}"],
                    textposition="top center",
                    hovertemplate=f"Initial: %{{x:.4f}}<br>f(x): %{{y:.4e}}<extra></extra>",
                )
            )

    # Mark the final computed root
    if final_root is not None and np.isfinite(final_root):
        y_root = f_vec(np.array([final_root]))[0]
        fig.add_trace(
            go.Scatter(
                x=[final_root],
                y=[0 if np.isnan(y_root) else y_root],
                mode="markers+text",
                name=f"Computed Root: {final_root:.6f}",
                marker=dict(
                    color=THEME_COLORS["root"],
                    size=14,
                    symbol="star",
                    line=dict(color="black", width=1.5),
                ),
                text=[f"Root ≈ {final_root:.5f}"],
                textposition="bottom center",
                hovertemplate="Root: %{x:.6f}<br>f(Root): %{y:.2e}<extra></extra>",
            )
        )

    # Clean academic styling
    fig.update_layout(
        title=f"Function Landscape: {equation_str} {title_suffix}",
        xaxis_title="x (independent variable)",
        yaxis_title="f(x)",
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_step_iteration_plot(
    result: SolverResult,
    step_index: int,
    f_vec: Callable[[np.ndarray], np.ndarray],
    x_range: Tuple[float, float],
    f_scalar: Optional[Callable[[float], float]] = None,
) -> go.Figure:
    """
    Generate the step-by-step geometric visualization for a specific iteration step.
    Renders chords, tangents, drop lines, or interval brackets according to algorithm type.
    """
    history = result.iteration_history
    if not history:
        return create_function_plot(f_vec, x_range, equation_str=result.equation_str)

    # Bound step_index to valid range [1, len(history)]
    idx = max(1, min(step_index, len(history)))
    rec: IterationRecord = history[idx - 1]

    # Dynamic viewing window centered on active points
    x_min, x_max = x_range
    pts_to_consider = [rec.current_estimate]
    if rec.lower_bound is not None:
        pts_to_consider.append(rec.lower_bound)
    if rec.upper_bound is not None:
        pts_to_consider.append(rec.upper_bound)
    if rec.previous_estimate is not None:
        pts_to_consider.append(rec.previous_estimate)

    valid_pts = [p for p in pts_to_consider if p is not None and np.isfinite(p)]
    if valid_pts:
        p_min, p_max = min(valid_pts), max(valid_pts)
        margin = max(abs(p_max - p_min) * 0.5, 1.0)
        x_min = min(x_min, p_min - margin)
        x_max = max(x_max, p_max + margin)

    x_pts = np.linspace(x_min, x_max, 500)
    y_pts = f_vec(x_pts)

    fig = go.Figure()

    # Function curve
    fig.add_trace(
        go.Scatter(
            x=x_pts,
            y=y_pts,
            mode="lines",
            name=f"f(x) = {result.equation_str}",
            line=dict(color=THEME_COLORS["curve"], width=2.5),
            hoverinfo="skip",
        )
    )

    # y = 0 axis
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[0, 0],
            mode="lines",
            name="y = 0",
            line=dict(color=THEME_COLORS["zero_line"], width=1.5, dash="dash"),
            hoverinfo="skip",
        )
    )

    method = result.method_name

    # =========================================================================
    # 1. Bisection Method Step Visualization
    # =========================================================================
    if method == "Bisection Method":
        a = rec.lower_bound
        b = rec.upper_bound
        c = rec.current_estimate
        fc = rec.function_value

        if a is not None and b is not None:
            fa = f_vec(np.array([a]))[0]
            fb = f_vec(np.array([b]))[0]

            # Shaded interval region
            fig.add_vrect(
                x0=a,
                x1=b,
                fillcolor=THEME_COLORS["interval"],
                opacity=0.5,
                layer="below",
                line_width=1,
                line_color=THEME_COLORS["point_a"],
                annotation_text=f"Bracket [{a:.3f}, {b:.3f}]",
                annotation_position="top left",
            )

            # Endpoints
            fig.add_trace(
                go.Scatter(
                    x=[a, b],
                    y=[fa, fb],
                    mode="markers+text",
                    name="Bracket Endpoints [a, b]",
                    marker=dict(size=11, color=THEME_COLORS["point_a"], symbol="square"),
                    text=[f"a={a:.3f}<br>(f(a)={fa:.2e})", f"b={b:.3f}<br>(f(b)={fb:.2e})"],
                    textposition=["bottom center", "top center"],
                )
            )

        # Midpoint c
        fig.add_trace(
            go.Scatter(
                x=[c, c],
                y=[0, fc],
                mode="lines+markers",
                name=f"Midpoint c = {c:.4f}",
                line=dict(color=THEME_COLORS["root"], width=2, dash="dot"),
                marker=dict(size=12, color=THEME_COLORS["root"], symbol="star"),
                text=["", f"c={c:.4f}<br>f(c)={fc:.2e}"],
                textposition="top right",
            )
        )

    # =========================================================================
    # 2. Regula-Falsi Method Step Visualization
    # =========================================================================
    elif method == "Regula-Falsi Method":
        a = rec.lower_bound
        b = rec.upper_bound
        c = rec.current_estimate
        fc = rec.function_value

        if a is not None and b is not None:
            fa = f_vec(np.array([a]))[0]
            fb = f_vec(np.array([b]))[0]

            # Secant chord connecting (a, f(a)) and (b, f(b))
            fig.add_trace(
                go.Scatter(
                    x=[a, b],
                    y=[fa, fb],
                    mode="lines+markers",
                    name="Interpolation Chord",
                    line=dict(color=THEME_COLORS["secant"], width=2.5),
                    marker=dict(size=11, color=THEME_COLORS["point_a"], symbol="circle"),
                    text=[f"A({a:.3f}, {fa:.2e})", f"B({b:.3f}, {fb:.2e})"],
                    textposition=["bottom left", "top right"],
                )
            )

            # Vertical guide line from (c, 0) to (c, f(c))
            fig.add_trace(
                go.Scatter(
                    x=[c, c],
                    y=[0, fc],
                    mode="lines",
                    name="Midpoint Drop Line",
                    line=dict(color="gray", width=1.5, dash="dot"),
                    showlegend=False,
                )
            )

        # Intercept estimate c
        fig.add_trace(
            go.Scatter(
                x=[c],
                y=[0],
                mode="markers+text",
                name=f"Chord Intercept c = {c:.4f}",
                marker=dict(size=12, color=THEME_COLORS["root"], symbol="triangle-up"),
                text=[f"c={c:.4f}"],
                textposition="bottom center",
            )
        )

    # =========================================================================
    # 3. Newton-Raphson Method Step Visualization
    # =========================================================================
    elif method == "Newton-Raphson Method":
        x_k = rec.current_estimate
        fx_k = rec.function_value
        fpx_k = rec.derivative_value

        if fpx_k is not None and abs(fpx_k) > 1e-14:
            x_next = x_k - fx_k / fpx_k

            # Tangent line segment spanning beyond x_k and x_next
            span = abs(x_next - x_k) * 1.5 + 0.5
            t_x0 = min(x_k, x_next) - span * 0.3
            t_x1 = max(x_k, x_next) + span * 0.3
            t_y0 = fpx_k * (t_x0 - x_k) + fx_k
            t_y1 = fpx_k * (t_x1 - x_k) + fx_k

            # Tangent line trace
            fig.add_trace(
                go.Scatter(
                    x=[t_x0, t_x1],
                    y=[t_y0, t_y1],
                    mode="lines",
                    name=f"Tangent at x_{idx} (slope f'={fpx_k:.2f})",
                    line=dict(color=THEME_COLORS["tangent"], width=2.5),
                )
            )

            # Drop line from (x_k, 0) to curve (x_k, f(x_k))
            fig.add_trace(
                go.Scatter(
                    x=[x_k, x_k],
                    y=[0, fx_k],
                    mode="lines+markers",
                    name=f"Current point (x_{idx}, f(x_{idx}))",
                    line=dict(color="purple", width=1.5, dash="dot"),
                    marker=dict(size=10, color="purple"),
                )
            )

            # Next root intercept x_{k+1}
            fig.add_trace(
                go.Scatter(
                    x=[x_next],
                    y=[0],
                    mode="markers+text",
                    name=f"Next estimate x_{idx+1} = {x_next:.4f}",
                    marker=dict(size=13, color=THEME_COLORS["root"], symbol="star"),
                    text=[f"x_{idx+1}={x_next:.4f}"],
                    textposition="bottom center",
                )
            )

    # =========================================================================
    # 4. Secant Method Step Visualization
    # =========================================================================
    elif method == "Secant Method":
        x_curr = rec.current_estimate
        f_curr = rec.function_value
        x_prev = rec.previous_estimate

        if x_prev is not None:
            f_prev = f_vec(np.array([x_prev]))[0]

            if abs(f_curr - f_prev) > 1e-15:
                x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

                # Secant line extended across x-axis
                span = abs(max(x_prev, x_curr, x_next) - min(x_prev, x_curr, x_next)) * 1.3 + 0.5
                s_x0 = min(x_prev, x_curr, x_next) - span * 0.2
                s_x1 = max(x_prev, x_curr, x_next) + span * 0.2
                slope = (f_curr - f_prev) / (x_curr - x_prev)
                s_y0 = slope * (s_x0 - x_curr) + f_curr
                s_y1 = slope * (s_x1 - x_curr) + f_curr

                fig.add_trace(
                    go.Scatter(
                        x=[s_x0, s_x1],
                        y=[s_y0, s_y1],
                        mode="lines",
                        name="Secant Line",
                        line=dict(color=THEME_COLORS["secant"], width=2.5),
                    )
                )

                # Secant base points
                fig.add_trace(
                    go.Scatter(
                        x=[x_prev, x_curr],
                        y=[f_prev, f_curr],
                        mode="markers+text",
                        name="Base Estimates",
                        marker=dict(size=10, color="indigo", symbol="circle"),
                        text=[f"x_{idx-1}={x_prev:.3f}", f"x_{idx}={x_curr:.3f}"],
                        textposition=["top left", "bottom right"],
                    )
                )

                # Intercept
                fig.add_trace(
                    go.Scatter(
                        x=[x_next],
                        y=[0],
                        mode="markers+text",
                        name=f"New Estimate x_{idx+1} = {x_next:.4f}",
                        marker=dict(size=12, color=THEME_COLORS["root"], symbol="star"),
                        text=[f"x_{idx+1}={x_next:.4f}"],
                        textposition="bottom center",
                    )
                )

    fig.update_layout(
        title=f"{method} — Iteration Step #{idx} of {len(history)}",
        xaxis_title="x",
        yaxis_title="f(x)",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.85)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_convergence_plot(
    result: SolverResult,
    log_scale: bool = True,
) -> go.Figure:
    """
    Generate an error decay vs iteration number convergence graph.
    """
    history = result.iteration_history
    if not history:
        fig = go.Figure()
        fig.update_layout(title="No iteration history available.")
        return fig

    iters = [rec.iteration for rec in history]
    e_abs_list = []
    e_rel_list = []
    res_list = []

    for rec in history:
        # Abs error
        val_abs = rec.approx_abs_error if rec.approx_abs_error is not None else np.nan
        e_abs_list.append(max(val_abs, 1e-16) if log_scale and np.isfinite(val_abs) else val_abs)

        # Rel error
        val_rel = rec.approx_rel_error if rec.approx_rel_error is not None else np.nan
        e_rel_list.append(max(val_rel, 1e-16) if log_scale and np.isfinite(val_rel) else val_rel)

        # Residual
        val_res = abs(rec.function_value)
        res_list.append(max(val_res, 1e-16) if log_scale and np.isfinite(val_res) else val_res)

    fig = go.Figure()

    # Absolute approximate error
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=e_abs_list,
            mode="lines+markers",
            name="Approx Abs Error (|x_k - x_{k-1}|)",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            hovertemplate="Iter %{x}: E_abs = %{y:.2e}<extra></extra>",
        )
    )

    # Relative approximate error
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=e_rel_list,
            mode="lines+markers",
            name="Approx Rel Error (E_rel)",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Iter %{x}: E_rel = %{y:.2e}<extra></extra>",
        )
    )

    # Residual
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=res_list,
            mode="lines+markers",
            name="Residual |f(x_k)|",
            line=dict(color="#d62728", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="Iter %{x}: Residual = %{y:.2e}<extra></extra>",
        )
    )

    # Tolerance reference line
    fig.add_hline(
        y=result.tolerance,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Tolerance ({result.tolerance:.1e})",
        annotation_position="bottom right",
    )

    y_axis_type = "log" if log_scale else "linear"

    fig.update_layout(
        title=f"Convergence Trajectory — {result.method_name}",
        xaxis_title="Iteration Number (k)",
        yaxis_title="Error Magnitude (log scale)" if log_scale else "Error Magnitude",
        yaxis_type=y_axis_type,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_multi_method_convergence_comparison(
    results: List[SolverResult],
    metric: str = "approx_abs_error",
) -> go.Figure:
    """
    Overlay convergence trajectories for multiple methods on the same plot.
    """
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, res in enumerate(results):
        if not res.iteration_history:
            continue

        iters = []
        errors = []
        for rec in res.iteration_history:
            iters.append(rec.iteration)
            if metric == "residual":
                val = abs(rec.function_value)
            elif metric == "approx_rel_error":
                val = rec.approx_rel_error if rec.approx_rel_error is not None else np.nan
            else:
                val = rec.approx_abs_error if rec.approx_abs_error is not None else np.nan

            errors.append(max(val, 1e-16) if np.isfinite(val) else np.nan)

        color = colors[idx % len(colors)]
        status_suffix = " (Converged)" if res.converged else " (Failed/Diverged)"

        fig.add_trace(
            go.Scatter(
                x=iters,
                y=errors,
                mode="lines+markers",
                name=f"{res.method_name}{status_suffix}",
                line=dict(color=color, width=2.5),
                marker=dict(size=6),
                hovertemplate=f"{res.method_name}<br>Iter: %{{x}}<br>Error: %{{y:.2e}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Multi-Method Convergence Rate Comparison (Logarithmic Scale)",
        xaxis_title="Iteration Number",
        yaxis_title=f"Error Metric: {metric} (log scale)",
        yaxis_type="log",
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def create_cost_comparison_barchart(results: List[SolverResult]) -> go.Figure:
    """
    Bar chart comparing iterations vs function evaluations across all methods.
    """
    methods = [res.method_name for res in results]
    iters = [res.iteration_count for res in results]
    fevs = [res.function_evaluations for res in results]

    fig = go.Figure(
        data=[
            go.Bar(name="Iterations (k)", x=methods, y=iters, marker_color="#1f77b4"),
            go.Bar(name="Function Evaluations (f & f')", x=methods, y=fevs, marker_color="#ff7f0e"),
        ]
    )

    fig.update_layout(
        barmode="group",
        title="Computational Cost: Iterations vs Function Evaluations",
        xaxis_title="Numerical Method",
        yaxis_title="Count",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig
