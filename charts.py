"""Gráficos Plotly de convergencia y plano complejo."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import plotly.graph_objects as go


COLORS = {"navy": "#0b1f33", "petrol": "#087f73", "mint": "#31c6a4", "red": "#e45858", "muted": "#78909c"}


def convergence_chart(iterations: list[dict[str, object]], tolerance: float) -> go.Figure:
    x = [int(row["iteration"]) for row in iterations]
    y = [float(row["ea"]) for row in iterations]
    positive = all(value > 0 and math.isfinite(value) for value in y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Ea", line=dict(color=COLORS["petrol"], width=3)))
    fig.add_hline(y=tolerance, line_dash="dash", line_color=COLORS["red"], annotation_text=f"ε = {tolerance:g}")
    fig.update_layout(xaxis_title="Iteración", yaxis_title="Error relativo estimado Ea",
                      yaxis_type="log" if positive else "linear", hovermode="x unified",
                      margin=dict(l=20, r=20, t=25, b=20), height=430)
    return fig


def z_plane_chart(roots: Sequence[complex], lagrange_radius: float | None = None) -> go.Figure:
    theta = np.linspace(0.0, 2.0 * math.pi, 361)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta),
                             mode="lines", name="Círculo unitario", line=dict(color=COLORS["petrol"], width=3)))
    if lagrange_radius is not None:
        fig.add_trace(go.Scatter(x=lagrange_radius * np.cos(theta),
                                 y=lagrange_radius * np.sin(theta), mode="lines",
                                 name="Cota de Lagrange", line=dict(color=COLORS["muted"], dash="dot"), visible="legendonly"))
    inside = [abs(root) < 1 for root in roots]
    fig.add_trace(go.Scatter(
        x=[root.real for root in roots], y=[root.imag for root in roots], mode="markers+text",
        text=[f"z{i}" for i in range(1, len(roots) + 1)], textposition="top center", name="Raíces",
        marker=dict(size=14, color=[COLORS["mint"] if flag else COLORS["red"] for flag in inside],
                    line=dict(color=COLORS["navy"], width=1.5)),
        customdata=[[i, root.real, root.imag, abs(root)] for i, root in enumerate(roots, start=1)],
        hovertemplate="z%{customdata[0]}<br>Re = %{customdata[1]:.8f}<br>Im = %{customdata[2]:.8f}<br>|z| = %{customdata[3]:.8f}<extra></extra>",
    ))
    extent = max(1.2, max((abs(root.real) for root in roots), default=1.0) * 1.25,
                 max((abs(root.imag) for root in roots), default=1.0) * 1.25)
    fig.add_hline(y=0, line_width=1, line_color="#aab7c4")
    fig.add_vline(x=0, line_width=1, line_color="#aab7c4")
    fig.update_layout(xaxis_title="Parte real", yaxis_title="Parte imaginaria", height=570,
                      margin=dict(l=20, r=20, t=25, b=20), xaxis=dict(range=[-extent, extent]),
                      yaxis=dict(range=[-extent, extent], scaleanchor="x", scaleratio=1), legend=dict(orientation="h"))
    return fig
