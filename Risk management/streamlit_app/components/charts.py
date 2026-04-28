"""Plotly-Chart-Helpers für die Overview-Page und andere Visualisierungen."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Farb-Konstanten (akademisch-zurückhaltend)
COLOR_BASELINE = "#1a365d"   # marineblau
COLOR_STRESS   = "#d97706"   # bernstein
COLOR_BRENT    = "#0d9488"   # teal
COLOR_RATE     = "#7c3aed"   # violett
COLOR_INTER    = "#dc2626"   # rot
COLOR_NEUTRAL  = "#94a3b8"   # slate-400


# ----------------------------------------------------------------------
# 1. Waterfall (Bridge): Baseline → +Brent → +Rate → +Inter → Stress
# ----------------------------------------------------------------------
def waterfall_decomposition(decomp: dict) -> go.Figure:
    """Plotly-Waterfall der PD/EL-Decomposition."""
    if not decomp:
        return go.Figure()

    base = decomp["baseline"]
    full = decomp["full"]
    unit = decomp["unit"]
    is_pd = "PD" in unit

    # Skalierung für PD: in Prozent; für EL: in Mio EUR
    if is_pd:
        scale = 100
        suffix = "%"
        title = (f"PD-Decomposition · {decomp.get('ticker', '')} "
                 f"(Baseline → Stress)")
        yaxis = "PD [%]"
    else:
        scale = 1e-6
        suffix = " m€"
        title = "Portfolio-EL-Decomposition (Baseline → Stress)"
        yaxis = "Expected Loss [m EUR]"

    measures = ["absolute", "relative", "relative", "relative", "total"]
    x_labels = ["Baseline", "+ ΔBrent", "+ Δr_10y", "+ Interaktion", "Stress (Total)"]
    y_values = [
        base * scale,
        decomp["brent"] * scale,
        decomp["dr"]    * scale,
        decomp["inter"] * scale,
        full * scale,
    ]
    text = [f"{v:+.3f}{suffix}" if i not in (0, 4) else f"{v:.3f}{suffix}"
            for i, v in enumerate(y_values)]

    fig = go.Figure(go.Waterfall(
        name="Decomposition",
        orientation="v",
        measure=measures,
        x=x_labels,
        textposition="outside",
        text=text,
        y=y_values,
        connector={"line": {"color": COLOR_NEUTRAL}},
        increasing={"marker": {"color": COLOR_STRESS}},
        decreasing={"marker": {"color": COLOR_BRENT}},
        totals={"marker": {"color": COLOR_BASELINE}},
    ))
    fig.update_layout(
        title=title,
        yaxis_title=yaxis,
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


# ----------------------------------------------------------------------
# 2. Top-15 Bar Chart: Baseline vs. Stressed PD
# ----------------------------------------------------------------------
def top_n_baseline_vs_stress(
    df: pd.DataFrame, n: int = 15,
    *,
    stress_col: str = "ScenarioPD",
    baseline_col: str = "BaselinePD",
) -> go.Figure:
    """Horizontales Bar-Chart, sortiert nach Stress-PD."""
    valid = df.dropna(subset=[stress_col, baseline_col]).copy()
    valid = valid.sort_values(stress_col, ascending=False).head(n)
    valid = valid.iloc[::-1]   # für plotly: oben = höchste PD

    labels = [f"{tk} · {row['Name']}" for tk, row in valid.iterrows()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=valid[baseline_col] * 100,
        name="Baseline",
        orientation="h",
        marker_color=COLOR_BASELINE,
        text=[f"{v*100:.3f}%" for v in valid[baseline_col]],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=valid[stress_col] * 100,
        name="Stress",
        orientation="h",
        marker_color=COLOR_STRESS,
        text=[f"{v*100:.3f}%" for v in valid[stress_col]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top-{n} Firmen · Baseline vs. Stress-PD",
        barmode="group",
        xaxis_title="PD [%]",
        height=max(400, 22 * n + 100),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    return fig


# ----------------------------------------------------------------------
# 3. Firm × Scenario Heatmap
# ----------------------------------------------------------------------
def firm_scenario_heatmap(scenarios_result: dict) -> go.Figure:
    """38 × 4 Heatmap: PD pro Firma pro Szenario."""
    hm = scenarios_result["heatmap"].copy()
    valid = hm.dropna(subset=[c for c in hm.columns
                              if c not in ("Name", "Sector")])
    if len(valid) == 0:
        return go.Figure()

    # Sortiere absteigend nach Baseline-PD
    valid = valid.sort_values("Baseline", ascending=True)
    scen_cols = [c for c in valid.columns if c not in ("Name", "Sector")]

    z = valid[scen_cols].to_numpy() * 100   # in Prozent
    y_labels = [f"{tk} · {row['Name'][:20]}" for tk, row in valid.iterrows()]

    fig = go.Figure(go.Heatmap(
        z=z, x=scen_cols, y=y_labels,
        colorscale=[[0, "#f0fdf4"], [0.3, "#84cc16"],
                    [0.6, "#facc15"], [1, "#dc2626"]],
        colorbar=dict(title="PD [%]"),
        text=[[f"{v:.3f}%" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 9},
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Firm × Scenario Heatmap (PD in %)",
        height=max(500, 18 * len(valid) + 100),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    return fig


# ----------------------------------------------------------------------
# 4. Sektor-Quantil-Heatmap (Sektoren × Risiko-Quantile)
# ----------------------------------------------------------------------
def sector_quantile_heatmap(mc_df: pd.DataFrame) -> go.Figure:
    """Sektoren × {Baseline, Mean, p50, p95, p99} EAD-gewichtete PD."""
    valid = mc_df.dropna(subset=["BaselinePD", "StressPDMean"]).copy()
    if len(valid) == 0:
        return go.Figure()

    # EAD-Gewichte aus dem Merton-DataFrame ziehen — hier vereinfacht
    # via gleiche Gewichtung pro Firma. Für genaue EAD-Gewichtung
    # müsste merton_df rangezogen werden; reicht für V1.
    grouped = valid.groupby("Sector", dropna=False).mean(numeric_only=True)

    cols_show = ["BaselinePD", "StressPDMean", "StressPDp50",
                 "StressPDp95", "StressPDp99"]
    col_labels = ["Baseline", "Stress μ", "Stress p50",
                  "Stress p95", "Stress p99"]
    z = grouped[cols_show].to_numpy() * 100

    fig = go.Figure(go.Heatmap(
        z=z, x=col_labels, y=grouped.index,
        colorscale=[[0, "#f0fdf4"], [0.3, "#84cc16"],
                    [0.6, "#facc15"], [1, "#dc2626"]],
        colorbar=dict(title="PD [%]"),
        text=[[f"{v:.3f}%" for v in row] for row in z],
        texttemplate="%{text}",
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Sektor-Risiko nach Quantilen (Mittelwert pro Sektor)",
        height=max(350, 35 * len(grouped) + 100),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    return fig


# ----------------------------------------------------------------------
# 5. Reverse Stress: kritische Schwellen-Übersicht
# ----------------------------------------------------------------------
def critical_thresholds_chart(rev_df: pd.DataFrame) -> go.Figure:
    """Scatter ΔBrent vs. Δr-Schwelle pro Firma (mit Sektor-Farbe)."""
    cb_col = next((c for c in rev_df.columns
                   if c.startswith("CritBrent_PD")), None)
    cr_col = next((c for c in rev_df.columns
                   if c.startswith("CritRate_PD")), None)
    if cb_col is None or cr_col is None:
        return go.Figure()

    valid = rev_df.dropna(subset=[cb_col, cr_col]).copy()
    if len(valid) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Keine Firma erreicht im Such-Intervall die Target-Schwelle.<br>"
                 "Modell-Realität — siehe MODEL_ASSUMPTIONS.md §6d.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#475569"),
        )
        fig.update_layout(
            title=f"Reverse Stress · Kritische Schwellen ({cb_col})",
            height=300, margin=dict(l=20, r=20, t=50, b=20),
            plot_bgcolor="white",
        )
        return fig

    fig = go.Figure(go.Scatter(
        x=valid[cb_col], y=valid[cr_col],
        mode="markers+text",
        text=valid.index, textposition="top center",
        marker=dict(size=12, color=COLOR_BRENT, line=dict(color="white", width=2)),
        hovertemplate=(
            "<b>%{text}</b><br>"
            f"Krit. ΔBrent_log: %{{x:+.3f}}<br>"
            f"Krit. Δr: %{{y:+.2f}}pp<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Reverse Stress · ΔBrent / Δr für PD-Target",
        xaxis_title="Krit. ΔBrent (log-Return)",
        yaxis_title="Krit. Δr_10y [pp]",
        height=400, margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1")
    fig.add_vline(x=0, line_dash="dot", line_color="#cbd5e1")
    return fig
