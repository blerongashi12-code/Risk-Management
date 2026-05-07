"""Sovereign Risk · doom-loop concentration + duration-shock P&L."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import (apply_theme, hero, eyebrow, insight, footer,
                              COLORS, PALETTE_DISCRETE, SEQ_COOL_TO_WARM)
from components.sidebar import render_sidebar
from components.methodology import render_sovereign_methodology
from components.backend_path import setup
setup()

from eba_loader import (                                            # type: ignore
    parse_sovereign_csv, sovereign_concentration,
    sovereign_maturity_ladder, domestic_share_per_bank,
    sovereign_kpis_per_bank, attach_country_names,
    rate_shock_pnl, rate_shock_pnl_per_bucket,
    load_bank_directory, load_country_dim,
    MATURITY_BUCKETS, DURATION_BY_BUCKET,
)
from config import EBA_RAW_DIR                                      # type: ignore


st.set_page_config(page_title="Sovereign risk · Tier 2", layout="wide")
apply_theme()
config = render_sidebar()

# === Load data (cached) ===============================================
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA sovereign exposures …")
def _load_sov_data():
    sov_raw = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    cty_dim = load_country_dim(EBA_RAW_DIR / "TR_Metadata.xlsx")
    conc = sovereign_concentration(sov_raw, period=202506)
    mat  = sovereign_maturity_ladder(sov_raw, period=202506)
    conc_named = attach_country_names(conc, cty_dim)
    return sov_raw, bank_dir, cty_dim, conc, conc_named, mat


try:
    sov_raw, bank_dir, cty_dim, conc, conc_named, mat = _load_sov_data()
    data_ok = len(conc) > 0 and len(mat) > 0
except FileNotFoundError as e:
    st.error(f"EBA sovereign file missing: {e}")
    st.stop()
    data_ok = False

if not data_ok:
    st.error("Sovereign data could not be loaded. Check EBA file presence.")
    st.stop()

# Live shock from sidebar
delta_r_pp = config["d_r_10y_pp"] * 100   # decimal → pp

hero(
    "Sovereign Risk View",
    eyebrow="Tier 2 · Sovereign book · EBA Transparency 2025",
    deck="Bank-level sovereign concentration and rate-sensitivity. The same "
         "Δr_10y shock that drives the structural and ASRF tiers now flows "
         "through a duration-weighted Mark-to-Market lens on the EU "
         "sovereign book.",
)

# === Methodology disclosure boxes (Critique 2 + 3) ====================
render_sovereign_methodology()

st.divider()

# === Aggregate KPI strip ==============================================
total_exposure = float(conc["exposure_eur"].sum())
n_banks = conc["LEI_Code"].nunique()

# Aggregate domestic share (volume-weighted)
dom = domestic_share_per_bank(conc, bank_dir, cty_dim)
weighted_dom_share = (
    (dom["domestic_eur"].sum() / dom["total_eur"].sum())
    if dom["total_eur"].sum() > 0 else 0.0
)

# System-wide P&L under live shock
pnl_df = rate_shock_pnl(mat, delta_r_pp=delta_r_pp)
system_pnl = float(pnl_df["delta_pnl_eur"].sum())

# Largest single bank-country pair
biggest = (conc_named.merge(bank_dir[["lei", "bank_name"]],
                            left_on="LEI_Code", right_on="lei")
           .sort_values("exposure_eur", ascending=False).head(1))
biggest_label = (
    f"{biggest.iloc[0]['bank_name'][:18]} → {biggest.iloc[0]['country_iso']}"
    if not biggest.empty else "—"
)
biggest_value = (
    f"€{biggest.iloc[0]['exposure_eur']/1e9:.0f} bn"
    if not biggest.empty else "—"
)

eyebrow("Aggregate sovereign exposure")
c1, c2, c3, c4, c5 = st.columns(5, gap="small")
c1.metric("Σ Sovereign book", f"€{total_exposure/1e12:.2f} tn",
          f"{n_banks} banks", delta_color="off")
c2.metric("Domestic share (vol-wtd)",
          f"{weighted_dom_share*100:.0f}%",
          "of EU banks' sovereign book", delta_color="off")
c3.metric("Largest pair", biggest_label, biggest_value,
          delta_color="off")
if abs(delta_r_pp) > 1e-3:
    c4.metric("System P&L · live shock",
              f"€{system_pnl/1e9:+.1f} bn",
              f"Δr = {delta_r_pp:+.0f} pp")
    # Bank with worst single-bank loss
    worst = (pnl_df.merge(bank_dir[["lei", "bank_name"]],
                          left_on="LEI_Code", right_on="lei")
             .sort_values("delta_pnl_eur").head(1))
    c5.metric("Hardest hit",
              worst.iloc[0]["bank_name"][:20] if not worst.empty else "—",
              f"€{worst.iloc[0]['delta_pnl_eur']/1e9:+.1f} bn"
                  if not worst.empty else "—",
              delta_color="off")
else:
    c4.metric("System P&L · live shock",
              "€0.0 bn", "Δr = 0 (move slider)", delta_color="off")
    c5.metric("Hardest hit", "—", "no shock applied",
              delta_color="off")

if abs(delta_r_pp) > 1e-3:
    insight(
        f"Under a parallel <strong>Δr = {delta_r_pp:+.0f} pp</strong> shock, the "
        f"EU top-{n_banks} sovereign book takes "
        f"<strong>€{system_pnl/1e9:+.0f} bn</strong> in mark-to-market P&L "
        f"({system_pnl/total_exposure*100:+.2f}% of book) — "
        f"approximated via per-bucket modified duration."
    )
else:
    insight(
        f"Σ sovereign exposure is <strong>€{total_exposure/1e12:.2f} tn</strong> "
        f"across {n_banks} European banks; volume-weighted domestic share is "
        f"<strong>{weighted_dom_share*100:.0f}%</strong>. Move the yield-curve "
        f"sliders to see Mark-to-Market P&L under a parallel rate shock."
    )

st.divider()

# === Doom-Loop heatmap (Bank × Country) ===============================
eyebrow("Concentration heatmap · top banks × top sovereigns")

# Pick top-15 banks and top-12 counterparty countries by total exposure
bank_totals = conc.groupby("LEI_Code")["exposure_eur"].sum().nlargest(15)
country_totals = (conc_named.groupby(["country_iso"])["exposure_eur"]
                  .sum().nlargest(12))

heatmap_df = (conc_named.merge(bank_dir[["lei", "bank_name", "country"]],
                               left_on="LEI_Code", right_on="lei")
              [conc_named["LEI_Code"].isin(bank_totals.index)
               & conc_named["country_iso"].isin(country_totals.index)])

# Build pivot
pivot = heatmap_df.pivot_table(
    index="bank_name", columns="country_iso",
    values="exposure_eur", aggfunc="sum", fill_value=0,
)
# Order rows by bank total exposure desc, columns by country total exposure desc
row_order = (heatmap_df.groupby("bank_name")["exposure_eur"].sum()
             .sort_values(ascending=True).index.tolist())  # ascending so big at top
col_order = (country_totals.index.tolist())
pivot = pivot.reindex(index=row_order, columns=col_order, fill_value=0)

# Bank country lookup for diagonal markers
bank_iso = (heatmap_df.drop_duplicates("bank_name").set_index("bank_name")
            ["country"].to_dict())

z = pivot.values / 1e9  # bn EUR
text = [[f"{v:.0f}" if v >= 1.0 else "" for v in row] for row in z]

fig_hm = go.Figure(go.Heatmap(
    z=z, x=pivot.columns, y=pivot.index,
    colorscale=SEQ_COOL_TO_WARM,
    colorbar=dict(title=dict(text="bn EUR",
                             font=dict(size=11, color=COLORS["navy"])),
                  thickness=10, outlinewidth=0),
    text=text,
    texttemplate="%{text}",
    textfont={"size": 9, "color": COLORS["navy"]},
    hovertemplate=("<b>%{y}</b><br>"
                   "Counterparty: %{x}<br>"
                   "Exposure: %{z:.1f} bn EUR<extra></extra>"),
    xgap=1, ygap=1,
))

# Highlight diagonal (bank country == counterparty country)
home_x, home_y = [], []
for bank, iso in bank_iso.items():
    if bank in pivot.index and iso in pivot.columns:
        home_x.append(iso)
        home_y.append(bank)
if home_x:
    fig_hm.add_trace(go.Scatter(
        x=home_x, y=home_y, mode="markers",
        marker=dict(symbol="square-open", size=18,
                    color=COLORS["amber"], line=dict(width=2.5)),
        name="Home country",
        hoverinfo="skip",
    ))

fig_hm.update_layout(
    title="Sovereign exposure (bn EUR) — amber outline marks home-country pairs",
    height=520,
    showlegend=False,
)
st.plotly_chart(fig_hm, use_container_width=True)

st.caption(
    "Diagonal pattern (home-country boxes) = doom-loop concentration. "
    "Italian, Spanish, French and Polish banks sit on the strongest "
    "domestic clusters."
)

st.divider()

# === Concentration ranking table ======================================
eyebrow("Bank-level concentration ranking")

# Build full KPI per bank, then merge with domestic share
kpis = sovereign_kpis_per_bank(conc, bank_dir)
kpis = kpis.merge(dom[["lei", "domestic_share"]], on="lei", how="left")

# Map top_country_code → country name
kpis["top_country_name"] = (
    kpis["top_country_code"].map(cty_dim.set_index("code")["country_name"])
)

display_kpis = (kpis.sort_values("total_eur", ascending=False).head(20).copy())
display_kpis["Total bn"]      = (display_kpis["total_eur"]/1e9).round(0).astype(int)
display_kpis["Domestic %"]    = (display_kpis["domestic_share"]*100).round(0).astype(int)
display_kpis["Top country"]   = display_kpis["top_country_name"].astype(str)
display_kpis["Top share %"]   = (display_kpis["top_share"]*100).round(0).astype(int)
display_kpis["Top-3 share %"] = (display_kpis["top3_share"]*100).round(0).astype(int)
display_kpis["HHI"]           = display_kpis["hhi"].round(3)
display_kpis["N countries"]   = display_kpis["n_countries"]

display_kpis = display_kpis[["bank_name", "bank_country", "Total bn",
                             "Domestic %", "Top country", "Top share %",
                             "Top-3 share %", "HHI", "N countries"]]
display_kpis.columns = ["Bank", "Bank country", "Total bn",
                        "Domestic %", "Top counterparty", "Top share %",
                        "Top-3 share %", "HHI", "N countries"]
st.dataframe(display_kpis, use_container_width=True, height=420,
             hide_index=True)

st.divider()

# === Live rate-shock P&L bar chart ====================================
eyebrow(f"Mark-to-market P&L under live Δr = {delta_r_pp:+.1f} pp")

if abs(delta_r_pp) > 1e-3:
    pnl_named = (pnl_df.merge(bank_dir[["lei", "bank_name"]],
                              left_on="LEI_Code", right_on="lei"))

    pnl_top = (pnl_named.assign(abs_loss=lambda d: d["delta_pnl_eur"].abs())
               .sort_values("abs_loss", ascending=False).head(15)
               .sort_values("delta_pnl_eur"))   # for plotly: most negative top

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(
        y=pnl_top["bank_name"], x=pnl_top["delta_pnl_eur"]/1e9,
        orientation="h",
        marker_color=[COLORS["crimson"] if v < 0 else COLORS["teal"]
                      for v in pnl_top["delta_pnl_eur"]],
        marker_line_width=0,
        text=[f"{v/1e9:+.1f} bn" for v in pnl_top["delta_pnl_eur"]],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["navy"]),
        hovertemplate=("<b>%{y}</b><br>"
                       "P&L: %{x:+.2f} bn EUR<extra></extra>"),
    ))
    fig_pnl.add_vline(x=0, line_color=COLORS["hairline"], line_width=1)
    fig_pnl.update_layout(
        title=f"Top 15 banks by absolute P&L · Δr = {delta_r_pp:+.1f} pp",
        xaxis_title="Mark-to-market P&L [bn EUR]",
        height=520, bargap=0.25,
    )
    st.plotly_chart(fig_pnl, use_container_width=True)
else:
    st.info("Apply a yield-curve shock (sidebar Svensson β-sliders or quick "
            "scenario) to see per-bank Mark-to-Market P&L.")

st.divider()

# === Bank drilldown ===================================================
eyebrow("Bank drilldown · maturity ladder & country composition")

bank_options = sorted(bank_totals.index.tolist(),
                      key=lambda lei: -bank_totals.loc[lei])
lei_to_name = bank_dir.set_index("lei")["bank_name"].to_dict()
sel_lei = st.selectbox(
    "Bank",
    bank_options,
    format_func=lambda lei: f"{lei_to_name.get(lei, lei[:8])}  "
                           f"(€{bank_totals.loc[lei]/1e9:.0f} bn sov book)",
    label_visibility="collapsed",
    key="sov_drill_bank",
)
sel_name = lei_to_name.get(sel_lei, sel_lei[:12])

drill_l, drill_r = st.columns([3, 2], gap="medium")

with drill_l:
    eyebrow(f"{sel_name} · maturity ladder")
    sub_mat = mat[mat["LEI_Code"] == sel_lei].sort_values("Maturity")
    if len(sub_mat) > 0:
        # Bar: exposure per bucket; compute per-bucket P&L if shock active
        bucket_pnl = rate_shock_pnl_per_bucket(sub_mat, delta_r_pp=delta_r_pp)

        fig_mat = go.Figure()
        fig_mat.add_trace(go.Bar(
            x=sub_mat["label"],
            y=sub_mat["exposure_eur"]/1e9,
            name="Exposure",
            marker_color=COLORS["mid_blue"],
            marker_line_width=0,
            text=[f"{v/1e9:.1f}" for v in sub_mat["exposure_eur"]],
            textposition="outside",
            textfont=dict(size=10, color=COLORS["navy"]),
            hovertemplate=("Bucket: %{x}<br>"
                           "Exposure: %{y:.1f} bn EUR<extra></extra>"),
        ))
        # Overlay P&L line if shock
        if abs(delta_r_pp) > 1e-3:
            fig_mat.add_trace(go.Scatter(
                x=bucket_pnl["label"],
                y=bucket_pnl["delta_pnl_eur"]/1e9,
                name=f"P&L @ Δr = {delta_r_pp:+.1f} pp",
                yaxis="y2",
                mode="lines+markers",
                line=dict(color=COLORS["crimson"], width=2.5),
                marker=dict(size=8, color=COLORS["crimson"],
                            line=dict(color=COLORS["white"], width=1.5)),
            ))
            fig_mat.update_layout(yaxis2=dict(
                title=dict(text="P&L [bn EUR]",
                           font=dict(color=COLORS["crimson"])),
                tickfont=dict(color=COLORS["crimson"]),
                overlaying="y", side="right",
                showgrid=False, zeroline=True,
                zerolinecolor=COLORS["hairline"],
            ))
        fig_mat.update_layout(
            xaxis_title="Maturity bucket",
            yaxis_title="Exposure [bn EUR]",
            height=400, bargap=0.25,
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
        )
        st.plotly_chart(fig_mat, use_container_width=True)

        # Summary table
        ladder_disp = sub_mat[["label", "exposure_eur",
                                "duration_years"]].copy()
        ladder_disp["Exposure bn"] = (ladder_disp["exposure_eur"]/1e9).round(1)
        ladder_disp["Duration y"] = ladder_disp["duration_years"]
        if abs(delta_r_pp) > 1e-3:
            ladder_disp["P&L bn"] = (
                bucket_pnl.set_index("Maturity").reindex(sub_mat["Maturity"])
                ["delta_pnl_eur"].values / 1e9
            ).round(2)
        ladder_disp = ladder_disp.drop(columns=["exposure_eur",
                                                  "duration_years"])
        ladder_disp.columns = (["Maturity bucket", "Exposure bn",
                                "Duration y", "P&L bn"]
                               if abs(delta_r_pp) > 1e-3
                               else ["Maturity bucket", "Exposure bn",
                                     "Duration y"])
        st.dataframe(ladder_disp, use_container_width=True, hide_index=True,
                     height=280)
    else:
        st.info("Maturity ladder unavailable for this bank.")

with drill_r:
    eyebrow(f"{sel_name} · country composition")
    sub_conc = (conc_named[conc_named["LEI_Code"] == sel_lei]
                .sort_values("exposure_eur", ascending=False))
    if len(sub_conc) > 0:
        # Top 8 + Other
        top_n = 8
        top_part = sub_conc.head(top_n).copy()
        if len(sub_conc) > top_n:
            other = sub_conc.iloc[top_n:]["exposure_eur"].sum()
            top_part = pd.concat([top_part, pd.DataFrame([{
                "country_iso": "Other", "country_name": "Other",
                "exposure_eur": other,
            }])], ignore_index=True)

        fig_pie = go.Figure(go.Pie(
            labels=top_part["country_iso"],
            values=top_part["exposure_eur"]/1e9,
            hole=0.5,
            marker=dict(colors=PALETTE_DISCRETE,
                        line=dict(color=COLORS["white"], width=2)),
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(family="Inter", size=10, color=COLORS["navy"]),
        ))
        fig_pie.update_layout(
            height=400,
            margin=dict(l=10, r=10, t=20, b=20),
            showlegend=False,
            annotations=[dict(
                text=f"€{sub_conc['exposure_eur'].sum()/1e9:.0f} bn",
                font=dict(family="Source Serif Pro", size=18,
                          color=COLORS["navy"]),
                showarrow=False,
            )],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Country composition unavailable.")

footer(
    f"Source: EBA Transparency 2025 · period 202506 · "
    f"Item 2520810 (on-balance gross carrying amount) · "
    f"Modified duration approximation via bucket midpoints · "
    f"P&L = -D · Δy · Exposure (parallel-shift assumption)"
)
