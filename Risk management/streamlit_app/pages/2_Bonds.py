"""Bonds · Sovereigns + Banking-Book + Trading-Book / ABS-MBS.

Drei Sub-Tabs:
  1. Sovereigns           — Maturity-Ladder, Country-Konzentration, FVOCI/AC
                            Accounting-Class-Split, CET1-Impact via OCI/P&L.
  2. Banking-Book Bonds   — Financials / Corporates / Covered Bonds aus
                            tr_cre.csv Exposure-Class. Aggregat enthält
                            sowohl Bonds als auch Loans (EBA-Limit).
                            CET1-Impact via Vasicek ΔRWA.
  3. Trading Book + ABS   — Market-RWA (Item 2520210) und
                            Securitisation-RWA (Item 2520209). FRTB-
                            Style Stress, kein Issuer-Detail.

Kein erfundener Daten. Wo etwas fehlt, klare "Datenbasis & Annahmen"-Box
mit Datenlücken-Disclaimer.
"""
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

from config import EBA_RAW_DIR, KAPPA_DOWNTURN_LGD                  # type: ignore
from eba_loader import (                                            # type: ignore
    parse_sovereign_csv, sovereign_concentration,
    sovereign_maturity_ladder, domestic_share_per_bank,
    sovereign_kpis_per_bank, attach_country_names,
    rate_shock_pnl,
    sovereign_by_accounting_class, sovereign_cet1_impact,
    parse_credit_risk_csv, loan_book_class_breakdown,
    parse_capital_overview, trading_book_stress,
    load_bank_directory, load_country_dim, load_eba_universe,
    EXPOSURE_TO_BOND_CATEGORY,
)


st.set_page_config(page_title="Bonds · Tier 2", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Bonds",
    eyebrow="Tier 2 · Sovereigns + Banking-Book + Trading-Book",
    deck="Drei Bond-Channels mit konsistenter CET1-Wirkung: Sovereigns "
         "granular nach Country × Maturity × IFRS-9-Accounting-Class, "
         "Banking-Book-Bonds (Financials / Corporates / Covered) aus dem "
         "EBA-IRB-Aggregat, Trading-Book + ABS/MBS via Market-Risk-RWA. "
         "Pro Sub-Tab: ΔFair Value oder ΔRWA → CET1-Quote vorher / nachher.",
)

# === Live macro shock from sidebar ===================================
delta_r_pp = config["d_r_10y_pp"] * 100
d_brent    = config["d_brent"]


# === Cached data loaders =============================================
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Bonds-data …")
def _load_data():
    sov_raw = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    cty_dim  = load_country_dim(EBA_RAW_DIR / "TR_Metadata.xlsx")
    cre_raw  = parse_credit_risk_csv(EBA_RAW_DIR / "tr_cre.csv", period=202506)
    cap_df   = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    return sov_raw, bank_dir, cty_dim, cre_raw, cap_df


sov_raw, bank_dir, cty_dim, cre_raw, cap_df = _load_data()

# Pre-compute the three derived dataframes used across sub-tabs
conc       = sovereign_concentration(sov_raw, period=202506)
conc_named = attach_country_names(conc, cty_dim)
mat        = sovereign_maturity_ladder(sov_raw, period=202506)
acct_split = sovereign_by_accounting_class(sov_raw, period=202506)
lb_class   = loan_book_class_breakdown(cre_raw, period=202506)


# =====================================================================
# Three tabs
# =====================================================================
tab_sov, tab_bb, tab_tb = st.tabs([
    "Sovereigns",
    "Banking-Book Bonds (Financials · Corporates · Covered)",
    "Trading Book + ABS / MBS",
])


# =====================================================================
# SUB-TAB 1 · Sovereigns
# =====================================================================
with tab_sov:
    render_sovereign_methodology()

    with st.expander("Datenbasis & Annahmen · Sovereigns", expanded=False):
        st.markdown("""
**Datenbasis:** EBA Transparency 2025, `tr_sov.csv`, Reporting-Stichtag
Juni 2025.
- **Total exposure** aus Item 2520810 (On-balance gross carrying amount)
- **Accounting-Class-Split** aus Items 2520812 (HfT), 2520813 (FVTPL),
  2520814 (FVOCI), 2520815 (AC). Pro Bank × Country × Maturity-Bucket.

**Wirkungskette für CET1:**
- HfT, FVTPL → ΔFV durchläuft P&L → CET1 (durchschlagend)
- FVOCI → ΔFV via OCI → CET1 (durchschlagend)
- AC, HtM → zu Buchwert, kein direkter CET1-Effekt unter Rate-Stress
  (latenter Verlust nicht erkannt)

**Approximationen:**
- Modified Duration via Bucket-Midpoint (bullet-bond at par)
- Parallel-Shift-Annahme — kein Slope/Curvature-Stress, kein Credit-Spread
- Kein Hedging-Effekt (Swaps/Futures aus EBA-Daten nicht rekonstruierbar)
""")

    # === Aggregate KPI strip ===
    total_exposure = float(conc["exposure_eur"].sum())
    n_banks = conc["LEI_Code"].nunique()
    dom = domestic_share_per_bank(conc, bank_dir, cty_dim)
    weighted_dom_share = (
        dom["domestic_eur"].sum() / dom["total_eur"].sum()
        if dom["total_eur"].sum() > 0 else 0.0
    )
    pnl_df = rate_shock_pnl(mat, delta_r_pp=delta_r_pp)
    system_pnl = float(pnl_df["delta_pnl_eur"].sum()) if abs(delta_r_pp) > 1e-3 else 0.0

    eyebrow("Aggregate sovereign exposure")
    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.metric("Σ Sovereign book", f"€{total_exposure/1e12:.2f} tn",
              f"{n_banks} banks", delta_color="off")
    s2.metric("Domestic share (vol-wtd)", f"{weighted_dom_share*100:.0f}%",
              "of EU sovereign book", delta_color="off")
    if abs(delta_r_pp) > 1e-3:
        s3.metric("System P&L · live shock", f"€{system_pnl/1e9:+.1f} bn",
                  f"Δr = {delta_r_pp:+.0f} pp")
    else:
        s3.metric("System P&L · live shock", "€0 bn",
                  "no shock", delta_color="off")
    s4.metric("Accounting classes", f"{acct_split['accounting_class'].nunique()}",
              "HfT / FVTPL / FVOCI / AC", delta_color="off")

    if abs(delta_r_pp) > 1e-3:
        cet1_acct = sovereign_cet1_impact(acct_split, delta_r_pp=delta_r_pp)
        cet1_total = float(cet1_acct["cet1_impact_eur"].sum())
        cet1_oci   = float(cet1_acct[cet1_acct["channel"]=="OCI"]["cet1_impact_eur"].sum())
        cet1_pnl   = float(cet1_acct[cet1_acct["channel"]=="P&L"]["cet1_impact_eur"].sum())
        ac_unrec   = float(cet1_acct[cet1_acct["channel"]=="none"]["delta_fv_eur"].sum())
        insight(
            f"<strong>Sovereign-CET1-Impact unter Δr = {delta_r_pp:+.0f} pp.</strong> "
            f"Total durchschlagend: <strong>€{cet1_total/1e9:+.1f} bn</strong>. "
            f"Davon via OCI (FVOCI): €{cet1_oci/1e9:+.1f} bn · via P&L "
            f"(HfT + FVTPL): €{cet1_pnl/1e9:+.1f} bn. "
            f"Latenter (nicht-erkannter) AC-Verlust: €{ac_unrec/1e9:+.1f} bn — "
            f"bleibt im HtM/AC-Buch zu Buchwert verborgen."
        )
    else:
        insight(
            "Apply a yield-curve shock in the sidebar to see the CET1-"
            "transmitting Sovereign-FV-Loss broken down by IFRS-9 "
            "accounting class."
        )

    st.divider()

    # === Doom-loop heatmap ===
    eyebrow("Doom-loop concentration · top-15 banks × top-12 sovereigns")
    bank_totals = conc.groupby("LEI_Code")["exposure_eur"].sum().nlargest(15)
    country_totals = (conc_named.groupby(["country_iso"])["exposure_eur"]
                      .sum().nlargest(12))
    heatmap_df = (conc_named.merge(bank_dir[["lei", "bank_name", "country"]],
                                   left_on="LEI_Code", right_on="lei")
                  [conc_named["LEI_Code"].isin(bank_totals.index)
                   & conc_named["country_iso"].isin(country_totals.index)])
    pivot = heatmap_df.pivot_table(
        index="bank_name", columns="country_iso",
        values="exposure_eur", aggfunc="sum", fill_value=0,
    )
    row_order = (heatmap_df.groupby("bank_name")["exposure_eur"].sum()
                 .sort_values(ascending=True).index.tolist())
    col_order = country_totals.index.tolist()
    pivot = pivot.reindex(index=row_order, columns=col_order, fill_value=0)
    z = pivot.values / 1e9
    text = [[f"{v:.0f}" if v >= 1.0 else "" for v in row] for row in z]

    fig_hm = go.Figure(go.Heatmap(
        z=z, x=pivot.columns, y=pivot.index,
        colorscale=SEQ_COOL_TO_WARM,
        colorbar=dict(title=dict(text="bn EUR",
                                 font=dict(size=11, color=COLORS["navy"])),
                      thickness=10, outlinewidth=0),
        text=text, texttemplate="%{text}",
        textfont={"size": 9, "color": COLORS["navy"]},
        xgap=1, ygap=1,
    ))
    bank_iso = (heatmap_df.drop_duplicates("bank_name").set_index("bank_name")
                ["country"].to_dict())
    home_x, home_y = [], []
    for bank, iso in bank_iso.items():
        if bank in pivot.index and iso in pivot.columns:
            home_x.append(iso); home_y.append(bank)
    if home_x:
        fig_hm.add_trace(go.Scatter(
            x=home_x, y=home_y, mode="markers",
            marker=dict(symbol="square-open", size=18,
                        color=COLORS["amber"], line=dict(width=2.5)),
            name="Home country", hoverinfo="skip",
        ))
    fig_hm.update_layout(
        title="Sovereign exposure (bn EUR) — amber = home-country pairs",
        height=520, showlegend=False,
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # === NEW · Accounting-class FV breakdown + CET1 impact ===
    eyebrow("IFRS-9 Accounting-Class · Fair-Value-Bestand & CET1-Impact")

    # Aggregate per accounting_class system-wide
    sys_by_class = (acct_split.groupby(["accounting_class", "channel"],
                                       as_index=False)["exposure_eur"]
                    .sum().sort_values("exposure_eur", ascending=False))

    a_l, a_r = st.columns([2, 3], gap="medium")

    with a_l:
        st.markdown("**System-wide Sovereign FV pro IFRS-9-Class**")
        disp = sys_by_class.copy()
        disp["FV bn"] = (disp["exposure_eur"]/1e9).round(0).astype(int)
        disp = disp.rename(columns={"accounting_class": "Class",
                                     "channel": "CET1 Channel"})
        st.dataframe(disp[["Class", "CET1 Channel", "FV bn"]],
                     use_container_width=True, hide_index=True, height=200)

        if abs(delta_r_pp) > 1e-3:
            st.markdown(f"**CET1-Impact pro Class @ Δr = {delta_r_pp:+.0f} pp**")
            cet1_acct = sovereign_cet1_impact(acct_split, delta_r_pp=delta_r_pp)
            sys_imp = (cet1_acct.groupby(["accounting_class", "channel"],
                                          as_index=False)
                                ["cet1_impact_eur"].sum())
            sys_imp["CET1 Δ bn"] = (sys_imp["cet1_impact_eur"]/1e9).round(2)
            sys_imp = sys_imp.rename(columns={"accounting_class":"Class",
                                               "channel":"Channel"})
            st.dataframe(sys_imp[["Class","Channel","CET1 Δ bn"]],
                         use_container_width=True, hide_index=True, height=200)

    with a_r:
        # Stacked bar: per top-15 bank, FV by accounting class
        top15 = (acct_split.groupby("LEI_Code")["exposure_eur"].sum()
                 .nlargest(15).index)
        sub = acct_split[acct_split["LEI_Code"].isin(top15)]
        sub = sub.merge(bank_dir[["lei", "bank_name"]],
                         left_on="LEI_Code", right_on="lei")
        pivoted = sub.pivot_table(index="bank_name", columns="accounting_class",
                                   values="exposure_eur", aggfunc="sum",
                                   fill_value=0)
        # Order by total
        pivoted = pivoted.assign(_tot=pivoted.sum(axis=1)).sort_values("_tot")
        pivoted = pivoted.drop(columns="_tot")

        fig_st = go.Figure()
        class_colors = {"AC":     COLORS["mid_blue"],
                        "FVOCI":  COLORS["bright_blue"],
                        "FVTPL":  COLORS["amber"],
                        "HfT":    COLORS["crimson"]}
        for cls in pivoted.columns:
            fig_st.add_trace(go.Bar(
                y=pivoted.index, x=pivoted[cls]/1e9,
                name=cls, orientation="h",
                marker_color=class_colors.get(cls, COLORS["stone"]),
                marker_line_width=0,
            ))
        fig_st.update_layout(
            title="Top-15 Banken · Sovereign FV nach IFRS-9-Class (bn EUR)",
            barmode="stack", height=480,
            xaxis_title="Fair Value [bn EUR]",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
        )
        st.plotly_chart(fig_st, use_container_width=True)

    st.divider()

    # === Per-bank drilldown ===
    eyebrow("Bank drilldown · Maturity-Ladder + CET1-Impact")
    bank_options = sorted(bank_totals.index.tolist(),
                          key=lambda lei: -bank_totals.loc[lei])
    lei_to_name = bank_dir.set_index("lei")["bank_name"].to_dict()
    sel_lei = st.selectbox(
        "Bank", bank_options,
        format_func=lambda lei: f"{lei_to_name.get(lei, lei[:8])}  "
                               f"(€{bank_totals.loc[lei]/1e9:.0f} bn sov book)",
        label_visibility="collapsed", key="sov_drill_bank",
    )
    sel_name = lei_to_name.get(sel_lei, sel_lei[:12])

    drill_l, drill_r = st.columns([3, 2], gap="medium")

    with drill_l:
        eyebrow(f"{sel_name} · maturity ladder")
        sub_mat = mat[mat["LEI_Code"] == sel_lei].sort_values("Maturity")
        if len(sub_mat) > 0:
            fig_mat = go.Figure(go.Bar(
                x=sub_mat["label"], y=sub_mat["exposure_eur"]/1e9,
                marker_color=COLORS["mid_blue"], marker_line_width=0,
                text=[f"{v/1e9:.1f}" for v in sub_mat["exposure_eur"]],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["navy"]),
            ))
            fig_mat.update_layout(
                xaxis_title="Maturity bucket", yaxis_title="Exposure [bn EUR]",
                height=360, bargap=0.25,
            )
            st.plotly_chart(fig_mat, use_container_width=True)

    with drill_r:
        eyebrow(f"{sel_name} · CET1-Impact pro Class")
        if abs(delta_r_pp) > 1e-3:
            cet1_acct = sovereign_cet1_impact(acct_split, delta_r_pp=delta_r_pp)
            bank_imp = cet1_acct[cet1_acct["LEI_Code"] == sel_lei]
            if len(bank_imp) > 0:
                disp = bank_imp.copy()
                disp["FV bn"]    = (disp["fair_value_eur"]/1e9).round(2)
                disp["ΔFV bn"]   = (disp["delta_fv_eur"]/1e9).round(2)
                disp["CET1 Δ bn"] = (disp["cet1_impact_eur"]/1e9).round(2)
                disp = disp.rename(columns={"accounting_class":"Class",
                                             "channel":"Channel"})
                st.dataframe(
                    disp[["Class","Channel","FV bn","ΔFV bn","CET1 Δ bn"]],
                    use_container_width=True, hide_index=True, height=240,
                )
                tot = float(bank_imp["cet1_impact_eur"].sum())
                st.metric("Σ CET1-Impact (bank)",
                          f"€{tot/1e9:+.2f} bn",
                          f"@ Δr = {delta_r_pp:+.0f} pp")
            else:
                st.info("Keine Accounting-Class-Daten für diese Bank.")
        else:
            st.info("Δr-Shock anwenden für CET1-Impact.")


# =====================================================================
# SUB-TAB 2 · Banking-Book Bonds (Financials / Corporates / Covered)
# =====================================================================
with tab_bb:

    with st.expander("Datenbasis & Annahmen · Banking-Book Bonds",
                     expanded=False):
        st.markdown("""
**Datenbasis:** EBA Transparency 2025, `tr_cre.csv`, Banking-Book IRB,
Reporting-Stichtag Juni 2025.
- **Financials**  = Exposure-Codes 202, 203, 204 (Banks / Institutions)
- **Corporates**  = Exposure-Codes 301–308, 311, 312 (alle Corporate-Subsets)
- **Covered Bonds** = Exposure-Code 603

**Wichtige Limitation (Critique 6 / Datenrealität).**
Die EBA-Public-Disclosure aggregiert in jeder dieser Klassen **Loans
+ Bonds + sonstige Banking-Book-Exposures zusammen**. Eine Trennung in
"reine Bond-Position" ist aus EBA-Daten **nicht möglich**. Wir zeigen
deshalb die **gesamte Banking-Book-IRB-Exposure** dieser Klassen.

**Stress-Wirkung:**
- ΔPD via Vasicek-Conditional-PD aus dem Macro-Shock M
- ΔLGD via Downturn-LGD-Funktion (κ = {kappa})
- ΔRWA → CET1-Impact via Capital-Bridge (siehe Credit-Risk-Tab)

**Was nicht modelliert wird:**
- Bond-spezifische Spread-Shocks (keine Marktspread-Daten im Projekt)
- Maturity-Effekte für Non-Sovereign-Bonds (EBA gibt keine Maturity)
- Issuer-Detail-Konzentration (keine Top-Issuer-Listen)
""".format(kappa=KAPPA_DOWNTURN_LGD))

    if lb_class.empty:
        st.error("Banking-Book-Bond-Class-Breakdown leer — Datenladelogik prüfen.")
    else:
        # Aggregate per category system-wide
        sys_cat = (lb_class.groupby("bond_category", as_index=False)
                          .agg(ead_eur=("ead_eur", "sum"),
                               rwa_eur=("rwa_eur", "sum"),
                               defaulted_eur=("defaulted_eur", "sum"),
                               oe_eur=("oe_eur", "sum")))
        sys_cat["rwa_density"] = sys_cat["rwa_eur"] / sys_cat["ead_eur"].replace(0, pd.NA)
        sys_cat["npl_ratio"]   = sys_cat["defaulted_eur"] / sys_cat["oe_eur"].replace(0, pd.NA)

        # === Aggregate KPIs per category ===
        eyebrow("System-wide Banking-Book Bond-Categories")
        b1, b2, b3 = st.columns(3, gap="small")
        for col, cat in zip([b1, b2, b3],
                             ["Financials", "Corporates", "Covered Bonds"]):
            row = sys_cat[sys_cat["bond_category"] == cat]
            if len(row) > 0:
                r = row.iloc[0]
                col.metric(cat,
                           f"€{r['ead_eur']/1e12:.2f} tn EAD",
                           f"RWA-Density {r['rwa_density']*100:.0f}% · "
                           f"NPL {r['npl_ratio']*100:.2f}%",
                           delta_color="off")
            else:
                col.metric(cat, "—", delta_color="off")

        st.divider()

        # === Per-category exposure breakdown table ===
        eyebrow("Top-15 Banken pro Bond-Category")

        cat_choice = st.selectbox(
            "Category",
            ["Financials", "Corporates", "Covered Bonds"],
            key="bb_cat",
        )
        sub_cat = (lb_class[lb_class["bond_category"] == cat_choice]
                   .merge(bank_dir[["lei", "bank_name"]],
                          left_on="LEI_Code", right_on="lei")
                   .sort_values("ead_eur", ascending=False).head(15))
        if len(sub_cat) > 0:
            disp = pd.DataFrame({
                "Bank":            sub_cat["bank_name"],
                "EAD bn":          (sub_cat["ead_eur"]/1e9).round(1),
                "RWA bn":          (sub_cat["rwa_eur"]/1e9).round(1),
                "RWA density":     (sub_cat["rwa_eur"]/sub_cat["ead_eur"].replace(0, pd.NA)*100).round(0).astype(str) + "%",
                "Defaulted bn":    (sub_cat["defaulted_eur"]/1e9).round(2),
                "Implied PD":      (sub_cat["implied_pd"]*100).round(2).astype(str) + "%",
            })
            st.dataframe(disp, use_container_width=True, hide_index=True,
                         height=440)
        else:
            st.info(f"Keine Daten für {cat_choice}.")

        st.divider()

        # === CET1-Impact via Vasicek-bridge for selected category ===
        eyebrow(f"{cat_choice} · CET1-Impact via Vasicek-RWA")

        if abs(config["d_brent"]) < 1e-3 and abs(delta_r_pp) < 1e-3:
            st.info(
                "Apply a macro shock in the sidebar to see ΔRWA → CET1-Impact "
                "via the Vasicek-Bridge (Capital-Adequacy-Tab zeigt die volle "
                "Decomposition aller drei Channels)."
            )
        else:
            # Use the universe-loaded portfolios for this; skip detailed
            # rebuild here. Refer to Capital-Adequacy page for full detail.
            st.markdown(f"""
**Hinweis:** Die volle CET1-Wirkungskette (PD-Shift × LGD-Shift × RWA-
Decomposition × CET1-Quote vorher/nachher) für **{cat_choice}** ist auf
der **Credit Risk · Loan Book**-Page sichtbar. Diese Sektion zeigt nur
die Banking-Book-Exposure-Größen, weil eine isolierte Re-Berechnung
ohne Segment-Granularität dieselben Vasicek-Inputs erneut bemühen
würde — siehe oben "Datenbasis & Annahmen".
""")
            # Show category share of total Banking Book RWA (allocation)
            tot_rwa_bb = float(sys_cat["rwa_eur"].sum())
            cat_row = sys_cat[sys_cat["bond_category"] == cat_choice]
            if len(cat_row) > 0:
                cat_rwa = float(cat_row.iloc[0]["rwa_eur"])
                share = cat_rwa / tot_rwa_bb if tot_rwa_bb > 0 else 0
                ci_l, ci_r = st.columns(2, gap="medium")
                ci_l.metric(f"{cat_choice} RWA",
                            f"€{cat_rwa/1e9:.0f} bn",
                            f"{share*100:.1f}% of BB-RWA",
                            delta_color="off")
                ci_r.metric("Allocation key for ΔCET1",
                            f"{share*100:.1f}%",
                            "of Loan-Book ΔRWA-Channel",
                            delta_color="off")


# =====================================================================
# SUB-TAB 3 · Trading Book + ABS / MBS
# =====================================================================
with tab_tb:

    with st.expander("Datenbasis & Annahmen · Trading Book + ABS/MBS",
                     expanded=False):
        st.markdown("""
**Datenbasis:** EBA Transparency 2025, `tr_oth.csv`, Reporting-Stichtag
Juni 2025.
- **Market RWA** aus Item 2520210 (Position, FX, Commodities) — Trading
  Book aggregat
- **Securitisation RWA** aus Item 2520209 (Securitisations in banking
  book, after the cap)
- Beides nur als **Bank-Aggregate**, keine Issuer-/Tranche-Detail

**Stress-Wirkung:**
- Market-RWA: FRTB-style Multiplier auf VaR/SVaR (κ_RWA = 0.30/2.5
  → +30% bei M = -2.5)
- Trading-Book P&L Haircut (κ_PnL = 0.50/2.5 → -50% bei M = -2.5)
- Securitisation-RWA: in V1 nicht stress-elastisch (Item bleibt
  konstant) — Disclaimer: Spread-Schock auf ABS/MBS ist ohne Tranche-
  Detail nicht modellierbar

**Limitation:** EBA-Public-Disclosure gibt **keine** Issuer-Granularität
für Trading-Book-Bonds. VaR/SVaR ist Bank-Aggregat. ABS/MBS-Tranche-
Struktur ist nicht publiziert.
""")

    # System totals
    sys_market = float(cap_df["rwa_market_eur"].sum())
    sys_secur  = float(cap_df.get("rwa_securitisation_eur",
                                  pd.Series([0])).sum())
    sys_tb_pnl = float(cap_df["tb_pnl_eur"].sum())

    eyebrow("Trading Book + ABS/MBS · System totals")
    t1, t2, t3 = st.columns(3, gap="small")
    t1.metric("Market RWA (Trading Book)",
              f"€{sys_market/1e9:.0f} bn",
              "Item 2520210", delta_color="off")
    t2.metric("Securitisation RWA",
              f"€{sys_secur/1e9:.0f} bn",
              "Item 2520209 · Banking Book", delta_color="off")
    t3.metric("Trading Book P&L (current)",
              f"€{sys_tb_pnl/1e9:+.1f} bn",
              "Item 2520311 · YTD", delta_color="off")

    st.divider()

    # FRTB stress
    if abs(config["d_brent"]) > 1e-3 or abs(delta_r_pp) > 1e-3:
        from macro_factor import anchor_from_eba, hybrid_mapping, factor_stats
        from components.data_loader import load_data_layer
        data = load_data_layer()
        fs = (factor_stats(data["brent"], data["svensson"], lookback=252)
              if data["brent"] is not None and data["svensson"] is not None
              else None)
        cov = fs["sigma"] if fs else np.array([[4e-4, 2e-5],[2e-5, 1e-4]])
        anchor = anchor_from_eba("2025")
        m = hybrid_mapping(d_brent, delta_r_pp, anchor=anchor,
                           cov_factors=cov, horizon_days=252)
        m_used = m["m_hybrid"]

        tb_stress = trading_book_stress(cap_df, m_factor=m_used)
        delta_mr_rwa = float(tb_stress["delta_rwa_market_eur"].sum())
        delta_tb_pnl = float(tb_stress["delta_tb_pnl_eur"].sum())

        insight(
            f"<strong>Trading-Book Stress @ M = {m_used:+.2f}.</strong> "
            f"Market-RWA: <strong>€{sys_market/1e9:.0f} bn → "
            f"€{(sys_market+delta_mr_rwa)/1e9:.0f} bn</strong> "
            f"({delta_mr_rwa/1e9:+.1f} bn). "
            f"Trading-Book P&L Δ: <strong>€{delta_tb_pnl/1e9:+.1f} bn</strong>. "
            f"Beide schlagen direkt durch CET1 — siehe Capital-Adequacy-Tab "
            f"für die volle 3-Channel-Decomposition."
        )

        # Top affected banks
        eyebrow("Top-10 banks · Market-RWA-Stress")
        tb_named = tb_stress.merge(bank_dir[["lei","bank_name"]],
                                    left_on="LEI_Code", right_on="lei")
        top_tb = tb_named.sort_values("delta_rwa_market_eur",
                                       ascending=False).head(10)
        disp_tb = pd.DataFrame({
            "Bank": top_tb["bank_name"],
            "MR-RWA base bn":   (top_tb["rwa_market_eur"]/1e9).round(1),
            "MR-RWA stress bn": (top_tb["rwa_market_eur_stress"]/1e9).round(1),
            "Δ MR-RWA bn":      (top_tb["delta_rwa_market_eur"]/1e9).round(2),
            "Δ TB-P&L bn":      (top_tb["delta_tb_pnl_eur"]/1e9).round(2),
        })
        st.dataframe(disp_tb, use_container_width=True, hide_index=True,
                     height=380)
    else:
        st.info(
            "Apply a macro shock in the sidebar to see the FRTB-style "
            "Market-RWA stress and Trading-Book P&L haircut."
        )

    st.divider()

    # ABS/MBS table
    eyebrow("Securitisation-RWA · top-10 banks (Item 2520209)")
    secur_df = (cap_df[["LEI_Code", "rwa_securitisation_eur"]]
                .merge(bank_dir[["lei","bank_name"]],
                       left_on="LEI_Code", right_on="lei")
                .sort_values("rwa_securitisation_eur", ascending=False).head(10))
    disp_secur = pd.DataFrame({
        "Bank":           secur_df["bank_name"],
        "Securit. RWA bn": (secur_df["rwa_securitisation_eur"]/1e9).round(2),
    })
    st.dataframe(disp_secur, use_container_width=True, hide_index=True,
                 height=380)
    st.caption(
        "ABS/MBS-Banking-Book-RWA. Issuer-/Tranche-Detail nicht im EBA-"
        "Public-Disclosure. V1 keine separate Stress-Funktion — "
        "Limitation in Annahmen-Tab dokumentiert."
    )


footer(
    f"Three sub-tabs · Sovereigns (full FVOCI/AC split) · Banking-Book-Bonds "
    f"(Financials / Corporates / Covered) · Trading Book + ABS-MBS · "
    f"Daten: EBA Transparency 2025, Stichtag Juni 2025"
)
