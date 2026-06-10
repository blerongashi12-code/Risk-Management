"""Marktbuch · Zins-/Sovereign-Kanal des 2-Kanal-Modells.

Zwei Sub-Tabs:
  1. Yield-Curve (Input)  — Bundesbank-Svensson-Kurve als zentraler
                            Makro-Input (Δr_10y) für beide Stress-Kanäle.
  2. Sovereigns           — Maturity-Ladder, Country-Konzentration (Doom-
                            Loop), IFRS-9-Klassen-Split (FVOCI/AC), CET1-
                            Impact via ΔMtM = −D·Δy·Exposure (OCI/P&L).
                            DIES ist der Sovereign-Kanal der CET1-Bridge.

Datenbasis durchgängig: die 10 kuratierten IRB-Banken (Pillar-3 EU-CR6,
31.12.2024). Kein erfundener Daten; Datenlücken werden offen markiert.
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

from components.theme import (tab_breadcrumb, apply_theme, hero, eyebrow, insight, footer,
                              COLORS, SEQ_COOL_TO_WARM)
from components.sidebar import render_sidebar
from components.methodology import render_sovereign_methodology
from components.legacy_views import render_yield_curve_tab
from components.backend_path import setup
setup()

from config import EBA_RAW_DIR                                      # type: ignore
from eba_loader import (                                            # type: ignore
    parse_sovereign_csv, sovereign_concentration,
    sovereign_maturity_ladder, domestic_share_per_bank,
    sovereign_kpis_per_bank, attach_country_names,
    rate_shock_pnl,
    sovereign_by_accounting_class, sovereign_cet1_impact,
    parse_capital_overview,
    load_bank_directory, load_country_dim,
)


st.set_page_config(page_title="Marktbuch · Sovereign-/Zins-Kanal", layout="wide")
apply_theme()
config = render_sidebar()

hero(
    "Marktbuch · Sovereign-/Zins-Kanal",
    eyebrow="Tab 3 · 2-Faktor-Modell · 10 IRB-Banken · 2 Sub-Tabs",
    deck="Der Marktbuch-Tab bildet den zins-getriebenen Sovereign-Kanal "
         "der CET1-Bridge ab — einen der zwei Kanäle unseres Modells "
         "(neben dem Kreditbuch). Zwei Sub-Tabs: (1) Yield-Curve als "
         "zentraler Makro-Input (Bundesbank Svensson Q4 2025) für den "
         "Δr_10y-Schock (mit historischer Episoden-Plausibilisierung), "
         "(2) Sovereigns granular nach Land × Restlaufzeit × "
         "IFRS-9-Klasse mit Duration-basiertem Mark-to-Market "
         "(ΔFV = −D·Δy·Exposure), Doom-Loop-Quantifizierung, latenten "
         "AC-Verlusten, Duration/BPV-Profil und Δr-Sensitivitätskurve. "
         "Nur HfT/FVTPL/FVOCI schlagen auf CET1 durch (AC bleibt zu "
         "Buchwert). Datenbasis durchgängig die 10 kuratierten Banken "
         "(Pillar-3 EU-CR6, 31.12.2024).",
)

tab_breadcrumb(3)
# === Live macro shock from sidebar ===================================
delta_r_pp = config["d_r_10y_pp"]    # already in pp (unit-bug fixed)


# === Cached data loaders =============================================
@st.cache_data(ttl=24*3600, show_spinner="Loading EBA Sovereign-data …")
def _load_data():
    sov_raw = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    cty_dim  = load_country_dim(EBA_RAW_DIR / "TR_Metadata.xlsx")
    # CET1-Kapital pro Bank aus tr_oth.csv — hier ausschliesslich als
    # NORMIERUNGSGROESSE (Sovereign-Exposure / CET1, Verlust in % des
    # CET1). KEIN Trading-Book-Stress: dieser Kanal bleibt entfernt.
    cap_raw  = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    return sov_raw, bank_dir, cty_dim, cap_raw


sov_raw, bank_dir, cty_dim, cap_raw = _load_data()

# === Top-10 universe filter =========================================
# Datenbasis aller Analysen: die 10 Banken aus pillar3_bank_pd_lgd.csv
# mit bank-spezifischen Pillar-3 EU-CR6-PDs am Stichtag 31.12.2024 (10/10
# Banken Pillar-3-verifiziert). Damit einheitliche Datenqualität über
# alle Tabs.
from eba_pd_loader import get_top10_leis                              # type: ignore
_IRB_LEIS = get_top10_leis()
N_IRB_BANKS = len(_IRB_LEIS)

# Pre-compute the derived sovereign dataframes — alle konsequent auf die
# 10 kuratierten IRB-Banken gefiltert (einheitliche Datenbasis über alle Tabs).
conc       = sovereign_concentration(sov_raw, period=202506)
conc       = conc[conc["LEI_Code"].isin(_IRB_LEIS)].copy()
conc_named = attach_country_names(conc, cty_dim)
mat        = sovereign_maturity_ladder(sov_raw, period=202506)
mat        = mat[mat["LEI_Code"].isin(_IRB_LEIS)].copy()
acct_split = sovereign_by_accounting_class(sov_raw, period=202506)
acct_split = acct_split[acct_split["LEI_Code"].isin(_IRB_LEIS)].copy()
cap10      = cap_raw[cap_raw["LEI_Code"].isin(_IRB_LEIS)].copy()

_LEI2NAME = bank_dir.set_index("lei")["bank_name"].to_dict()


def _short(name: str, n: int = 24) -> str:
    """Kürzt lange Banknamen für Achsen-Labels."""
    return name if len(name) <= n else name[: n - 1] + "…"


# =====================================================================
# Two tabs · Yield-Curve (Input) + Sovereigns (der Sovereign-Kanal)
# =====================================================================
tab_yc, tab_sov = st.tabs([
    "1 · Yield-Curve (Input)",
    "2 · Sovereigns",
])


# =====================================================================
# SUB-TAB 0 · Yield-Curve (zentraler Input für alle Bond-Channels)
# =====================================================================
with tab_yc:
    render_yield_curve_tab(config)

    st.divider()

    # === Historische Zins-Episoden · Plausibilisierung der Δr-Range ==
    eyebrow("Historische Zins-Episoden · wie realistisch ist die Δr-Schock-Range?")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #034B6F;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Ökonomische Einordnung.</strong> Ein Stress-Szenario ist '
        'nur dann aussagekräftig, wenn die unterstellte Schock-Größe '
        'historisch plausibel ist. Die Sidebar erlaubt Δr_10y von '
        '<strong>−3 bis +5 pp</strong> — die Tabelle zeigt, dass die '
        'großen Zins-Episoden der letzten Jahrzehnte genau in diese '
        'Größenordnung fallen. Ein <strong>+2 pp</strong>-Schock ist damit '
        'kein Tail-Fantasiewert, sondern wurde 2022 real beobachtet; '
        '<strong>+3 pp</strong> entspricht etwa der kumulierten '
        'EZB-Zinswende 2021–2023.'
        '<br><br>'
        '<table style="font-size:0.86rem;border-collapse:collapse;width:100%;">'
        '<thead><tr style="background:#F4F4F4;">'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Episode</th>'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Markt</th>'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Bewegung (ca.)</th>'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Zeitraum</th>'
        '</tr></thead><tbody>'
        '<tr><td style="padding:0.4rem 0.6rem;">EZB-Zinswende 2021–2023</td>'
        '<td style="padding:0.4rem 0.6rem;">10y Bund</td>'
        '<td style="padding:0.4rem 0.6rem;font-weight:600;">≈ −0,2 % → +3,0 % '
        '(≈ +320 bp)</td>'
        '<td style="padding:0.4rem 0.6rem;">Dez 2021 – Okt 2023</td></tr>'
        '<tr><td style="padding:0.4rem 0.6rem;">„Bond-Massaker" 1994</td>'
        '<td style="padding:0.4rem 0.6rem;">10y US-Treasury</td>'
        '<td style="padding:0.4rem 0.6rem;font-weight:600;">≈ +240 bp</td>'
        '<td style="padding:0.4rem 0.6rem;">Jan – Nov 1994</td></tr>'
        '<tr><td style="padding:0.4rem 0.6rem;">Taper Tantrum 2013</td>'
        '<td style="padding:0.4rem 0.6rem;">10y US-Treasury</td>'
        '<td style="padding:0.4rem 0.6rem;font-weight:600;">≈ +140 bp</td>'
        '<td style="padding:0.4rem 0.6rem;">Mai – Sep 2013</td></tr>'
        '<tr><td style="padding:0.4rem 0.6rem;">UK-Gilt-Krise (Mini-Budget)</td>'
        '<td style="padding:0.4rem 0.6rem;">30y UK-Gilt</td>'
        '<td style="padding:0.4rem 0.6rem;font-weight:600;">≈ +130 bp</td>'
        '<td style="padding:0.4rem 0.6rem;">3 Handelstage, Sep 2022</td></tr>'
        '</tbody></table>'
        '<div style="margin-top:0.7rem;font-size:0.82rem;color:#6E6E6E;">'
        '<strong>Quellen:</strong> Deutsche Bundesbank (Tagesrenditen '
        'börsennotierter Bundeswertpapiere, Zeitreihen-Datenbank) · '
        'Federal Reserve H.15 (Treasury Constant Maturities) · '
        'Bank of England (Gilt-Renditen, Financial Stability Report '
        'Okt 2022). Werte gerundet auf ±10 bp.<br>'
        '<strong>Methodische Anmerkung.</strong> Unser Modell unterstellt '
        'einen <em>Parallel-Shift</em> der Kurve am 10y-Punkt; reale '
        'Episoden waren teils Steepener/Flattener. Der Parallel-Shift '
        'mit Modified Duration ist die Standard-Erstordnungs-Näherung '
        'für Zinsrisiko (Tuckman/Serrat 2012, Kap. 4) — '
        'laufzeitdifferenzierte Twists wären eine V2-Erweiterung.'
        '</div></div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# SUB-TAB 1 · Sovereigns
# =====================================================================
with tab_sov:
    # === Sub-Tab Intro ================================================
    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #034B6F;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.6;">'
        '<strong>Was zeigt dieser Sub-Tab?</strong> Den '
        '<em>Sovereign-Channel</em> unseres Stress-Modells: wie hart trifft '
        'ein steigender 10-Jahres-Zins die CET1-Quote der EU-Banken über '
        'ihre Staatsanleihen-Bestände. Wir zeigen (1) wie groß das '
        'Gesamt-Sovereign-Buch ist, (2) wer welche Länder hält '
        '(<em>Doom-Loop</em>), (3) wie die Bestände nach IFRS-9 '
        'klassifiziert sind (entscheidet, ob ein Marktverlust überhaupt '
        'in der CET1 erscheint) und (4) die Wirkung pro einzelner Bank.<br><br>'
        f'<strong>Datenbasis.</strong> Alle Analysen filtern auf die '
        f'<strong>{N_IRB_BANKS} IRB-tauglichen Banken</strong> der EBA '
        f'Transparency Exercise 2025 — dieselbe Universe wie im Kreditbuch. '
        'Quelle: <code>tr_sov.csv</code>, Stichtag Juni 2025.'
        '</div>',
        unsafe_allow_html=True,
    )

    render_sovereign_methodology()

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

    eyebrow("Sovereign-Bestand auf einen Blick")

    st.markdown(
        '<div style="background:#F4F4F4;border-radius:6px;'
        'padding:0.75rem 1.0rem;margin:0.3rem 0 0.9rem 0;'
        'color:#051C2C;font-size:0.86rem;line-height:1.55;">'
        '<strong>Was zeigen die vier Kacheln?</strong><br>'
        '<strong>Σ Sovereign-Buch</strong> — Summe aller Staatsanleihen-Positionen '
        f'der {N_IRB_BANKS} IRB-Banken in Bilanzwert.<br>'
        '<strong>Home-Country-Anteil</strong> — wie viel jede Bank in '
        'Anleihen <em>ihres eigenen Heimatlandes</em> hält (z.B. Deutsche '
        'Bank in Bundesanleihen). Volumengewichtet über das EU-System. '
        'Hoher Anteil = klassisches <em>Doom-Loop-Signal</em>.<br>'
        '<strong>System-P&amp;L unter Live-Schock</strong> — der '
        'gesamte EUR-Mark-to-Market-Verlust unter dem aktuell in der '
        'Sidebar gesetzten Δr_10y.<br>'
        '<strong>Anzahl IFRS-9-Klassen</strong> — wie viele '
        'Bilanzkategorien beobachtet werden (HfT / FVTPL / FVOCI / AC, '
        'erklärt im Block weiter unten).'
        '</div>',
        unsafe_allow_html=True,
    )

    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.metric("Σ Sovereign-Buch", f"€{total_exposure/1e12:.2f} tn",
              f"{n_banks} IRB-Banken", delta_color="off")
    s2.metric("Home-Country-Anteil",
              f"{weighted_dom_share*100:.0f}%",
              "Anteil heimischer Staatsanleihen",
              delta_color="off")
    if abs(delta_r_pp) > 1e-3:
        s3.metric("System-P&L · Live-Schock",
                  f"€{system_pnl/1e9:+.1f} bn",
                  f"Δr = {delta_r_pp:+.0f} pp")
    else:
        s3.metric("System-P&L · Live-Schock", "€0 bn",
                  "kein Schock aktiv", delta_color="off")
    s4.metric("IFRS-9-Klassen",
              f"{acct_split['accounting_class'].nunique()}",
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
    eyebrow("Doom-Loop · welche Bank hält welches Land?")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #A52F4D;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.6;">'
        '<strong>Was ist der „Doom Loop"?</strong> Der Begriff beschreibt '
        'den Teufelskreis zwischen einem Staat und seinen heimischen '
        'Banken: gerät ein Staat in fiskalische Schieflage (siehe Italien '
        '2011/12, Griechenland 2010-12), brechen die Kurse seiner '
        'Staatsanleihen ein. Weil heimische Banken überproportional viele '
        'dieser Bonds halten, schwächt das ihr Eigenkapital — was '
        'wiederum den Staat zwingt einzuspringen, was die Staatsfinanzen '
        'weiter belastet. Eine Abwärtsspirale.<br><br>'
        '<strong>Was zeigt die Heatmap?</strong> Zeilen = alle '
        f'{N_IRB_BANKS} kuratierten IRB-Banken, Spalten = die 12 '
        'größten Schuldnerländer im EU-System. Zellwert = Sovereign-'
        'Exposure in Mrd. EUR. Dunkelrot = hohes Engagement. '
        '<span style="color:#C9A227;font-weight:600;">Amber-Quadrate</span> '
        'markieren die Home-Country-Paare (Bank im Land ihres '
        'Hauptsitzes). Konzentrierte Amber-Bereiche im roten Bereich = '
        'Doom-Loop-Hotspot.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ISO -> country-name map (out of country_dim) for axis labels
    iso_to_name = (cty_dim.set_index("iso")["country_name"].to_dict()
                   if "iso" in cty_dim.columns else {})

    bank_totals = conc.groupby("LEI_Code")["exposure_eur"].sum().nlargest(N_IRB_BANKS)
    country_totals = (conc_named.groupby(["country_iso"])["exposure_eur"]
                      .sum().nlargest(12))
    # First merge, then filter on the merged DataFrame (Index-aligned).
    heatmap_df = conc_named.merge(
        bank_dir[["lei", "bank_name", "country"]],
        left_on="LEI_Code", right_on="lei",
    )
    heatmap_df = heatmap_df[
        heatmap_df["LEI_Code"].isin(bank_totals.index)
        & heatmap_df["country_iso"].isin(country_totals.index)
    ]
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

    # Better x-axis labels: "DE · Germany" instead of just "DE"
    x_labels = [f"{iso} · {iso_to_name.get(iso, iso)}" for iso in pivot.columns]

    fig_hm = go.Figure(go.Heatmap(
        z=z, x=x_labels, y=pivot.index,
        colorscale=SEQ_COOL_TO_WARM,
        colorbar=dict(title=dict(text="bn EUR",
                                 font=dict(size=11, color=COLORS["navy"])),
                      thickness=10, outlinewidth=0),
        text=text, texttemplate="%{text}",
        textfont={"size": 9, "color": COLORS["navy"]},
        xgap=1, ygap=1,
        hovertemplate="<b>%{y}</b><br>%{x}<br>%{z:.1f} bn EUR<extra></extra>",
    ))
    # Map original iso to labelled x for home-country markers
    iso_to_label = {iso: lbl for iso, lbl in zip(pivot.columns, x_labels)}
    bank_iso = (heatmap_df.drop_duplicates("bank_name").set_index("bank_name")
                ["country"].to_dict())
    home_x, home_y = [], []
    for bank, iso in bank_iso.items():
        if bank in pivot.index and iso in iso_to_label:
            home_x.append(iso_to_label[iso]); home_y.append(bank)
    if home_x:
        fig_hm.add_trace(go.Scatter(
            x=home_x, y=home_y, mode="markers",
            marker=dict(symbol="square-open", size=18,
                        color=COLORS["amber"], line=dict(width=2.5)),
            name="Home country", hoverinfo="skip",
        ))
    fig_hm.update_layout(
        title="Sovereign-Exposure (Mrd. EUR) · Amber-Quadrate = Home-Country-Paare",
        height=560, showlegend=False,
        xaxis=dict(tickangle=-30, tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Country-code → name lookup table beneath
    with st.expander("Länder-Codes ausgeschrieben", expanded=False):
        codes_used = list(pivot.columns)
        lookup_df = pd.DataFrame({
            "ISO-Code": codes_used,
            "Land": [iso_to_name.get(c, "—") for c in codes_used],
            "Σ Exposure (Mrd. EUR)": [round(float(
                conc_named[conc_named["country_iso"] == c]["exposure_eur"].sum()
            )/1e9, 1) for c in codes_used],
        }).sort_values("Σ Exposure (Mrd. EUR)", ascending=False)
        st.dataframe(lookup_df, hide_index=True,
                     use_container_width=True, height=420)

    st.divider()

    # === Doom-Loop quantifiziert · Home-Bias × Kapital-Exponierung ===
    eyebrow("Doom-Loop quantifiziert · Home-Bias × Sovereign-Exposure relativ zum CET1-Kapital")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #A52F4D;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Ökonomische Interpretation.</strong> Die Heatmap oben zeigt, '
        '<em>wo</em> die Bestände liegen — diese Grafik beantwortet, '
        '<em>wie gefährlich</em> sie sind. Der Doom-Loop ist umso bindender, '
        'je stärker zwei Dimensionen zusammenkommen: (x) der '
        '<strong>Home-Bias</strong> — wie viel des Sovereign-Buchs im '
        'eigenen Heimatland steckt — und (y) der '
        '<strong>Kapital-Hebel</strong> — wie groß das Sovereign-Buch '
        'relativ zum CET1-Verlustpuffer ist. Eine Bank rechts oben hält '
        'viel heimisches Staatsrisiko und hat wenig Kapital dagegen: '
        'genau die Konstellation, die in der Eurokrise 2011/12 die '
        'Banken-Staaten-Spirale getrieben hat.<br><br>'
        '<strong>Mathematische Definition.</strong> '
        'Home-Bias<sub>i</sub> = E<sub>domestic,i</sub> / E<sub>total,i</sub> · '
        'Kapital-Hebel<sub>i</sub> = E<sub>total,i</sub> / CET1<sub>i</sub>. '
        'CET1-Kapital aus EBA <code>tr_oth.csv</code> (Item 2520102), '
        'Sovereign-Exposure aus <code>tr_sov.csv</code> (Item 2520810), '
        'beide Stichtag Juni 2025.<br>'
        '<span style="font-size:0.82rem;color:#6E6E6E;">'
        '<strong>Quellen:</strong> Brunnermeier et al. (2016): „The '
        'Sovereign-Bank Diabolic Loop", <em>American Economic Review '
        'P&amp;P</em> 106(5) · Acharya, Drechsler &amp; Schnabl (2014): '
        '„A Pyrrhic Victory? Bank Bailouts and Sovereign Credit Risk", '
        '<em>Journal of Finance</em> 69(6) · ESRB (2015): „Report on the '
        'regulatory treatment of sovereign exposures".'
        '</span></div>',
        unsafe_allow_html=True,
    )

    code_to_iso = cty_dim.set_index("code")["iso"].to_dict()
    kpis = sovereign_kpis_per_bank(conc, bank_dir)
    dl = (kpis.merge(dom[["lei", "domestic_share"]], on="lei", how="left")
              .merge(cap10[["LEI_Code", "cet1_eur"]],
                     left_on="lei", right_on="LEI_Code", how="left"))
    dl = dl[dl["cet1_eur"] > 0].copy()
    dl["sov_over_cet1"] = dl["total_eur"] / dl["cet1_eur"]
    dl["top_iso"] = dl["top_country_code"].map(code_to_iso)

    dl_l, dl_r = st.columns([3, 2], gap="medium")

    with dl_l:
        fig_dl = go.Figure(go.Scatter(
            x=dl["domestic_share"] * 100,
            y=dl["sov_over_cet1"] * 100,
            mode="markers+text",
            text=[_short(n, 16) for n in dl["bank_name"]],
            textposition="top center",
            textfont=dict(size=9, color=COLORS["navy"]),
            marker=dict(
                size=dl["total_eur"] / 1e9,
                sizemode="area",
                sizeref=2.0 * float(dl["total_eur"].max() / 1e9) / (38 ** 2),
                sizemin=8,
                color=COLORS["crimson"], opacity=0.75,
                line=dict(width=1, color=COLORS["white"]),
            ),
            customdata=np.stack([dl["total_eur"] / 1e9,
                                 dl["cet1_eur"] / 1e9], axis=-1),
            hovertemplate=("<b>%{text}</b><br>Home-Bias: %{x:.0f}%<br>"
                           "Sov/CET1: %{y:.0f}%<br>"
                           "Σ Sovereign: %{customdata[0]:.0f} bn €<br>"
                           "CET1: %{customdata[1]:.0f} bn €<extra></extra>"),
        ))
        fig_dl.add_vline(x=float(dl["domestic_share"].median() * 100),
                         line_dash="dot", line_color=COLORS["stone"],
                         annotation_text="Median", annotation_font_size=9)
        fig_dl.add_hline(y=float(dl["sov_over_cet1"].median() * 100),
                         line_dash="dot", line_color=COLORS["stone"])
        fig_dl.update_layout(
            title="Doom-Loop-Map · Blasengröße = Σ Sovereign-Buch",
            xaxis_title="Home-Bias [% des Sovereign-Buchs im Heimatland]",
            yaxis_title="Sovereign-Exposure [% des CET1-Kapitals]",
            height=460, showlegend=False,
        )
        st.plotly_chart(fig_dl, use_container_width=True)

    with dl_r:
        st.markdown("**Konzentrations-KPIs pro Bank**")
        st.caption(
            "**Sov/CET1** = Kapital-Hebel · **Top-1** = größtes "
            "Schuldnerland (Anteil) · **HHI** = Herfindahl-Index über die "
            "Länderverteilung (>2500 = hoch konzentriert, DOJ/FTC-"
            "Konvention) · **n** = Anzahl Schuldnerländer."
        )
        kpi_disp = pd.DataFrame({
            "Bank":        [_short(n, 22) for n in dl["bank_name"]],
            "Σ Sov bn":    (dl["total_eur"] / 1e9).round(0).astype(int),
            "Sov/CET1":    (dl["sov_over_cet1"] * 100).round(0).astype(int).astype(str) + "%",
            "Home-Bias":   (dl["domestic_share"] * 100).round(0).astype(int).astype(str) + "%",
            "Top-1":       dl["top_iso"].fillna("—") + " " + (dl["top_share"] * 100).round(0).astype(int).astype(str) + "%",
            "HHI":         (dl["hhi"] * 10000).round(0).astype(int),
            "n":           dl["n_countries"],
        }).sort_values("Σ Sov bn", ascending=False)
        st.dataframe(kpi_disp, use_container_width=True, hide_index=True,
                     height=420)

    _max_lev = dl.loc[dl["sov_over_cet1"].idxmax()]
    insight(
        f"<strong>Lesart.</strong> Der größte Kapital-Hebel liegt bei "
        f"<strong>{_max_lev['bank_name']}</strong> mit einem Sovereign-Buch "
        f"von {_max_lev['sov_over_cet1']*100:.0f}% des CET1-Kapitals "
        f"(Home-Bias {_max_lev['domestic_share']*100:.0f}%). Je weiter "
        f"rechts oben eine Bank steht, desto direkter überträgt sich ein "
        f"Spread-/Zinsschock ihres Heimatlandes auf ihre Kapitalquote — "
        f"der Mechanismus, den Brunnermeier et al. (2016) als "
        f"<em>diabolic loop</em> formalisieren."
    )

    st.divider()

    # === NEW · Accounting-class FV breakdown + CET1 impact ===
    eyebrow("IFRS-9-Klassifizierung · welcher Bond schlägt überhaupt auf CET1 durch?")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #2251FF;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Warum diese Sektion entscheidend ist.</strong> Eine Bank '
        'kann denselben Bond zu unterschiedlichen Bilanzwerten halten. '
        'Erst die IFRS-9-Klasse entscheidet, <em>ob</em> ein Marktverlust '
        'überhaupt in der CET1-Quote sichtbar wird:<br><br>'
        '<table style="font-size:0.86rem;border-collapse:collapse;width:100%;">'
        '<thead><tr style="background:#F4F4F4;">'
        '<th style="text-align:left;padding:0.4rem 0.6rem;width:90px;">Kürzel</th>'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Voller Name</th>'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Bilanzierung</th>'
        '<th style="text-align:left;padding:0.4rem 0.6rem;">Wirkung auf CET1</th>'
        '</tr></thead><tbody>'
        '<tr><td style="padding:0.4rem 0.6rem;font-weight:600;">HfT</td>'
        '<td style="padding:0.4rem 0.6rem;">Held for Trading</td>'
        '<td style="padding:0.4rem 0.6rem;">Fair Value, über GuV</td>'
        '<td style="padding:0.4rem 0.6rem;color:#A52F4D;">'
        'Verlust schlägt <strong>sofort</strong> auf P&amp;L → CET1</td></tr>'
        '<tr><td style="padding:0.4rem 0.6rem;font-weight:600;">FVTPL</td>'
        '<td style="padding:0.4rem 0.6rem;">Fair Value Through P&amp;L</td>'
        '<td style="padding:0.4rem 0.6rem;">Fair Value, über GuV</td>'
        '<td style="padding:0.4rem 0.6rem;color:#A52F4D;">'
        'Verlust schlägt <strong>sofort</strong> auf P&amp;L → CET1</td></tr>'
        '<tr><td style="padding:0.4rem 0.6rem;font-weight:600;">FVOCI</td>'
        '<td style="padding:0.4rem 0.6rem;">Fair Value Through OCI</td>'
        '<td style="padding:0.4rem 0.6rem;">Fair Value, via Eigenkapital (OCI)</td>'
        '<td style="padding:0.4rem 0.6rem;color:#A52F4D;">'
        'Verlust geht via OCI in Reserve → reduziert CET1 <strong>direkt</strong></td></tr>'
        '<tr><td style="padding:0.4rem 0.6rem;font-weight:600;">AC</td>'
        '<td style="padding:0.4rem 0.6rem;">Amortised Cost (≈ HtM, „Held-to-Maturity")</td>'
        '<td style="padding:0.4rem 0.6rem;">zum fortgeführten Anschaffungswert</td>'
        '<td style="padding:0.4rem 0.6rem;color:#00A9A5;">'
        '<strong>Kein</strong> CET1-Effekt — latenter Verlust bleibt verborgen</td></tr>'
        '</tbody></table>'
        '<div style="margin-top:0.7rem;font-size:0.82rem;color:#6E6E6E;">'
        '<strong>Wichtige Abgrenzung — IFRS-9 hat zwei separate Dimensionen:</strong> '
        '(a)&nbsp;<em>Klassifizierung</em> (HfT/FVTPL/FVOCI/AC) bestimmt, wie ein '
        'Asset bilanziert wird. (b)&nbsp;<em>Stages 1/2/3</em> sind eine '
        'separate Impairment-Dimension für Kreditforderungen (kein Default → '
        'Stage 1, signifikante Verschlechterung → Stage 2, Default → Stage 3). '
        'Wir zeigen hier <strong>nur die Klassifizierung</strong> '
        '(a) — sie ist für die Mark-to-Market-Wirkung auf Sovereign-Bonds '
        'allein entscheidend.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="background:#F4F4F4;padding:0.7rem 1.0rem;'
        'border-radius:6px;margin:0.2rem 0 1rem 0;color:#051C2C;'
        'font-size:0.86rem;line-height:1.55;">'
        '<strong>Woher kommen die Zahlen?</strong> Direkt aus den '
        'EBA-Items der <code>tr_sov.csv</code>: <code>2520812</code> (HfT), '
        '<code>2520813</code> (FVTPL), <code>2520814</code> (FVOCI), '
        '<code>2520815</code> (AC). Jede IRB-Bank meldet pro Land × Bucket '
        'die Aufteilung — wir aggregieren über alle Länder und Buckets, '
        'damit man auf einen Blick sieht, wie viel des Sovereign-Buchs '
        'CET1-relevant ist (HfT + FVTPL + FVOCI) versus zu Buchwert '
        'gehalten wird (AC).'
        '</div>',
        unsafe_allow_html=True,
    )

    # Aggregate per accounting_class system-wide
    sys_by_class = (acct_split.groupby(["accounting_class", "channel"],
                                       as_index=False)["exposure_eur"]
                    .sum().sort_values("exposure_eur", ascending=False))

    a_l, a_r = st.columns([2, 3], gap="medium")

    with a_l:
        st.markdown("**System-weiter Sovereign-Bestand pro IFRS-9-Klasse**")
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
        # Stacked bar: alle 10 Banken, FV by accounting class
        top_banks = (acct_split.groupby("LEI_Code")["exposure_eur"].sum()
                     .nlargest(N_IRB_BANKS).index)
        sub = acct_split[acct_split["LEI_Code"].isin(top_banks)]
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
            title=f"Alle {N_IRB_BANKS} Banken · Sovereign FV nach IFRS-9-Class (bn EUR)",
            barmode="stack", height=480,
            xaxis_title="Fair Value [bn EUR]",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
        )
        st.plotly_chart(fig_st, use_container_width=True)

    st.divider()

    # === Erkannt vs. verborgen · latente AC-Verluste (SVB-Lektion) ===
    eyebrow("Erkannt vs. verborgen · latente AC-Verluste (die SVB-Lektion)")

    _svb_shock = delta_r_pp if abs(delta_r_pp) > 1e-3 else 2.0
    _svb_label = (f"Live-Schock Δr = {delta_r_pp:+.1f} pp"
                  if abs(delta_r_pp) > 1e-3
                  else "Beispiel-Schock Δr = +2.0 pp (kein Live-Schock aktiv)")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #C9A227;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Ökonomische Interpretation — warum „latent" nicht '
        '„harmlos" heißt.</strong> Die IFRS-9-Tabelle oben zeigt: '
        'AC-Bestände (Amortised Cost) erzeugen unter einem Zinsschock '
        '<em>keinen</em> CET1-Effekt — der Marktwertverlust existiert '
        'ökonomisch trotzdem, er ist nur bilanziell unsichtbar. Der '
        'Kollaps der Silicon Valley Bank (März 2023) hat gezeigt, wie '
        'dieser latente Verlust real wird: Muss eine Bank wegen '
        'Liquiditätsabflüssen AC-Bestände <em>vorzeitig verkaufen</em>, '
        'wird der gesamte aufgelaufene Marktwertverlust schlagartig '
        'GuV-wirksam. Jiang et al. (2023) beziffern die unrealisierten '
        'Verluste des US-Bankensystems nach der Zinswende auf rund '
        '2 Billionen USD — der Auslöser des Runs auf SVB.<br><br>'
        '<strong>Mathematik.</strong> Identische MtM-Formel für alle '
        'Klassen: ΔFV = −D·Δy·E pro Laufzeit-Bucket. Aufteilung: '
        '<em>erkannt</em> = ΔFV der HfT/FVTPL/FVOCI-Bestände '
        '(CET1-wirksam) · <em>latent</em> = ΔFV der AC-Bestände '
        '(nicht CET1-wirksam, aber ökonomisch real).<br>'
        '<span style="font-size:0.82rem;color:#6E6E6E;">'
        '<strong>Quellen:</strong> Jiang, Matvos, Piskorski &amp; Seru '
        '(2023): „Monetary Tightening and U.S. Bank Fragility in 2023: '
        'Mark-to-Market Losses and Uninsured Depositor Runs?", NBER '
        'Working Paper 31048 · FDIC (2023): „FDIC\'s Supervision of '
        'Silicon Valley Bank" · Duration-MtM: Tuckman/Serrat (2012), '
        'Kap. 4.</span></div>',
        unsafe_allow_html=True,
    )

    _imp_svb = sovereign_cet1_impact(acct_split, delta_r_pp=_svb_shock)
    _rec = (_imp_svb[_imp_svb["channel"].isin(["P&L", "OCI"])]
            .groupby("LEI_Code")["cet1_impact_eur"].sum())
    _lat = (_imp_svb[_imp_svb["channel"] == "none"]
            .groupby("LEI_Code")["delta_fv_eur"].sum())
    svb = pd.DataFrame({"erkannt_eur": _rec, "latent_eur": _lat}).fillna(0.0)
    svb["total_eur"] = svb["erkannt_eur"] + svb["latent_eur"]
    svb = svb.sort_values("total_eur", ascending=False)
    svb["name"] = [_short(_LEI2NAME.get(l, l[:8]), 22) for l in svb.index]

    sv_l, sv_r = st.columns([3, 2], gap="medium")

    with sv_l:
        fig_svb = go.Figure()
        fig_svb.add_trace(go.Bar(
            y=svb["name"], x=svb["erkannt_eur"] / 1e9, orientation="h",
            name="Erkannt · HfT/FVTPL/FVOCI (CET1-wirksam)",
            marker_color=COLORS["crimson"], marker_line_width=0,
        ))
        fig_svb.add_trace(go.Bar(
            y=svb["name"], x=svb["latent_eur"] / 1e9, orientation="h",
            name="Latent · AC (bilanziell unsichtbar)",
            marker_color=COLORS["amber"], marker_line_width=0,
        ))
        fig_svb.update_layout(
            title=f"MtM-Wirkung pro Bank · {_svb_label}",
            barmode="relative", height=440,
            xaxis_title="ΔFair Value [Mrd. €] · negativ = Verlust",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
        )
        st.plotly_chart(fig_svb, use_container_width=True)

    with sv_r:
        sys_rec = float(svb["erkannt_eur"].sum())
        sys_lat = float(svb["latent_eur"].sum())
        sys_tot = sys_rec + sys_lat
        lat_share = sys_lat / sys_tot if abs(sys_tot) > 1 else 0.0
        st.metric("Σ MtM-Wirkung (alle 10 Banken)",
                  f"€{sys_tot/1e9:+.1f} bn", _svb_label,
                  delta_color="off")
        st.metric("davon CET1-wirksam erkannt",
                  f"€{sys_rec/1e9:+.1f} bn",
                  "HfT + FVTPL + FVOCI", delta_color="off")
        st.metric("davon latent (AC, verborgen)",
                  f"€{sys_lat/1e9:+.1f} bn",
                  f"{lat_share*100:.0f}% der Gesamt-MtM-Wirkung",
                  delta_color="off")
        st.caption(
            "AC schützt die CET1-Quote **buchhalterisch**, nicht "
            "**ökonomisch**: Solange die Bank halten kann, läuft der "
            "Verlust per Pull-to-Par aus. Gerät sie unter "
            "Liquiditätsdruck und muss verkaufen, materialisiert sich "
            "der latente Verlust sofort — der SVB-Mechanismus. Für die "
            "CET1-Bridge (Tab 4) zählt konsistent nur der erkannte Teil."
        )

    st.divider()

    # === Worked Example · konkret durchgerechnet =====================
    eyebrow("Worked Example · ein +200 bp Zinsschock konkret durchgerechnet")

    st.markdown(
        '<div style="background:#FAFAFA;border:1px solid #E6E6E6;'
        'border-left:2px solid #051C2C;padding:0.95rem 1.2rem;'
        'border-radius:4px;margin:0.4rem 0 1.0rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        'Wir nehmen einen <strong>Δr_10y = +2.0 pp</strong>-Schock und '
        'rechnen exemplarisch durch, was er für vier Banken bedeutet. '
        'Formel pro Bank: '
        '<code>ΔFair Value = − Σ_buckets Modified Duration · Δy · '
        'Exposure_bucket</code>. Wir aggregieren über alle '
        'Restlaufzeit-Buckets und alle Counterparty-Länder. Nur die '
        'IFRS-9-Klassen HfT + FVTPL + FVOCI schlagen auf CET1 durch '
        '(AC bleibt zu Buchwert).'
        '</div>',
        unsafe_allow_html=True,
    )

    # Compute concrete examples for 4 representative banks
    _delta_r_example = 2.0  # +200 bp
    _example_leis = [
        ("Crédit Agricole", "FR969500TJ5KRTCJQWXH"),
        ("Deutsche Bank",   "7LTWFZYICNSX8D621K86"),
        ("UniCredit",       "549300TRUWO2CD2G5692"),
        ("Banco Santander", "5493006QMFDDMYWIAM13"),
    ]
    _wf_rows = []
    _pnl_at_shock = rate_shock_pnl(mat, delta_r_pp=_delta_r_example)
    _acct_at_shock = sovereign_cet1_impact(acct_split,
                                           delta_r_pp=_delta_r_example)
    for bank_label, lei in _example_leis:
        sub_pnl = _pnl_at_shock[_pnl_at_shock["LEI_Code"] == lei]
        sub_acct = _acct_at_shock[_acct_at_shock["LEI_Code"] == lei]
        if len(sub_pnl) == 0:
            continue
        total_sov = float(sub_pnl["exposure_eur"].sum()) \
                    if "exposure_eur" in sub_pnl.columns else \
                    float(conc[conc["LEI_Code"]==lei]["exposure_eur"].sum())
        total_mtm = float(sub_pnl["delta_pnl_eur"].sum())
        cet1_eff  = float(sub_acct["cet1_impact_eur"].sum())
        cet1_share = (cet1_eff / total_mtm * 100) if abs(total_mtm) > 1 else 0.0
        _wf_rows.append({
            "Bank":                      bank_label,
            "Σ Sovereign-Buch (Mrd. €)": f"€{total_sov/1e9:,.0f}",
            "MtM-Verlust (Mrd. €)":      f"€{total_mtm/1e9:+,.2f}",
            "Davon CET1-wirksam":        f"€{cet1_eff/1e9:+,.2f}",
            "% durchschlagend":          f"{cet1_share:.0f}%",
        })
    if _wf_rows:
        st.dataframe(pd.DataFrame(_wf_rows), hide_index=True,
                     use_container_width=True, height=200)
        st.markdown(
            '<div style="background:#FFFFFF;border-left:2px solid #A52F4D;'
            'padding:0.7rem 1.0rem;margin:0.4rem 0 0.6rem 0;color:#051C2C;'
            'font-size:0.86rem;line-height:1.55;">'
            '<strong>Drei Lehren aus dem Beispiel:</strong><br>'
            '• Banken mit großem Sovereign-Buch haben absolut die '
            'größten MtM-Verluste — die Schock-Magnitude skaliert '
            'linear mit dem Bestand.<br>'
            '• Der CET1-wirksame Anteil hängt stark von der IFRS-9-'
            'Klassifizierung ab: italienische und französische Banken '
            'halten typisch mehr in HfT/FVTPL/FVOCI, was den Anteil in '
            'die Höhe treibt.<br>'
            '• AC-(Buchwert-)Anteile sind in der ΔFV-Spalte enthalten, '
            'erscheinen aber NICHT in der CET1-Wirksam-Spalte — der '
            'latente Verlust ist ökonomisch real, regulatorisch aber '
            'unsichtbar.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # === Zinsrisiko-Profil · gewichtete Duration + BPV ===============
    eyebrow("Zinsrisiko-Profil · gewichtete Duration & Basis-Punkt-Wert (BPV)")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #2251FF;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Ökonomische Interpretation.</strong> Zwei Banken mit '
        'gleich großem Sovereign-Buch können völlig unterschiedlich '
        'zinssensitiv sein — entscheidend ist, <em>wo auf der Kurve</em> '
        'sie investiert sind. Die Kennzahl dafür ist die '
        '<strong>exposure-gewichtete Modified Duration</strong> D̄ und der '
        'daraus abgeleitete <strong>Basis-Punkt-Wert (BPV</strong>, auch '
        'DV01): der €-Verlust des Portfolios bei +1 Basispunkt '
        'Zinsanstieg. Der Balken zerlegt den BPV zusätzlich nach '
        'Laufzeit-Buckets — so sieht man, ob das Zinsrisiko einer Bank '
        'aus dem langen Ende (&gt;10Y) oder aus der Breite des Buchs kommt.'
        '<br><br>'
        '<strong>Mathematik.</strong> '
        'D̄<sub>i</sub> = Σ<sub>b</sub> E<sub>b</sub>·D<sub>b</sub> / '
        'Σ<sub>b</sub> E<sub>b</sub> &nbsp;·&nbsp; '
        'BPV<sub>i</sub> = Σ<sub>b</sub> E<sub>b</sub>·D<sub>b</sub>·10⁻⁴. '
        'Bucket-Durationen per Mittelpunkt-Approximation: &lt;3M 0.125 · '
        '3M–1Y 0.625 · 1–2Y 1.5 · 2–3Y 2.5 · 3–5Y 4.0 · 5–10Y 7.5 · '
        '&gt;10Y 15.0 Jahre (dokumentiert in MODEL_ASSUMPTIONS §7).<br>'
        '<span style="font-size:0.82rem;color:#6E6E6E;">'
        '<strong>Quellen:</strong> Tuckman &amp; Serrat (2012): „Fixed '
        'Income Securities", Kap. 4 (DV01/Duration) · Fabozzi (2007): '
        '„Fixed Income Analysis", Kap. Duration &amp; Convexity.'
        '</span></div>',
        unsafe_allow_html=True,
    )

    risk = mat.copy()
    risk["bpv_eur"] = risk["duration_years"] * risk["exposure_eur"] * 1e-4
    bank_risk = (risk.groupby("LEI_Code")
                     .agg(exp_eur=("exposure_eur", "sum"),
                          bpv_eur=("bpv_eur", "sum")))
    bank_risk["dur_w"] = bank_risk["bpv_eur"] * 1e4 / bank_risk["exp_eur"]
    bank_risk = bank_risk.sort_values("bpv_eur")
    bank_risk["name"] = [_short(_LEI2NAME.get(l, l[:8]), 22)
                         for l in bank_risk.index]

    bucket_order = (mat[["Maturity", "label"]].drop_duplicates()
                    .sort_values("Maturity")["label"].tolist())
    _BUCKET_COLORS = ["#BFD7EA", "#8FB8DC", "#5E9ACD", "#2D7CBE",
                      "#0F5C97", "#0A3D64", "#A52F4D"]

    bp_l, bp_r = st.columns([3, 2], gap="medium")

    with bp_l:
        fig_bpv = go.Figure()
        for b_label, b_color in zip(bucket_order, _BUCKET_COLORS):
            sub_b = (risk[risk["label"] == b_label]
                     .set_index("LEI_Code")["bpv_eur"]
                     .reindex(bank_risk.index).fillna(0.0))
            fig_bpv.add_trace(go.Bar(
                y=bank_risk["name"], x=sub_b / 1e6, orientation="h",
                name=b_label, marker_color=b_color, marker_line_width=0,
            ))
        fig_bpv.update_layout(
            title="BPV pro Bank, zerlegt nach Laufzeit-Bucket [Mio. € pro +1 bp]",
            barmode="stack", height=440,
            xaxis_title="BPV [Mio. € / bp]",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0, font=dict(size=9)),
        )
        st.plotly_chart(fig_bpv, use_container_width=True)

    with bp_r:
        st.markdown("**Duration & BPV pro Bank**")
        st.caption(
            "**D̄** = exposure-gewichtete Modified Duration in Jahren · "
            "**BPV** = €-Verlust bei +1 bp. Hohes D̄ bei kleinem Buch "
            "kann denselben BPV erzeugen wie niedriges D̄ bei großem Buch."
        )
        bpv_disp = pd.DataFrame({
            "Bank":          bank_risk["name"],
            "Σ Sov bn":      (bank_risk["exp_eur"] / 1e9).round(0).astype(int),
            "D̄ (Jahre)":     bank_risk["dur_w"].round(2),
            "BPV (Mio €/bp)": (bank_risk["bpv_eur"] / 1e6).round(1),
        }).iloc[::-1]
        st.dataframe(bpv_disp, use_container_width=True, hide_index=True,
                     height=420)

    st.divider()

    # === Sensitivität · CET1-wirksamer Verlust über die Δr-Range =====
    eyebrow("Sensitivität · CET1-wirksamer Sovereign-Verlust über die gesamte Δr-Range")

    st.markdown(
        '<div style="background:#FFFFFF;border:1px solid #E6E6E6;'
        'border-left:4px solid #034B6F;padding:0.85rem 1.1rem;'
        'border-radius:6px;margin:0.4rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.65;">'
        '<strong>Ökonomische Interpretation.</strong> Statt eines '
        'einzelnen Szenarios zeigt diese Sektion die volle '
        '<strong>Verlustfunktion</strong> jeder Bank über die gesamte '
        'Slider-Range (−3 bis +5 pp). Die Steigung jeder Linie ist das '
        'bankspezifische Zinsrisiko des CET1-wirksamen Teilbuchs '
        '(HfT/FVTPL/FVOCI) — sie kombiniert Buchgröße, Duration und '
        'IFRS-9-Mix in einer Zahl. Rechts dieselbe Wirkung normiert auf '
        'das CET1-Kapital: Wie viele Prozentpunkte ihres Verlustpuffers '
        'kostet der Schock jede Bank allein über den Sovereign-Kanal?'
        '<br><br>'
        '<strong>Mathematik.</strong> Verlust<sub>i</sub>(Δr) = '
        'Σ<sub>b∈CET1-wirksam</sub> D<sub>b</sub>·E<sub>b</sub> · (Δr/100) '
        '— linear in Δr (Erstordnungs-Näherung). Der '
        'Konvexitätsterm ½·C·(Δy)² wird vernachlässigt; bei steigenden '
        'Zinsen <em>überschätzt</em> die lineare Näherung den Verlust '
        'leicht — die Schätzung ist also konservativ.<br>'
        '<span style="font-size:0.82rem;color:#6E6E6E;">'
        '<strong>Quellen:</strong> Tuckman &amp; Serrat (2012), Kap. 4 · '
        'Fabozzi (2007), Kap. Duration &amp; Convexity (Konvexitäts-'
        'Korrektur zweiter Ordnung).</span></div>',
        unsafe_allow_html=True,
    )

    eff = acct_split[acct_split["channel"].isin(["P&L", "OCI"])].copy()
    eff["dxe"] = eff["duration_years"] * eff["exposure_eur"]
    slope_per_pp = eff.groupby("LEI_Code")["dxe"].sum() / 100.0

    sn_l, sn_r = st.columns([3, 2], gap="medium")

    with sn_l:
        r_grid = np.linspace(-3.0, 5.0, 33)
        fig_sw = go.Figure()
        _LINE_COLORS = [COLORS["navy"], COLORS["mid_blue"],
                        COLORS["bright_blue"], COLORS["crimson"],
                        COLORS["amber"], "#00A9A5", "#6E6E6E",
                        "#8FB8DC", "#A52F4D", "#0A3D64"]
        _ranked = slope_per_pp.sort_values(ascending=False)
        for i, (lei, slope) in enumerate(_ranked.items()):
            fig_sw.add_trace(go.Scatter(
                x=r_grid, y=-slope * r_grid / 1e9, mode="lines",
                name=_short(_LEI2NAME.get(lei, lei[:8]), 18),
                line=dict(width=2,
                          color=_LINE_COLORS[i % len(_LINE_COLORS)]),
            ))
        if abs(delta_r_pp) > 1e-3:
            fig_sw.add_vline(x=delta_r_pp, line_dash="dash",
                             line_color=COLORS["crimson"],
                             annotation_text=f"Live: {delta_r_pp:+.1f} pp",
                             annotation_font_size=10)
        fig_sw.update_layout(
            title="CET1-wirksamer Sovereign-MtM pro Bank [Mrd. €] über Δr",
            xaxis_title="Δr_10y [Prozentpunkte]",
            yaxis_title="CET1-wirksamer ΔFV [Mrd. €]",
            height=460,
            legend=dict(font=dict(size=9)),
        )
        st.plotly_chart(fig_sw, use_container_width=True)

    with sn_r:
        _pct_shock = delta_r_pp if abs(delta_r_pp) > 1e-3 else 2.0
        pct = pd.DataFrame({"slope": slope_per_pp})
        pct = pct.merge(cap10[["LEI_Code", "cet1_eur"]],
                        left_index=True, right_on="LEI_Code")
        pct = pct[pct["cet1_eur"] > 0].copy()
        pct["loss_pct_cet1"] = (pct["slope"] * _pct_shock
                                / pct["cet1_eur"] * 100)
        pct["name"] = [_short(_LEI2NAME.get(l, l[:8]), 20)
                       for l in pct["LEI_Code"]]
        pct = pct.sort_values("loss_pct_cet1")
        fig_pct = go.Figure(go.Bar(
            y=pct["name"], x=pct["loss_pct_cet1"], orientation="h",
            marker_color=COLORS["crimson"], marker_line_width=0,
            text=[f"{v:.1f}%" for v in pct["loss_pct_cet1"]],
            textposition="outside",
            textfont=dict(size=9, color=COLORS["navy"]),
        ))
        fig_pct.update_layout(
            title=(f"Sovereign-Verlust in % des CET1-Kapitals "
                   f"(Δr = {_pct_shock:+.1f} pp)"),
            xaxis_title="CET1-wirksamer Verlust [% des CET1]",
            height=460,
        )
        st.plotly_chart(fig_pct, use_container_width=True)
        st.caption(
            "Isolierter Sovereign-Kanal — die volle 2-Kanal-Bridge "
            "(Kreditbuch + Sovereign, inkl. ΔRWA-Nenner-Effekt) zeigt "
            "Tab 4 · Eigenkapital."
        )

    st.divider()

    # === Per-bank drilldown ===
    eyebrow("Bank-Drilldown · Maturity-Ladder + CET1-Impact pro Klasse")

    st.markdown(
        '<div style="background:#F4F4F4;padding:0.75rem 1.0rem;'
        'border-radius:6px;margin:0.3rem 0 0.9rem 0;color:#051C2C;'
        'font-size:0.86rem;line-height:1.55;">'
        '<strong>Wozu der Drilldown?</strong> Alle bisherigen Grafiken '
        'zeigen aggregierte Systemzahlen. Hier kann man eine einzelne der '
        f'{N_IRB_BANKS} IRB-Banken auswählen und sieht die zwei '
        'entscheidenden bankspezifischen Verteilungen: '
        '(1)&nbsp;<em>Maturity-Ladder</em> — wie sich ihr '
        'Staatsanleihen-Bestand auf Restlaufzeiten verteilt, '
        '(2)&nbsp;<em>CET1-Impact pro IFRS-9-Klasse</em> — welche '
        'Bilanzkategorie wie viel zur Eigenkapitalwirkung des '
        'Zinsschocks beiträgt.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Alle 10 kuratierten IRB-Banken im Dropdown, nach Sovereign-Exposure
    # sortiert, Type-to-Search via Selectbox.
    sov_per_bank = (conc.groupby("LEI_Code")["exposure_eur"].sum()
                       .sort_values(ascending=False))
    lei_to_name = bank_dir.set_index("lei")["bank_name"].to_dict()
    bank_options = sov_per_bank.index.tolist()

    sel_lei = st.selectbox(
        f"Bank wählen ({len(bank_options)} IRB-Banken · tippen zum Suchen)",
        bank_options,
        format_func=lambda lei: (
            f"{lei_to_name.get(lei, lei[:8])}  "
            f"(€{sov_per_bank.loc[lei]/1e9:.1f} bn Sovereign-Buch)"
        ),
        key="sov_drill_bank",
    )
    sel_name = lei_to_name.get(sel_lei, sel_lei[:12])

    drill_l, drill_r = st.columns([3, 2], gap="medium")

    with drill_l:
        eyebrow(f"{sel_name} · Maturity-Ladder")
        st.caption(
            "Verteilt das Sovereign-Buch dieser Bank auf die sieben "
            "Restlaufzeit-Buckets der EBA-Disclosure. Längere Laufzeiten "
            "(höhere Duration) → stärkere Mark-to-Market-Reaktion auf "
            "einen Zinsschock. Eine Bank mit Schwerpunkt im 5–10y- und "
            ">10y-Bucket ist deutlich zinssensitiver als eine mit "
            "Konzentration im kurzen Ende."
        )
        sub_mat = mat[mat["LEI_Code"] == sel_lei].sort_values("Maturity")
        if len(sub_mat) > 0:
            fig_mat = go.Figure(go.Bar(
                x=sub_mat["label"], y=sub_mat["exposure_eur"]/1e9,
                marker_color=COLORS["mid_blue"], marker_line_width=0,
                text=[f"{v/1e9:.1f}" for v in sub_mat["exposure_eur"]],
                textposition="outside",
                textfont=dict(size=10, color=COLORS["navy"]),
                hovertemplate="<b>%{x}</b><br>%{y:.2f} bn EUR<extra></extra>",
            ))
            fig_mat.update_layout(
                xaxis_title="Restlaufzeit-Bucket",
                yaxis_title="Exposure [Mrd. EUR]",
                height=380, bargap=0.25,
            )
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info("Keine Maturity-Daten für diese Bank im Datensatz.")

    with drill_r:
        eyebrow(f"{sel_name} · CET1-Impact pro IFRS-9-Klasse")
        st.caption(
            "Spaltenbedeutung: **FV bn** = aktueller Bilanzwert in der "
            "jeweiligen Klasse · **ΔFV bn** = Mark-to-Market-Verlust unter "
            "dem Δr-Schock · **CET1 Δ bn** = davon CET1-wirksam "
            "(HfT/FVTPL/FVOCI: voll; AC: 0). Die Σ-Kennzahl unten "
            "summiert über alle Klassen."
        )
        if abs(delta_r_pp) > 1e-3:
            cet1_acct = sovereign_cet1_impact(acct_split, delta_r_pp=delta_r_pp)
            bank_imp = cet1_acct[cet1_acct["LEI_Code"] == sel_lei]
            if len(bank_imp) > 0:
                disp = bank_imp.copy()
                disp["FV bn"]    = (disp["fair_value_eur"]/1e9).round(2)
                disp["ΔFV bn"]   = (disp["delta_fv_eur"]/1e9).round(2)
                disp["CET1 Δ bn"] = (disp["cet1_impact_eur"]/1e9).round(2)
                disp = disp.rename(columns={"accounting_class":"Klasse",
                                             "channel":"Kanal"})
                st.dataframe(
                    disp[["Klasse","Kanal","FV bn","ΔFV bn","CET1 Δ bn"]],
                    use_container_width=True, hide_index=True, height=240,
                )
                tot = float(bank_imp["cet1_impact_eur"].sum())
                st.metric("Σ CET1-Impact dieser Bank",
                          f"€{tot/1e9:+.2f} bn",
                          f"unter Δr = {delta_r_pp:+.0f} pp")
            else:
                st.info("Keine IFRS-9-Klassen-Daten für diese Bank.")
        else:
            st.info(
                "Setze einen Δr-Schock in der Sidebar, um den CET1-Impact "
                "zu sehen."
            )


footer(
    f"Zwei Sub-Tabs · Yield-Curve (Bundesbank-Svensson-Input) · Sovereigns "
    f"(Doom-Loop-Map · IFRS-9-Split · latente AC-Verluste · Duration/BPV · "
    f"Δr-Sensitivität — der ΔMtM-Kanal der CET1-Bridge) · "
    f"Daten: EBA Transparency 2025 (tr_sov.csv + CET1 aus tr_oth.csv, "
    f"Stichtag Juni 2025) · 10 IRB-Banken"
)
