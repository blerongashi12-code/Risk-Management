"""Marktbuch · Zins-/Sovereign-Kanal des 2-Kanal-Modells.

Der Tab erklärt den Sovereign-Marktwertkanal der CET1-Bridge:
Zinsschock -> Duration-basierter Marktwertverlust -> IFRS-9-Klasse
entscheidet, welcher Teil in CET1 sichtbar wird.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import (
    tab_breadcrumb,
    apply_theme,
    eyebrow,
    insight,
    footer,
    COLORS,
)
from components.sidebar import render_sidebar
from components.methodology import render_sovereign_methodology
from components.legacy_views import render_yield_curve_tab
from components.backend_path import setup

setup()

from config import EBA_RAW_DIR                                      # type: ignore
from eba_loader import (                                            # type: ignore
    parse_sovereign_csv,
    sovereign_concentration,
    sovereign_maturity_ladder,
    domestic_share_per_bank,
    sovereign_kpis_per_bank,
    rate_shock_pnl,
    sovereign_by_accounting_class,
    sovereign_cet1_impact,
    parse_capital_overview,
    load_bank_directory,
    load_country_dim,
)
from eba_pd_loader import get_top10_leis                            # type: ignore


st.set_page_config(page_title="Marktbuch · Sovereign-/Zins-Kanal", layout="wide")
apply_theme()
config = render_sidebar()

st.markdown(
    '<div style="margin:0.35rem 0 0.75rem 0;padding:0.95rem 1.1rem;'
    'background:#FFFFFF;border:1px solid #E6E6E6;border-left:4px solid #051C2C;'
    'border-top:4px solid #2251FF;color:#051C2C;">'
    '<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;'
    'text-transform:uppercase;color:#6E6E6E;margin-bottom:0.25rem;">'
    'Tab 3 · Marktbuch-Stressmodell · Sovereign-Zinskanal</div>'
    '<div style="font-size:1.28rem;font-weight:800;line-height:1.2;'
    'font-family:Inter,-apple-system,Segoe UI,sans-serif;'
    'letter-spacing:0.005em;margin-bottom:0.35rem;">'
    'Marktbuch: Sovereign-Zinsschock, IFRS-9-Wirkung und CET1</div>'
    '<div style="font-size:0.90rem;line-height:1.55;color:#3A4A57;'
    'max-width:1080px;">'
    'Dieser Tab übersetzt den aktiven 10-Jahres-Zinsschock in Marktwertänderungen '
    'der gemeldeten Staatsanleihen-Bestände. Die Duration bestimmt die Höhe '
    'des Marktwerteffekts; der bankindividuelle IFRS-9-Split entscheidet, '
    'welcher Anteil über Gewinn und Verlust oder über OCI/Eigenkapital in '
    'der CET1-Bridge sichtbar wird.'
    '</div>'
    '<div style="display:flex;gap:0.45rem;flex-wrap:wrap;margin-top:0.65rem;">'
    '<span style="font-size:0.70rem;font-weight:700;color:#034B6F;'
    'background:#EEF6FA;border:1px solid #D8E6ED;padding:0.25rem 0.45rem;">'
    'EBA Sovereign Template</span>'
    '<span style="font-size:0.70rem;font-weight:700;color:#034B6F;'
    'background:#EEF6FA;border:1px solid #D8E6ED;padding:0.25rem 0.45rem;">'
    'Modified Duration</span>'
    '<span style="font-size:0.70rem;font-weight:700;color:#034B6F;'
    'background:#EEF6FA;border:1px solid #D8E6ED;padding:0.25rem 0.45rem;">'
    'IFRS-9 Split</span>'
    '<span style="font-size:0.70rem;font-weight:700;color:#051C2C;'
    'background:#F4F4F4;border:1px solid #E0E0E0;padding:0.25rem 0.45rem;">'
    'CET1 Bridge</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

tab_breadcrumb(3)
delta_r_pp = float(config["d_r_10y_pp"])


@st.cache_data(ttl=24 * 3600, show_spinner="Loading EBA Sovereign-data ...")
def _load_data():
    sov_raw = parse_sovereign_csv(EBA_RAW_DIR / "tr_sov.csv", period=202506)
    bank_dir = load_bank_directory(EBA_RAW_DIR / "TR_Metadata.xlsx")
    cty_dim = load_country_dim(EBA_RAW_DIR / "TR_Metadata.xlsx")
    cap_raw = parse_capital_overview(EBA_RAW_DIR / "tr_oth.csv", period=202506)
    return sov_raw, bank_dir, cty_dim, cap_raw


sov_raw, bank_dir, cty_dim, cap_raw = _load_data()
_IRB_LEIS = get_top10_leis()
N_IRB_BANKS = len(_IRB_LEIS)

conc = sovereign_concentration(sov_raw, period=202506)
conc = conc[conc["LEI_Code"].isin(_IRB_LEIS)].copy()
mat = sovereign_maturity_ladder(sov_raw, period=202506)
mat = mat[mat["LEI_Code"].isin(_IRB_LEIS)].copy()
acct_split = sovereign_by_accounting_class(sov_raw, period=202506)
acct_split = acct_split[acct_split["LEI_Code"].isin(_IRB_LEIS)].copy()
cap10 = cap_raw[cap_raw["LEI_Code"].isin(_IRB_LEIS)].copy()

_LEI2NAME = bank_dir.set_index("lei")["bank_name"].to_dict()


def _short(name: str, n: int = 24) -> str:
    return name if len(name) <= n else name[: n - 1] + "..."


def _bn(value: float, digits: int = 1) -> str:
    return f"€{value / 1e9:+.{digits}f} bn"


def _metric_bn(value: float, digits: int = 1) -> str:
    return f"€{value / 1e9:.{digits}f} bn"


def _info_box(html: str, border: str = "#034B6F", tone: str = "#FFFFFF") -> None:
    st.markdown(
        f'<div style="background:{tone};border:1px solid #E6E6E6;'
        f'border-left:4px solid {border};padding:0.9rem 1.05rem;'
        'border-radius:6px;margin:0.35rem 0 1rem 0;color:#051C2C;'
        'font-size:0.88rem;line-height:1.6;">'
        f'{html}</div>',
        unsafe_allow_html=True,
    )


def _tab_overview_cards() -> None:
    st.markdown(
        """
<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.75rem;margin:0.2rem 0 1.1rem 0;">
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-top:3px solid #034B6F;border-radius:6px;padding:0.85rem;">
    <div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;letter-spacing:0.08em;text-transform:uppercase;">Input · Bestand</div>
    <div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">Sovereign-Buch</div>
    <div style="font-size:0.78rem;color:#536774;line-height:1.45;margin-top:0.25rem;">EBA-Bestände nach Bank, Land, Laufzeit und IFRS-9-Klasse.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-top:3px solid #2251FF;border-radius:6px;padding:0.85rem;">
    <div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;letter-spacing:0.08em;text-transform:uppercase;">Input · Markt</div>
    <div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">Zinsschock</div>
    <div style="font-size:0.78rem;color:#536774;line-height:1.45;margin-top:0.25rem;">Aktiver 10-Jahres-Schock aus der Sidebar als Parallel-Shift.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-top:3px solid #051C2C;border-radius:6px;padding:0.85rem;">
    <div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;letter-spacing:0.08em;text-transform:uppercase;">Analyse · MtM</div>
    <div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">Marktwertwirkung</div>
    <div style="font-size:0.78rem;color:#536774;line-height:1.45;margin-top:0.25rem;">Duration übersetzt den Zinsschock in Fair-Value-Änderungen.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-top:3px solid #6E6E6E;border-radius:6px;padding:0.85rem;">
    <div style="font-size:0.72rem;font-weight:800;color:#6E6E6E;letter-spacing:0.08em;text-transform:uppercase;">Output · CET1</div>
    <div style="font-weight:760;color:#051C2C;margin-top:0.2rem;">IFRS-9-Wirkung</div>
    <div style="font-size:0.78rem;color:#536774;line-height:1.45;margin-top:0.25rem;">P&L und OCI gehen in CET1; AC bleibt im Basismodell nicht aktiv.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _weighted_duration(df: pd.DataFrame) -> float:
    total = float(df["exposure_eur"].sum())
    if total <= 0:
        return 0.0
    return float((df["duration_years"] * df["exposure_eur"]).sum() / total)


tab_yc, tab_sov = st.tabs([
    "1 · Yield-Curve Input",
    "2 · Sovereign-Kanal",
])


with tab_yc:
    render_yield_curve_tab(config)

    st.divider()
    eyebrow("Plausibilitätsanker · Zinsschock-Range")
    _info_box(
        '<strong>Warum die Sidebar-Range fachlich tragfähig ist.</strong> '
        'Die Range von -3 bis +5 Prozentpunkten ist ein Stresstest-Rahmen, '
        'kein Forecast. Historische Zinsphasen zeigen, dass Bewegungen von '
        '+1 bis +3 Prozentpunkten innerhalb weniger Monate bis Quartale '
        'realistisch sind; größere Werte bilden bewusst einen schweren '
        'adversen Tail ab. Methodisch bleibt der Tab transparent: Wir '
        'modellieren einen Parallel-Shift und übersetzen ihn mit Modified '
        'Duration in Marktwertverluste.',
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["EZB-Zinswende 2021-2023", "10y Bund", "ca. +320 bp", "sehr schwerer realer Anker"],
                ["Bond-Massaker 1994", "10y US Treasury", "ca. +240 bp", "globaler Zinsstress"],
                ["Taper Tantrum 2013", "10y US Treasury", "ca. +140 bp", "schneller Neubewertungsschock"],
                ["UK-Gilt-Krise 2022", "30y UK Gilt", "ca. +130 bp", "Liquiditätsstress am langen Ende"],
            ],
            columns=["Episode", "Markt", "Bewegung", "Einordnung"],
        ),
        use_container_width=True,
        hide_index=True,
        height=180,
    )
    st.caption(
        "Quellen: Deutsche Bundesbank Zinszeitreihen, Federal Reserve H.15, "
        "Bank of England Financial Stability Report 2022. Werte gerundet; "
        "die exakte Kurvenform realer Episoden wird in V1 bewusst als "
        "Parallel-Shift approximiert."
    )


with tab_sov:
    eyebrow("Tab-Übersicht · was im Marktbuch gezeigt wird")
    _info_box(
        f'<strong>Was dieser Tab macht.</strong> Er isoliert den Sovereign-'
        f'Kanal des Marktbuchs für die {N_IRB_BANKS} IRB-Banken. Wir fragen '
        'nicht, ob ein Staat ausfällt. Wir fragen: Was passiert mit dem '
        'Marktwert der gemeldeten Staatsanleihen, wenn der 10-Jahres-Zins '
        'steigt oder fällt, und welcher Teil dieses Marktwertverlustes ist '
        'nach IFRS 9 CET1-wirksam?',
    )
    _tab_overview_cards()

    total_exposure = float(conc["exposure_eur"].sum())
    gross_live = float(rate_shock_pnl(mat, delta_r_pp=delta_r_pp)["delta_pnl_eur"].sum())
    acct_live = sovereign_cet1_impact(acct_split, delta_r_pp=delta_r_pp)
    cet1_live = float(acct_live["cet1_impact_eur"].sum()) if not acct_live.empty else 0.0
    cet1_relevant_eur = float(
        acct_split[acct_split["channel"].isin(["P&L", "OCI"])]["exposure_eur"].sum()
    )
    cet1_relevant_share = cet1_relevant_eur / total_exposure if total_exposure else 0.0
    dur_system = _weighted_duration(mat)

    dom = domestic_share_per_bank(conc, bank_dir, cty_dim)
    weighted_dom_share = (
        float(dom["domestic_eur"].sum() / dom["total_eur"].sum())
        if float(dom["total_eur"].sum()) > 0 else 0.0
    )

    eyebrow("System Snapshot · aktive Modellwirkung")
    c1, c2, c3, c4 = st.columns(4, gap="small")
    c1.metric("Sovereign-Buch", f"€{total_exposure / 1e12:.2f} tn", f"{N_IRB_BANKS} Banken", delta_color="off")
    c2.metric("CET1-relevanter Anteil", f"{cet1_relevant_share * 100:.0f}%", "HfT + FVTPL + FVOCI", delta_color="off")
    c3.metric("Gewichtete Duration", f"{dur_system:.1f} Jahre", "über alle Laufzeiten", delta_color="off")
    c4.metric("Live CET1-Effekt", _bn(cet1_live), f"Δr = {delta_r_pp:+.1f} pp")

    if abs(delta_r_pp) > 1e-3:
        insight(
            f"<strong>Live-Schock.</strong> Der aktuelle Zinsschock von "
            f"<strong>{delta_r_pp:+.1f} Prozentpunkten</strong> erzeugt "
            f"einen gesamten Marktwerteffekt von <strong>{_bn(gross_live)}</strong>. "
            f"CET1-wirksam werden davon <strong>{_bn(cet1_live)}</strong>, "
            f"weil nur HfT, FVTPL und FVOCI in der aktiven Bridge durchschlagen."
        )
    else:
        insight(
            "Aktuell ist kein Zinsschock aktiv. Der Tab zeigt die Struktur "
            "des Sovereign-Buchs; sobald der Sidebar-Schock verändert wird, "
            "rechnen Snapshot, Worked Example und Drilldown live mit."
        )

    st.divider()

    eyebrow("Doom-Loop-Indikator · Home-Sovereign-Konzentration")
    _info_box(
        '<strong>Wie liest man die Grafik?</strong> Jeder Punkt ist eine '
        'Bank. Die x-Achse zeigt, welcher Anteil ihres Sovereign-Buchs im '
        'eigenen Heimatland liegt. Die y-Achse zeigt, wie groß das gesamte '
        'Sovereign-Buch relativ zum CET1-Kapital ist. Die Blasengröße ist '
        'das absolute Sovereign-Volumen. Rechts oben stehen Banken mit '
        'hoher Heimatland-Konzentration und großem Sovereign-Buch im '
        'Verhältnis zum Kapitalpuffer.',
        border=COLORS["crimson"],
    )

    code_to_iso = cty_dim.set_index("code")["iso"].to_dict() if "code" in cty_dim.columns else {}
    kpis = sovereign_kpis_per_bank(conc, bank_dir)
    dl = (
        kpis.merge(dom[["lei", "domestic_share"]], on="lei", how="left")
        .merge(cap10[["LEI_Code", "cet1_eur"]], left_on="lei", right_on="LEI_Code", how="left")
    )
    dl = dl[dl["cet1_eur"] > 0].copy()
    dl["sov_over_cet1"] = dl["total_eur"] / dl["cet1_eur"]
    dl["top_iso"] = dl["top_country_code"].map(code_to_iso).fillna(dl["top_country_code"].astype(str))

    dl_l, dl_r = st.columns([3, 2], gap="medium")
    with dl_l:
        fig_dl = go.Figure(
            go.Scatter(
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
                    sizemin=9,
                    color=COLORS["crimson"],
                    opacity=0.72,
                    line=dict(width=1, color=COLORS["white"]),
                ),
                customdata=pd.concat(
                    [dl["total_eur"] / 1e9, dl["cet1_eur"] / 1e9, dl["top_iso"]],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>Home-Bias: %{x:.0f}%<br>"
                    "Sovereign/CET1: %{y:.0f}%<br>"
                    "Sovereign-Buch: %{customdata[0]:.0f} bn €<br>"
                    "CET1: %{customdata[1]:.0f} bn €<br>"
                    "Top-Land: %{customdata[2]}<extra></extra>"
                ),
            )
        )
        fig_dl.add_vline(
            x=float(dl["domestic_share"].median() * 100),
            line_dash="dot",
            line_color=COLORS["stone"],
            annotation_text="Median",
            annotation_font_size=9,
        )
        fig_dl.add_hline(
            y=float(dl["sov_over_cet1"].median() * 100),
            line_dash="dot",
            line_color=COLORS["stone"],
        )
        fig_dl.update_layout(
            title="Home-Sovereign-Konzentration relativ zum CET1-Kapital",
            xaxis_title="Home-Bias [% des Sovereign-Buchs im Heimatland]",
            yaxis_title="Sovereign-Buch [% des CET1-Kapitals]",
            height=440,
            showlegend=False,
        )
        st.plotly_chart(fig_dl, use_container_width=True)

    with dl_r:
        st.markdown("**Konzentrationslesart**")
        st.caption(
            "Sov/CET1 zeigt die Kapitalgröße des Sovereign-Buchs. Home-Bias "
            "zeigt, wie stark dieses Buch am eigenen Staat hängt. Top-1 ist "
            "das größte einzelne Schuldnerland."
        )
        dl_table = dl.sort_values("sov_over_cet1", ascending=False).copy()
        kpi_disp = pd.DataFrame(
            {
                "Bank": [_short(n, 24) for n in dl_table["bank_name"]],
                "Sov/CET1": (dl_table["sov_over_cet1"] * 100).round(0).astype(int).astype(str) + "%",
                "Home-Bias": (dl_table["domestic_share"] * 100).round(0).astype(int).astype(str) + "%",
                "Top-Land": (
                    dl_table["top_iso"].fillna("-")
                    + " "
                    + (dl_table["top_share"] * 100).round(0).astype(int).astype(str)
                    + "%"
                ),
            }
        )
        st.dataframe(kpi_disp, use_container_width=True, hide_index=True, height=360)

    st.caption(
        "Quellenkonzept: Brunnermeier et al. (2016), Acharya/Drechsler/Schnabl "
        "(2014) und ESRB (2015) beschreiben die Bank-Staaten-Rückkopplung. "
        "Unser Tab operationalisiert sie als Konzentrationsindikator, nicht "
        "als Staatsausfallmodell."
    )

    st.divider()

    eyebrow("IFRS-9-Klassifizierung · was wird CET1-wirksam?")
    _info_box(
        '<strong>Die ökonomische Kernaussage.</strong> Der Zinsschock erzeugt '
        'zunächst für alle Bonds einen Marktwertverlust. Für die CET1-Bridge '
        'zählt aber nur der Teil, der bilanziell sichtbar wird. HfT und '
        'FVTPL laufen über Gewinn und Verlust. FVOCI läuft über das sonstige '
        'Ergebnis und damit über Eigenkapital. Amortised Cost bleibt im '
        'Basismodell zu Buchwert und ist deshalb kein aktiver CET1-Kanal.',
        border=COLORS["bright_blue"],
    )
    st.markdown(
        """
<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.75rem;margin:0.2rem 0 1.1rem 0;">
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-radius:6px;padding:0.8rem;border-left:4px solid #051C2C;">
    <div style="font-size:0.68rem;font-weight:800;color:#051C2C;letter-spacing:0.08em;text-transform:uppercase;">P&L · bilanziell sichtbar</div>
    <div style="font-weight:800;color:#051C2C;margin-top:0.2rem;">HfT</div>
    <div style="font-size:0.76rem;color:#536774;line-height:1.45;">Held for Trading · Fair Value über Gewinn und Verlust · CET1-wirksam.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-radius:6px;padding:0.8rem;border-left:4px solid #034B6F;">
    <div style="font-size:0.68rem;font-weight:800;color:#034B6F;letter-spacing:0.08em;text-transform:uppercase;">P&L · bilanziell sichtbar</div>
    <div style="font-weight:800;color:#051C2C;margin-top:0.2rem;">FVTPL</div>
    <div style="font-size:0.76rem;color:#536774;line-height:1.45;">Fair Value Through Profit or Loss · direkter P&L-Effekt · CET1-wirksam.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-radius:6px;padding:0.8rem;border-left:4px solid #2251FF;">
    <div style="font-size:0.68rem;font-weight:800;color:#2251FF;letter-spacing:0.08em;text-transform:uppercase;">Eigenkapital · OCI</div>
    <div style="font-weight:800;color:#051C2C;margin-top:0.2rem;">FVOCI</div>
    <div style="font-size:0.76rem;color:#536774;line-height:1.45;">Fair Value Through Other Comprehensive Income · OCI-Reserve · CET1-wirksam.</div>
  </div>
  <div style="background:#FFFFFF;border:1px solid #E1E6EA;border-radius:6px;padding:0.8rem;border-left:4px solid #6E6E6E;">
    <div style="font-size:0.68rem;font-weight:800;color:#6E6E6E;letter-spacing:0.08em;text-transform:uppercase;">Buchwert · nicht aktiv</div>
    <div style="font-weight:800;color:#051C2C;margin-top:0.2rem;">AC</div>
    <div style="font-size:0.76rem;color:#536774;line-height:1.45;">Amortised Cost · im Basismodell kein direkter CET1-Effekt ohne Verkauf oder Impairment.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    sys_by_class = (
        acct_split.groupby(["accounting_class", "channel"], as_index=False)["exposure_eur"]
        .sum()
        .sort_values("exposure_eur", ascending=False)
    )
    class_order = ["AC", "FVOCI", "FVTPL", "HfT"]
    class_colors = {
        "AC": COLORS["stone"],
        "FVOCI": COLORS["bright_blue"],
        "FVTPL": COLORS["mid_blue"],
        "HfT": COLORS["navy"],
    }
    class_labels = {
        "HfT": "P&L · HfT",
        "FVTPL": "P&L · FVTPL",
        "FVOCI": "OCI/Eigenkapital · FVOCI",
        "AC": "nicht CET1-aktiv · AC",
    }
    class_items = {
        "HfT": "2520812",
        "FVTPL": "2520813",
        "FVOCI": "2520814",
        "AC": "2520815",
    }
    channel_labels = {
        "P&L": "P&L -> CET1",
        "OCI": "OCI/Eigenkapital -> CET1",
        "none": "nicht aktiv",
    }

    ifrs_l, ifrs_r = st.columns([2, 3], gap="medium")
    with ifrs_l:
        disp = sys_by_class.copy()
        disp["Exposure bn"] = (disp["exposure_eur"] / 1e9).round(0).astype(int)
        disp["EBA-Item"] = disp["accounting_class"].map(class_items)
        disp["Durchleitung"] = disp["channel"].map(channel_labels)
        disp = disp.rename(columns={"accounting_class": "Klasse"})
        st.dataframe(
            disp[["Klasse", "EBA-Item", "Durchleitung", "Exposure bn"]],
            use_container_width=True,
            hide_index=True,
            height=205,
        )
        st.markdown(
            '<div style="background:#F4F4F4;border-left:3px solid #2251FF;'
            'padding:0.65rem 0.8rem;margin:0.35rem 0 0.75rem 0;'
            'font-size:0.78rem;line-height:1.45;color:#051C2C;">'
            '<strong>So steht es in der Datengrundlage:</strong> '
            '<code>tr_sov.csv</code> enthält pro Bank, Schuldnerland und '
            'Laufzeit-Bucket separate EBA-Items für HfT, FVTPL, FVOCI und '
            'AC. Die Spalte <code>Amount</code> ist der gemeldete Betrag '
            'in Mio. EUR; wir aggregieren diese Zeilen auf Bank × Klasse.'
            '</div>',
            unsafe_allow_html=True,
        )
        if abs(delta_r_pp) > 1e-3:
            impact_by_channel = (
                acct_live.groupby("channel", as_index=False)["cet1_impact_eur"].sum()
            )
            impact_by_channel["CET1 Δ bn"] = (impact_by_channel["cet1_impact_eur"] / 1e9).round(2)
            impact_by_channel["Kanal"] = impact_by_channel["channel"].map(channel_labels)
            st.dataframe(
                impact_by_channel[["Kanal", "CET1 Δ bn"]],
                use_container_width=True,
                hide_index=True,
                height=150,
            )

    with ifrs_r:
        st.markdown(
            '<div style="font-size:0.78rem;line-height:1.45;color:#536774;'
            'margin:0.1rem 0 0.55rem 0;">'
            '<span style="color:#051C2C;font-weight:760;">P&amp;L</span> '
            'und <span style="color:#2251FF;font-weight:760;">OCI/Eigenkapital</span> '
            'gehen in die CET1-Bridge; '
            '<span style="color:#6E6E6E;font-weight:760;">AC</span> bleibt '
            'im Basismodell nicht aktiv.'
            '</div>',
            unsafe_allow_html=True,
        )
        sub = acct_split.merge(bank_dir[["lei", "bank_name"]], left_on="LEI_Code", right_on="lei")
        pivoted = sub.pivot_table(
            index="bank_name",
            columns="accounting_class",
            values="exposure_eur",
            aggfunc="sum",
            fill_value=0,
        )
        pivoted = pivoted.assign(_total=pivoted.sum(axis=1)).sort_values("_total").drop(columns="_total")
        fig_ifrs = go.Figure()
        for cls in class_order:
            if cls in pivoted.columns:
                fig_ifrs.add_trace(
                    go.Bar(
                        y=pivoted.index,
                        x=pivoted[cls] / 1e9,
                        name=class_labels.get(cls, cls),
                        orientation="h",
                        marker_color=class_colors.get(cls, COLORS["stone"]),
                        marker_line_width=0.4,
                        marker_line_color=COLORS["white"],
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            + class_labels.get(cls, cls)
                            + "<br>Exposure: %{x:.1f} bn EUR<extra></extra>"
                        ),
                    )
                )
        st.markdown("**Bilanzielle Durchleitung des Sovereign-Buchs nach IFRS 9**")
        fig_ifrs.update_layout(
            title="",
            barmode="stack",
            height=455,
            xaxis_title="Exposure [Mrd. EUR]",
            margin=dict(t=20, b=86),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="left",
                x=0,
                font=dict(size=10),
            ),
        )
        st.plotly_chart(fig_ifrs, use_container_width=True)

    st.caption(
        "Datenquelle: EBA Transparency 2025, `tr_sov.csv`, Items 2520812 "
        "(HfT), 2520813 (FVTPL), 2520814 (FVOCI), 2520815 (AC). Der Split "
        "ist bankindividuell gemeldet und keine Modellannahme."
    )

    st.divider()

    eyebrow("Live Worked Example · aktueller Zinsschock")
    _info_box(
        f'<strong>Rechnung mit dem aktuell eingestellten Schock.</strong> '
        f'Für jede Bank wird zuerst der gesamte Marktwerteffekt berechnet: '
        f'<code>ΔFV = - Duration × Δr × Exposure</code>. Danach zeigt der '
        f'IFRS-9-Split, welcher Anteil in der CET1-Bridge ankommt. Der '
        f'aktuelle Sidebar-Wert ist <strong>Δr = {delta_r_pp:+.1f} Prozentpunkte</strong>.',
        border=COLORS["navy"],
        tone="#FAFAFA",
    )

    pnl_live = rate_shock_pnl(mat, delta_r_pp=delta_r_pp).set_index("LEI_Code")
    bank_duration = (
        mat.assign(dxe=mat["duration_years"] * mat["exposure_eur"])
        .groupby("LEI_Code")
        .agg(exposure_eur=("exposure_eur", "sum"), dxe=("dxe", "sum"))
    )
    bank_duration["duration_w"] = bank_duration["dxe"] / bank_duration["exposure_eur"]

    live_bank = (
        acct_live.groupby("LEI_Code")
        .agg(cet1_impact_eur=("cet1_impact_eur", "sum"), delta_fv_eur=("delta_fv_eur", "sum"))
        .join(pnl_live[["delta_pnl_eur"]], how="left")
        .join(bank_duration[["exposure_eur", "duration_w"]], how="left")
    )
    live_bank["non_cet1_eur"] = live_bank["delta_fv_eur"] - live_bank["cet1_impact_eur"]
    live_bank["name"] = [_short(_LEI2NAME.get(lei, lei[:8]), 22) for lei in live_bank.index]
    if abs(delta_r_pp) > 1e-3:
        live_bank = live_bank.reindex(live_bank["cet1_impact_eur"].abs().sort_values(ascending=False).index)
    else:
        live_bank = live_bank.sort_values("exposure_eur", ascending=False)
    live_top = live_bank.head(6).copy()

    ex_l, ex_r = st.columns([3, 2], gap="medium")
    with ex_l:
        fig_ex = go.Figure()
        if abs(delta_r_pp) <= 1e-3:
            fig_ex.add_trace(
                go.Bar(
                    y=live_top["name"],
                    x=live_top["exposure_eur"] / 1e9,
                    orientation="h",
                    name="Sovereign-Buch",
                    marker_color=COLORS["mid_blue"],
                    marker_line_width=0,
                )
            )
            chart_title = "Kein aktiver Zinsschock: größte Beispielbanken nach Sovereign-Buch"
            x_title = "Exposure [Mrd. EUR]"
            barmode = "group"
        else:
            fig_ex.add_trace(
                go.Bar(
                    y=live_top["name"],
                    x=live_top["cet1_impact_eur"] / 1e9,
                    orientation="h",
                    name="CET1-wirksam",
                    marker_color=COLORS["crimson"],
                    marker_line_width=0,
                )
            )
            fig_ex.add_trace(
                go.Bar(
                    y=live_top["name"],
                    x=live_top["non_cet1_eur"] / 1e9,
                    orientation="h",
                    name="nicht CET1-wirksam (AC)",
                    marker_color="#AEBCC5",
                    marker_line_width=0,
                )
            )
            chart_title = "Live-Marktwerteffekt: CET1-wirksam vs. nicht aktiv in der Bridge"
            x_title = "ΔFair Value [Mrd. EUR]"
            barmode = "relative"
        fig_ex.update_layout(
            title=chart_title,
            barmode=barmode,
            height=390,
            xaxis_title=x_title,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_ex, use_container_width=True)

    with ex_r:
        ex_table = pd.DataFrame(
            {
                "Bank": live_top["name"],
                "Sov bn": (live_top["exposure_eur"] / 1e9).round(0).astype(int),
                "Duration": live_top["duration_w"].round(1),
                "ΔFV bn": (live_top["delta_fv_eur"] / 1e9).round(2),
                "CET1 Δ bn": (live_top["cet1_impact_eur"] / 1e9).round(2),
            }
        )
        st.dataframe(ex_table, use_container_width=True, hide_index=True, height=250)
        st.caption(
            "Die Tabelle ist bewusst knapp: Größe des Buchs, Zinshebel "
            "(Duration), gesamter Marktwerteffekt und der tatsächlich "
            "CET1-wirksame Anteil."
        )

    if abs(delta_r_pp) <= 1e-3:
        st.info("Der aktuell eingestellte Zinsschock ist 0.0 Prozentpunkte; deshalb sind die Live-Verluste null.")

    st.divider()

    eyebrow("Bank-Drilldown · wo sitzt das Zinsrisiko?")
    _info_box(
        '<strong>Nutzen des Drilldowns.</strong> Für eine einzelne Bank sieht '
        'man die beiden Treiber, die im Aggregat oft verdeckt werden: '
        'erstens die Laufzeitenstruktur des Sovereign-Buchs, zweitens die '
        'IFRS-9-Klassen, über die der Live-Schock in CET1 sichtbar wird.',
        tone="#F4F4F4",
    )

    sov_per_bank = conc.groupby("LEI_Code")["exposure_eur"].sum().sort_values(ascending=False)
    lei_to_name = bank_dir.set_index("lei")["bank_name"].to_dict()
    bank_options = sov_per_bank.index.tolist()
    sel_lei = st.selectbox(
        f"Bank wählen ({len(bank_options)} IRB-Banken · tippen zum Suchen)",
        bank_options,
        format_func=lambda lei: (
            f"{lei_to_name.get(lei, lei[:8])}  "
            f"({_metric_bn(float(sov_per_bank.loc[lei]), 1)} Sovereign-Buch)"
        ),
        key="sov_drill_bank",
    )
    sel_name = lei_to_name.get(sel_lei, sel_lei[:12])

    drill_l, drill_r = st.columns([3, 2], gap="medium")
    with drill_l:
        st.markdown(f"**{sel_name} · Restlaufzeiten**")
        sub_mat = mat[mat["LEI_Code"] == sel_lei].sort_values("Maturity")
        if len(sub_mat) > 0:
            fig_mat = go.Figure(
                go.Bar(
                    x=sub_mat["label"],
                    y=sub_mat["exposure_eur"] / 1e9,
                    marker_color=COLORS["mid_blue"],
                    marker_line_width=0,
                    text=[f"{v / 1e9:.1f}" for v in sub_mat["exposure_eur"]],
                    textposition="outside",
                    textfont=dict(size=10, color=COLORS["navy"]),
                    hovertemplate="<b>%{x}</b><br>%{y:.2f} bn EUR<extra></extra>",
                )
            )
            fig_mat.update_layout(
                xaxis_title="Restlaufzeit-Bucket",
                yaxis_title="Exposure [Mrd. EUR]",
                height=360,
                bargap=0.25,
            )
            st.plotly_chart(fig_mat, use_container_width=True)
        else:
            st.info("Keine Maturity-Daten für diese Bank im Datensatz.")

    with drill_r:
        st.markdown(f"**{sel_name} · IFRS-9 und CET1-Effekt**")
        bank_imp = acct_live[acct_live["LEI_Code"] == sel_lei].copy()
        if len(bank_imp) > 0:
            bank_imp["Exposure bn"] = (bank_imp["fair_value_eur"] / 1e9).round(2)
            bank_imp["ΔFV bn"] = (bank_imp["delta_fv_eur"] / 1e9).round(2)
            bank_imp["CET1 Δ bn"] = (bank_imp["cet1_impact_eur"] / 1e9).round(2)
            bank_imp["Kanal"] = bank_imp["channel"].replace({"none": "nicht aktiv"})
            bank_imp = bank_imp.rename(columns={"accounting_class": "Klasse"})
            st.dataframe(
                bank_imp[["Klasse", "Kanal", "Exposure bn", "ΔFV bn", "CET1 Δ bn"]],
                use_container_width=True,
                hide_index=True,
                height=230,
            )
            st.metric(
                "Σ CET1-Effekt dieser Bank",
                _bn(float(bank_imp["cet1_impact_eur"].sum()), 2),
                f"unter Δr = {delta_r_pp:+.1f} pp",
            )
        else:
            st.info("Keine IFRS-9-Klassen-Daten für diese Bank.")

    st.divider()

    eyebrow("Methodischer Appendix · bewusst nicht im Hauptmodell")
    with st.expander("Warum latente AC-Verluste nicht als eigener Kanal modelliert werden", expanded=False):
        st.markdown(
            """
Latente Verluste auf Amortised-Cost-Beständen sind ökonomisch relevant,
aber sie sind **kein eigener CET1-Kanal** in diesem Modell. Sie würden erst
modellwirksam, wenn wir zusätzlich einen Verkauf, eine Liquiditätskrise,
einen Run oder ein Impairment-Szenario annehmen.

Deshalb zeigen die Hauptgrafiken AC nur als Abgrenzung: Der Marktwerteffekt
existiert rechnerisch, wird aber nicht in die CET1-Bridge übernommen. Diese
Trennung hält das Modell sauber: Marktbuch-Zinsschock ja, zusätzlicher
Liquiditäts-/Forced-Sale-Stress nein.
"""
        )

    render_sovereign_methodology()


footer(
    "Zwei Sub-Tabs · Yield-Curve Input · Sovereign-Kanal mit Doom-Loop-"
    "Indikator, IFRS-9-Wirkungslogik, Live Worked Example und Bank-"
    "Drilldown · Daten: EBA Transparency 2025 `tr_sov.csv` + CET1 aus "
    "`tr_oth.csv`, Stichtag Juni 2025 · 10 IRB-Banken"
)
