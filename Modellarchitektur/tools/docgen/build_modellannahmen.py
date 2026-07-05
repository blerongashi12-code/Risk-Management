# -*- coding: utf-8 -*-
"""Modellannahmen.docx — kuratierte, verständliche Abgabefassung.

Nur das AKTUELLE Modell (2-Faktor + CET1-Walk-Forward), granular aber
übersichtlich, mit vier didaktischen Schaubildern. Quelle der Fachinhalte:
docs/MODEL_ASSUMPTIONS.md (V2.1) + live SENSITIVITY_MATRIX.
"""
import sys, os, datetime
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
from two_factor_stress import SENSITIVITY_MATRIX, stress_pd, stress_lgd
from vasicek import irb_capital_requirement

FIGS = os.path.dirname(os.path.abspath(__file__)) + "/figs"
OUT = str(pathlib.Path(__file__).resolve().parents[3] / "Abgabe-Files" / "Abgabedokumente" / "Modellannahmen.docx")

NAVY = RGBColor(0x05, 0x1C, 0x2C)
MID = RGBColor(0x03, 0x4B, 0x6F)
GREY = RGBColor(0x6E, 0x6E, 0x6E)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Arial"; st.font.size = Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
st.paragraph_format.space_after = Pt(7)

for hid, size, col in [("Heading 1", 16, NAVY), ("Heading 2", 12.5, MID),
                        ("Heading 3", 11, NAVY)]:
    h = doc.styles[hid]
    h.font.name = "Arial"; h.font.size = Pt(size); h.font.bold = True
    h.font.color.rgb = col
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")

sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, m, Cm(2.2))

foot = sec.footer.paragraphs[0]
foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = foot.add_run("Modellannahmen · EU Banking Credit Stress Cockpit · Seite ")
r.font.size = Pt(8); r.font.color.rgb = GREY; r.font.name = "Arial"
fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
foot._p.append(fld)


# ---------------- Helpers ----------------
def shade(el, hexcol):
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear")
    sh.set(qn("w:fill"), hexcol); el.append(sh)


def body(text, bold_parts=(), size=10.5, italic=False, color=None,
         space_after=7):
    """Absatz; **…** wird fett gesetzt."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    for i, tok in enumerate(text.split("**")):
        if not tok:
            continue
        r = p.add_run(tok)
        r.bold = (i % 2 == 1)
        r.italic = italic
        r.font.size = Pt(size)
        if color:
            r.font.color.rgb = color
    return p


def bullet(text):
    p = doc.add_paragraph(style="List Bullet")
    for i, tok in enumerate(text.split("**")):
        if not tok:
            continue
        r = p.add_run(tok); r.bold = (i % 2 == 1); r.font.size = Pt(10.5)
    return p


def h1(t): doc.add_paragraph(t, style="Heading 1")
def h2(t): doc.add_paragraph(t, style="Heading 2")


def figure(fname, width_cm, caption):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(f"{FIGS}/{fname}", width=Cm(width_cm))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(12)
    r = c.add_run(caption); r.font.size = Pt(8.5); r.font.color.rgb = GREY
    r.italic = True


def table(headers, rows, col_pct, fontsize=9):
    tab = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tab.style = "Table Grid"; tab.alignment = WD_TABLE_ALIGNMENT.CENTER
    total = Cm(16.6)
    for c, htxt in enumerate(headers):
        cell = tab.rows[0].cells[c]
        cell.width = Cm(16.6 * col_pct[c] / 100)
        cell.paragraphs[0].text = ""
        r = cell.paragraphs[0].add_run(htxt)
        r.bold = True; r.font.size = Pt(fontsize)
        shade(cell._tc.get_or_add_tcPr(), "D9E4EC")
    for ri, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tab.rows[1 + ri].cells[c]
            cell.width = Cm(16.6 * col_pct[c] / 100)
            cell.paragraphs[0].text = ""
            for i, tok in enumerate(str(val).split("**")):
                if not tok:
                    continue
                r = cell.paragraphs[0].add_run(tok)
                r.bold = (i % 2 == 1); r.font.size = Pt(fontsize)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return tab


# ---------------- Titelseite ----------------
t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
t.paragraph_format.space_before = Pt(150)
r = t.add_run("Modellannahmen"); r.font.size = Pt(32); r.bold = True
r.font.color.rgb = NAVY
s2 = doc.add_paragraph(); s2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s2.add_run("EU Banking Credit Stress Cockpit\n"
               "2-Faktor-Kreditstress (Basel-III-IRB) · Zielgröße CET1-Quote · "
               "Pillar-3-Datenbasis")
r.font.size = Pt(13); r.font.color.rgb = MID
meta = doc.add_paragraph(); meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
meta.paragraph_format.space_before = Pt(30)
r = meta.add_run("Kuratierte Abgabefassung · Stand "
                 + datetime.date.today().strftime("%d.%m.%Y")
                 + "\nTechnische Referenz: Modellarchitektur/docs/MODEL_ASSUMPTIONS.md (V2.1)")
r.font.size = Pt(9.5); r.font.color.rgb = GREY
doc.add_page_break()

# ---------------- Inhaltsverzeichnis ----------------
p = doc.add_paragraph(); r = p.add_run("Inhaltsverzeichnis")
r.font.size = Pt(16); r.bold = True; r.font.color.rgb = NAVY
tocp = doc.add_paragraph()
fld = OxmlElement("w:fldSimple")
fld.set(qn("w:instr"), 'TOC \\o "1-2" \\h \\z \\u')
run = OxmlElement("w:r"); tel = OxmlElement("w:t")
tel.text = "Inhaltsverzeichnis: Rechtsklick → Felder aktualisieren (F9)."
run.append(tel); fld.append(run); tocp._p.append(fld)
doc.add_page_break()

# ================= 1 · Das Modell auf einen Blick =================
h1("1 · Das Modell auf einen Blick")
body("Das Cockpit beantwortet eine aufsichtlich zentrale Frage: "
     "**Wie stünde die harte Kernkapitalquote (CET1) der zehn größten "
     "EU-IRB-Banken, wenn ein Ölpreis- und/oder Zinsschock einträte?** "
     "Die CET1-Quote ist die Solvenz-Kennzahl der Bankenaufsicht — fällt sie "
     "unter die gestaffelten Mindestanforderungen, greifen aufsichtliche "
     "Maßnahmen bis hin zu Ausschüttungssperren.")
body("Vier Prinzipien tragen das Modell:")
bullet("**Quellenreine Datenbasis** — alle Risikoparameter (PD, LGD, EAD) "
       "stammen aus den offiziellen Pillar-3-Offenlegungen (EU CR6) der "
       "Banken selbst; kein Wert wird geschätzt oder abgeleitet.")
bullet("**Zwei getrennte, messbare Faktoren** — Ölpreis (ΔBrent) und "
       "10-Jahres-Zins (Δr_10y) werden nicht zu einem Kunstfaktor "
       "aggregiert, sondern wirken einzeln und beobachtbar.")
bullet("**Sektor-differenzierte Transmission** — jede Kreditklasse reagiert "
       "gemäß quellenbelegter Sensitivitäten unterschiedlich (eine "
       "Hypothek anders als ein Kreditkartenbuch).")
bullet("**Konservative Auslegung** — im Zweifel überschätzt das Modell den "
       "Kapitalbedarf; entlastende Effekte (z. B. Zinsüberschuss) werden "
       "bewusst nicht gegengerechnet.")
figure("fig1_pipeline.png", 16.2,
       "Abb. 1 · Modell-Pipeline: Zwei Makro-Schocks laufen über die "
       "β-Transmission und die Basel-III-Kapitalformel in das Kreditbuch; "
       "der Zinsschock wirkt zusätzlich über den Sovereign-Kanal. Zielgröße "
       "ist stets die CET1-Quote unter Stress.")

# ================= 2 · Die zwei Risikofaktoren =================
h1("2 · Die zwei Risikofaktoren")
body("Beide Faktoren sind täglich beobachtbare Marktgrößen — das Modell "
     "braucht keine latenten Zustandsvariablen:")
table(
    ["Faktor", "Einheit", "Ökonomische Wirkung", "Lesebeispiel"],
    [["**ΔBrent** (Ölpreis)", "log-Return",
      "Energie-/Kostenschock: verteuert Vorleistungen, drückt "
      "Schuldner-Cashflows und Haushaltsbudgets",
      "+0,50 ≈ +65 % Ölpreis"],
     ["**Δr_10y** (10-J-Zins)", "Prozentpunkte",
      "Refinanzierungs-/Bewertungsschock: verteuert Anschluss-"
      "finanzierung, senkt Sicherheiten- und Anleihewerte",
      "+2,0 = +200 Basispunkte"]],
    [16, 12, 50, 22])
body("**Warum zwei getrennte Faktoren?** Über fünf Jahre (1 242 Handelstage) "
     "beträgt die Korrelation der beiden Faktoren nur ρ = +0,07 "
     "(95 %-Konfidenzintervall [+0,01; +0,12]; OLS-R² = 0,005). Sie tragen "
     "also nahezu unabhängige Information — eine Aggregation zu einem "
     "Einheitsfaktor würde genau die Unterscheidung wegwerfen, die z. B. "
     "eine zinsgetriebene Hypothekenkrise von einem Energiekostenschock "
     "trennt (Annahme A-10).")
h2("Datenquellen und ihre Rollen")
table(
    ["Quelle", "Inhalt", "Rolle im Modell"],
    [["Pillar-3-Reports (EU CR6)", "PD, LGD, EAD je Bank × Kreditklasse",
      "**Input** — Risikoparameter (Live-Snapshot 31.12.2024 + "
      "Backtest-Reihe FY2021–2024)"],
     ["EBA Transparency Exercise", "CET1, RWA je Bank und Quartal",
      "**Vergleichsseite** — realisierte Werte für die Validierung; nie "
      "PD/LGD-Quelle"],
     ["Brent (ICE), täglich", "Ölpreis-Historie", "Faktor 1"],
     ["Bundesbank-Svensson-Parameter", "Zinsstrukturkurve, täglich "
      "(Nelson/Siegel 1987, erweitert um Svensson 1994)",
      "Faktor 2 (10-J-Zins)"],
     ["EBA IFRS-9-Split", "HfT/FVTPL/FVOCI-Anteile der Staatsanleihen",
      "Sovereign-Kanal (CET1-wirksamer Anteil)"]],
    [24, 33, 43])

# ================= 3 · Stress-Transmission =================
h1("3 · Sektor-differenzierte Stress-Transmission")
body("Kern des Modells ist die Übersetzung der beiden Schocks in "
     "Risikoparameter — **mit eigenen Sensitivitäten je Kreditklasse** "
     "(Annahme A-04):")
body("ΔPD [pp] = β_Öl · ΔBrent + β_Zins · Δr_10y      "
     "ΔLGD [pp] = γ_Öl · ΔBrent + γ_Zins · Δr_10y",
     size=10.5, color=MID)
body("Die gestressten Werte werden begrenzt — jede Grenze hat einen "
     "Grund: Die **PD-Untergrenze von 0,03 %** ist der regulatorische "
     "Basel-Floor (CRR Art. 160/163 — keine Bank darf einem lebenden "
     "Portfolio eine geringere Ausfallwahrscheinlichkeit zuweisen). Die "
     "**PD-Obergrenze von 50 %** dient der numerischen Stabilität: jenseits "
     "davon wäre ein Portfolio faktisch im Ausfallstatus und die "
     "IRB-Formel nicht mehr die richtige Linse. Die **LGD-Untergrenze von "
     "5 %** verhindert unplausibel verlustfreie Ausfälle selbst bei bester "
     "Besicherung; die **Obergrenze 100 %** ist definitorisch (mehr als "
     "das Exposure kann nicht verloren gehen).")
figure("fig2_beta.png", 14.5,
       "Abb. 2 · Die Sensitivitäts-Matrix als Bild: Jede Zelle ist eine "
       "dokumentierte Annahme — rot = Risiko steigt mit dem Schock, blau = "
       "sinkt, weiß = kein Effekt. Auffällig: Hypotheken sind zinsgetrieben, "
       "Banken profitieren leicht von Zinsanstiegen (Zinsüberschuss), "
       "Staaten reagieren nicht über das Kreditbuch (separater "
       "Sovereign-Kanal).")
h2("Ökonomische Logik je Kreditklasse")
_logik = {
    "corporate": "Energie verteuert Produktion, Zinsen die Refinanzierung; "
                 "LGD steigt mit fallenden Sicherheitenwerten.",
    "sme_corporate": "Wie Corporate, aber ≈2× stärker — KMU haben dünnere "
                     "Puffer und weniger Preissetzungsmacht (ECB WP 2897).",
    "mortgage": "Zins dominiert: Raten steigen (Floating/Anschluss), "
                "Hauspreise fallen → höchste LGD-Zins-Sensitivität.",
    "qrre": "Kreditkarten/Revolving: hängt am Haushaltsbudget → "
            "energie-/inflationssensitiv, kaum zins-sensitiv.",
    "other_retail": "Konsum-/Autokredite: Mischprofil zwischen Mortgage "
                    "und QRRE.",
    "bank": "Leicht **negatives** Zins-β: steigende Zinsen stützen den "
            "Zinsüberschuss der Gegenpartei-Banken (NIM-Effekt).",
    "sovereign": "Im Kreditbuch bewusst inert — der Zins wirkt auf Staaten "
                 "über den Marktwert-Kanal (Abschnitt 4), nicht über die PD.",
}
rows = []
for c in ["corporate", "sme_corporate", "mortgage", "qrre",
          "other_retail", "bank", "sovereign"]:
    m = SENSITIVITY_MATRIX[c]
    fmt = lambda v: f"{v:+.2f}".replace(".", ",")
    rows.append([c, fmt(m['pd_rate']), fmt(m['pd_oil']),
                 fmt(m['lgd_rate']), fmt(m['lgd_oil']), _logik[c]])
table(["Klasse", "β PD·Zins", "β PD·Öl", "γ LGD·Zins", "γ LGD·Öl",
       "Ökonomische Logik (Kurzform)"],
      rows, [14, 9, 9, 10, 9, 49], fontsize=8.5)
body("Quellen der Kalibrierung: EBA (2024) *2025 EU-wide Stress Test — "
     "Methodological Note* §2.4.2 (sektorale Sensitivitäten sind der "
     "aufsichtlich vorgesehene Ansatz); ECB WP 2897 (Unternehmens-PD auf "
     "Angebots-/Zinsschocks, KMU-Verstärkung), ECB WP 3112 "
     "(Hypotheken-Defaults und Zins), ECB Financial Stability Review Mai "
     "2024 (Haushalte). Die β sind als belegte, per Regler übersteuerbare "
     "Defaults ausgewiesen — Konfidenzklasse [estimate].", size=9.5,
     italic=True)

h2("Durchgerechnetes Beispiel: ein Schock läuft durch das Modell")
# Live aus der Engine gerechnet — keine handgepflegten Zahlen.
_pd0, _lgd0 = 0.025, 0.35
_dr, _db = 2.0, 0.30
_pd1 = stress_pd(_pd0, _db, _dr, "corporate")
_lgd1 = stress_lgd(_lgd0, _db, _dr, "corporate")
_r0 = irb_capital_requirement(_pd0, _lgd0, "corporate", maturity_years=2.5)
_r1 = irb_capital_requirement(_pd1, _lgd1, "corporate", maturity_years=2.5)
_g = lambda v, d=2: f"{v:.{d}f}".replace(".", ",")
body("Angenommen wird ein Corporate-Segment mit PD = 2,50 % und "
     "LGD = 35,00 %; als Szenario tritt ein Zinsanstieg von +2,0 "
     "Prozentpunkten bei gleichzeitig +30 % Ölpreis ein. Dann rechnet das "
     "Modell in drei Schritten:")
bullet("**Schritt 1 — PD-Transmission:** ΔPD = 0,20 · 2,0 + 0,30 · 0,30 = "
       f"**+{_g((_pd1-_pd0)*100)} pp** → gestresste PD "
       f"**{_g(_pd1*100)} %**.")
bullet("**Schritt 2 — LGD-Transmission:** ΔLGD = 1,00 · 2,0 + 0,50 · 0,30 = "
       f"**+{_g((_lgd1-_lgd0)*100)} pp** → gestresste LGD "
       f"**{_g(_lgd1*100)} %**.")
bullet("**Schritt 3 — Kapitalwirkung (IRB-Formel):** der erwartete Verlust "
       f"steigt von {_g(_pd0*_lgd0*100, 3)} % auf {_g(_pd1*_lgd1*100, 3)} % "
       f"des Exposures (**+{_g((_pd1*_lgd1/(_pd0*_lgd0)-1)*100, 0)} %**), "
       "die RWA-Dichte von "
       f"{_g(float(_r0['rwa_density'])*100, 1)} % auf "
       f"{_g(float(_r1['rwa_density'])*100, 1)} % "
       f"(**+{_g((float(_r1['rwa_density'])/float(_r0['rwa_density'])-1)*100, 1)} %**).")
body("Auf Portfolioebene wiederholt sich diese Rechnung für jede Bank und "
     "jede Klasse mit den echten Pillar-3-Parametern; die Summe der "
     "ΔRWA- und ΔEL-Beiträge speist die CET1-Bridge (Abschnitt 4). Die "
     "Beispielwerte sind live aus der Modell-Engine gerechnet, nicht "
     "handgepflegt.", size=9.5, italic=True)

# ================= 4 · Kapitalrechnung =================
h1("4 · Kapitalrechnung und CET1-Bridge")
body("Die gestressten PD/LGD laufen durch die **regulatorische "
     "Basel-III-IRB-Kapitalformel** (Vasicek/ASRF; BCBS 2017, §272–284) — "
     "bewusst der aufsichtliche Standard, keine Eigenkonstruktion:")
body("K = [ LGD · N( (N⁻¹(PD) + √ρ · N⁻¹(0,999)) / √(1−ρ) ) − PD · LGD ] · MA"
     "      →      RWA = K · 12,5 · EAD", size=10, color=MID)
body("Dabei ist ρ die Basel-Asset-Korrelation je Klasse (Corporates "
     "PD-abhängig 12–24 %, Hypotheken 15 %, QRRE 4 %), MA das "
     "Laufzeit-Adjustment und N die Standardnormalverteilung. Die "
     "theoretische Fundierung des ASRF-Rahmens liefern Vasicek (1991, "
     "2002) und Gordy (2003). Der **Sovereign-Kanal** bewertet parallel "
     "die Staatsanleihebestände: ΔMtM = −Duration · Δr · Exposure, "
     "CET1-wirksam nur für den FVOCI/FVTPL-Anteil je Bank "
     "(EBA-IFRS-9-Split, Annahme A-06; Durationslogik nach Tuckman/Serrat "
     "2012). Die begleitenden Marktbuch-Analysen des Cockpits — "
     "Staaten-Banken-Verflechtung (Doom Loop) und latente Bewertungs-"
     "verluste — stützen sich auf Brunnermeier et al. (2016), "
     "Acharya/Drechsler/Schnabl (2014) und Jiang et al. (2023).")
figure("fig3_bridge.png", 15.8,
       "Abb. 3 · Die CET1-Bridge: Das Kreditbuch erhöht den Nenner (ΔRWA) "
       "und senkt über erwartete Verluste den Zähler (ΔEL); der "
       "Sovereign-Kanal senkt den Zähler zusätzlich (ΔMtM). Entlastende "
       "Gewinn-Effekte werden bewusst nicht angesetzt — die Quote unter "
       "Stress ist eine konservative Untergrenze.")

# ================= 5 · Datenbasis =================
h1("5 · Datenbasis und Integritätssicherung")
h2("Live-Snapshot (Cockpit)")
body("**70 Parameter-Zeilen** = 10 Banken × 7 IRB-Klassen zum einheitlichen "
     "Stichtag **31.12.2024**, jede Zeile direkt aus dem EU-CR6-Sub-total "
     "des jeweiligen Pillar-3-Reports mit Seiten- und URL-Beleg "
     "(Konfidenzklasse [published]). Santander-Sovereign läuft "
     "regulatorisch im Standardansatz und ist daher aus dem IRB-Kreditbuch "
     "ausgenommen.")
h2("Historische Backtest-Reihe")
body("Für die Validierung existiert eine eigene Roh-Reihe: **10 Banken × "
     "FY2021–FY2024, 225 Datenpunkte** (PD, LGD und EAD je Klasse aus "
     "derselben Report-Vintage). Das entspricht **88 % Abdeckung relativ "
     "zum bank-eigenen Meldeumfang** — Klassen, die eine Bank gar nicht im "
     "A-IRB führt, zählen nicht als Lücke. Die verbleibenden Zellen sind "
     "dokumentierte Quellgrenzen (z. B. auf ganze Prozent gerundete PD im "
     "BNP-Report 2021) und werden **nicht** mit geschätzten Werten gefüllt.")
h2("Vier Integritäts-Sicherungen je Datenpunkt")
bullet("**Kalibrier-Anker:** Jeder Extraktions-Parser muss zunächst alle "
       "hand-verifizierten 2024-Werte exakt reproduzieren, bevor Vorjahre "
       "gelesen werden.")
bullet("**Dichte-Cross-Check:** RWA ÷ EAD jeder Zeile muss die im Report "
       "ausgewiesene RWA-Dichte treffen.")
bullet("**EAD-Kontinuität:** Sprünge der Exposure-Reihe über die Jahre "
       "werden gegen den Report geprüft (echte Effekte wie IRB→SA-Umstellung "
       "werden dokumentiert, Ausreißer verworfen).")
bullet("**Adversariale Quellprüfung:** Ein unabhängiger Prüfdurchlauf je "
       "Bank gegen die Quell-PDFs (protokolliert in "
       "tools/pillar3_backfill/VERIFICATION_REPORT.json).")

# ================= 6 · Annahmen-Inventar =================
h1("6 · Annahmen-Inventar (A-01 bis A-10)")
body("Jede Annahme trägt eine Konfidenzklasse: **[published]** = "
     "veröffentlichte Messung · **[estimate]** = belegte Schätzung · "
     "**[approximation]** = strukturelle Vereinfachung · **[assumption]** = "
     "gesetzte Konvention. Die Tabelle gibt den Überblick; darunter wird jede Annahme einzeln begründet — inklusive der verworfenen Alternative und der Wirkung einer Fehlspezifikation.")
table(
    ["ID", "Annahme", "Festlegung im Modell", "Klasse", "Quelle/Beleg"],
    [["A-01", "PD-Quelle", "Pillar-3 EU-CR6-Sub-totals je Bank×Klasse, "
      "Stichtag 31.12.2024", "published", "EBA-ITS/2020/04; CRR Art. 431–455"],
     ["A-02", "LGD-Quelle", "identisch A-01 (EU-CR6-LGD)", "published",
      "wie A-01"],
     ["A-02c", "Backtest-Zeitreihe", "eigene Pillar-3-Roh-Reihe, 10 Banken × "
      "FY2021–2024 (225 Punkte, 88 %), strikt no-look-ahead", "published",
      "Bank-Pillar-3-Reports FY2021–2024"],
     ["A-03", "EAD", "EBA-Transparency Item 2520522 (post-CCF)", "published",
      "EBA Transparency 2025"],
     ["A-04", "Stress-Transmission", "2-Faktor-β/γ je Klasse (Abschnitt 3); "
      "PD-Floor 3 bp, Cap 50 %; LGD-Floor 5 %", "estimate",
      "EBA 2025 §2.4.2; ECB WP 2897/3112; FSR 5/2024"],
     ["A-05", "Kapitalformel", "Basel-III-IRB (Vasicek/ASRF), α = 99,9 %, "
      "Basel-ρ je Klasse, Laufzeit-Adjustment", "published",
      "BCBS 2017 §272–284; CRR Art. 153/154"],
     ["A-06", "Sovereign-Kanal", "ΔMtM = −Duration·Δr·Exposure; CET1-wirksam "
      "nur FVOCI/FVTPL-Anteil", "approximation",
      "Tuckman/Serrat 2012; EBA-IFRS-9-Split"],
     ["A-07", "Trading Book", "kein eigener Kanal (entfernt — kleine "
      "Handelsbücher, keine belastbare FRTB-Kalibrierung)", "assumption",
      "Änderungsprotokoll 06/2026"],
     ["A-08", "Universum", "die 10 größten EU-IRB-Banken (einheitliche "
      "Datenqualität statt Vollabdeckung)", "approximation",
      "EBA Transparency"],
     ["A-09", "Klassen-Raster", "7 IRB-Klassen (corporate, sme_corporate, "
      "mortgage, qrre, other_retail, bank, sovereign)", "approximation",
      "EU-CR6-Struktur"],
     ["A-10", "Faktor-Unabhängigkeit", "ρ(ΔBrent, Δr_10y) = +0,07 → separate "
      "Modellierung ohne Kreuzterm", "estimate",
      "eigene Regression, 1 242 Handelstage"]],
    [7, 17, 40, 12, 24], fontsize=8.5)

h2("Die Annahmen im Detail")

def annahme(aid, titel, text):
    doc.add_paragraph(f"{aid} · {titel}", style="Heading 3")
    body(text, size=9.5, space_after=9)

annahme("A-01", "PD-Quelle: Pillar-3 EU-CR6 (31.12.2024)",
    "**Festlegung:** Je Bank und Klasse die EAD-gewichtete Durchschnitts-PD "
    "aus dem EU-CR6-Sub-total — die regulatorische 1-Jahres-PD nach CRR "
    "Art. 180, inklusive des 100-%-Default-Bands. **Warum:** Es ist die "
    "einzige öffentliche, bankindividuelle und geprüfte PD-Quelle mit "
    "identischer Definition über alle zehn Banken. **Verworfene "
    "Alternative:** PD-Ableitung aus EBA-Transparency-Ausfallquoten "
    "(vermischt Bestands- und Flussgrößen, keine 1-Jahres-PD) sowie "
    "Rating-Agentur-PDs (nicht je Bank × Klasse verfügbar). **Wirkung bei "
    "Fehlspezifikation:** Das PD-Niveau skaliert den erwarteten Verlust "
    "nahezu proportional, die Kapitalanforderung unterproportional — eine "
    "Niveauverzerrung verschiebt also Ergebnisse, kippt aber keine "
    "Richtungsaussage.")
annahme("A-02", "LGD-Quelle: Pillar-3 EU-CR6 (31.12.2024)",
    "**Festlegung:** EAD-gewichtete Durchschnitts-LGD aus derselben "
    "EU-CR6-Tabelle wie A-01. **Warum:** Bankintern geschätzte A-IRB-LGD "
    "spiegeln die tatsächliche Besicherung des Portfolios wider. "
    "**Verworfene Alternative:** die pauschale F-IRB-Aufsichts-LGD von "
    "45 % — sie würde besicherte Portfolien (v. a. Hypotheken) massiv "
    "überzeichnen. **Wirkung bei Fehlspezifikation:** LGD wirkt linear auf "
    "erwarteten Verlust und fast linear auf die Kapitalanforderung — der "
    "direkteste Hebel unter den Parametern.")
annahme("A-02c", "Backtest-Zeitreihe (nur Pillar-3, no-look-ahead)",
    "**Festlegung:** Eigene Roh-Reihe über 10 Banken × FY2021–FY2024 (225 "
    "Punkte); ein Quartal im Jahr Y nutzt ausschließlich den Stichtag "
    "31.12.(Y−1) — den jüngsten, der damals publiziert war. **Warum:** Nur "
    "so ist die Validierung echt out-of-sample. **Verworfene Alternative:** "
    "den 2024-Snapshot rückwirkend anzuwenden (Look-ahead-Verzerrung) oder "
    "zwischen Stichtagen zu interpolieren (nicht quellengestützt). "
    "**Wirkung bei Fehlspezifikation:** Mit Look-ahead wäre jedes "
    "Backtest-Ergebnis systematisch geschönt und wertlos.")
annahme("A-03", "EAD: EBA Transparency, post-CCF",
    "**Festlegung:** Exposure at Default je Klasse nach "
    "Kreditumrechnungsfaktoren (post-CCF; EBA-Item 2520522). **Warum:** "
    "Post-CCF ist exakt die Bezugsgröße der IRB-Formel. **Verworfene "
    "Alternative:** Original Exposure — überzeichnet nicht gezogene "
    "Kreditlinien. **Wirkung bei Fehlspezifikation:** EAD wirkt als "
    "Portfoliogewicht linear auf Volumeneffekte, verzerrt aber die "
    "Quoten-Richtung nicht.")
annahme("A-04", "2-Faktor-Transmission (β/γ je Klasse)",
    "**Festlegung:** Lineare Übersetzung der Schocks in ΔPD/ΔLGD über die "
    "Matrix aus Abschnitt 3, mit begründeten Unter-/Obergrenzen. **Warum:** "
    "Die EBA-Stresstest-Methodik sieht sektorale Sensitivitäten explizit "
    "vor; die lineare Form ist transparent, prüfbar und per Regler "
    "übersteuerbar. **Verworfene Alternative:** ein Ein-Faktor-Aggregat "
    "(verliert die Sektor-Differenzierung und überzeichnete in Tests die "
    "Magnitude) sowie bankindividuell geschätzte Ökonometrie (Datenlage "
    "öffentlich nicht ausreichend). **Wirkung bei Fehlspezifikation:** Die "
    "β sind der größte Unsicherheitsträger des Modells — deshalb "
    "Konfidenzklasse [estimate] und Sichtbarkeit als eigene Matrix; ein "
    "Fehler skaliert die Stress-Wirkung in etwa proportional.")
annahme("A-05", "IRB-Kapitalformel (Basel III / ASRF)",
    "**Festlegung:** Die unveränderte regulatorische Formel mit α = 99,9 %, "
    "Basel-Asset-Korrelationen je Klasse und Laufzeit-Adjustment. "
    "**Warum:** Nur der regulatorische Standard macht Modell-Output und "
    "gemeldete RWA vergleichbar. **Verworfene Alternative:** interne "
    "Portfoliomodelle (z. B. Migrationsmatrizen) — nicht gegen die "
    "CET1-Meldung abgleichbar. **Wirkung bei Fehlspezifikation:** Die "
    "Formel selbst ist fixiert; das Modellrisiko liegt vollständig in den "
    "Inputs (A-01 bis A-04).")
annahme("A-06", "Sovereign-Kanal (Duration-MtM, IFRS-9-Split)",
    "**Festlegung:** ΔMtM = −Duration · Δr · Exposure; CET1-wirksam nur der "
    "FVOCI/FVTPL-Anteil je Bank. **Warum:** Das folgt der "
    "IFRS-9-Bewertungslogik — zu fortgeführten Anschaffungskosten (AC) "
    "gehaltene Bestände berühren die CET1-Quote laufend nicht. "
    "**Verworfene Alternative:** volle Marktbewertung aller Bestände "
    "(überzeichnet) oder Verzicht auf den Kanal (unterschlägt das "
    "Zinsrisiko). **Wirkung bei Fehlspezifikation:** Latente AC-Verluste "
    "bleiben unsichtbar — eine bewusste Untererfassung, die das Cockpit "
    "als separate Analyse ausweist (SVB-Lektion, Jiang et al. 2023).")
annahme("A-07", "Kein Trading-Book-Kanal",
    "**Festlegung:** Das Handelsbuch ist kein eigener Stress-Kanal. "
    "**Warum:** Die Handelsbücher der zehn Banken sind relativ klein, und "
    "eine belastbare öffentliche FRTB-Kalibrierung existiert nicht — ein "
    "pauschaler Multiplikator wäre Scheingenauigkeit. **Verworfene "
    "Alternative:** ein Kanal mit angenommenem Marktrisiko-Multiplikator "
    "(frühere Version, entfernt). **Wirkung:** Für Banken mit größerem "
    "Handelsbuch wird der Stress tendenziell unterschätzt — als "
    "Scope-Grenze dokumentiert (Abschnitt 8).")
annahme("A-08", "Universum: die 10 größten EU-IRB-Banken",
    "**Festlegung:** Genau zehn Institute, ausgewählt nach Größe und "
    "A-IRB-Offenlegungsqualität. **Warum:** Einheitliche, vollständige "
    "Pillar-3-Daten schlagen eine breite, aber heterogene Abdeckung. "
    "**Verworfene Alternative:** alle ~67 IRB-Banken der Transparency "
    "(uneinheitliche Offenlegung, Konzern-/Tochter-Doppelzählungen). "
    "**Wirkung:** Aussagen gelten für diese Gruppe — nicht für den "
    "gesamten EU-Bankenmarkt.")
annahme("A-09", "Klassen-Raster: 7 IRB-Klassen",
    "**Festlegung:** corporate, sme_corporate, mortgage, qrre, "
    "other_retail, bank, sovereign — das EU-CR6-Raster. **Warum:** Es ist "
    "der kleinste gemeinsame Nenner, den alle zehn Banken identisch "
    "offenlegen. **Verworfene Alternative:** feinere Branchenraster "
    "(NACE) — öffentlich nicht flächendeckend verfügbar. **Wirkung:** "
    "Heterogenität innerhalb einer Klasse wird auf den gewichteten "
    "Durchschnitt gemittelt.")
annahme("A-10", "Faktor-Unabhängigkeit (kein Kreuzterm)",
    "**Festlegung:** Öl- und Zinsschock wirken additiv, ohne "
    "Interaktionsterm. **Warum:** Empirisch sind die Faktoren nahezu "
    "unkorreliert (ρ = +0,07 über 1 242 Handelstage). **Verworfene "
    "Alternative:** Copula- oder Interaktionsmodelle — ohne empirischen "
    "Träger nur zusätzliche Komplexität. **Wirkung bei Fehlspezifikation:** "
    "Bei gleichzeitigen Extremschocks entstünde ein Effekt zweiter "
    "Ordnung; im beobachteten Wertebereich ist er vernachlässigbar.")

# ================= 7 · Validierung =================
h1("7 · Validierung: der CET1-Walk-Forward-Backtest")
body("Das Modell wird nicht an dem Datenstand geprüft, mit dem es gebaut "
     "wurde, sondern **rollierend durch die Vergangenheit**: Für jedes Jahr "
     "wird das Portfolio mit den damals bekannten Pillar-3-Parametern des "
     "Vorjahres eingefroren, der tatsächlich eingetretene Öl-/Zins-Schock "
     "des Jahres eingespeist und die prognostizierte CET1-Quote mit der "
     "später gemeldeten verglichen — ohne jeden Blick in die Zukunft.")
figure("fig4_walkforward.png", 16.2,
       "Abb. 4 · No-Look-Ahead-Prinzip: einfrieren (31.12. des Vorjahres) → "
       "realen Schock einspeisen → gegen die gemeldete CET1-Quote halten → "
       "ein Jahr weiterrollen.")
h2("Ergebnis (29 Bank-Jahre, 2022–2024)")
table(
    ["Kennzahl", "Wert", "Bedeutung"],
    [["MAE (mittlerer absoluter Fehler)", "**≈ 1,3 pp**",
      "Ø-Abstand Prognose ↔ gemeldete CET1-Quote; auf ~15 %-Niveau "
      "≈ 9 % relativer Abstand"],
     ["Treffer ≤ 1 pp", "**65 %**", "Anteil der Bank-Jahre mit höchstens "
      "1 Prozentpunkt Abweichung"],
     ["Konservativ-Anteil", "**72 %**", "Prognose ≤ gemeldete Quote — das "
      "Modell irrt auf der sicheren Seite"],
     ["Bias", "**−1,0 pp**", "systematisch vorsichtig (gewollt: kein "
      "NII-Gegeneffekt angesetzt)"]],
    [30, 14, 56], fontsize=9)
h2("Ehrliche Grenze: Point-in-Time vs. Through-the-Cycle")
body("Die einzelnen Melde-Kanäle (PD, Kredit-RWA) sind **nicht "
     "punktprognostizierbar** — und zwar aus einem dokumentierten "
     "Daten-Grund, nicht wegen eines Modellfehlers: Das Modell rechnet "
     "**Point-in-Time** (sofortige Schockreaktion), die gemeldeten "
     "A-IRB-Parameter sind **Through-the-Cycle** — regulatorisch geglättet "
     "und antizyklisch (CRR Art. 180) sowie durch Portfoliosteuerung "
     "überlagert. Beleg 2022: Zins +2,8 pp → Modell-PD +0,5 pp, gemeldete "
     "PD −0,2 pp. Validiert ist das Modell deshalb als **konservatives "
     "Solvenz-/Frühwarn-Instrument** (im CET1-Niveau nah und auf der "
     "sicheren Seite) — nicht als Prognose einzelner Meldegrößen.")

# ================= 8 · Scope =================
h1("8 · Bewusste Scope-Grenzen")
for b in [
    "**Operational Risk** bleibt unter Stress konstant.",
    "**CVA / Kontrahentenrisiko** nicht im Szenario.",
    "**Sovereign-Spread-Risiko** (z. B. Italien vs. Bund) nicht modelliert — "
    "nur der Niveauzins wirkt.",
    "**IFRS-9-Lifetime-EL** (Stage-2-Migration) außerhalb des Scopes; "
    "1-Jahres-Horizont.",
    "**Mehrjahres-Stresspfade** (3-Jahres-EBA-Logik) nicht abgebildet.",
    "**Hedging** (Swaps, CDS) aus Offenlegungen nicht rekonstruierbar.",
    "**Ertrags-Gegeneffekte** (NII, Gebühren) bewusst nicht angesetzt — "
    "Ergebnis ist eine konservative Untergrenze der CET1-Quote.",
    "**Bank-individuelle β** — alle Banken teilen die Klassen-β; die "
    "Differenzierung kommt aus dem realen Portfolio-Mix je Bank.",
]:
    bullet(b)
body("Verwendungs-Scope: ICAAP-Validierungs-Übung und Lehre/Demonstration. "
     "**Keine Anlageempfehlung**; Modell-Output ist eine konservative "
     "Szenario-Schätzung, keine bank-individuelle Punktprognose.",
     italic=True, size=9.5)

# ================= 9 · Bibliographie =================
h1("9 · Bibliographie")
body("Vollständige Liste aller im Modell verwendeten Quellen — gegliedert "
     "nach Regulatorik, wissenschaftlicher Literatur und Datenquellen. Jede "
     "Quelle ist an der Stelle ihrer Verwendung im Dokument bzw. im "
     "Modell-Code referenziert.")
h2("Regulatorische Quellen")
for b in [
    "BCBS (2017). Basel III: Finalising post-crisis reforms. Basel Committee "
    "on Banking Supervision, §272–284 (IRB-Kapitalformel, Asset-Korrelationen).",
    "Board of Governors / OCC (2011). SR 11-7: Supervisory Guidance on Model "
    "Risk Management (Outcomes-Analysis des Backtests).",
    "EBA (2020). ITS/2020/04 — Implementing Technical Standards on public "
    "disclosures (EU-CR6-Template, Pillar-3-Datenbasis).",
    "EBA (2024). 2025 EU-wide Stress Test — Methodological Note, §2.4.2 "
    "(sektorale Sensitivitäten als aufsichtlicher Projektionsansatz).",
    "EBA/ESRB (2025). 2025 EU-wide Stress Test — Macro-financial Scenario, "
    "§4.1.6 (adverse Zinspfade als Plausibilitätsanker).",
    "EBA (2025). 2025 EU-wide Stress Test — Results, Fig. 22 (relative "
    "Verlustquoten je Portfolio als Quervalidierung der β-Struktur).",
    "EBA (2025). Report on the 2024 Credit Risk Benchmarking Exercise "
    "(PD-/LGD-Niveaus je IRB-Klasse).",
    "EBA GL 2014/14. Guidelines on ICAAP/Stress-Testing (Governance-Rahmen).",
    "Verordnung (EU) Nr. 575/2013 (CRR) — insb. Art. 153/154 "
    "(IRB-Risikogewichte), Art. 180/181 (PD-/LGD-Schätzung, TTC-Charakter), "
    "Art. 431–455 (Offenlegung).",
]:
    bullet(b)
h2("Wissenschaftliche Literatur")
for b in [
    "Acharya, V., Drechsler, I. & Schnabl, P. (2014). A Pyrrhic Victory? "
    "Bank Bailouts and Sovereign Credit Risk. Journal of Finance "
    "(Staaten-Banken-Verflechtung, Marktbuch-Analyse).",
    "Bandoni, E., Fourné, M. & Jarmulska, B. (2025). Mortgage loan rates and "
    "the defaults of variable rate mortgages. ECB Working Paper 3112 "
    "(Zins-Sensitivität der Hypotheken-PD).",
    "Brunnermeier, M. et al. (2016). The Sovereign-Bank Diabolic Loop and "
    "ESBies. American Economic Review P&P (Doom-Loop, Marktbuch-Analyse).",
    "Gordy, M. (2003). A Risk-Factor Model Foundation for Ratings-Based Bank "
    "Capital Rules. Journal of Financial Intermediation (ASRF-Fundierung).",
    "Hyndman, R. J. & Athanasopoulos, G. (2021). Forecasting: Principles and "
    "Practice, 3. Aufl., Kap. 5.8 (Prognose-Evaluation, MAE).",
    "Jiang, E., Matvos, G., Piskorski, T. & Seru, A. (2023). Monetary "
    "Tightening and U.S. Bank Fragility. NBER Working Paper 31048 (latente "
    "Bewertungsverluste, Marktbuch-Analyse).",
    "Konietschke, P., Metzler, J. & Ponte Marques, A. (2026). A quantile "
    "probability model for sectoral corporate defaults in Europe. ECB "
    "Working Paper 3207 (Sektor-Heterogenität der Unternehmens-PD).",
    "Lo Duca, M., Moccero, D. & Parlapiano, F. (2024). The impact of "
    "macroeconomic and monetary policy shocks on credit risk in the euro "
    "area corporate sector. ECB Working Paper 2897 (Corporate-/KMU-β).",
    "Nelson, C. & Siegel, A. (1987). Parsimonious Modeling of Yield Curves. "
    "Journal of Business (Zinsstrukturmodell).",
    "Pesaran, M. H. & Timmermann, A. (1992). A Simple Nonparametric Test of "
    "Predictive Performance. Journal of Business & Economic Statistics "
    "(Richtungs-Trefferquote).",
    "Svensson, L. (1994). Estimating and Interpreting Forward Interest "
    "Rates: Sweden 1992–1994. NBER Working Paper 4871 "
    "(Zinsstrukturmodell der Bundesbank-Parameter).",
    "Tuckman, B. & Serrat, A. (2012). Fixed Income Securities, 3. Aufl. "
    "(Modified Duration, Sovereign-Kanal).",
    "Vasicek, O. (1991). Limiting Loan Loss Probability Distribution. KMV "
    "Working Paper (Portfolio-Verlustverteilung).",
    "Vasicek, O. (2002). Loan Portfolio Value. Risk Magazine, Dezember "
    "(geschlossene Form der IRB-Formel).",
    "ECB (2024). Financial Stability Review, Mai 2024 (Energie-/Lebens"
    "haltungskosten-Wirkung auf Haushalte; Retail-β).",
]:
    bullet(b)
h2("Datenquellen")
for b in [
    "Pillar-3-Berichte (EU CR6) der zehn Banken, FY2021–FY2024 — "
    "Risikoparameter PD/LGD/EAD (je Zeile mit Seiten-/URL-Beleg).",
    "EBA EU-wide Transparency Exercise (Vintages 2020–2025) — realisierte "
    "CET1- und RWA-Werte, IFRS-9-Split der Staatsanleihen.",
    "ICE Brent Crude (täglich) — Ölpreis-Faktor.",
    "Deutsche Bundesbank — täglich geschätzte Svensson-Parameter der "
    "Zinsstrukturkurve (10-Jahres-Zins-Faktor).",
]:
    bullet(b)

doc.save(OUT)
print("geschrieben:", OUT)
