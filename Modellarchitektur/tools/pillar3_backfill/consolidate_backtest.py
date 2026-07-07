# -*- coding: utf-8 -*-
"""Consolidate the reliable raw EU-CR6 PD/LGD/EAD rows extracted so far into
data/pillar3_backtest_pdlgd.csv (the consistent raw backtest series).
Only verified rows: column-calibrated + EAD-continuous. Known fixes applied:
- UniCredit: the 'mortgage' slot holds the Central-governments (sovereign) row
  (header-bleed); relabel -> sovereign. Real Retail-mortgage is a documented gap.
- DENSITY? flags on ING/SocGen are the fraction-vs-percent artefact; values are
  EAD-continuous + column-calibrated -> kept.
"""
import pandas as pd, os

RM = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/"
DL = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/backtest_raw/"
rows = []   # dicts: LEI, bank_name, vasicek_class, vintage_date, pd_pct, lgd_pct, ead_eur_m, source, note

def add(lei, bank, cls, vint, pd_, lgd, ead, src, note=""):
    rows.append({"LEI": lei, "bank_name": bank, "vasicek_class": cls,
                 "vintage_date": vint, "pd_pct": f"{float(pd_):.2f}",
                 "lgd_pct": f"{float(lgd):.2f}",
                 "ead_eur_m": ("" if ead in ("", None) else f"{float(ead):.0f}"),
                 "source": src, "note": note})

# --- Deutsche Bank: from committed CSV (pd/lgd) + portfolio sidecar (ead) ---
pdcsv = pd.read_csv(RM + "pillar3_bank_pd_lgd.csv", dtype=str)
port = pd.read_csv(RM + "pillar3_portfolio_timeseries.csv", dtype=str)
port_ead = {(r["LEI"], r["vasicek_class"], r["vintage_date"]): r["ead_eur_m"] for _, r in port.iterrows()}
DB = "7LTWFZYICNSX8D621K86"; INGL = "549300NYKK9MWM7GGW15"
for _, r in pdcsv[pdcsv.LEI == DB].iterrows():
    ead = port_ead.get((DB, r["vasicek_class"], r["vintage_date"]), "")
    add(DB, "Deutsche Bank", r["vasicek_class"], r["vintage_date"], r["pd_pct"], r["lgd_pct"], ead,
        "DB Pillar 3 EU CR6 AIRB (raw subtotal)")

# --- ING: 2021 from CSV+sidecar; 2024 raw anchors (verified) ---
for _, r in pdcsv[(pdcsv.LEI == INGL) & (pdcsv.vintage_date == "2021-12-31")].iterrows():
    ead = port_ead.get((INGL, r["vasicek_class"], "2021-12-31"), "")
    add(INGL, "ING Groep", r["vasicek_class"], "2021-12-31", r["pd_pct"], r["lgd_pct"], ead,
        "ING Additional Pillar III 2021 EU CR6 IRB (raw)")
ING24 = {"corporate": (1.81, 14.43, 327826), "sme_corporate": (6.12, 27.39, 47644),
         "mortgage": (1.46, 22.61, 336546), "qrre": (3.75, 22.50, 14592),
         "other_retail": (6.39, 39.90, 4636), "bank": (0.39, 19.79, 47295)}
for c, (p, l, e) in ING24.items():
    add(INGL, "ING Groep", c, "2024-12-31", p, l, e, "ING Additional Pillar III 2024 EU CR6 IRB (raw)")
# ING 2022 + 2023: raw 6/6 from the class-total rows (ing_2022.py, robust detector
# calibrated to reproduce FY2024 exactly).
for yr in ("2022", "2023"):
    fp = DL + f"ing_{yr}_rows.csv"
    if os.path.exists(fp):
        iy = pd.read_csv(fp, dtype=str)
        for _, r in iy.iterrows():
            add(INGL, "ING Groep", r["vasicek_class"], f"{yr}-12-31", r["pd_pct"], r["lgd_pct"],
                r["ead_eur_m"], f"ING Additional Pillar III {yr} EU CR6 IRB (raw class-total)")

# --- UniCredit: relabel mortgage-slot -> sovereign; keep 5 header classes ---
UC = "549300TRUWO2CD2G5692"
uc = pd.read_csv(DL + "unicredit_raw_rows.csv", dtype=str)
for _, r in uc.iterrows():
    c = r["vasicek_class"]
    if c == "mortgage":
        c = "sovereign"; note = "Central-gov row (header-bleed relabel); verified low-PD"
    else:
        note = ""
    add(UC, "UniCredit", c, r["vintage_date"], r["pd_pct"], r["lgd_pct"], r["ead_eur_m"],
        "UniCredit Pillar III EU CR6 AIRB (raw subtotal)", note)
# UC 2021 sovereign + other_retail: the flowing-table extractor missed these; values
# read + density-verified from the 2021 report (p134 Central-gov, p138 Retail-Other-non-SME).
add(UC, "UniCredit", "sovereign", "2021-12-31", 0.10, 21.48, 24824,
    "UniCredit Pillar III 2021 EU CR6 AIRB (Central-gov subtotal, source-verified)")
add(UC, "UniCredit", "other_retail", "2021-12-31", 4.98, 42.84, 22912,
    "UniCredit Pillar III 2021 EU CR6 AIRB (Retail-Other-non-SME subtotal, source-verified)")

# --- SocGen: 2022/2023 from extract csv; 2024 raw anchors (verified) ---
SG = "O2RNE8IBXP4R0TD8PU41"
sg = pd.read_csv(DL + "socgen_timeseries_rows.csv", dtype=str)
for _, r in sg.iterrows():
    add(SG, "Societe Generale", r["vasicek_class"], r["vintage_date"], r["pd_pct"], r["lgd_pct"],
        r["ead_eur_m"], "SocGen Pillar 3 EU CR6 AIRB (raw subtotal)", r.get("note", ""))
SG24 = {"corporate": (2.39, 32.30, 169993),  # large "Corporate - Other" (was missing)
        "sme_corporate": (3.57, 19.06, 67967), "mortgage": (1.18, 16.37, 116982),
        "qrre": (7.89, 48.48, 4149), "other_retail": (8.06, 31.89, 28572),
        "bank": (0.36, 22.68, 40633), "sovereign": (0.09, 1.54, 310916)}
for c, (p, l, e) in SG24.items():
    add(SG, "Societe Generale", c, "2024-12-31", p, l, e, "SocGen Pillar 3 2024 EU CR6 AIRB (raw)")
# SocGen bank 2021/2023: Institutions subtotal exists + density-verified, but the
# extractor missed it (empty learned context -> overlap<1). FY2022 not extractable
# (concatenated-text PDF). Source-read p150 (2021) / p142 (2023).
add(SG, "Societe Generale", "bank", "2021-12-31", 0.89, 23.38, 39906,
    "SocGen Pillar 3 2021 EU CR6 AIRB Institutions subtotal (source-verified)")
add(SG, "Societe Generale", "bank", "2023-12-31", 0.60, 23.48, 37138,
    "SocGen Pillar 3 2023 EU CR6 AIRB Institutions subtotal (source-verified)")

# --- BPCE: FY2024 clean (7/7 RAW-OK); prior years pending (multi-block) ---
BP = "FR9695005MSX1OYEMGDF"
if os.path.exists(DL + "bpce_raw_rows.csv"):
    bp = pd.read_csv(DL + "bpce_raw_rows.csv", dtype=str)
    for _, r in bp[bp.vintage_date == "2024-12-31"].iterrows():
        add(BP, "Groupe BPCE", r["vasicek_class"], "2024-12-31", r["pd_pct"], r["lgd_pct"],
            r["ead_eur_m"], "BPCE Pillar III 2024 EU CR6 AIRB (raw subtotal)")
# BPCE 2022/2023: only the 4 distinctive A-IRB classes (sovereign/bank/mortgage/other_retail)
# from block-1, density-verified + EAD-continuous. corporate (PD discontinuity) and
# sme/qrre (multi-subclass collisions across the 4-block table) remain documented gaps.
if os.path.exists(DL + "bpce_prior_rows.csv"):
    bpp = pd.read_csv(DL + "bpce_prior_rows.csv", dtype=str)
    for _, r in bpp.iterrows():
        add(BP, "Groupe BPCE", r["vasicek_class"], r["vintage_date"], r["pd_pct"], r["lgd_pct"],
            r["ead_eur_m"], "BPCE Pillar III EU CR6 A-IRB block-1 (distinctive class, source-verified)")
# BPCE 2021: from the June-2022 half-year UPDATE report (31.12.2021 A-IRB comparative,
# block-6, p80-83). Abbreviated report -> only sovereign/bank/other_retail present
# (mortgage absent); all density-verified + EAD-continuous with 2022/2023.
add(BP, "Groupe BPCE", "sovereign", "2021-12-31", 0.33, 9.00, 58551,
    "BPCE Pillar III H1-2022 update, 31.12.2021 A-IRB comparative (source-verified)")
add(BP, "Groupe BPCE", "bank", "2021-12-31", 2.83, 32.32, 7480,
    "BPCE Pillar III H1-2022 update, 31.12.2021 A-IRB comparative (source-verified)")
add(BP, "Groupe BPCE", "other_retail", "2021-12-31", 14.67, 24.15, 45285,
    "BPCE Pillar III H1-2022 update, 31.12.2021 A-IRB comparative (source-verified)")

# --- Credit Mutuel: FY2024 (5/7) + FY2023 {corp,sme,mortgage,other_retail}.
#     FY2023 bank/sovereign/qrre were mis-assigned (all grabbed the revolving row) -> dropped.
#     sovereign DROPPED for ALL years: CM has NO IRB sovereign (central govts permanently on
#     standard approach); the 'sovereign 2024' row was a byte-identical duplicate of bank
#     (verification finding) -> phantom, removed. ---
CM = "9695000CG7B84NLR5984"
if os.path.exists(DL + "cm_raw_rows.csv"):
    cm = pd.read_csv(DL + "cm_raw_rows.csv", dtype=str)
    keep23 = {"corporate", "sme_corporate", "mortgage", "other_retail"}
    for _, r in cm.iterrows():
        v, c = r["vintage_date"], r["vasicek_class"]
        if c == "sovereign":                       # CM discloses no IRB sovereign
            continue
        if v == "2023-12-31" and c not in keep23:
            continue
        add(CM, "Credit Mutuel", c, v, r["pd_pct"], r["lgd_pct"], r["ead_eur_m"],
            "Credit Mutuel Pilier 3 EU CR6 AIRB (raw subtotal)")
# CM 2021/2022: only the two distinctive-LGD classes are reliably mappable (the
# degenerate-anchor classes stay wrong-row). corporate (LGD 45, EAD continuous
# 67->70->73k, anchor-matched) + mortgage (LGD 15-16, EAD continuous ~214-291k).
add(CM, "Credit Mutuel", "corporate", "2022-12-31", 2.39, 45.00, 66725,
    "Credit Mutuel Pilier 3 2022 EU CR6 AIRB corporate (LGD-distinctive, continuity-verified)")
add(CM, "Credit Mutuel", "mortgage", "2021-12-31", 2.14, 15.00, 214247,
    "Credit Mutuel Pilier 3 2021 EU CR6 AIRB mortgage (LGD-distinctive, continuity-verified)")
add(CM, "Credit Mutuel", "mortgage", "2022-12-31", 2.10, 15.00, 229540,
    "Credit Mutuel Pilier 3 2022 EU CR6 AIRB mortgage (LGD-distinctive, continuity-verified)")

# --- BNP Paribas: EU-CR6 IRBA-by-PD-scale subtotals, anchor+density+EAD-continuity
#     verified. bnp_2024.pdf yields 2024 + the 2023 comparative; bnp_2023/2021 add
#     2022/2021. Extractor class 'sme' -> canonical 'sme_corporate'. ---
BNP = "R0MUWSFPU8MPRO8K5P83"
if os.path.exists(DL + "bnp_raw_rows.csv"):
    bnp = pd.read_csv(DL + "bnp_raw_rows.csv", dtype=str)
    for _, r in bnp.iterrows():
        c = "sme_corporate" if r["vasicek_class"] == "sme" else r["vasicek_class"]
        add(BNP, "BNP Paribas", c, r["vintage_date"], r["pd_pct"], r["lgd_pct"],
            r["ead_eur_m"], "BNP Paribas URD EU CR6 IRBA-by-PD-scale (raw subtotal)")

# --- Banco Santander: EU-CR6 AIRB subtotals, label+year-driven, 6 classes
#     (no sovereign = Standardised), 2021-2024. Each report gives reporting date +
#     prior-year comparative; primary disclosure preferred, overlaps cross-validated.
#     NB: bank/corporate AIRB-EAD shrinks sharply 2022->2023 (real IRB roll-back to
#     FIRB), verified correct (density-checked); not an extraction error. ---
SAN = "5493006QMFDDMYWIAM13"
if os.path.exists(DL + "santander_raw_rows.csv"):
    san = pd.read_csv(DL + "santander_raw_rows.csv", dtype=str)
    for _, r in san.iterrows():
        note = ("AIRB-Perimeter 2022->2023 verkleinert (IRB-Rollback zu FIRB), Werte verifiziert"
                if r["vasicek_class"] in ("bank", "corporate") else "")
        add(SAN, "Banco Santander", r["vasicek_class"], r["vintage_date"], r["pd_pct"], r["lgd_pct"],
            r["ead_eur_m"], "Banco Santander Pillar 3 EU CR6 AIRB (raw subtotal)", note)

# --- Rabobank: EU-CR6 A-IRB from the dedicated text-based "Pillar 3 Year Report"
#     (the annual-report PDF is image-only). 2022 = 6/6 clean; 2024 = mortgage+sme
#     continuous, sovereign/corporate are post-rollback A-IRB residuals (A-IRB->F-IRB,
#     density-verified, EAD shrank sharply -> note), other_retail-2024 dropped (parse). ---
RABO = "DG3RU1DBUFHT4ZF9WN62"
if os.path.exists(DL + "rabobank_raw_rows.csv"):
    rb = pd.read_csv(DL + "rabobank_raw_rows.csv", dtype=str)
    for _, r in rb.iterrows():
        note = ("A-IRB-Residuum nach IRB-Rollback (A->F-IRB), EAD stark verkleinert, dichte-verifiziert"
                if (r["vintage_date"] == "2024-12-31" and r["vasicek_class"] in ("sovereign", "corporate", "bank"))
                else "")
        add(RABO, "Rabobank", r["vasicek_class"], r["vintage_date"], r["pd_pct"], r["lgd_pct"],
            r["ead_eur_m"], "Rabobank Pillar 3 Year Report EU CR6 AIRB (raw subtotal)", note)

# --- Crédit Agricole S.A.: EU-CR6 Advanced-IRB. CA *S.A.* (not CA Group) is the entity
#     in the EBA Transparency panel -> the backtestable CA entity. The year-end 2024 report
#     yields 31.12.2024 + the 31.12.2023 comparative (7/7 each, density-verified). ---
CASA = "F0HUI1NY1AZMJMD8LP67"
if os.path.exists(DL + "ca_raw_rows.csv"):
    ca = pd.read_csv(DL + "ca_raw_rows.csv", dtype=str)
    for _, r in ca.iterrows():
        add(CASA, "Credit Agricole SA", r["vasicek_class"], r["vintage_date"], r["pd_pct"],
            r["lgd_pct"], r["ead_eur_m"], "Credit Agricole S.A. Pillar 3 EU CR6 A-IRB (raw subtotal)")

# ====================================================================
# Lueckenschluss-Sweep Juli 2026: 27 Zellen, alle wort-woertlich aus den
# Quell-PDFs belegt (Seite + Rohzeile), Dichte-Check (RWA/EAD vs. gedruckte
# Dichte <=1pp) und EAD-Kontinuitaet gegen Nachbarjahre bestanden.
# Verbleibende echte Quellgrenzen (NICHT extrahierbar):
#   - SocGen sovereign 2022: PD/LGD/Dichte als woertliche "0"-Platzhalter
#     gedruckt, doppelt geprueft (socgen_2022 S.145 UND socgen_2023 S.147).
#   - CM corporate+sme 2021: kombinierte NI-Tabelle bricht A-/F-IRB-Konvention.
#   - CA sme 2021: echter A-IRB-Perimeterbruch (Kontinuitaetsregel).
# ====================================================================

# --- BNP 2021 (7/7): 31.12.2021-Vergleichsspalten der URD 2022 (bnp_2022.pdf,
#     Euronext-Download), S.411-423. Dezimalwerte, Dichte-Checks bestanden. ---
for cls, pd_, lgd, ead, page in [
    ("sovereign",     0.05,  2.00, 469143, 411),
    ("bank",          0.67, 28.00,  43767, 411),
    ("corporate",     2.86, 34.00, 350196, 415),
    ("sme_corporate", 7.30, 29.00,  43188, 414),
    ("mortgage",      2.19, 12.00, 179316, 419),
    ("qrre",          9.54, 53.00,  12425, 422),
    ("other_retail",  7.24, 40.00,  47407, 423),
]:
    add(BNP, "BNP Paribas", cls, "2021-12-31", pd_, lgd, ead,
        f"BNP Paribas URD 2022 EU CR6 IRBA, 31.12.2021-Vergleich (S.{page}, quellverifiziert)")

# --- SocGen bank 2022: Institutions-Subtotal, doppelt belegt (socgen_2022.pdf
#     S.145 primaer + socgen_2023.pdf S.147 Vorjahres-Vergleich, identisch). ---
add(SG, "Societe Generale", "bank", "2022-12-31", 1.24, 25.01, 38844,
    "SocGen Pillar 3 2022 EU CR6 AIRB Institutions subtotal (S.145; doppelt belegt via 2023-Bericht S.147)")

# --- Rabobank 2024: bank = A-IRB-Residuum nach Rollback (S.76, EAD 5,783->465->55,
#     woertlich + dichte-verifiziert); other_retail S.82 (Parse-Luecke geschlossen). ---
add(RABO, "Rabobank", "bank", "2024-12-31", 2.23, 10.65, 55,
    "Rabobank Pillar 3 2024 EU CR6 (S.76, quellverifiziert)",
    "A-IRB-Residuum nach IRB-Rollback (A->F-IRB), EAD stark verkleinert, dichte-verifiziert")
add(RABO, "Rabobank", "other_retail", "2024-12-31", 5.38, 24.00, 1771,
    "Rabobank Pillar 3 2024 EU CR6 (S.82, quellverifiziert)")

# --- Credit Mutuel 2021-2023: bank-Serie ist durchgaengig F-IRB 'Etablissements'
#     (A-IRB weist keine Institute aus; 2024-Anker 30,410 = F-IRB-Zeile).
#     other_retail = 'Clientele de detail'-Subtotal (Serienkonvention wie 2023/24). ---
add(CM, "Credit Mutuel", "bank", "2021-12-31", 0.16, 30.00, 27374,
    "Credit Mutuel Pilier 3 2021 EU CR6 NI (S.52, quellverifiziert)",
    "F-IRB-Zeile (Serienkonvention: CM meldet Institute nur F-IRB)")
add(CM, "Credit Mutuel", "sme_corporate", "2021-12-31", 4.17, 29.00, 134578,
    "Credit Mutuel Pilier 3 2021 EU CR6 A-IRB Entreprises (S.52, quellverifiziert)",
    "A-IRB-Entreprises-Zeile; Serienkonvention wie 2022/2023/2024")
add(CM, "Credit Mutuel", "other_retail", "2021-12-31", 2.55, 17.00, 361722,
    "Credit Mutuel Pilier 3 2021 EU CR6 (S.52, quellverifiziert)")
add(CM, "Credit Mutuel", "bank", "2022-12-31", 0.19, 35.00, 25423,
    "Credit Mutuel Pilier 3 2022 EU CR6 F-IRB Etablissements (S.51, quellverifiziert)",
    "F-IRB-Zeile (Serienkonvention: CM meldet Institute nur F-IRB)")
add(CM, "Credit Mutuel", "sme_corporate", "2022-12-31", 4.89, 23.00, 86416,
    "Credit Mutuel Pilier 3 2022 EU CR6 A-IRB Entreprises (S.48, quellverifiziert)")
add(CM, "Credit Mutuel", "other_retail", "2022-12-31", 2.52, 16.00, 385061,
    "Credit Mutuel Pilier 3 2022 EU CR6 Clientele de detail (S.48, quellverifiziert)")
add(CM, "Credit Mutuel", "bank", "2023-12-31", 0.19, 35.00, 29543,
    "Credit Mutuel Pilier 3 2023 EU CR6 F-IRB Etablissements (S.50, quellverifiziert)",
    "F-IRB-Zeile (Serienkonvention: CM meldet Institute nur F-IRB; 2024-Anker 30,410 kontinuierlich)")

# --- Credit Agricole S.A. qrre 2022: 31.12.2022-Vergleich im H1-2023-Bericht
#     (ca_sa_2023h1.pdf S.42), Dichte-Check bestanden. ---
add(CASA, "Credit Agricole SA", "qrre", "2022-12-31", 5.13, 79.12, 11827,
    "Credit Agricole S.A. Pillar 3 H1-2023, 31.12.2022 EU CR6 A-IRB Vergleich (S.42, quellverifiziert)")
add(CASA, "Credit Agricole SA", "sme_corporate", "2021-12-31", 26.43, 33.29, 3074,
    "Credit Agricole S.A. Pillar 3 H1-2022, 31.12.2021 EU CR6 A-IRB Vergleich (S.40, quellverifiziert)",
    "MODEL_EXCLUDED: echter A-IRB-Perimeterbruch; ab 2022 werden >1m-SME-Exposures aus Retail nach Corporates-SME umklassifiziert")

# --- SocGen sovereign 2022: source row exists, but PD/LGD/Density are printed as
#     literal zero placeholders in the 2022 report. Keep the source data point for
#     audit/coverage, but exclude it from model input instead of inferring PD/LGD
#     from RWA or neighbouring years. ---
add(SG, "Societe Generale", "sovereign", "2022-12-31", 0.00, 0.00, 271679,
    "SocGen Pillar 3 2022 EU CR6 AIRB Central governments subtotal (S.145, gedruckte 0-Platzhalter)",
    "MODEL_EXCLUDED: PD/LGD nur als gedruckte 0-Platzhalter; keine Ableitung aus RWA/Nachbarjahren")

# --- BPCE corporate/sme/qrre 2021-2023: Block-Anker-Matching. 2021 aus dem
#     H1-2022-Update (31.12.2021-Vergleich, S.81-85, umbrochene Subtotal-Labels);
#     2022 aus bpce_2022 S.128-132, doppelt belegt via bpce_2023 S.157-161;
#     2023 aus bpce_2023 S.149-153. Dichte-Checks exakt. Corporate-PD-Sprung
#     2022->2023 (17,90->4,07) ist wie-gemeldet in beiden Publikationen. ---
for vint, cells in [
    ("2021-12-31", [("corporate", 16.54, 33.74, 78604), ("sme_corporate", 8.34, 22.25, 5527),
                    ("mortgage", 19.91, 10.81, 278290), ("qrre", 9.28, 21.20, 11217)]),
    ("2022-12-31", [("corporate", 17.90, 33.58, 80498), ("sme_corporate", 7.15, 21.99, 6041),
                    ("qrre", 9.70, 21.64, 11189)]),
    ("2023-12-31", [("corporate", 4.07, 33.99, 85550), ("sme_corporate", 7.81, 23.06, 6735),
                    ("qrre", 9.50, 37.08, 23933)]),
]:
    src_map = {"2021-12-31": "BPCE Pillar III H1-2022 update, 31.12.2021 A-IRB Vergleich (S.81-85, quellverifiziert)",
               "2022-12-31": "BPCE Pillar III 2022 EU CR6 A-IRB (S.128-132; doppelt belegt via 2023-Bericht S.157-161)",
               "2023-12-31": "BPCE Pillar III 2023 EU CR6 A-IRB (S.149-153, quellverifiziert)"}
    for cls, pd_, lgd, ead in cells:
        note = ("PD-Niveausprung 2022->2023 wie-gemeldet (in beiden Publikationen identisch gedruckt)"
                if cls == "corporate" else "")
        add(BP, "Groupe BPCE", cls, vint, pd_, lgd, ead, src_map[vint], note)

# --- Post-process: ING has NO Qualifying Revolving (QRRE) A-IRB block in any year;
#     its 'qrre' rows are actually "Retail - Secured by immovable property SME"
#     (verification finding). Relabel -> mortgage_sme so it does not contaminate the
#     real cross-bank QRRE class (DB/Santander/BNP/UniCredit qrre are genuine). ---
for r in rows:
    if r["LEI"] == INGL and r["vasicek_class"] == "qrre":
        r["vasicek_class"] = "mortgage_sme"
        r["note"] = (r.get("note") or "") + " [ING SRE-SME, nicht QRRE: umbenannt]"

df = pd.DataFrame(rows).drop_duplicates(subset=["LEI", "vasicek_class", "vintage_date"], keep="first")
df["include_in_backtest"] = 1
df["quality_flag"] = "source_verified"
excluded = df["note"].fillna("").str.contains("MODEL_EXCLUDED", regex=False)
df.loc[excluded, "include_in_backtest"] = 0
df.loc[excluded, "quality_flag"] = "source_verified_model_excluded"
df = df.sort_values(["bank_name", "vintage_date", "vasicek_class"])
out = RM + "pillar3_backtest_pdlgd.csv"
df.to_csv(out, index=False)
print(f"WROTE {out}  ({len(df)} rows)")
print("\nCoverage (rows per bank x year):")
print(df.groupby(["bank_name", "vintage_date"]).size().to_string())
print("\nclasses per bank:")
for b in df.bank_name.unique():
    cs = sorted(df[df.bank_name == b].vasicek_class.unique())
    print(f"  {b:18} {cs}")
