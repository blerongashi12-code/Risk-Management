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

# --- Post-process: ING has NO Qualifying Revolving (QRRE) A-IRB block in any year;
#     its 'qrre' rows are actually "Retail - Secured by immovable property SME"
#     (verification finding). Relabel -> mortgage_sme so it does not contaminate the
#     real cross-bank QRRE class (DB/Santander/BNP/UniCredit qrre are genuine). ---
for r in rows:
    if r["LEI"] == INGL and r["vasicek_class"] == "qrre":
        r["vasicek_class"] = "mortgage_sme"
        r["note"] = (r.get("note") or "") + " [ING SRE-SME, nicht QRRE: umbenannt]"

df = pd.DataFrame(rows).drop_duplicates(subset=["LEI", "vasicek_class", "vintage_date"], keep="first")
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
