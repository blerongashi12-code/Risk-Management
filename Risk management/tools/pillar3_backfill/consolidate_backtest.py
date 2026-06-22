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

RM = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/"
DL = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/backtest_raw/"
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

# --- SocGen: 2022/2023 from extract csv; 2024 raw anchors (verified) ---
SG = "O2RNE8IBXP4R0TD8PU41"
sg = pd.read_csv(DL + "socgen_timeseries_rows.csv", dtype=str)
for _, r in sg.iterrows():
    add(SG, "Societe Generale", r["vasicek_class"], r["vintage_date"], r["pd_pct"], r["lgd_pct"],
        r["ead_eur_m"], "SocGen Pillar 3 EU CR6 AIRB (raw subtotal)", r.get("note", ""))
SG24 = {"sme_corporate": (3.57, 19.06, 67967), "mortgage": (1.18, 16.37, 116982),
        "qrre": (7.89, 48.48, 4149), "other_retail": (8.06, 31.89, 28572),
        "bank": (0.36, 22.68, 40633), "sovereign": (0.09, 1.54, 310916)}
for c, (p, l, e) in SG24.items():
    add(SG, "Societe Generale", c, "2024-12-31", p, l, e, "SocGen Pillar 3 2024 EU CR6 AIRB (raw)")

# --- BPCE: FY2024 clean (7/7 RAW-OK); prior years pending (multi-block) ---
BP = "FR9695005MSX1OYEMGDF"
if os.path.exists(DL + "bpce_raw_rows.csv"):
    bp = pd.read_csv(DL + "bpce_raw_rows.csv", dtype=str)
    for _, r in bp[bp.vintage_date == "2024-12-31"].iterrows():
        add(BP, "Groupe BPCE", r["vasicek_class"], "2024-12-31", r["pd_pct"], r["lgd_pct"],
            r["ead_eur_m"], "BPCE Pillar III 2024 EU CR6 AIRB (raw subtotal)")

# --- Credit Mutuel: FY2024 (6/7, no qrre) + FY2023 {corp,sme,mortgage,other_retail}.
#     FY2023 bank/sovereign/qrre were mis-assigned (all grabbed the revolving row) -> dropped ---
CM = "9695000CG7B84NLR5984"
if os.path.exists(DL + "cm_raw_rows.csv"):
    cm = pd.read_csv(DL + "cm_raw_rows.csv", dtype=str)
    keep23 = {"corporate", "sme_corporate", "mortgage", "other_retail"}
    for _, r in cm.iterrows():
        v, c = r["vintage_date"], r["vasicek_class"]
        if v == "2023-12-31" and c not in keep23:
            continue
        add(CM, "Credit Mutuel", c, v, r["pd_pct"], r["lgd_pct"], r["ead_eur_m"],
            "Credit Mutuel Pilier 3 EU CR6 AIRB (raw subtotal)")

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
