# -*- coding: utf-8 -*-
"""Merge ING FY2021 (6 classes) into the repo PD/LGD CSV + portfolio sidecar.
5 rows auto-extracted (ing_timeseries_rows.csv) + mortgage from the verified
raw sub-total row (p43: EAD 307,236 / PD 1.57 / LGD 16.08; density 0.11 ok)."""
import pandas as pd

PD_CSV = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_bank_pd_lgd.csv"
PORT_CSV = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_portfolio_timeseries.csv"
URL = "https://www.ing.com/Investor-relations/Annual-reports.htm (Additional Pillar III Report 2021)"

LABEL = {"corporate": "Corporates-Other", "sme_corporate": "Corporates-SME",
         "mortgage": "Retail - Secured by immovable property (non-SME)",
         "qrre": "Retail - Secured by immovable property SME (QRRE proxy)",
         "other_retail": "Retail Other SME", "bank": "Institutions"}

# auto-extracted (5) + manual verified mortgage (1)
ing21 = pd.read_csv("C:/Users/blero/Downloads/ing_timeseries_rows.csv")
rows = {r["vasicek_class"]: (r["pd_pct"], r["lgd_pct"], r["ead_eur_m"], r["source_page"])
        for _, r in ing21.iterrows()}
rows["mortgage"] = (1.57, 16.08, 307236, 43)   # verified from p43 raw sub-total

LEI = "549300NYKK9MWM7GGW15"

# ---- PD/LGD CSV ----
pddf = pd.read_csv(PD_CSV, dtype=str)
if "note" not in pddf.columns:
    pddf["note"] = ""
pddf["note"] = pddf["note"].fillna("")
# drop any pre-existing ING 2021 rows (idempotent)
mask = ~((pddf["LEI"] == LEI) & (pddf["vintage_date"] == "2021-12-31"))
pddf = pddf[mask]
new_pd = []
for cls, (pdv, lgd, ead, pg) in rows.items():
    new_pd.append({"bank_name": "ING Groep", "LEI": LEI, "bank_country": "NL",
                   "vasicek_class": cls, "pd_pct": f"{float(pdv):.2f}", "lgd_pct": f"{float(lgd):.2f}",
                   "vintage_date": "2021-12-31", "status": "pillar3_verified",
                   "source": "ING Group Additional Pillar III Report 2021",
                   "source_period": "2021-12-31",
                   "source_table": f"EU CR6 - IRB approach (2021) - {LABEL[cls]} sub-total [col d=EAD, e=PD, g=LGD]",
                   "source_page": str(int(pg)), "source_url": URL, "note": ""})
new_pd = pd.DataFrame(new_pd)[list(pddf.columns)]
pddf = pd.concat([pddf, new_pd], ignore_index=True)
cls_order = {c: i for i, c in enumerate(["corporate","sme_corporate","mortgage","qrre","other_retail","bank","sovereign"])}
pddf["_co"] = pddf["vasicek_class"].map(cls_order).fillna(99)
pddf = pddf.sort_values(["bank_name", "vintage_date", "_co"]).drop(columns="_co")
pddf.to_csv(PD_CSV, index=False)

# ---- portfolio sidecar (EAD) ----
po = pd.read_csv(PORT_CSV, dtype=str)
mask = ~((po["LEI"] == LEI) & (po["vintage_date"] == "2021-12-31"))
po = po[mask]
new_po = []
for cls, (pdv, lgd, ead, pg) in rows.items():
    new_po.append({"LEI": LEI, "bank_name": "ING Groep", "vasicek_class": cls,
                   "vintage_date": "2021-12-31", "ead_eur_m": str(int(float(ead))),
                   "maturity_years": "2.50", "rwa_eur_m": "", "density_pct": "",
                   "source": f"ING Group Additional Pillar III Report 2021 EU CR6 IRB col d=EAD (p{int(pg)})"})
new_po = pd.DataFrame(new_po)[list(po.columns)]
po = pd.concat([po, new_po], ignore_index=True)
po.to_csv(PORT_CSV, index=False)

print(f"PD/LGD CSV rows now: {len(pddf)} | ING 2021 rows: {len(new_pd)}")
print(f"portfolio CSV rows now: {len(po)} | ING 2021 rows: {len(new_po)}")
print("ING 2021 PD/LGD:")
for cls in ["corporate","sme_corporate","mortgage","qrre","other_retail","bank"]:
    print(f"  {cls:14} {rows[cls][0]} / {rows[cls][1]}")
