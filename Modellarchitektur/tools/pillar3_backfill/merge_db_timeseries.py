# -*- coding: utf-8 -*-
"""Merge DB FY2021-2023 time-series rows into data/pillar3_bank_pd_lgd.csv.
Keeps existing 70 rows (2024-12-31) byte-identical in their values; adds a
'note' column (empty for existing) and 21 DB historical rows."""
import pandas as pd

TARGET = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_bank_pd_lgd.csv"
DBROWS = "C:/Users/blero/Downloads/db_timeseries_rows.csv"

CLASS_LABEL = {
    "sovereign":     "Central governments and central banks",
    "bank":          "Institutions",
    "corporate":     "Corporates - Other (non-SME)",
    "sme_corporate": "Corporates - of which SMEs",
    "mortgage":      "Retail - Secured by real estate, non-SME",
    "qrre":          "Retail - Qualifying Revolving (QRRE)",
    "other_retail":  "Retail - Other, non-SME",
}

existing = pd.read_csv(TARGET, dtype=str)
if "note" not in existing.columns:
    existing["note"] = ""
existing["note"] = existing["note"].fillna("")

db = pd.read_csv(DBROWS, dtype=str)
# clean source_table to canonical label per class
def clean_table(r):
    y = r["vintage_date"][:4]
    return (f"EU CR6 - AIRB approach (Dec 31, {y}) - "
            f"{CLASS_LABEL[r['vasicek_class']]} Sub-total "
            f"[EAD-weighted; col e=PD, g=LGD]")
db["source_table"] = db.apply(clean_table, axis=1)
db = db.rename(columns={"flag": "note"})
db["note"] = db["note"].fillna("")

# align columns to existing schema
cols = list(existing.columns)
for c in cols:
    if c not in db.columns:
        db[c] = ""
db = db[cols]

merged = pd.concat([existing, db], ignore_index=True)
# sort: bank, then vintage ascending, then a stable class order
class_order = {c: i for i, c in enumerate(
    ["corporate","sme_corporate","mortgage","qrre","other_retail","bank","sovereign"])}
merged["_co"] = merged["vasicek_class"].map(class_order).fillna(99)
merged = merged.sort_values(["bank_name", "vintage_date", "_co"]).drop(columns="_co")
merged.to_csv(TARGET, index=False)

print(f"Existing rows: {len(existing)} | DB historical added: {len(db)} | total: {len(merged)}")
print("Vintages:", sorted(merged["vintage_date"].unique()))
print("DB rows per vintage:",
      merged[merged.LEI=="7LTWFZYICNSX8D621K86"].groupby("vintage_date").size().to_dict())
dup = merged.duplicated(subset=["LEI","vasicek_class","vintage_date"]).sum()
print("duplicate (LEI,class,vintage):", dup)
