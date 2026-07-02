# -*- coding: utf-8 -*-
"""DB Pillar-3 EU-CR6 AIRB extractor (TEMPLATE / Muster).

Selects the 'Dec 31, YYYY' AIRB block, maps the 7 model classes to their
EU-CR6 sub-total rows, reads PD (col e / idx4) and LGD (col g / idx6),
and runs an internal density cross-check (RWA/EAD vs reported density)
plus default-band detection. Prints a verifiable time series.

NOT committed yet — prototype to validate before productionising.
"""
import re, sys
import pdfplumber

sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
YEARS = [2021, 2022, 2023, 2024]

# hand-verified DB FY2024 values (from pillar3_bank_pd_lgd.csv) for validation
KNOWN_2024 = {
    "corporate": (4.12, 30.86), "sme_corporate": (6.77, 33.03),
    "mortgage": (2.63, 20.03), "qrre": (2.66, 57.60),
    "other_retail": (9.50, 58.41), "bank": (0.19, 48.15),
    "sovereign": (0.31, 66.37),
}

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def nums(s):
    out = []
    for t in NUM.findall(s):
        c = t.replace(",", "")
        try:
            out.append(float(c))
        except ValueError:
            pass
    return out

def find_block_start(pdf, year):
    """Page index of the 'EU CR6 - AIRB' table headed 'Dec 31, <year>'."""
    target = f"Dec 31, {year}"
    for i, pg in enumerate(pdf.pages):
        t = pg.extract_text() or ""
        if "CR6" in t and "AIRB" in t and target in t:
            return i
    return None

def extract_year(path, year):
    pdf = pdfplumber.open(path)
    start = find_block_start(pdf, year)
    if start is None:
        pdf.close()
        return None, f"no 'Dec 31, {year}' AIRB block found"
    # walk pages from start until next block ('Jun 30') or 'All exposure classes'
    section = None          # 'corporates' | 'retail'
    recent = []             # last few label lines
    found = {}
    raw = {}
    stop = False
    for i in range(start, min(start + 8, len(pdf.pages))):
        t = pdf.pages[i].extract_text() or ""
        for ln in t.split("\n"):
            s = ln.strip()
            if not s:
                continue
            # stop if we hit the comparative (Jun 30) block
            if re.search(r"Jun 30, \d{4}", s) and i > start:
                stop = True
                break
            low = s.lower()
            letters = re.sub(r"[^A-Za-z]", "", s)
            digits = re.sub(r"[^0-9]", "", s)
            is_label = len(letters) > 3 and len(digits) <= 3 and "sub-total" not in low and not low.startswith("total")
            if is_label:
                recent.append(s)
                recent = recent[-4:]
                if low == "corporates" or low.startswith("corporates"):
                    section = "corporates"
                if low == "retail" or (low.startswith("retail") and "secured" not in low):
                    section = "retail"
            if re.match(r"^(Sub-total|Subtotal|Total)\b", s):
                n = nums(s)
                if len(n) < 10:
                    recent = []
                    continue
                ctx = " | ".join(recent).lower()
                cls = classify(section, ctx)
                if cls and cls not in found:   # take FIRST (year-end) occurrence
                    found[cls] = n
                    raw[cls] = (i + 1, " | ".join(recent)[-60:], s[:120])
                # reset context: labels after a sub-total describe the NEXT row
                recent = []
        if stop:
            break
    pdf.close()
    return (found, raw), None

def classify(section, ctx):
    if "central bank" in ctx or "central government" in ctx:
        return "sovereign"
    if "institution" in ctx:
        return "bank"
    nonsme = "non-sme" in ctx
    sme = ("sme" in ctx) and not nonsme
    if "revolving" in ctx:
        return "qrre"
    if "real estate" in ctx or "immovable" in ctx or "property" in ctx:
        return "mortgage" if nonsme else None
    if "other retail" in ctx:
        return "other_retail" if nonsme else None
    if section == "corporates":
        if "specialized" in ctx or "specialised" in ctx:
            return None
        if "of which" in ctx or sme:
            return "sme_corporate"
        if "other" in ctx:
            return "corporate"
    return None

CLASSES = ["sovereign", "bank", "corporate", "sme_corporate",
           "mortgage", "qrre", "other_retail"]

print("=" * 78)
print("DEUTSCHE BANK · Pillar-3 EU-CR6 AIRB · PD/LGD time series (col e=PD, g=LGD)")
print("=" * 78)
series = {}
for y in YEARS:
    res, err = extract_year(f"{DIR}deutsche_bank_{y}.pdf", y)
    if err:
        print(f"FY{y}: ERROR {err}")
        continue
    found, raw = res
    series[y] = {}
    for c in CLASSES:
        if c not in found:
            series[y][c] = None
            continue
        n = found[c]
        ead, pdv, lgd, rwa, dens = n[3], n[4], n[6], n[8], n[9]
        dens_check = (rwa / ead * 100) if ead else 0
        flag = ""
        if abs(dens_check - dens) > 2.0:
            flag = "DENSITY?"
        # default-band signature: very high avg PD vs low density
        if pdv > 8 and dens < 40 and c in ("bank", "sovereign"):
            flag = (flag + " DEFAULT-BAND").strip()
        series[y][c] = (pdv, lgd, ead, dens, flag)

# print table
hdr = f"{'class':14} " + " ".join(f"FY{y}".rjust(16) for y in YEARS)
print(hdr); print("-" * len(hdr))
for c in CLASSES:
    cells = []
    for y in YEARS:
        v = series.get(y, {}).get(c)
        cells.append(f"{v[0]:5.2f}/{v[1]:5.2f}".rjust(16) if v else "—".rjust(16))
    print(f"{c:14} " + " ".join(cells))

print("\nValidation vs known FY2024:")
ok = True
for c in CLASSES:
    v = series.get(2024, {}).get(c)
    k = KNOWN_2024[c]
    if v and abs(v[0]-k[0]) < 0.05 and abs(v[1]-k[1]) < 0.05:
        print(f"  [OK]   {c:14} {v[0]:.2f}/{v[1]:.2f}  == csv {k[0]}/{k[1]}")
    else:
        ok = False
        got = f"{v[0]:.2f}/{v[1]:.2f}" if v else "MISSING"
        print(f"  [FAIL] {c:14} got {got}  vs csv {k[0]}/{k[1]}")
print("\nCALIBRATION", "PASSED" if ok else "FAILED", "- if passed, prior-year rows use identical logic.")

print("\nFlags / notes:")
for y in YEARS:
    for c in CLASSES:
        v = series.get(y, {}).get(c)
        if v and v[4]:
            print(f"  FY{y} {c}: {v[4]}  (PD {v[0]:.2f}, density {v[3]:.1f}%)")

# --- write CSV rows for FY2021-2023 (FY2024 already in repo CSV) ---
import csv as _csv
URLS = {
    2021: "https://investor-relations.db.com/files/documents/regulatory-reporting/Pillar_3_Report_as_of_December_31_2021.pdf",
    2022: "https://investor-relations.db.com/files/documents/regulatory-reporting/Pillar-3-Report-Q4-2022.pdf",
    2023: "https://investor-relations.db.com/files/documents/regulatory-reporting/Pillar-3-Report-Q4-2023.pdf",
}
# re-extract with raw provenance
out_rows = []
for y in (2021, 2022, 2023):
    res, err = extract_year(f"{DIR}deutsche_bank_{y}.pdf", y)
    if err:
        print("ERR", y, err); continue
    found, raw = res
    for c in CLASSES:
        if c not in found:
            continue
        n = found[c]
        pg, ctxtail, rawrow = raw[c]
        v = series[y][c]
        out_rows.append({
            "bank_name": "Deutsche Bank",
            "LEI": "7LTWFZYICNSX8D621K86",
            "bank_country": "DE",
            "vasicek_class": c,
            "pd_pct": f"{n[4]:.2f}",
            "lgd_pct": f"{n[6]:.2f}",
            "vintage_date": f"{y}-12-31",
            "status": "pillar3_verified",
            "source": f"Deutsche Bank Pillar 3 Report as of December 31, {y}",
            "source_period": f"{y}-12-31",
            "source_table": f"EU CR6 - AIRB approach (Dec 31, {y} block) - {ctxtail.strip()} Sub-total [col e=PD, g=LGD]",
            "source_page": pg,
            "source_url": URLS[y],
            "flag": (v[4] if v else ""),
        })
with open("db_timeseries_rows.csv", "w", newline="", encoding="utf-8") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)
print(f"\nWROTE db_timeseries_rows.csv with {len(out_rows)} rows (FY2021-2023 x 7 classes).")

# --- portfolio sidecar: EAD (col d, nums[3]) + maturity (col h, nums[7]) per class/vintage ---
# Frozen-portfolio weights for the walk-forward backtest. EAD reported in EUR m.
port_rows = []
for y in (2021, 2022, 2023, 2024):
    res, err = extract_year(f"{DIR}deutsche_bank_{y}.pdf", y)
    if err:
        print("PORT ERR", y, err); continue
    found, raw = res
    for c in CLASSES:
        if c not in found:
            continue
        n = found[c]
        port_rows.append({
            "LEI": "7LTWFZYICNSX8D621K86",
            "bank_name": "Deutsche Bank",
            "vasicek_class": c,
            "vintage_date": f"{y}-12-31",
            "ead_eur_m": f"{n[3]:.0f}",
            "maturity_years": f"{n[7]:.2f}",
            "rwa_eur_m": f"{n[8]:.0f}",
            "density_pct": f"{n[9]:.2f}",
            "source": f"Deutsche Bank Pillar 3 EU CR6 AIRB (Dec 31, {y}) col d=EAD, h=maturity, i=RWA",
        })
with open("db_portfolio_rows.csv", "w", newline="", encoding="utf-8") as fh:
    w = _csv.DictWriter(fh, fieldnames=list(port_rows[0].keys()))
    w.writeheader()
    w.writerows(port_rows)
print(f"WROTE db_portfolio_rows.csv with {len(port_rows)} rows (FY2021-2024 x 7 classes).")
