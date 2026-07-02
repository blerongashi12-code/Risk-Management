# -*- coding: utf-8 -*-
"""Precise ING EU-CR6 extractor (forensic, layout-verified).
ING: one exposure class per page; "Subtotal (exposure class)" row;
columns idx3=EAD, idx4=PD, idx6=LGD, idx8=RWA, idx9=density (DECIMAL fraction).
Class label = line after each "EU CR6 - IRB approach" title.
Density verified as fraction OR percent. FY2024 validated against knowns.
"""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
FILES = {2021: "ing-group-additional-pillar-iii-report-2021-.pdf",
         2022: "ing-group-additional-pillar-iii-report-2022.pdf",
         2024: "ing_2024.pdf"}
KNOWN24 = {"corporate": (1.81, 14.43), "sme_corporate": (6.12, 27.39),
           "mortgage": (1.46, 22.61), "qrre": (3.75, 22.50),
           "other_retail": (6.39, 39.90), "bank": (0.39, 19.79)}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def nums(s):
    out = []
    for t in NUM.findall(s):
        c = t.replace(",", "")
        try: out.append(float(c))
        except ValueError: pass
    return out

def ing_class(label):
    l = label.lower()
    if "spec" in l and "lending" in l:
        return None
    if "corporates-other" in l or ("corporate" in l and "other" in l):
        return "corporate"
    if "corporates-sme" in l or ("corporate" in l and "sme" in l):
        return "sme_corporate"
    if "immovable" in l or "secured by" in l:
        if "non-sme" in l:
            return "mortgage"
        if "sme" in l:
            return "qrre"           # ING proxy: SRE-SME
        return "mortgage"
    if "retail other" in l or "other retail" in l:
        if "non-sme" in l:
            return None             # not used by baseline
        return "other_retail"       # ING baseline uses "Retail Other SME"
    if "institution" in l:
        return "bank"
    return None

def extract(path):
    """Return {class: (pd, lgd, ead, rwa, dens, page)} (first occurrence)."""
    found = {}
    pdf = pdfplumber.open(path)
    cur_label = None
    for i, pg in enumerate(pdf.pages):
        t = pg.extract_text() or ""
        lines = t.split("\n")
        for j, ln in enumerate(lines):
            s = ln.strip()
            if "EU CR6" in s and "IRB" in s:
                # class label = next non-empty line that isn't another title/column header
                for k in range(j + 1, min(j + 4, len(lines))):
                    nx = lines[k].strip()
                    if nx and "EU CR6" not in nx and not nx[0].isdigit():
                        cur_label = nx
                        break
            if re.match(r"^Subtotal \(exposure class\)", s, re.I):
                cls = ing_class(cur_label or "")
                n = nums(s)
                if cls and cls not in found and len(n) >= 10:
                    ead, pdv, lgd, rwa, dens = n[3], n[4], n[6], n[8], n[9]
                    found[cls] = (pdv, lgd, ead, rwa, dens, i + 1, cur_label)
    pdf.close()
    return found

def dens_ok(ead, rwa, dens):
    if ead <= 0:
        return False
    return abs(rwa / ead * 100 - dens) < 2.0 or abs(rwa / ead - dens) < 0.02

CLS = ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank"]
out = []
for y in sorted(FILES):
    path = DIR + FILES[y]
    if not os.path.exists(path):
        print(f"FY{y}: MISSING"); continue
    f = extract(path)
    print(f"\n=== ING FY{y} ===")
    for c in CLS:
        if c not in f:
            print(f"  {c:14} MISS"); continue
        pdv, lgd, ead, rwa, dens, pg, lab = f[c]
        ok = dens_ok(ead, rwa, dens)
        tag = ""
        if y == 2024:
            k = KNOWN24[c]
            tag = "CALIB-OK" if abs(pdv - k[0]) < 0.05 and abs(lgd - k[1]) < 0.05 else f"CALIB-FAIL(vs {k})"
        flag = "" if ok else "DENSITY?"
        print(f"  {c:14} PD={pdv:6.2f} LGD={lgd:6.2f} ead={ead:>10,.0f} dens={dens:6.3f} p{pg} {tag}{flag} '{lab[:34]}'")
        if y != 2024 and ok:
            out.append({"short": "ing", "LEI": "549300NYKK9MWM7GGW15", "vasicek_class": c,
                        "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgd:.2f}",
                        "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}",
                        "source_page": pg, "note": ""})
if out:
    with open("ing_timeseries_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWROTE ing_timeseries_rows.csv ({len(out)} rows)")
