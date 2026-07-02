# -*- coding: utf-8 -*-
"""Final ING EU-CR6 extractor. One class per page; columns idx3=EAD, idx4=PD,
idx6=LGD (consistent across corporate rows [12 tok] and retail rows [11 tok,
no maturity]). Class = first matching ING header line on the page."""
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

def page_class(t):
    """Identify the exposure class from the page (one class per CR6 page).
    Class names appear after the column-header block, so search full text."""
    head = t.lower()
    if "spec" in head and "lending" in head:
        return None
    if "corporates-other" in head:
        return "corporate"
    if "corporates-sme" in head:
        return "sme_corporate"
    if "secured by immovable" in head or "immovable property" in head:
        return "qrre" if "sme" in head and "non-sme" not in head else "mortgage"
    if "retail other" in head:
        return None if "non-sme" in head else "other_retail"
    if "institution" in head:
        return "bank"
    return None

def extract(path):
    found = {}
    pdf = pdfplumber.open(path)
    for i, pg in enumerate(pdf.pages):
        t = pg.extract_text() or ""
        # ING label drift: "Subtotal (exposure class)" (2024) vs "sub-total" (2021/22)
        m = re.search(r"(?im)^\s*sub-?total\b[^\n]*", t)
        if not m:
            continue
        n = nums(m.group(0))
        if len(n) < 9:
            continue
        cls = page_class(t)
        if cls and cls not in found:
            ead, pdv, lgd = n[3], n[4], n[6]
            if 0 <= pdv < 40 and 0 < lgd <= 100 and ead > 500:
                found[cls] = (pdv, lgd, ead, i + 1)
    pdf.close()
    return found

CLS = ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank"]
out = []
for y in sorted(FILES):
    path = DIR + FILES[y]
    if not os.path.exists(path):
        print(f"FY{y}: MISSING"); continue
    f = extract(path)
    print(f"\n=== ING FY{y} ({len(f)}/6) ===")
    for c in CLS:
        if c not in f:
            print(f"  {c:14} MISS"); continue
        pdv, lgd, ead, pg = f[c]
        tag = ""
        if y == 2024:
            k = KNOWN24[c]
            tag = "CALIB-OK" if abs(pdv - k[0]) < 0.05 and abs(lgd - k[1]) < 0.05 else f"FAIL(vs {k})"
        print(f"  {c:14} PD={pdv:6.2f} LGD={lgd:6.2f} ead={ead:>10,.0f} p{pg} {tag}")
        if y != 2024:
            out.append({"short": "ing", "LEI": "549300NYKK9MWM7GGW15", "vasicek_class": c,
                        "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgd:.2f}",
                        "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}",
                        "source_page": pg, "note": ""})
if out:
    with open("ing_timeseries_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWROTE ing_timeseries_rows.csv ({len(out)} rows)")
