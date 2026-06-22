# -*- coding: utf-8 -*-
"""UniCredit raw EU-CR6 A-IRB extractor (flowing multi-class table, DB-style).
Track class header across pages; at each 'Subtotal' assign the current class
(first occurrence in the year-end A-IRB block). Columns idx3=EAD, idx4=PD,
idx6=LGD, idx9=density(%). Stop at the first 'Total (all exposures classes)'.
Verifies FY2024 against the mapped raw values; extracts 2021-2024 for the
consistent raw backtest series (within-bank continuity = verification)."""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"
FILES = {2021: "unicredit_2021.pdf", 2022: "unicredit_2022.pdf",
         2023: "unicredit_2023.pdf", 2024: "unicredit_2024_official.pdf"}
# mapped raw FY2024 values (from layout dumps) for validation
RAW24 = {"sovereign": (0.22, 20.75), "bank": (0.11, 25.90),
         "sme_corporate": (6.82, 23.14), "corporate": (3.25, 31.89),
         "mortgage": (2.66, 21.15), "qrre": (3.67, 70.86),
         "other_retail": (4.90, 39.02)}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def nums(s):
    out = []
    for t in NUM.findall(s):
        c = t.replace(",", "")
        try: out.append(float(c))
        except ValueError: pass
    return out

def header_class(s):
    l = s.lower()
    if "central government" in l or "central gov" in l:
        return "sovereign"
    if "institution" in l:
        return "bank"
    if "specialised" in l or "specialized" in l:
        return None        # skip marker
    if "corporate" in l and "sme" in l:
        return "sme_corporate"
    if "corporate" in l and "other" in l:
        return "corporate"
    if "secured by real estate" in l or "secured by immovable" in l or "real estate" in l:
        return "mortgage"
    if "qualifying revolving" in l or "revolving" in l:
        return "qrre"
    if "other" in l and "non sme" in l.replace("-", " "):
        return "other_retail"
    return "KEEP"          # not a class header -> keep current

def extract(path):
    found = {}
    pdf = pdfplumber.open(path)
    cur = None
    stop = False
    in_block = False     # only record once we've entered the A-IRB block at "Central governments"
    for pg in pdf.pages:
        if stop:
            break
        t = pg.extract_text() or ""
        for ln in t.split("\n"):
            s = ln.strip()
            if not s:
                continue
            if in_block and re.search(r"Total \(all exposures? classes\)", s, re.I):
                stop = True
                break
            letters = re.sub(r"[^A-Za-z]", "", s); digits = re.sub(r"[^0-9]", "", s)
            if len(letters) > 5 and len(digits) <= 2 and "subtotal" not in s.lower():
                hc = header_class(s)
                if hc == "sovereign" and not in_block:
                    in_block = True       # A-IRB block starts at Central governments
                    cur = "sovereign"
                elif in_block and hc != "KEEP":
                    cur = hc
            if in_block and re.match(r"^Subtotal\b", s, re.I):
                n = nums(s)
                if cur and cur not in found and len(n) >= 9:
                    ead, pdv, lgd = n[3], n[4], n[6]
                    rwa = n[8] if len(n) > 8 else None
                    dens = n[9] if len(n) > 9 else None
                    if 0 <= pdv < 40 and 0 < lgd <= 100 and ead > 200:
                        dok = (rwa and dens and abs(rwa / ead * 100 - dens) < 2.0)
                        found[cur] = (pdv, lgd, ead, dok)
    pdf.close()
    return found

CLS = ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank", "sovereign"]
out = []
for y in sorted(FILES):
    path = DIR + FILES[y]
    if not os.path.exists(path):
        print(f"FY{y}: MISSING"); continue
    f = extract(path)
    print(f"\n=== UniCredit FY{y} ({len(f)}/7) ===")
    for c in CLS:
        if c not in f:
            print(f"  {c:14} MISS"); continue
        pdv, lgd, ead, dok = f[c]
        tag = ""
        if y == 2024:
            k = RAW24[c]
            tag = "RAW-OK" if abs(pdv - k[0]) < 0.05 and abs(lgd - k[1]) < 0.05 else f"DIFF(vs {k})"
        flag = "" if dok else "DENS?"
        print(f"  {c:14} PD={pdv:6.2f} LGD={lgd:6.2f} ead={ead:>10,.0f} {tag}{flag}")
        out.append({"short": "unicredit", "LEI": "549300TRUWO2CD2G5692", "vasicek_class": c,
                    "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgd:.2f}",
                    "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}", "note": flag})
if out:
    with open("unicredit_raw_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWROTE unicredit_raw_rows.csv ({len(out)} rows)")
