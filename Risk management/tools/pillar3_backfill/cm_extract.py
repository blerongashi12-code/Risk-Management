# -*- coding: utf-8 -*-
"""Credit Mutuel raw EU-CR6 A-IRB extractor (FRENCH number format).
FR numbers: space thousands ('83 159'), comma decimal ('5,69%'). Rows start
'Sous-total'. Columns idx3=EAD, idx4=PD, idx6=LGD. FY2024 value-matches the
curated baseline (=raw); FY2023 by EAD-continuity to the 2024 anchor."""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"
FILES = {2023: "cm_2023.pdf", 2024: "cm_2024.pdf"}
KN = {"corporate": (2.59, 45.00), "sme_corporate": (5.69, 25.00),
      "mortgage": (1.79, 16.00), "qrre": (3.13, 33.00),
      "other_retail": (2.59, 19.00), "bank": (0.12, 34.00),
      "sovereign": (0.12, 34.00)}
FRNUM = re.compile(r"\d{1,3}(?:\s\d{3})*(?:,\d+)?")

def nums(s):
    out = []
    for m in FRNUM.findall(s):
        v = m.replace(" ", "").replace(",", ".")
        try: out.append(float(v))
        except ValueError: pass
    return out

def subrows(path):
    rows = []
    pdf = pdfplumber.open(path)
    for i, pg in enumerate(pdf.pages):
        for ln in (pg.extract_text() or "").split("\n"):
            s = ln.strip()
            if re.match(r"^Sous-?total", s, re.I):
                n = nums(s)
                if len(n) >= 10:        # main A-IRB block has 12 cols
                    rows.append((i + 1, n))
    pdf.close()
    return rows

CLS = ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank", "sovereign"]
out = []
anchors = {}
for y in sorted(FILES):
    path = DIR + FILES[y]
    if not os.path.exists(path):
        print(f"FY{y}: MISSING"); continue
    rows = subrows(path)
    print(f"\n=== Credit Mutuel FY{y} ({len(rows)} sous-total rows, >=10 cols) ===")
    for c in CLS:
        pdk, lgk = KN[c]
        pick = None
        if y == 2024:
            for (pg, n) in rows:
                if abs(n[4] - pdk) < 0.05 and abs(n[6] - lgk) < 0.05:
                    pick = (pg, n); break
            if pick:
                anchors[c] = pick[1][3]
        else:
            anc = anchors.get(c)
            cands = [(pg, n) for (pg, n) in rows
                     if abs(n[6] - lgk) < 1.5 and (not anc or 0.5 <= n[3] / anc <= 2.0)]
            if anc:
                cands.sort(key=lambda x: abs(x[1][3] - anc))
            pick = cands[0] if cands else None
        if not pick:
            print(f"  {c:14} MISS"); continue
        pg, n = pick
        ead, pdv, lgd, rwa, dens = n[3], n[4], n[6], n[8], n[9]
        dok = abs(rwa / ead * 100 - dens) < 3 if ead else False
        tag = "RAW-OK" if y == 2024 else ""
        print(f"  {c:14} PD={pdv:6.2f} LGD={lgd:6.2f} ead={ead:>9,.0f} p{pg} {tag}{'' if dok else ' DENS?'}")
        out.append({"short": "cm", "LEI": "9695000CG7B84NLR5984", "vasicek_class": c,
                    "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgd:.2f}",
                    "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}", "note": ""})
if out:
    with open("cm_raw_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWROTE cm_raw_rows.csv ({len(out)} rows)")
