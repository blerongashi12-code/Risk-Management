# -*- coding: utf-8 -*-
"""Forensic ING EU-CR6 extraction with DENSITY-ANCHORED column location.

Per sub-total row, locate the EAD column k such that RWA(k+5)/EAD(k)*100 ≈
density(k+6) (standard EU-CR6 ITS order d,e,f,g,h,i,j = EAD,PD,obligors,LGD,
maturity,RWA,density). Then PD=n[k+1], LGD=n[k+3] — robust to column drift.
Only rows that pass the density check are emitted (never guessed).
Rows are mapped to classes by EAD-continuity to the FY2024 anchors + context.
"""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"
FILES = {2021: "ing-group-additional-pillar-iii-report-2021-.pdf",
         2022: "ing-group-additional-pillar-iii-report-2022.pdf",
         2024: "ing_2024.pdf"}
# FY2024 anchors (ead €m, pd, lgd) from the verified baseline
ANCH = {"corporate": (327826, 1.81, 14.43), "sme_corporate": (47644, 6.12, 27.39),
        "mortgage": (336546, 1.46, 22.61), "qrre": (14592, 3.75, 22.50),
        "other_retail": (4636, 6.39, 39.90), "bank": (47295, 0.39, 19.79)}
KW = {"corporate": ["corporate"], "sme_corporate": ["sme"],
      "mortgage": ["immovable", "real estate", "mortgage", "secured"],
      "qrre": ["revolving", "qrre", "qualifying"],
      "other_retail": ["other retail", "retail other"],
      "bank": ["institution"]}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def nums(s):
    out = []
    for t in NUM.findall(s):
        c = t.replace(",", "")
        try: out.append(float(c))
        except ValueError: pass
    return out

def locate_cols(n):
    """Return (k, ead, pd, lgd, rwa, dens) where density verifies; else None."""
    best, best_res = None, 1e9
    for k in range(0, len(n) - 6):
        ead, rwa, dens = n[k], n[k + 5], n[k + 6]
        pdv, lgdv = n[k + 1], n[k + 3]
        if ead <= 500 or not (0 <= pdv < 40) or not (0 < lgdv <= 100):
            continue
        if not (0 < dens <= 150) or rwa <= 0 or rwa > ead * 1.5:
            continue
        res = abs(rwa / ead * 100 - dens)
        if res < 1.5 and res < best_res:
            best_res, best = res, (k, ead, pdv, lgdv, rwa, dens)
    return best

def verified_rows(path):
    rows = []
    pdf = pdfplumber.open(path)
    for i, pg in enumerate(pdf.pages):
        t = pg.extract_text() or ""
        ctx = []
        for ln in t.split("\n"):
            s = ln.strip()
            if not s:
                continue
            letters = re.sub(r"[^A-Za-z]", "", s); digits = re.sub(r"[^0-9]", "", s)
            if len(letters) > 3 and len(digits) <= 4 and not re.match(r"^(Sub-?total|Total)\b", s, re.I):
                ctx.append(s.lower()); ctx = ctx[-3:]
            if re.match(r"^(Sub-?total|Subtotal|Total)\b", s, re.I):
                loc = locate_cols(nums(s))
                if loc:
                    rows.append((i + 1, " | ".join(ctx), *loc))
                ctx = []
    pdf.close()
    return rows

def map_class(cls, rows):
    """Pick the density-verified row best matching class by keyword + EAD-continuity."""
    anch_ead = ANCH[cls][0]
    kws = KW[cls]
    cands = []
    for (pg, ctx, k, ead, pdv, lgdv, rwa, dens) in rows:
        kw_hit = any(w in ctx for w in kws)
        excl = ("sme" in ctx and cls in ("corporate", "mortgage", "other_retail")
                and "sme" not in KW[cls])  # avoid SME variant for non-SME classes
        if cls == "mortgage" and "sme" in ctx:
            excl = True
        if cls == "other_retail" and "sme" in ctx:
            excl = True
        if not kw_hit or excl:
            continue
        ratio = ead / anch_ead if anch_ead else 0
        ead_ok = 0.5 <= ratio <= 2.0
        cands.append((abs(ead - anch_ead), ead_ok, pg, ctx, pdv, lgdv, ead, dens))
    cands = [c for c in cands if c[1]]    # require EAD continuity
    if not cands:
        return None
    cands.sort(key=lambda c: c[0])
    _, _, pg, ctx, pdv, lgdv, ead, dens = cands[0]
    return (pdv, lgdv, ead, dens, pg, ctx[:46])

out = []
for y in sorted(FILES):
    path = DIR + FILES[y]
    if not os.path.exists(path):
        print(f"FY{y}: MISSING"); continue
    rows = verified_rows(path)
    print(f"\n=== ING FY{y}: {len(rows)} density-verified sub-total rows ===")
    for cls in ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank"]:
        m = map_class(cls, rows)
        if m is None:
            print(f"  {cls:14} MISS"); continue
        pdv, lgdv, ead, dens, pg, ctx = m
        a = ANCH[cls]
        tag = "CALIB-OK" if y == 2024 and abs(pdv - a[1]) < 0.05 and abs(lgdv - a[2]) < 0.05 else ""
        print(f"  {cls:14} PD={pdv:6.2f} LGD={lgdv:6.2f} ead={ead:>10,.0f} dens={dens:5.1f} p{pg} {tag} '{ctx}'")
        if y != 2024:
            out.append({"short": "ing", "LEI": "549300NYKK9MWM7GGW15", "vasicek_class": cls,
                        "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgdv:.2f}",
                        "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}",
                        "source_page": pg, "note": ""})
if out:
    with open("ing_timeseries_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWROTE ing_timeseries_rows.csv ({len(out)} rows)")
