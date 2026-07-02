# -*- coding: utf-8 -*-
"""ING FY2022 EU-CR6 extractor. The 2022 'Additional Pillar III' report drops the
'sub-total' label: each one-class-per-page CR6 table ends with an UNLABELLED
class-total row (just the numbers). Detect it as the line with >=10 numeric
tokens that is NOT a PD-band row ('X to <Y' / '(Default)'). Columns as 2021/2024:
idx3=EAD, idx4=PD, idx6=LGD, idx8=RWA, idx9=density(fraction)."""
import re, sys, os, logging, csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
YEAR = sys.argv[1] if len(sys.argv) > 1 else "2022"
FILES = {"2021": "ing-group-additional-pillar-iii-report-2021-.pdf",
         "2022": "ing-group-additional-pillar-iii-report-2022.pdf",
         "2023": "ing-group-additional-pillar-iii-report-2023.pdf",
         "2024": "ing_2024.pdf"}
PDF = DIR + FILES[YEAR]
OUT = f"C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/backtest_raw/ing_{YEAR}_rows.csv"
LEI = "549300NYKK9MWM7GGW15"
# 2021 and 2024 reference (PD,LGD) to sanity-bound 2022
REF = {"corporate": (1.80, 16.0), "sme_corporate": (5.7, 26.0), "mortgage": (1.4, 22.0),
       "qrre": (4.0, 20.0), "other_retail": (6.5, 45.0), "bank": (0.65, 21.0)}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
PDBAND = re.compile(r"^\s*\d[\d.]*\s+to\s+<|\(Default\)|^\s*\d{1,3}\.\d{2}\s+to")

def nums(s):
    out = []
    for t in NUM.findall(s):
        try: out.append(float(t.replace(",", "")))
        except ValueError: pass
    return out

def page_class(t):
    h = t.lower()
    if "spec" in h and "lending" in h:
        return None
    if "corporates - other" in h or "corporates-other" in h:
        return "corporate"
    if "corporates - sme" in h or "corporates-sme" in h:
        return "sme_corporate"
    if "secured by immovable" in h or "immovable property" in h:
        return "qrre" if ("sme" in h and "non-sme" not in h) else "mortgage"
    if "retail other" in h or "other retail" in h:
        return None if "non-sme" in h else "other_retail"
    if "institution" in h:
        return "bank"
    return None

def total_row(t):
    """Return the unlabelled class-total row's numbers (the non-PD-band line with
    the most numeric tokens)."""
    best = None
    for ln in t.splitlines():
        s = ln.strip()
        if not s or PDBAND.search(s):
            continue
        n = nums(s)
        # total row: >=10 numbers, first value is a sizeable amount (on-balance EAD)
        if len(n) >= 10 and n[0] > 200:
            # prefer the LAST such row on the page (the aggregation)
            best = n
    return best

def extract():
    rows = {}
    with pdfplumber.open(PDF) as pdf:
        for i, pg in enumerate(pdf.pages):
            t = pg.extract_text() or ""
            if "EU CR6" not in t or "PD range" not in t and "PD Scale" not in t:
                continue
            cls = page_class(t)
            if not cls or cls in rows:
                continue
            n = total_row(t)
            if not n:
                continue
            # corporate/institutions rows carry a maturity column (small int 1-5)
            # at idx7 -> RWA=idx8, dens=idx9. Retail rows have no maturity -> shift.
            mat = n[7] < 20
            ead, pd, lgd = n[3], n[4], n[6]
            rwa = n[8] if mat else n[7]
            dens = n[9] if mat else n[8]
            rows[cls] = dict(page=i + 1, ead=ead, pd=pd, lgd=lgd, rwa=rwa, dens=dens)
    return rows

if __name__ == "__main__":
    rows = extract()
    print(f"== ING FY{YEAR} ({len(rows)}/6) ==")
    good = []
    for c in ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank"]:
        if c not in rows:
            print(f"  {c:14} MISS"); continue
        r = rows[c]
        # density cross-check (fraction): RWA/EAD vs idx9
        dchk = r["rwa"] / r["ead"] if r["ead"] else 0
        densflag = "" if abs(dchk - r["dens"]) < 0.03 else f" DENS? {dchk:.3f} vs {r['dens']}"
        # plausibility vs 2021/2024 neighbourhood
        ref = REF[c]
        pf = "" if (0 <= r["pd"] < 40 and abs(r["lgd"] - ref[1]) < 18) else " PLAUS?"
        print(f"  {c:14} PD={r['pd']:6.2f} LGD={r['lgd']:6.2f} EAD={r['ead']:>10,.0f} "
              f"RWA={r['rwa']:>9,.0f} dens={r['dens']:.3f} p{r['page']}{densflag}{pf}")
        good.append((c, r))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEI", "bank_name", "vasicek_class", "vintage_date", "pd_pct", "lgd_pct",
                    "ead_eur_m", "rwa_eur_m", "density", "source", "page"])
        for c, r in good:
            w.writerow([LEI, "ING Groep", c, f"{YEAR}-12-31", f"{r['pd']:.2f}", f"{r['lgd']:.2f}",
                        f"{r['ead']:.0f}", f"{r['rwa']:.0f}", f"{r['dens']:.4f}",
                        f"ING Additional Pillar III {YEAR} EU CR6 IRB (raw class-total)", r["page"]])
    print(f"wrote {OUT} ({len(good)} rows)")
