#!/usr/bin/env python3
"""Rabobank EU-CR6 A-IRB extractor (dedicated 'Pillar 3 Year Report', text-based;
the annual-report version is image-only). One PD-band row per line, prefixed by a
row number; each class block ends with a 'Subtotal <class>' row whose figures are
on that line OR on the wrapped next line. With the row-number prefix the value
layout is: [row#, on, off, CCF, EAD(idx4), PD(idx5), obligors(idx6), LGD(idx7),
M(idx8), RWA(idx9), density(idx10), ...].

Class map (A-IRB): Central Governments->sovereign, Institutions->bank,
Corporates - Other->corporate, Corporates - SME->sme_corporate,
Retail - Secured ... non-SME->mortgage, Retail - Other non-SME->other_retail.
Skipped (no canonical slot): Corporates - Specialised Lending, Retail - Other SME,
Retail - Secured ... SME (=mortgage_sme proxy, optional). No QRRE (Rabobank has none).
Verification: density cross-check (RWA/EAD vs idx10) + EAD-continuity across years."""
import sys, re, csv, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/"
OUT = DIR + "backtest_raw/rabobank_raw_rows.csv"
LEI = "DG3RU1DBUFHT4ZF9WN62"   # Cooperatieve Rabobank U.A.
FILES = {"2020": "rabobank_2020.pdf", "2021": "rabobank_2021.pdf",
         "2022": "rabobank_2022.pdf", "2024": "rabobank_2024.pdf"}

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
SUB = re.compile(r"Subtotal\s+(.*)", re.I)
AIRB = re.compile(r"EU\s*CR6\s*[–-]\s*AIRB", re.I)
STOP = re.compile(r"Total\s*A-?IRB", re.I)

def nums(s):
    out = []
    for t in NUM.findall(s):
        try: out.append(float(t.replace(",", "")))
        except ValueError: pass
    return out

def pick_layout(v):
    """Layout drifts two ways: (a) some reports prefix each row with a row number
    (EAD=idx4) vs not (EAD=idx3); (b) corporate/sovereign rows carry a maturity
    column but retail rows do not (RWA/density shift by one). EAD/PD/LGD sit at the
    same relative spots; only RWA/density move. Try all combinations and keep the
    one whose RWA/EAD reconciles with the reported density (cross-check)."""
    for base in (1, 0):                      # base=1: row# present; base=0: none
        if len(v) < base + 9:
            continue
        ead, pd, lgd = v[base+3], v[base+4], v[base+6]
        if not (ead > 100 and 0 < pd < 40 and 0 < lgd <= 100):
            continue
        for rwa_i in (base + 7, base + 8):   # no-maturity / maturity present
            if rwa_i + 1 >= len(v):
                continue
            rwa, dens = v[rwa_i], v[rwa_i + 1]
            if dens >= 0 and abs(rwa / ead * 100 - dens) < 1.0:
                return ead, pd, lgd, rwa, dens
    return None

def classify(label):
    l = label.lower().replace("- ", "-").replace(" -", "-")
    if "central government" in l:
        return "sovereign"
    if "institution" in l:
        return "bank"
    if "specialised" in l or "specialized" in l:
        return None
    if "corporates" in l and "sme" in l:
        return "sme_corporate"
    if "corporates" in l and "other" in l:
        return "corporate"
    if "secured by immovable" in l or "secured" in l:
        if "non-sme" in l:
            return "mortgage"
        return None              # secured SME = mortgage_sme proxy -> skip (no clean slot)
    if "other" in l and "non-sme" in l:
        return "other_retail"
    return None

def extract_year(path):
    rows = {}
    with pdfplumber.open(path) as pdf:
        in_airb = False
        for i, pg in enumerate(pdf.pages):
            t = pg.extract_text() or ""
            if AIRB.search(t):
                in_airb = True
            if not in_airb:
                continue
            lines = t.splitlines()
            for j, ln in enumerate(lines):
                if STOP.search(ln):
                    in_airb = False
                    break
                m = SUB.search(ln)
                if not m:
                    continue
                after = m.group(1)
                n = nums(ln)
                if len(n) >= 10:                 # figures on the Subtotal line
                    label = after
                    vals = n
                else:                            # wrapped: tail + figures on next line
                    nxt = lines[j + 1] if j + 1 < len(lines) else ""
                    label = (after + " " + re.sub(r"^\s*\d+\s*", "", nxt)).strip()
                    vals = nums(nxt)
                got = pick_layout(vals)
                if not got:
                    continue
                cls = classify(label)
                if not cls or cls in rows:
                    continue
                ead, pd, lgd, rwa, dens = got
                rows[cls] = dict(page=i + 1, ead=ead, pd=pd, lgd=lgd, rwa=rwa, dens=dens,
                                 label=label[:40])
    return rows

if __name__ == "__main__":
    CLS = ["sovereign", "bank", "corporate", "sme_corporate", "mortgage", "other_retail"]
    out = []
    for yr, fn in FILES.items():
        rows = extract_year(DIR + "pillar3_reports/" + fn)
        print(f"\n== Rabobank FY{yr} ({len(rows)}/6) ==")
        for c in CLS:
            if c not in rows:
                print(f"  {c:14} MISS"); continue
            r = rows[c]
            dchk = r["rwa"] / r["ead"] if r["ead"] else 0
            flag = "" if abs(dchk - r["dens"] / 100) < 0.03 else f" DENS? {dchk*100:.2f} vs {r['dens']}"
            ok = 0 <= r["pd"] < 40 and 0 < r["lgd"] <= 100 and r["ead"] > 100
            print(f"  {c:14} PD={r['pd']:6.2f} LGD={r['lgd']:6.2f} EAD={r['ead']:>10,.0f} "
                  f"RWA={r['rwa']:>9,.0f} dens={r['dens']:.2f} p{r['page']} '{r['label']}'{flag}{'' if ok else ' IMPLAUS'}")
            if ok:
                out.append([LEI, "Rabobank", c, f"{yr}-12-31", f"{r['pd']:.2f}", f"{r['lgd']:.2f}",
                            f"{r['ead']:.0f}", f"{r['rwa']:.0f}", f"{r['dens']:.3f}",
                            f"Rabobank Pillar 3 Year Report {yr} EU CR6 AIRB (raw subtotal)", r["page"]])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEI", "bank_name", "vasicek_class", "vintage_date", "pd_pct", "lgd_pct",
                    "ead_eur_m", "rwa_eur_m", "density", "source", "page"])
        w.writerows(out)
    print(f"\nwrote {OUT} ({len(out)} rows)")
