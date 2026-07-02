#!/usr/bin/env python3
"""BPCE prior-year (2022/2023) extractor for the RELIABLY-identifiable A-IRB classes.

BPCE's EU-CR6 IRB table repeats classes across 4 sub-blocks (A-IRB current,
F-IRB current, A-IRB prior, F-IRB prior). Earlier blind value-match grabbed the
wrong block. Here we restrict to BLOCK 1 (the A-IRB current-year section, before
the 2nd 'Central governments' header) and only take the 4 classes whose signature
is distinctive enough to map without ambiguity:
  sovereign (low LGD), bank ('Institutions'), mortgage (LGD~10.7 + EAD>200k),
  other_retail (EAD ~40-50k, LGD ~24-29, PD>10).
corporate (PD discontinuity), sme, qrre (multi-subclass collisions) are left as
documented gaps. Each kept row is density-checked (RWA/EAD vs reported density).
Columns: idx3=EAD, idx4=PD, idx6=LGD, idx8=RWA, idx9=density(%)."""
import sys, re, csv, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/"
OUT = DIR + "backtest_raw/bpce_prior_rows.csv"
LEI = "FR9695005MSX1OYEMGDF"
JOBS = {"2022": "bpce_2022.pdf", "2023": "bpce_2023.pdf"}

def nums_pos(s):
    out = []
    for tok in s.split():
        t = tok.replace(",", "").replace("%", "").replace("(", "-").replace(")", "")
        if t in ("-", "–", "—", ""):
            out.append(0.0); continue
        try: out.append(float(t))
        except ValueError: pass
    return out

def block1_subtotals(path):
    """All subtotals in the FIRST 'Central governments' block of the EU-CR6 IRB table."""
    rows = []
    with pdfplumber.open(path) as pdf:
        start = None
        for i in range(len(pdf.pages)):
            if re.search(r"EU CR6.*IRB", pdf.pages[i].extract_text() or "", re.I):
                start = i; break
        if start is None:
            return rows
        cg = 0
        for i in range(start, start + 8):
            for ln in (pdf.pages[i].extract_text() or "").splitlines():
                s = ln.strip()
                if re.search(r"Central governments", s, re.I) and len(re.sub(r"[0-9,.% ]", "", s)) > 8:
                    cg += 1
                if cg >= 2:
                    return rows
                m = re.search(r"sub-?\s?total", s, re.I)
                if m:
                    n = nums_pos(s[m.end():])
                    if len(n) >= 10:
                        rows.append((s[:m.start()].lower(), n))
    return rows

def dens_ok(n):
    ead, rwa, dens = n[3], n[8], n[9]
    return ead > 0 and abs(rwa / ead * 100 - dens) < 3

# 2024 anchors (distinctive classes only)
ANCH = {"sovereign": (0.00, 7.26), "bank": (0.41, 40.42),
        "mortgage": (14.68, 10.70), "other_retail": (21.55, 28.88)}

def pick(rows, cls):
    pa, la = ANCH[cls]
    cands = []
    for (lab, n) in rows:
        ead, pd, lgd = n[3], n[4], n[6]
        if ead < 1000 or not dens_ok(n):
            continue
        if cls == "sovereign" and lgd < 15 and pd < 2 and ead > 30000:
            cands.append((abs(lgd - la), lab, n))
        elif cls == "bank" and ("institution" in lab or 33 < lgd < 42) and ead < 15000:
            cands.append((abs(lgd - la), lab, n))
        elif cls == "mortgage" and lgd < 13 and ead > 200000:
            cands.append((abs(lgd - la), lab, n))
        elif cls == "other_retail" and 23 < lgd < 30 and 30000 < ead < 60000 and pd > 10:
            cands.append((abs(lgd - la) + abs(pd - pa) * 0.3, lab, n))
    cands.sort()
    return cands[0] if cands else None

if __name__ == "__main__":
    out = []
    for yr, fn in JOBS.items():
        rows = block1_subtotals(DIR + "pillar3_reports/" + fn)
        print(f"\n== BPCE FY{yr} block-1 ({len(rows)} subtotals) ==")
        for cls in ("sovereign", "bank", "mortgage", "other_retail"):
            p = pick(rows, cls)
            if not p:
                print(f"  {cls:14} MISS"); continue
            _, lab, n = p
            ead, pd, lgd, rwa, dens = n[3], n[4], n[6], n[8], n[9]
            print(f"  {cls:14} PD={pd:6.2f} LGD={lgd:6.2f} EAD={ead:>9,.0f} RWA={rwa:>9,.0f} dens={dens:.1f} (RWA/EAD={rwa/ead*100:.1f})")
            out.append([LEI, "Groupe BPCE", cls, f"{yr}-12-31", f"{pd:.2f}", f"{lgd:.2f}",
                        f"{ead:.0f}", f"{rwa:.0f}", f"{dens:.3f}",
                        f"BPCE Pillar III {yr} EU CR6 A-IRB block-1 (distinctive class, density-verified)", 0])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEI", "bank_name", "vasicek_class", "vintage_date", "pd_pct", "lgd_pct",
                    "ead_eur_m", "rwa_eur_m", "density", "source", "page"])
        w.writerows(out)
    print(f"\nwrote {OUT} ({len(out)} rows)")
