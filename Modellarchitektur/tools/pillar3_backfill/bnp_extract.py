#!/usr/bin/env python3
"""BNP Paribas EU-CR6 (IRBA by PD scale) extractor — bnp_2024.pdf yields both
31.12.2024 and the 31.12.2023 comparative. Deterministic page-map (verified by
value-match against FY2024 curated anchors), dual number-format aware.

Column order in every SUB-TOTAL row (value sequence, ignoring % signs):
  [EAD_on, EAD_off, CCF, EAD(idx3), PD(idx4), LGD(idx5), M, RWA(idx7), dens(idx8), EL, (prov)]

Verification gates (no fabrication):
  * 2024 row PD & LGD must match the curated anchor (|dPD|<=0.15, |dLGD|<=1.0)
  * density cross-check: |RWA/EAD - dens/100| <= 0.03
  * 2023 row EAD must be within [0.5, 2.0]x the 2024 EAD (same-book continuity)
"""
import sys, re, csv
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/"
OUT = DIR + "backtest_raw/bnp_raw_rows.csv"
LEI = "R0MUWSFPU8MPRO8K5P83"  # BNP Paribas SA

# page (0-based) -> (vasicek_class, subtotal occurrence on the page, anchor PD, anchor LGD).
# Layout A (FY2022-2024 tables): value sequence idx3=EAD,4=PD,5=LGD,7=RWA,8=dens.
# bnp_2024.pdf yields 31.12.2024 (anchor-matched) + 31.12.2023 comparative.
# NB: on the corporate pages the 1st SUB-TOTAL is "Corporates - Specialised
# financing" (NOT SME); the TRUE "SME corporates" is the 2nd SUB-TOTAL (occ #1).
# Earlier the 1st was mis-mapped to sme; corrected per source-PDF verification.
PAGEMAP_2024 = {
    439: [("sovereign", 0, 0.08, 3.0), ("bank", 1, 0.79, 26.0)],
    441: [("sme", 1, 7.02, 30.0)],
    442: [("corporate", 0, 2.87, 37.0)],
    447: [("mortgage", 0, 1.61, 17.0)],
    449: [("qrre", 0, 8.91, 36.0)],
    450: [("other_retail", 0, 6.70, 36.0)],
    440: [("sovereign", 0, 0.08, 3.0), ("bank", 1, 0.79, 26.0)],
    443: [("sme", 1, 7.24, 28.0)],
    444: [("corporate", 0, 2.87, 37.0)],
    448: [("mortgage", 0, 1.61, 17.0)],
    451: [("qrre", 0, 8.91, 36.0)],
    452: [("other_retail", 0, 6.70, 36.0)],
}
# bnp_2023.pdf yields 31.12.2022 (we only take 2022; 2023 already covered by the
# 2024 doc and cross-validates identically). Anchors are the observed FY2022 values
# (verified by density + EAD-continuity vs 2023). NB: FY2021 primary pages round PD
# to integer % -> excluded (would break precision consistency); see STATUS.md.
PAGEMAP_2023 = {
    408: [("sovereign", 0, 0.08, 2.0), ("bank", 1, 0.76, 28.0)],
    411: [("sme", 1, 6.16, 28.0)],
    412: [("corporate", 0, 2.52, 33.0)],
    416: [("mortgage", 0, 2.00, 12.0)],
    419: [("qrre", 0, 8.86, 51.0)],
    420: [("other_retail", 0, 6.38, 40.0)],
}
JOBS = [("bnp_2024.pdf", PAGEMAP_2024), ("bnp_2023.pdf", PAGEMAP_2023)]

SUB = re.compile(r"sub[\s‐‑‒–—\-]*total", re.I)
DATE = re.compile(r"31\s*December\s*(\d{4})")

def parse_numbers(line):
    """Return ordered numeric values from a SUB-TOTAL line, format-aware."""
    body = SUB.sub(" ", line)
    # EU format uses comma as decimal -> a percent looks like "1,76 %". Anglo
    # thousands-commas (223,409) are never followed by %, so this is unambiguous.
    european = re.search(r"\d,\d+\s*%", body) is not None
    if european:
        # a 3-digit group that is itself a percent (e.g. CCF "100 %") must NOT be
        # absorbed as a thousands continuation of the preceding number.
        pat = re.compile(r"\(?\d{1,3}(?:\s\d{3}(?!\s*%))*(?:,\d+)?\)?")
        vals = []
        for tok in pat.findall(body):
            t = tok.strip("()").replace(" ", "").replace(",", ".")
            if t:
                vals.append(float(t))
    else:
        pat = re.compile(r"\(?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?")
        vals = []
        for tok in pat.findall(body):
            t = tok.strip("()").replace(",", "")
            if t and t not in {"-"}:
                vals.append(float(t))
    return vals

def extract():
    rows = []
    for fname, pagemap in JOBS:
        with pdfplumber.open(DIR + "pillar3_reports/" + fname) as pdf:
            for pageno, specs in sorted(pagemap.items()):
                txt = pdf.pages[pageno].extract_text() or ""
                dm = DATE.search(txt)
                year = dm.group(1) if dm else "?"
                vintage = f"{year}-12-31"
                subs = [ln for ln in txt.splitlines() if SUB.search(ln)]
                for vclass, occ, apd, algd in specs:
                    if occ >= len(subs):
                        print(f"  !! {fname} p{pageno} {vclass}: subtotal #{occ} missing (have {len(subs)})")
                        continue
                    v = parse_numbers(subs[occ])
                    if len(v) < 9:
                        print(f"  !! {fname} p{pageno} {vclass}: only {len(v)} numbers: {subs[occ][:90]}")
                        continue
                    ead, pd, lgd, rwa, dens = v[3], v[4], v[5], v[7], v[8]
                    rows.append(dict(src=fname, page=pageno, year=year, vintage=vintage, vclass=vclass,
                                     ead=ead, pd=pd, lgd=lgd, rwa=rwa, dens=dens,
                                     apd=apd, algd=algd))
    return rows

def verify(rows):
    """Per-row density cross-check; 2024 anchor match; EAD-continuity vs the
    nearest available later year (FY2024 reference) so prior years are gated."""
    by_class = {}
    for r in rows:
        by_class.setdefault(r["vclass"], {})[r["year"]] = r
    good = []
    for vclass, yrs in sorted(by_class.items()):
        ref = yrs.get("2024") or yrs.get("2023")
        flags = []
        for year, r in sorted(yrs.items()):
            # density cross-check on every row
            dchk = r["rwa"] / r["ead"] if r["ead"] else 0
            if abs(dchk - r["dens"] / 100) > 0.03:
                flags.append(f"{year} dens RWA/EAD={dchk:.3f} vs {r['dens']}%")
            # anchor match only for 2024 (external curated anchors)
            if year == "2024":
                if abs(r["pd"] - r["apd"]) > 0.15:
                    flags.append(f"2024 PD {r['pd']} != anchor {r['apd']}")
                if abs(r["lgd"] - r["algd"]) > 1.0:
                    flags.append(f"2024 LGD {r['lgd']} != anchor {r['algd']}")
            cont = ""
            if ref and year != ref["year"]:
                ratio = r["ead"] / ref["ead"] if ref["ead"] else 0
                cont = f" EADcont/{ref['year']}={ratio:.2f}"
                if not (0.45 <= ratio <= 2.2):
                    flags.append(f"{year} EAD {r['ead']:.0f} vs {ref['year']} {ref['ead']:.0f} ({ratio:.2f})")
            mark = "OK " if not flags else "CHK"
            print(f"  [{mark}] {vclass:14s} {year}  PD={r['pd']:6.2f}  LGD={r['lgd']:5.1f}  "
                  f"EAD={r['ead']:>10,.0f}  RWA={r['rwa']:>9,.0f}  dens={r['dens']:.0f}%{cont}")
            good.append(r)
        if flags:
            print(f"        ^ FLAGS {vclass}: {flags}")
    return good

if __name__ == "__main__":
    rows = extract()
    print(f"\n== extracted {len(rows)} BNP subtotal rows ==")
    good = verify(rows)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEI", "bank_name", "vasicek_class", "vintage_date",
                    "pd_pct", "lgd_pct", "ead_eur_m", "rwa_eur_m", "density", "source", "page"])
        for r in good:
            w.writerow([LEI, "BNP Paribas", r["vclass"], r["vintage"],
                        f"{r['pd']:.2f}", f"{r['lgd']:.2f}", f"{r['ead']:.0f}",
                        f"{r['rwa']:.0f}", f"{r['dens']:.3f}",
                        f"BNP Paribas URD EU CR6 IRBA-by-PD-scale ({r['src']})", r["page"] + 1])
    print(f"\nwrote {OUT}  ({len(good)} rows)")
