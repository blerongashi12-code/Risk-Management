#!/usr/bin/env python3
"""Banco Santander EU-CR6 AIRB extractor. Each report shows the reporting date
plus the prior-year comparative; every CR6-AIRB page is self-labelled with a
year marker and an exposure-class label, and ends with 'Subtotal (exposure class)'.

Column layout (value sequence, % signs are not tokens, '—' is a kept placeholder):
  [on, off, CCF, EAD(idx3), PD(idx4), obligors(idx5), LGD(idx6), M(idx7),
   RWA(idx8), density(idx9), EL, prov]

No sovereign (Santander sovereign is Standardised, excluded from IRB).
Verification: density cross-check (RWA/EAD vs idx9) + EAD-continuity across years.
NB: curated CSV *retail* values differ from raw subtotals (different obligor
counts -> different source); the consistent raw series uses the report subtotals.
"""
import sys, re, csv, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/"
OUT = DIR + "backtest_raw/santander_raw_rows.csv"
LEI = "5493006QMFDDMYWIAM13"

REPORTS = [("santander_2024.pdf", 60, 160), ("santander_2023.pdf", 50, 150),
           ("santander_2022.pdf", 50, 170), ("santander_2021.pdf", 50, 200)]

AIRB = re.compile(r"CR6\s*-\s*AIRB approach", re.I)   # excludes FIRB / standardised
FIRB = re.compile(r"CR6\s*-\s*FIRB", re.I)
YR = re.compile(r"^(20\d\d)$")
SUB = re.compile(r"Subtotal \(exposure class\)", re.I)
# label -> vasicek class (order matters: match most specific first)
LABELS = [
    (re.compile(r"Institutions", re.I), "bank"),
    (re.compile(r"Corporates\s*[-–]\s*Other", re.I), "corporate"),
    (re.compile(r"Corporates\s*[-–]?\s*SMEs", re.I), "sme_corporate"),
    (re.compile(r"Mortgages no SMEs", re.I), "mortgage"),
    (re.compile(r"Qualifying Revolving", re.I), "qrre"),
    (re.compile(r"Other non SMEs", re.I), "other_retail"),
]
SKIP = re.compile(r"Mortgages SMEs|Retail\s*[-–]\s*SMEs", re.I)  # SME-retail sub-classes we don't use

def tokvals(line):
    body = SUB.sub(" ", line)
    out = []
    for tok in re.findall(r"—|–|\(?-?\d[\d,]*(?:\.\d+)?\)?", body):
        if tok in ("—", "–"):
            out.append(None)
        else:
            out.append(float(tok.strip("()").replace(",", "")))
    return out

def classify(page_text):
    """Return vasicek class for a CR6-AIRB page, or None."""
    if SKIP.search(page_text):
        return None
    for rx, cls in LABELS:
        if rx.search(page_text):
            return cls
    return None

def extract():
    recs = []
    for fname, lo, hi in REPORTS:
        try:
            pdf = pdfplumber.open(DIR + "pillar3_reports/" + fname)
        except Exception as e:
            print(f"  !! cannot open {fname}: {e}")
            continue
        report_year = re.search(r"(20\d\d)", fname).group(1)
        with pdf:
            for i in range(lo, min(hi, len(pdf.pages))):
                t = pdf.pages[i].extract_text() or ""
                if not AIRB.search(t) or FIRB.search(t):
                    continue
                subs = [ln for ln in t.splitlines() if SUB.search(ln)]
                if not subs:
                    continue
                cls = classify(t)
                if not cls:
                    continue
                yr = next((ln.strip() for ln in t.splitlines() if YR.match(ln.strip())), None)
                if not yr:
                    continue
                v = tokvals(subs[0])
                if len(v) < 10 or v[3] is None or v[4] is None or v[6] is None:
                    print(f"  !! {fname} p{i} {cls}: bad parse {subs[0][:80]}")
                    continue
                recs.append(dict(report=report_year, page=i, year=yr, vclass=cls,
                                 ead=v[3], pd=v[4], lgd=v[6], rwa=v[8], dens=v[9]))
    return recs

def build_series(recs):
    """Dedup to one row per (class, year): prefer the primary disclosure
    (report_year == page_year). Cross-validate overlaps."""
    best = {}
    xval = []
    for r in recs:
        key = (r["vclass"], r["year"])
        primary = (r["report"] == r["year"])
        if key not in best:
            best[key] = (primary, r)
        else:
            was_primary, prev = best[key]
            # cross-validation log if both exist
            if abs(prev["pd"] - r["pd"]) > 0.05 or abs(prev["lgd"] - r["lgd"]) > 0.1:
                xval.append((key, prev["pd"], prev["lgd"], r["pd"], r["lgd"]))
            if primary and not was_primary:
                best[key] = (primary, r)
    return {k: v[1] for k, v in best.items()}, xval

def verify(series):
    by_class = {}
    for (cls, yr), r in series.items():
        by_class.setdefault(cls, {})[yr] = r
    good = []
    for cls, yrs in sorted(by_class.items()):
        ref = yrs.get("2024") or yrs.get("2023")
        for yr, r in sorted(yrs.items()):
            flags = []
            if r["ead"] and r["rwa"] is not None:
                dchk = r["rwa"] / r["ead"]
                if r["dens"] is not None and abs(dchk - r["dens"] / 100) > 0.03:
                    flags.append(f"dens {dchk:.3f} vs {r['dens']}%")
            cont = ""
            if ref and yr != ref["year"] and ref["ead"]:
                ratio = r["ead"] / ref["ead"]
                cont = f" cont/{ref['year']}={ratio:.2f}"
                if not (0.4 <= ratio <= 2.5):
                    flags.append(f"EAD {r['ead']:.0f} vs {ref['year']} {ref['ead']:.0f}")
            mark = "OK " if not flags else "CHK"
            print(f"  [{mark}] {cls:14s} {yr}(rep{r['report']}p{r['page']})  "
                  f"PD={r['pd']:6.2f} LGD={r['lgd']:6.2f} EAD={r['ead']:>10,.0f} "
                  f"RWA={r['rwa']:>9,.0f} dens={r['dens']:.0f}%{cont}"
                  + ("  << " + ";".join(flags) if flags else ""))
            good.append(r)
    return good

if __name__ == "__main__":
    recs = extract()
    print(f"\n== {len(recs)} raw AIRB subtotal records across reports ==")
    series, xval = build_series(recs)
    if xval:
        print("cross-validation deltas (same class/year, two reports):")
        for k, p1, l1, p2, l2 in xval:
            print(f"   {k}: ({p1},{l1}) vs ({p2},{l2})")
    print(f"\n== series: {len(series)} (class,year) cells ==")
    good = verify(series)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEI", "bank_name", "vasicek_class", "vintage_date",
                    "pd_pct", "lgd_pct", "ead_eur_m", "rwa_eur_m", "density", "source", "page"])
        for r in sorted(good, key=lambda x: (x["vclass"], x["year"])):
            w.writerow([LEI, "Banco Santander", r["vclass"], f"{r['year']}-12-31",
                        f"{r['pd']:.2f}", f"{r['lgd']:.2f}", f"{r['ead']:.0f}",
                        "" if r["rwa"] is None else f"{r['rwa']:.0f}",
                        "" if r["dens"] is None else f"{r['dens']:.3f}",
                        f"Banco Santander Pillar 3 CR6-AIRB (raw subtotal, report {r['report']})", r["page"] + 1])
    print(f"\nwrote {OUT}  ({len(good)} rows)")
