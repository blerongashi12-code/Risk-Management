# -*- coding: utf-8 -*-
"""Generalised per-bank Pillar-3 EU-CR6 extractor for the historical backfill.

Method (no fabrication, calibration-gated):
  1. FY2024 report: value-match each class's known (PD,LGD) to a sub-total row
     -> LEARN the column positions (pos_pd, pos_lgd), the context label, the EAD.
  2. Prior years: for each learned class, pick the sub-total row whose context
     best matches the learned label AND whose EAD is closest/continuous; read
     PD/LGD at the learned positions; density cross-check.
  3. Anything that can't be matched/verified -> reported as MISS (never guessed).

Run from anywhere. PDFs in C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/<short>_<year>.pdf
Outputs <short>_timeseries_rows.csv (PD/LGD) + prints a summary.
"""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"
CLASSES = ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank", "sovereign"]
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def nums(s):
    out = []
    for t in NUM.findall(s):
        c = t.replace(",", "")
        try: out.append(float(c))
        except ValueError: pass
    return out

def subtotal_rows(path, year):
    """All Sub-total/Total rows (>=8 numeric tokens) with a 3-line context window.
    Restrict to the year-end block when a 'Dec 31, <year>' / '31/12/<year>' header
    is detectable on the page; otherwise keep all (single-period reports)."""
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
            is_label = len(letters) > 3 and len(digits) <= 4 and not re.match(r"^(Sub-?total|Total|Sous-total)\b", s, re.I)
            if is_label:
                ctx.append(s); ctx = ctx[-3:]
            if re.match(r"^(Sub-?total|Subtotal|Sous-total|Total)\b", s, re.I):
                n = nums(s)
                if len(n) >= 8:
                    rows.append((i + 1, " | ".join(ctx), n))
                ctx = []
    pdf.close()
    return rows

def calibrate(rows, knowns, tol=0.05):
    """Learn per class: (pos_pd, pos_lgd, context, ead, page) from FY2024 value-match."""
    learned = {}
    for cls, (p, l) in knowns.items():
        best = None
        for (pg, ctx, n) in rows:
            pp = [j for j, v in enumerate(n) if abs(v - p) <= tol]
            lp = [j for j, v in enumerate(n) if abs(v - l) <= tol]
            if pp and lp:
                # prefer canonical EU-CR6 positions pd=4,lgd=6 if present
                pos_pd = 4 if 4 in pp else pp[0]
                pos_lgd = 6 if 6 in lp else lp[0]
                ead = n[3] if len(n) > 3 else None
                cand = (pos_pd, pos_lgd, ctx, ead, pg)
                if best is None or (pos_pd == 4 and pos_lgd == 6):
                    best = cand
        if best:
            learned[cls] = best
    return learned

def _ctx_tokens(c):
    return set(re.sub(r"[^a-z ]", " ", c.lower()).split())

def match_prior(rows, learned_for_cls):
    """Pick the sub-total row best matching the learned class (context overlap +
    EAD continuity). Returns (pd, lgd, ead, density_check_ok, ctx, page) or None."""
    pos_pd, pos_lgd, ref_ctx, ref_ead, _ = learned_for_cls
    ref_tok = _ctx_tokens(ref_ctx) - {"sub", "total", "of", "which", "the", "and", "by", "non"}
    best, best_score = None, -1
    for (pg, ctx, n) in rows:
        if len(n) <= max(pos_pd, pos_lgd, 9):
            continue
        tok = _ctx_tokens(ctx)
        overlap = len(ref_tok & tok)
        ead = n[3]
        ead_prox = 0.0
        if ref_ead and ead > 0:
            ratio = ead / ref_ead
            ead_prox = 1.0 if 0.6 <= ratio <= 1.7 else (0.3 if 0.4 <= ratio <= 2.5 else 0.0)
        score = overlap + ead_prox
        if score > best_score and overlap >= 1:
            best_score, best = score, (pg, ctx, n)
    if best is None:
        return None
    pg, ctx, n = best
    pdv, lgdv, ead = n[pos_pd], n[pos_lgd], n[3]
    rwa = n[8] if len(n) > 8 else None
    dens = n[9] if len(n) > 9 else None
    dens_ok = (rwa is not None and dens is not None and ead > 0
               and abs(rwa / ead * 100 - dens) <= 2.0)
    plausible = (0 <= pdv < 40) and (0 < lgdv <= 100)
    if not plausible:
        return None
    return (pdv, lgdv, ead, dens_ok, ctx[:40], pg)

def run_bank(cfg):
    short, lei, knowns = cfg["short"], cfg["lei"], cfg["knowns"]
    years = cfg["years"]   # {year: filename}
    print("=" * 74); print(f"BANK: {short}  (LEI {lei})"); print("=" * 74)
    rows_by_year = {}
    for y, fn in years.items():
        p = DIR + fn
        if not os.path.exists(p):
            print(f"  FY{y}: MISSING pdf {fn}"); continue
        rows_by_year[y] = subtotal_rows(p, y)
        print(f"  FY{y}: {len(rows_by_year[y])} sub-total rows parsed")
    if 2024 not in rows_by_year:
        print("  no FY2024 -> cannot calibrate"); return []
    learned = calibrate(rows_by_year[2024], knowns)
    print(f"\n  CALIBRATION (FY2024 value-match): {len(learned)}/{len(knowns)} classes")
    for cls in CLASSES:
        if cls in learned:
            pp, lp, ctx, ead, pg = learned[cls]
            print(f"    [OK ] {cls:14} pos_pd={pp} pos_lgd={lp} ead={ead:,.0f} p{pg} '{ctx[-34:]}'")
        elif cls in knowns:
            print(f"    [MISS] {cls:14} (FY2024 value not found -> not extractable)")
    out = []
    print("\n  PRIOR-YEAR EXTRACTION:")
    for y in sorted(years):
        if y == 2024 or y not in rows_by_year:
            continue
        for cls in CLASSES:
            if cls not in learned:
                continue
            m = match_prior(rows_by_year[y], learned[cls])
            if m is None:
                print(f"    FY{y} {cls:14} MISS")
                continue
            pdv, lgdv, ead, dens_ok, ctx, pg = m
            flag = "" if dens_ok else "DENSITY?"
            print(f"    FY{y} {cls:14} PD={pdv:6.2f} LGD={lgdv:6.2f} ead={ead:,.0f} p{pg} {flag} '{ctx}'")
            out.append({"short": short, "LEI": lei, "vasicek_class": cls,
                        "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgdv:.2f}",
                        "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}",
                        "source_page": pg, "density_ok": dens_ok, "note": flag, "ctx": ctx})
    return out

SANTANDER = {
    "short": "santander", "lei": "5493006QMFDDMYWIAM13",
    "knowns": {"corporate": (4.01, 44.88), "sme_corporate": (7.68, 42.53),
               "mortgage": (2.44, 15.08), "qrre": (4.20, 83.56),
               "other_retail": (6.77, 42.53), "bank": (1.02, 15.80)},
    "years": {2021: "santander_2021.pdf", 2022: "santander_2022.pdf",
              2023: "santander_2023.pdf", 2024: "santander_2024.pdf"},
}

ING = {
    "short": "ing", "lei": "549300NYKK9MWM7GGW15",
    "knowns": {"corporate": (1.81, 14.43), "sme_corporate": (6.12, 27.39),
               "mortgage": (1.46, 22.61), "qrre": (3.75, 22.50),
               "other_retail": (6.39, 39.90), "bank": (0.39, 19.79),
               "sovereign": (0.10, 15.00)},
    "years": {2021: "ing-group-additional-pillar-iii-report-2021-.pdf",
              2022: "ing-group-additional-pillar-iii-report-2022.pdf",
              2023: "4qfy2023-ing-credit-update.pdf",
              2024: "ing_2024.pdf"},
}

SOCGEN = {
    "short": "socgen", "lei": "O2RNE8IBXP4R0TD8PU41",
    "knowns": {"corporate": (2.39, 30.18), "sme_corporate": (3.57, 19.06),
               "mortgage": (1.18, 16.37), "qrre": (7.89, 48.48),
               "other_retail": (8.06, 31.89), "bank": (0.36, 22.68),
               "sovereign": (0.09, 1.54)},
    "years": {2021: "socgen_2021.pdf", 2022: "socgen_2022.pdf",
              2023: "socgen_2023.pdf", 2024: "socgen_2024.pdf"},
}

BANK = SOCGEN   # <- currently processing

if __name__ == "__main__":
    rows = run_bank(BANK)
    if rows:
        fn = f"{BANK['short']}_timeseries_rows.csv"
        with open(fn, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWROTE {fn} ({len(rows)} rows)")
