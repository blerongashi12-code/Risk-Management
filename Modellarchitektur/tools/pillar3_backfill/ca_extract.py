#!/usr/bin/env python3
"""Crédit Agricole S.A. EU-CR6 Advanced-IRB extractor. CA S.A. (LEI
F0HUI1NY1AZMJMD8LP67) is the entity in the EBA Transparency panel (CA *Group* is
not), so it is the backtestable CA entity. The year-end report ("RISK REPORT
PILLAR 3 31 DECEMBER 2024") carries the A-IRB CR6 for 31.12.2024 AND the
31.12.2023 comparative; the 30-June "Disclosures Review" reports add mid-year.

Layout: flowing table, ONE class per block. The class name is a left-column label
that wraps VERTICALLY across several lines ("Central" / "governments" / "and
Central" / "Banks") interspersed with PD-band rows; the block ends with SUB-TOTAL.
We accumulate the leading-alpha fragment of each line since the last SUB-TOTAL ->
class name. SUB-TOTAL value layout: idx3=EAD, idx4=PD, idx5=LGD, idx7=RWA, idx8=dens.
"""
import sys, re, csv, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/"
OUT = DIR + "backtest_raw/ca_raw_rows.csv"
LEI = "F0HUI1NY1AZMJMD8LP67"            # Crédit Agricole S.A.

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
FRAG = re.compile(r"^([A-Za-z][A-Za-z &/–\-]*?)(?=\s+\d|\s*$)")
DATE = re.compile(r"31 DECEMBER (\d{4})", re.I)
# page furniture / column-header band that must NOT be taken as a class label
FURNITURE = re.compile(
    r"RISKS AND PILLAR|Pillar 3|^Risk$|weighted|Off-|Exposure|balance|^average|maturity"
    r"|IRB-A|PD range|in millions|of euros|supporting|Expected|CR.DIT AGRICOLE|Density|^Value"
    r"|Number|Amount|CCF|post|pre-|factors|^loss|provision|approach|category|RATINGS-BASED"
    r"|DEFAULT|FOUNDATION|of which|exposure", re.I)

def nums(s):
    # fix concatenated "off-balance + CCF%" cells (e.g. "2,387100.98%"): a 3-digit
    # comma-group immediately followed by another digit -> insert a separator.
    s = re.sub(r"(,\d{3})(\d)", r"\1 \2", s)
    out = []
    for t in NUM.findall(s):
        try: out.append(float(t.replace(",", "")))
        except ValueError: pass
    return out

def classify(section, frags):
    """Class = persistent SECTION (Central/Institutions/Corporates/Retail) + wrapped
    sub-qualifier fragments."""
    f = " ".join(frags).lower()
    if section == "central":
        return "sovereign"
    if section == "institutions":
        return "bank"
    if section == "corporates":
        if "specialised" in f or "specialized" in f:
            return None
        if "sme" in f:
            return "sme_corporate"
        if "other" in f:
            return "corporate"
        return None
    if section == "retail":
        if "secured" in f or "immovable" in f or "real estate" in f:
            return "mortgage" if "non" in f else None     # secured non-SME = residential mortgage
        if "qualifying" in f or "revolving" in f:
            return "qrre"
        if "non" in f and "sme" in f:
            return "other_retail"                          # Other non-SME
        return None
    return None

def set_section(frag, cur):
    low = frag.lower()
    if "central" in low:
        return "central"
    if "institution" in low:
        return "institutions"
    if "corporate" in low:
        return "corporates"
    if low.startswith("retail"):
        return "retail"
    return cur

def extract(path, want_years):
    """Return {year: {vclass: (ead,pd,lgd,rwa,dens,page)}}. Only A-IRB CR6 tables."""
    res = {}
    with pdfplumber.open(path) as pdf:
        in_airb = False
        year = None
        frags = []
        section = ""
        for i, pg in enumerate(pdf.pages):
            t = pg.extract_text() or ""
            head = "\n".join(t.splitlines()[:8])
            mt = re.search(r"ADVANCED INTERNAL\s+RATINGS-BASED APPROACH AT 31 DECEMBER (\d{4})", t, re.I)
            if mt:
                in_airb = True
                if mt.group(1) != year:
                    year = mt.group(1); frags = []; section = ""
            elif re.search(r"FOUNDATION INTERNAL", head, re.I) or re.search(r"30 JUNE|RATINGS-BASED APPROACH AT 30", head, re.I):
                in_airb = False
            if not in_airb or year not in want_years:
                continue
            for ln in t.splitlines():
                s = ln.strip()
                if not s:
                    continue
                if re.match(r"^SUB-?TOTAL", s, re.I):
                    n = nums(s)
                    if len(n) >= 9:
                        cls = classify(section, frags)
                        ead, pd, lgd, rwa, dens = n[3], n[4], n[5], n[7], n[8]
                        if cls and ead > 100 and 0 <= pd < 40 and 0 < lgd <= 100:
                            res.setdefault(year, {})
                            if cls not in res[year]:
                                res[year][cls] = (ead, pd, lgd, rwa, dens, i + 1)
                    frags = []
                    continue
                if re.match(r"^TOTAL\b", s, re.I):          # grand total ends the table
                    frags = []
                    continue
                if FURNITURE.search(s):                     # skip header/furniture
                    continue
                m = FRAG.match(s)
                if m:
                    frag = m.group(1).strip()
                    if frag.upper() not in ("SUB", "TOTAL") and len(frag) > 1:
                        section = set_section(frag, section)
                        frags.append(frag)
    return res

if __name__ == "__main__":
    CLS = ["sovereign", "bank", "corporate", "sme_corporate", "mortgage", "qrre", "other_retail"]
    # year-end report -> 2024 + 2023 comparative; each half-year "Review" carries the
    # prior 31-December comparative -> 2022 / 2021 / 2020.
    jobs = [("ca_sa_2024.pdf", {"2024", "2023"}),
            ("ca_sa_2023h1.pdf", {"2022"}),
            ("ca_sa_2022h1.pdf", {"2021"}),
            ("ca_sa_2021h1.pdf", {"2020"})]
    # collect all years first, then drop half-year outliers vs the trusted year-end
    # (2023/2024) reference per class (catches granular-subblock mis-grabs, e.g. FY2021 sme).
    allres = {}
    for fn, years in jobs:
        for yr, d in extract(DIR + "pillar3_reports/" + fn, years).items():
            allres.setdefault(yr, {}).update({c: v for c, v in d.items() if c not in allres.get(yr, {})})
    ref = {}   # class -> (ref_ead, ref_pd) from year-end 2023/2024
    for yr in ("2023", "2024"):
        for c, v in allres.get(yr, {}).items():
            ref.setdefault(c, []).append((v[0], v[1]))
    ref = {c: (sum(e for e, _ in lst) / len(lst), sum(p for _, p in lst) / len(lst)) for c, lst in ref.items()}
    dropped = []
    for yr in list(allres):
        if yr in ("2023", "2024"):
            continue
        for c in list(allres[yr]):
            if c in ref:
                ead, pd = allres[yr][c][0], allres[yr][c][1]
                re_ead, re_pd = ref[c]
                if not (0.4 <= ead / re_ead <= 2.5) or pd > 3 * re_pd + 3:
                    dropped.append(f"{yr} {c} (EAD={ead:.0f} vs ref {re_ead:.0f}, PD={pd:.2f} vs {re_pd:.2f})")
                    del allres[yr][c]
    if dropped:
        print("DROPPED half-year outliers (vs year-end reference):")
        for d in dropped:
            print("   ", d)

    rows = []
    for yr in sorted(allres):
        res = {yr: allres[yr]}
        if True:
            print(f"\n== CA S.A. FY{yr} ({len(res[yr])}/7) ==")
            for c in CLS:
                if c not in res[yr]:
                    print(f"  {c:14} MISS"); continue
                ead, pd, lgd, rwa, dens, pg = res[yr][c]
                dchk = rwa / ead * 100 if ead else 0
                flag = "" if abs(dchk - dens) < 1.5 else f" DENS? {dchk:.2f} vs {dens}"
                print(f"  {c:14} PD={pd:6.2f} LGD={lgd:6.2f} EAD={ead:>10,.0f} RWA={rwa:>9,.0f} dens={dens:.2f} p{pg}{flag}")
                rows.append([LEI, "Credit Agricole SA", c, f"{yr}-12-31", f"{pd:.2f}", f"{lgd:.2f}",
                             f"{ead:.0f}", f"{rwa:.0f}", f"{dens:.3f}",
                             f"Credit Agricole S.A. Pillar 3 {yr} EU CR6 A-IRB (raw subtotal)", pg])
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["LEI", "bank_name", "vasicek_class", "vintage_date", "pd_pct", "lgd_pct",
                    "ead_eur_m", "rwa_eur_m", "density", "source", "page"])
        w.writerows(rows)
    print(f"\nwrote {OUT} ({len(rows)} rows)")
