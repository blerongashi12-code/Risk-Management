# -*- coding: utf-8 -*-
"""Credit Mutuel raw EU-CR6 A-IRB extractor (FRENCH number format).
FR numbers: space thousands ('83 159'), comma decimal ('5,69%'). Rows start
'Sous-total'. Columns idx3=EAD, idx4=PD, idx6=LGD. FY2024 value-matches the
curated baseline (=raw); FY2023 by EAD-continuity to the 2024 anchor."""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
# NB: cm_2021/cm_2022.pdf (Groupe CM Pilier 3, heruntergeladen) sind vorhanden, aber
# NICHT in der Reihe: CM hat Spalten-Layout-Drift (2021/22 ohne Schuldnerzahl-Spalte,
# LGD@idx5; 2023/24 mit, LGD@idx6) UND degenerierte kuratierte Anker (bank==sovereign
# 0,12/34; corporate/other_retail kollidieren auf PD) -> Zeilen->Klasse-Mapping ueber
# Jahre unzuverlaessig (other_retail-EAD springt 135k/36k/395k/396k). Nur 2023/24 vetted.
FILES = {2023: "cm_2023.pdf", 2024: "cm_2024.pdf"}
KN = {"corporate": (2.59, 45.00), "sme_corporate": (5.69, 25.00),
      "mortgage": (1.79, 16.00), "qrre": (3.13, 33.00),
      "other_retail": (2.59, 19.00), "bank": (0.12, 34.00),
      "sovereign": (0.12, 34.00)}
FRNUM = re.compile(r"\d{1,3}(?:\s\d{3})*(?:,\d+)?")
# position-preserving token: FR number OR a dash placeholder
TOKEN = re.compile(r"\d{1,3}(?:\s\d{3})*(?:,\d+)?|[-–—]")

def nums(s):
    out = []
    for m in FRNUM.findall(s):
        v = m.replace(" ", "").replace(",", ".")
        try: out.append(float(v))
        except ValueError: pass
    return out

def cols(s):
    """Position-preserving parse + layout-robust column detection.
    CM layout drifts: 2021/22 have NO obligor column (LGD at idx5), 2023/24 add an
    obligor count (LGD at idx6); maturity is '2,5' or a dash. Returns
    (ead, pd, lgd, rwa, dens) using: EAD=idx3, PD=idx4, LGD=first token<=100 after
    PD, density=position d where RWA(d-1)/EAD ~= dens(d) (cross-check). None if unsure."""
    body = re.sub(r"^\s*Sous-?total\s*", "", s, flags=re.I)  # drop label (its '-' is not a column)
    toks = []
    for m in TOKEN.findall(body):
        if m in ("-", "–", "—"):
            toks.append(None)
        else:
            try: toks.append(float(m.replace(" ", "").replace(",", ".")))
            except ValueError: toks.append(None)
    if len(toks) < 8 or toks[3] is None or toks[4] is None:
        return None
    ead, pd = toks[3], toks[4]
    # LGD: first token after PD that is a percentage (0<v<=100), skipping an obligor count (>100)
    lgd_i = None
    for j in range(5, min(len(toks), 8)):
        if toks[j] is not None and 0 < toks[j] <= 100:
            lgd_i = j; break
    if lgd_i is None:
        return None
    lgd = toks[lgd_i]
    # density via RWA/EAD cross-check: scan positions after LGD
    rwa = dens = None
    for d in range(lgd_i + 1, len(toks)):
        if toks[d] is None or toks[d - 1] is None or ead <= 0:
            continue
        if 0 < toks[d] <= 100 and abs(toks[d - 1] / ead * 100 - toks[d]) < 2.5:
            rwa, dens = toks[d - 1], toks[d]; break
    return (ead, pd, lgd, rwa, dens)

def subrows(path):
    rows = []
    pdf = pdfplumber.open(path)
    for i, pg in enumerate(pdf.pages):
        for ln in (pg.extract_text() or "").split("\n"):
            s = ln.strip()
            if re.match(r"^Sous-?total", s, re.I):
                c = cols(s)
                if c and c[2] is not None:        # need at least EAD/PD/LGD
                    rows.append((i + 1, c))
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
            for (pg, n) in rows:                       # n = (ead, pd, lgd, rwa, dens)
                if abs(n[1] - pdk) < 0.05 and abs(n[2] - lgk) < 0.05:
                    pick = (pg, n); break
            if pick:
                anchors[c] = pick[1][0]                 # ead anchor
        else:
            anc = anchors.get(c)
            cands = [(pg, n) for (pg, n) in rows
                     if abs(n[2] - lgk) < 2.0 and (not anc or 0.5 <= n[0] / anc <= 2.0)]
            if anc:
                cands.sort(key=lambda x: abs(x[1][0] - anc))
            pick = cands[0] if cands else None
        if not pick:
            print(f"  {c:14} MISS"); continue
        pg, n = pick
        ead, pdv, lgd, rwa, dens = n[0], n[1], n[2], n[3], n[4]
        dok = (rwa is not None and dens is not None and ead and abs(rwa / ead * 100 - dens) < 3)
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
