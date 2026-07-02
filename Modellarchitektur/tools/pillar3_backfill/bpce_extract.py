# -*- coding: utf-8 -*-
"""BPCE raw EU-CR6 A-IRB extractor. BPCE writes the class label IN the subtotal
row ('Corporates - Other sub-total 70,973 ...'). Columns idx3=EAD, idx4=PD,
idx6=LGD, idx9=density(%). FY2024 value-matches the curated baseline (=raw);
prior years matched by label-keyword + EAD-continuity to the 2024 anchor."""
import re, sys, os, logging, csv as _csv
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
FILES = {2021: "bpce_2021.pdf", 2022: "bpce_2022.pdf",
         2023: "bpce_2023.pdf", 2024: "bpce_2024.pdf"}
KN = {"corporate": (5.68, 33.28), "sme_corporate": (10.69, 26.23),
      "mortgage": (14.68, 10.70), "qrre": (9.63, 33.85),
      "other_retail": (21.55, 28.88), "bank": (0.41, 40.42),
      "sovereign": (0.00, 7.26)}
KW = {"sovereign": lambda l: "central government" in l,
      "bank": lambda l: "institution" in l,
      "sme_corporate": lambda l: "corporate" in l and "sme" in l,
      "corporate": lambda l: "corporate" in l and "other" in l,
      "mortgage": lambda l: "real estate" in l or "immovable" in l,
      "qrre": lambda l: "revolving" in l,
      "other_retail": lambda l: "retail" in l and "other" in l and "sme" in l and "non-sme" not in l}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def nums(s):
    out = []
    for t in NUM.findall(s):
        c = t.replace(",", "")
        try: out.append(float(c))
        except ValueError: pass
    return out

def nums_pos(s):
    """Position-preserving tokenizer for the numeric part of a subtotal row.
    '-'/'–' placeholders -> 0.0 (keep column positions); '(875)' -> -875;
    strips %, commas. Skips non-numeric label tokens."""
    out = []
    for tok in s.split():
        t = tok.strip().replace(",", "").replace("%", "").replace("(", "-").replace(")", "")
        if t in ("-", "–", "—", ""):
            out.append(0.0); continue
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out

def subrows(path):
    """Subtotal rows incl. BPCE's line-wrapped case (label line ends 'sub-',
    data line starts 'total ...'). Returns (page, label_lower, position-nums)."""
    rows = []
    pdf = pdfplumber.open(path)
    for i, pg in enumerate(pdf.pages):
        lines = (pg.extract_text() or "").split("\n")
        for j, ln in enumerate(lines):
            s = ln.strip()
            m = re.search(r"sub-?\s?total", s, re.I)
            if m:
                n = nums_pos(s[m.end():])
                if len(n) >= 9:
                    rows.append((i + 1, s[:m.start()].lower(), n))
            elif re.match(r"^total\b", s, re.I) and j > 0 and \
                    re.search(r"sub-?\s*$", lines[j - 1].strip(), re.I):
                n = nums_pos(s[re.match(r"^total\b", s, re.I).end():])
                if len(n) >= 9:
                    label = re.sub(r"sub-?\s*$", "", lines[j - 1].strip(), flags=re.I).lower()
                    rows.append((i + 1, label, n))
    pdf.close()
    return rows

CLS = ["corporate", "sme_corporate", "mortgage", "qrre", "other_retail", "bank", "sovereign"]
out = []
anchors = {}   # class -> ead from 2024 value-match
for y in sorted(FILES):
    path = DIR + FILES[y]
    if not os.path.exists(path):
        print(f"FY{y}: MISSING"); continue
    rows = subrows(path)
    print(f"\n=== BPCE FY{y} ({len(rows)} subtotal rows) ===")
    for c in CLS:
        pdk, lgk = KN[c]
        pick = None
        if y == 2024:
            # value-match: idx4≈pd AND idx6≈lgd
            for (pg, lab, n) in rows:
                if abs(n[4] - pdk) < 0.05 and abs(n[6] - lgk) < 0.05:
                    pick = (pg, lab, n); break
            if pick:
                anchors[c] = pick[2][3]
        else:
            # prior year: label-keyword + EAD nearest 2024 anchor
            anc = anchors.get(c)
            cands = [(pg, lab, n) for (pg, lab, n) in rows if KW[c](lab)]
            if anc:
                cands = [x for x in cands if 0.5 <= x[2][3] / anc <= 2.0]
                cands.sort(key=lambda x: abs(x[2][3] - anc))
            pick = cands[0] if cands else None
        if not pick:
            print(f"  {c:14} MISS"); continue
        pg, lab, n = pick
        ead, pdv, lgd, rwa, dens = n[3], n[4], n[6], n[8], n[9]
        dok = abs(rwa / ead * 100 - dens) < 2.5 if ead else False
        tag = "RAW-OK" if y == 2024 else ""
        print(f"  {c:14} PD={pdv:6.2f} LGD={lgd:6.2f} ead={ead:>9,.0f} p{pg} {tag}{'' if dok else ' DENS?'}")
        out.append({"short": "bpce", "LEI": "FR9695005MSX1OYEMGDF", "vasicek_class": c,
                    "pd_pct": f"{pdv:.2f}", "lgd_pct": f"{lgd:.2f}",
                    "vintage_date": f"{y}-12-31", "ead_eur_m": f"{ead:.0f}", "note": ""})
if out:
    with open("bpce_raw_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"\nWROTE bpce_raw_rows.csv ({len(out)} rows)")
