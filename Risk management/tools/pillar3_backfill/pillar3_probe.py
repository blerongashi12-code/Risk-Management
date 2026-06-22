# -*- coding: utf-8 -*-
"""Throwaway probe: calibrate EU-CR6 AIRB sub-total extraction against
known DB-2024 values, then see if same logic reads DB-2023 / DB-2022.
NOT committed to the repo."""
import re, sys
import pdfplumber

sys.stdout.reconfigure(encoding="utf-8")

TR = ("C:/Users/blero/.claude/projects/"
      "C--Users-blero-Downloads-RiskMgmt--claude-worktrees-elegant-jemison-28f26f/"
      "9bb2b4a2-d718-400b-844b-8a5142e0522a/tool-results/")
PDFS = {
    "DB-2024": TR + "webfetch-1781961793731-f0q29c.pdf",
    "DB-2023": TR + "webfetch-1781961794778-3n2y3b.pdf",
    "DB-2022": TR + "webfetch-1781961531760-zwdu0t.pdf",
}

# known hand-verified DB-2024 values from pillar3_bank_pd_lgd.csv
KNOWN_2024 = {
    "corporate":     (4.12, 30.86),
    "sme_corporate": (6.77, 33.03),
    "mortgage":      (2.63, 20.03),
    "qrre":          (2.66, 57.60),
    "other_retail":  (9.50, 58.41),
    "bank":          (0.19, 48.15),
    "sovereign":     (0.31, 66.37),
}

NUM = re.compile(r"-?\d[\d,]*\.?\d*")

def parse_nums(line):
    """Return list of floats from a row (commas stripped)."""
    toks = NUM.findall(line)
    out = []
    for t in toks:
        try:
            out.append(float(t.replace(",", "")))
        except ValueError:
            pass
    return out

def airb_pages(pdf):
    pages = []
    for i, pg in enumerate(pdf.pages):
        t = pg.extract_text() or ""
        if "CR6" in t and ("Average PD" in t or "PD scale" in t or "A-IRB" in t or "AIRB" in t):
            pages.append(i)
    return pages

def subtotal_rows(pdf, pageidxs=None):
    """Yield (page, context_header, raw_line, nums) for Sub-total/Total rows.
    pageidxs=None -> scan ALL pages."""
    rows = []
    idxs = range(len(pdf.pages)) if pageidxs is None else pageidxs
    for i in idxs:
        t = pdf.pages[i].extract_text() or ""
        ctx = ""
        for ln in t.split("\n"):
            s = ln.strip()
            letters = re.sub(r"[^A-Za-z]", "", s)
            digits = re.sub(r"[^0-9]", "", s)
            # context header = mostly-text line, no big number sequence
            if len(letters) > 4 and len(digits) <= 3 and "Sub-total" not in s and "Total" not in s:
                ctx = s
            if re.match(r"^(Sub-total|Subtotal|Total)\b", s):
                nums = parse_nums(s)
                rows.append((i + 1, ctx, s[:120], nums))
    return rows

def find_match(rows, pd_t, lgd_t, tol=0.03):
    """Find rows whose numeric tokens contain both pd and lgd (any positions)."""
    hits = []
    for (pg, ctx, raw, nums) in rows:
        pd_pos = [j for j, v in enumerate(nums) if abs(v - pd_t) <= tol]
        lgd_pos = [j for j, v in enumerate(nums) if abs(v - lgd_t) <= tol]
        if pd_pos and lgd_pos:
            hits.append((pg, ctx, pd_pos, lgd_pos, raw))
    return hits

print("=" * 70)
print("CALIBRATION on DB-2024 (match known PD/LGD to rows)")
print("=" * 70)
pdf = pdfplumber.open(PDFS["DB-2024"])
pages = airb_pages(pdf)
print("AIRB CR6 candidate pages (1-based):", [p + 1 for p in pages])
rows = subtotal_rows(pdf, None)   # scan ALL pages, match by value
rows = [r for r in rows if len(r[3]) >= 10]
print(f"total sub-total/total rows (>=10 nums, all pages): {len(rows)}\n")
for cls, (pd_t, lgd_t) in KNOWN_2024.items():
    hits = find_match(rows, pd_t, lgd_t)
    print(f"[{cls}] target PD={pd_t} LGD={lgd_t}  -> {len(hits)} candidate row(s)")
    for (pg, ctx, pdp, lgdp, raw) in hits[:4]:
        print(f"     p{pg} pdpos={pdp} lgdpos={lgdp} ctx='{ctx[:45]}'")
        print(f"        raw: {raw}")
pdf.close()

# 2024 EAD anchors (nums[3]) per class, for continuity matching in prior years
EAD_2024 = {
    "corporate": 237810, "sme_corporate": 25888, "mortgage": 166929,
    "qrre": 9248, "other_retail": 29688, "bank": 16495, "sovereign": 135860,
}

def dump_year(tag):
    print("\n" + "=" * 70)
    print(f"{tag}: sub-total rows (EAD>2000), col3=EAD col4=PD col6=LGD")
    print("=" * 70)
    pdf = pdfplumber.open(PDFS[tag])
    rows = [r for r in subtotal_rows(pdf, None) if len(r[3]) >= 10]
    shown = []
    for (pg, ctx, raw, nums) in rows:
        ead, pdv, lgdv = nums[3], nums[4], nums[6]
        if ead > 2000 and 0 <= pdv < 40 and 0 < lgdv <= 100:
            shown.append((pg, ctx, ead, pdv, lgdv, raw))
    for (pg, ctx, ead, pdv, lgdv, raw) in shown:
        print(f"  p{pg:<3} EAD={ead:>10,.0f} PD={pdv:>6.2f} LGD={lgdv:>6.2f}  ctx='{ctx[:38]}'")
    pdf.close()

dump_year("DB-2023")
dump_year("DB-2022")
