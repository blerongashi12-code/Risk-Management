#!/usr/bin/env python3
"""Locate BNP EU-CR6 / IRBA-by-PD-scale subtotal rows in the 952-page URD.

Strategy: scan the credit-risk Pillar-3 region (~p408-490), find pages whose
text mentions PD scale / PD range / EU CR6 / IRBA exposure, and dump every line
that looks like a class SUB-TOTAL / TOTAL with its numbers. We then value-match
the FY2024 known PD/LGD to fix column positions before extracting.
"""
import sys, re
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
pdf_path = DIR + "bnp_2024.pdf"

# FY2024 curated knowns (for value-match calibration of columns)
KNOWN = {
    "corporate": (2.87, 37.00), "sme": (3.78, 25.00), "mortgage": (1.61, 17.00),
    "qrre": (8.91, 36.00), "other_retail": (6.70, 36.00), "bank": (0.79, 26.00),
    "sovereign": (0.08, 3.00),
}

HDR_RE = re.compile(r"(pd\s*scale|pd\s*range|EU\s*CR6|IRBA?\s+EXPOSURE|exposures?\s+by\s+pd)", re.I)
TOTAL_RE = re.compile(r"(sub[-\s]?total|total)\b", re.I)

with pdfplumber.open(pdf_path) as pdf:
    n = len(pdf.pages)
    lo, hi = 405, 495
    print(f"pages={n} scanning {lo}-{hi}")
    for i in range(lo, min(hi, n)):
        pg = pdf.pages[i]
        txt = pg.extract_text() or ""
        if not HDR_RE.search(txt):
            continue
        lines = txt.splitlines()
        hdr = " ".join(lines[:2])[:90].replace("\n", " ")
        # collect TOTAL-ish lines that carry >=3 numbers
        hits = []
        for ln in lines:
            if TOTAL_RE.search(ln):
                nums = re.findall(r"-?\d[\d,\.]*", ln)
                if len(nums) >= 3:
                    hits.append(ln.strip()[:140])
        if hits:
            print(f"\n=== p{i} hdr='{hdr}'")
            for h in hits[:25]:
                print("   ", h)
