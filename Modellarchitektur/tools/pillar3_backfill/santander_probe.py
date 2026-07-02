#!/usr/bin/env python3
"""Locate Santander EU-CR6 AIRB subtotal rows by value-matching the FY2024
curated anchors. Santander breaks IRB by geography (many subtotals) + a
consolidated total; the precise anchor (PD,LGD) pairs disambiguate the group row."""
import sys, re, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
fname = sys.argv[1] if len(sys.argv) > 1 else "santander_2024.pdf"

# distinctive anchor tokens to hunt for (PD or LGD that are unlikely coincidences)
ANCHORS = ["44.88", "83.56", "15.08", "42.53", "4.01", "7.68", "2.44", "4.20", "6.77", "1.02", "15.80"]
TOK = re.compile("|".join(re.escape(a) for a in ANCHORS))

with pdfplumber.open(DIR + fname) as pdf:
    n = len(pdf.pages)
    print(f"{fname} pages={n}")
    for i in range(n):
        txt = pdf.pages[i].extract_text() or ""
        hits = set(TOK.findall(txt))
        if len(hits) >= 2:  # page carrying >=2 anchors is likely the EU-CR6 table
            head = next((l.strip()[:70] for l in txt.splitlines() if l.strip()), "")
            print(f"  p{i}: anchors={sorted(hits)}  head='{head}'")
