#!/usr/bin/env python3
"""Show BNP IRBA block labels: each class block starts with a labelled
'0.00 to < 0.15%' row and ends with SUB-TOTAL. Interleave them with the page
date so we can map label -> subtotal unambiguously across 2024/2023."""
import sys, re
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
path = DIR + (sys.argv[1] if len(sys.argv) > 1 else "bnp_2024.pdf")
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 439
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 453

DATE = re.compile(r"31\s*December\s*(\d{4})")
START = re.compile(r"0[.,]00\s*to\s*<\s*0[.,]15")
SUB = re.compile(r"sub[‑\-\s]?total", re.I)

with pdfplumber.open(path) as pdf:
    for i in range(lo, min(hi, len(pdf.pages))):
        txt = pdf.pages[i].extract_text() or ""
        date = None
        out = []
        for ln in txt.splitlines():
            s = ln.strip()
            m = DATE.search(s)
            if m and date is None:
                date = m.group(1)
            if START.search(s):
                # label = text before the "0.00 to" token
                lab = s[:START.search(s).start()].strip()
                out.append(("START", lab[:60]))
            elif SUB.search(s):
                out.append(("SUB", s[:130]))
        print(f"\n--- p{i}  date={date}")
        for tag, s in out:
            print(f"   [{tag}] {s}")
