#!/usr/bin/env python3
"""Dump BNP IRBA table page titles + the date line to map page->class->date."""
import sys, re
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"
path = DIR + (sys.argv[1] if len(sys.argv) > 1 else "bnp_2024.pdf")
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 437
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 454

with pdfplumber.open(path) as pdf:
    for i in range(lo, min(hi, len(pdf.pages))):
        txt = pdf.pages[i].extract_text() or ""
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        # Title-ish lines: contain 'TABLE', 'IRBA', 'EU CR6', a class word, or a date
        keys = re.compile(r"(TABLE\s*\d|EU\s*CR6|IRBA|exposure class|31\s*December|Central gov|Institution|Corporate|Retail|secured by|revolving|SME|Specialised|sovereign)", re.I)
        picks = [l for l in lines[:12] if keys.search(l)]
        print(f"\n--- p{i}")
        for l in picks[:8]:
            print("   ", l[:120])
