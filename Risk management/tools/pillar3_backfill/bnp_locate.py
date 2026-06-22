#!/usr/bin/env python3
"""Locate 'IRBA EXPOSURE BY PD SCALE' table pages in any BNP URD + show the
page date and the class-block labels, so we can build a PAGEMAP per year."""
import sys, re
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"
path = DIR + sys.argv[1]
TITLE = re.compile(r"IRBA\s+EXPOSURE\s+BY\s+PD\s+SCALE", re.I)
DATE = re.compile(r"31\s*December\s*(\d{4})")
START = re.compile(r"0[.,]00\s*to\s*<\s*0[.,]15")
SUB = re.compile(r"sub[\s‐‑‒–—\-]*total", re.I)

with pdfplumber.open(path) as pdf:
    n = len(pdf.pages)
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else n
    print(f"{sys.argv[1]} pages={n} scanning {lo}-{hi}", flush=True)
    # find pages with the table title, then report a window after each
    title_pages = []
    for i in range(lo, min(hi, n)):
        t = pdf.pages[i].extract_text() or ""
        if TITLE.search(t):
            title_pages.append(i)
    print("title pages:", title_pages)
    seen = set()
    for tp in title_pages:
        for i in range(tp, min(tp + 3, n)):
            if i in seen:
                continue
            seen.add(i)
            t = pdf.pages[i].extract_text() or ""
            dm = DATE.search(t)
            labels = []
            for ln in t.splitlines():
                if START.search(ln):
                    labels.append(ln[:START.search(ln).start()].strip()[:32])
            subs = [ln.strip()[:120] for ln in t.splitlines() if SUB.search(ln)]
            ttl = next((ln.strip()[:70] for ln in t.splitlines() if TITLE.search(ln)), "")
            print(f"\n p{i} date={dm.group(1) if dm else '?'} title='{ttl}'")
            for lab in labels:
                print(f"    START {lab}")
            for s in subs:
                print(f"    SUB   {s}")
