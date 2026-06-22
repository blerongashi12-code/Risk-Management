# -*- coding: utf-8 -*-
"""Cheap generic calibration probe across all downloaded FY2024 Pillar-3 PDFs.
For each bank, scan sub-total rows and try to locate each known (PD,LGD) pair.
Tells us which banks the deterministic extractor handles + column positions.
No LLM agents — pure pdfplumber."""
import re, sys, glob, os, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
import pdfplumber
sys.stdout.reconfigure(encoding="utf-8")

DIR = "C:/Users/blero/Downloads/RiskMgmt/Risk management/data/pillar3_reports/"

# knowns: class -> (pd, lgd)   order corp, sme, mort, qrre, other, bank, sov
KN = {
 "ing":      {"corporate":(1.81,14.43),"sme_corporate":(6.12,27.39),"mortgage":(1.46,22.61),"qrre":(3.75,22.50),"other_retail":(6.39,39.90),"bank":(0.39,19.79),"sovereign":(0.10,15.00)},
 "socgen":   {"corporate":(2.39,30.18),"sme_corporate":(3.57,19.06),"mortgage":(1.18,16.37),"qrre":(7.89,48.48),"other_retail":(8.06,31.89),"bank":(0.36,22.68),"sovereign":(0.09,1.54)},
 "rabobank": {"corporate":(9.98,22.20),"sme_corporate":(5.05,12.82),"mortgage":(1.50,15.00),"qrre":(2.90,15.37),"other_retail":(5.38,36.34),"bank":(2.23,10.65),"sovereign":(0.05,26.55)},
 "unicredit":{"corporate":(2.42,38.42),"sme_corporate":(3.93,14.08),"mortgage":(1.41,14.07),"qrre":(2.97,54.10),"other_retail":(5.68,25.87),"bank":(2.74,34.12),"sovereign":(0.06,8.25)},
 "bnp":      {"corporate":(2.87,37.00),"sme_corporate":(3.78,25.00),"mortgage":(1.61,17.00),"qrre":(8.91,36.00),"other_retail":(6.70,36.00),"bank":(0.79,26.00),"sovereign":(0.08,3.00)},
 "bpce":     {"corporate":(5.68,33.28),"sme_corporate":(10.69,26.23),"mortgage":(14.68,10.70),"qrre":(9.63,33.85),"other_retail":(21.55,28.88),"bank":(0.41,40.42),"sovereign":(0.00,7.26)},
 "cm":       {"corporate":(2.59,45.00),"sme_corporate":(5.69,25.00),"mortgage":(1.79,16.00),"qrre":(3.13,33.00),"other_retail":(2.59,19.00),"bank":(0.12,34.00),"sovereign":(0.12,34.00)},
 "santander":{"corporate":(4.01,44.88),"sme_corporate":(7.68,42.53),"mortgage":(2.44,15.08),"qrre":(4.20,83.56),"other_retail":(6.77,42.53),"bank":(1.02,15.80)},
}
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def nums(s):
    out=[]
    for t in NUM.findall(s):
        c=t.replace(",","")
        try: out.append(float(c))
        except: pass
    return out

def subrows(path):
    rows=[]
    pdf=pdfplumber.open(path)
    for i,pg in enumerate(pdf.pages):
        t=pg.extract_text() or ""
        ctx=""
        for ln in t.split("\n"):
            s=ln.strip()
            if not s: continue
            letters=re.sub(r"[^A-Za-z]","",s); digits=re.sub(r"[^0-9]","",s)
            if len(letters)>3 and len(digits)<=3 and "sub-total" not in s.lower() and "sous-total" not in s.lower():
                ctx=s
            if re.match(r"^(Sub-total|Subtotal|Total|Sous-total)\b", s, re.I):
                n=nums(s)
                if len(n)>=8: rows.append((i+1,ctx,n,s[:90]))
    pdf.close()
    return rows

for short in KN:
    path=f"{DIR}{short}_2024.pdf"
    if not os.path.exists(path):
        print(f"\n### {short}: NO FY2024 PDF"); continue
    rows=subrows(path)
    print(f"\n### {short}  ({len(rows)} sub-total rows)")
    for cls,(p,l) in KN[short].items():
        hits=[]
        for (pg,ctx,n,raw) in rows:
            pp=[j for j,v in enumerate(n) if abs(v-p)<=0.05]
            lp=[j for j,v in enumerate(n) if abs(v-l)<=0.05]
            if pp and lp: hits.append((pg,ctx[:34],pp,lp))
        tag="OK " if len(hits)>=1 else "MISS"
        extra=""
        if hits:
            pg,ctx,pp,lp=hits[0]
            extra=f"p{pg} pdpos={pp} lgdpos={lp} '{ctx}'"+(f"  (+{len(hits)-1} more)" if len(hits)>1 else "")
        print(f"  [{tag}] {cls:14} PD={p} LGD={l}  {extra}")
