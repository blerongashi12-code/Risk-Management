# -*- coding: utf-8 -*-
"""Batch-download prior-year Pillar-3 PDFs to pillar3_reports/<bank>_<year>.pdf."""
import urllib.request, os, ssl
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
DIR = "C:/Users/blero/Downloads/RiskMgmt/Modellarchitektur/data/pillar3_reports/"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

JOBS = {
 "socgen_2023": "https://www.societegenerale.com/sites/default/files/documents/2024-03/societe-generale-pillar-3-31122023-en.pdf",
 "socgen_2022": "https://www.societegenerale.com/sites/default/files/documents/2023-03/societe-generale-pillar-3-31122022-en.pdf",
 "socgen_2021": "https://www.societegenerale.com/sites/default/files/documents/2022-03/societe-generale-pillar-3-31122021-en.pdf",
 "unicredit_2023": "https://www.unicreditgroup.eu/content/dam/unicreditgroup-eu/documents/en/investors/third-pillar-basel/2023/UniCredit-Group-Disclosure-Pillar-III-as-at-31-December-2023.pdf",
 "unicredit_2022": "https://www.unicreditgroup.eu/content/dam/unicreditgroup-eu/documents/en/investors/third-pillar-basel/2022/UniCredit-Group-Disclosure-Pillar-III-as-at-31-December-2022.pdf",
 "unicredit_2021": "https://www.unicreditgroup.eu/content/dam/unicreditgroup-eu/documents/en/investors/third-pillar-basel/2021/UniCredit-Group-Disclosure-Pillar-III-as-at-31-December-2021.pdf",
 "bpce_2022": "https://www.ppr.groupebpce.com/app/uploads/2024/02/BPCE2022_PIII_EN_BAT_MEL1_23-03-31.pdf",
}
for name, url in JOBS.items():
    out = DIR + name + ".pdf"
    if os.path.exists(out) and os.path.getsize(out) > 100000:
        print(f"SKIP {name} (exists)"); continue
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
            data = r.read()
        with open(out, "wb") as f:
            f.write(data)
        print(f"OK   {name:18} {len(data)/1e6:6.1f} MB")
    except Exception as e:
        print(f"FAIL {name:18} {type(e).__name__}: {str(e)[:60]}")
