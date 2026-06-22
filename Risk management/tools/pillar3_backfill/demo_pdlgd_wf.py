# -*- coding: utf-8 -*-
"""End-to-end demo of the PD/LGD-driven walk-forward on Deutsche Bank.
Run from the 'Risk management' dir. Scratch — not committed."""
import sys, numpy as np, pandas as pd, logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)
RM = "C:/Users/blero/Downloads/RiskMgmt/Risk management"
sys.path.insert(0, RM)            # root → config.py
sys.path.insert(0, RM + "/backend")
sys.stdout.reconfigure(encoding="utf-8")

from config import EBA_RAW_DIR
from svensson import zero_rate, params_from_row
from macro_factor import factor_covariance
from eba_loader import load_historical_capital_panel, panel_to_wide
import backtesting_walkforward as wf

DB = "7LTWFZYICNSX8D621K86"

brent = pd.read_parquet("data/cache/brent_crude.parquet")
sven  = pd.read_parquet("data/cache/svensson_params.parquet")
bclose = brent["Close_USD"] if "Close_USD" in brent.columns else brent["Close"]
rates = pd.Series([zero_rate(10.0, params_from_row(r), as_decimal=False)
                   for _, r in sven.iterrows()], index=sven.index)
cov = factor_covariance(np.log(bclose).diff(), rates.diff())

panel = load_historical_capital_panel(EBA_RAW_DIR)
cw = panel_to_wide(panel)
periods = sorted(int(p) for p in cw["Period"].unique())
print("capital_wide periods:", periods)
print("DB rows in capital_wide:", len(cw[cw.LEI_Code == DB]))

# (a) prove the time series flows through: RWA scale under fixed adverse M=-2
print("\n(a) Frozen-portfolio RWA scale under M=-2 (adverse), per DB Pillar-3 vintage:")
port = wf.load_portfolio_timeseries()
for v in ["2021-12-31", "2022-12-31", "2023-12-31", "2024-12-31"]:
    fp = wf.build_frozen_portfolio(DB, v, port_df=port)
    base = fp.portfolio_kpis()
    s = wf.frozen_rwa_scale(fp, -2.0)
    print(f"  {v}: scale(M=-2)={s:.4f}  recon-baseRWA={base['rwa']/1e9:6.1f}bn  segs={len(fp.segments)}")

# (b) full PD/LGD-driven walk-forward
macro_q = wf.compute_quarterly_macro(brent, sven, periods)
macro_m = wf.attach_m_factor_quarterly(macro_q, cov_factors=cov, horizon_days=63)
pan = wf.build_pdlgd_walkforward(DB, macro_m, cw)
print(f"\n(b) PD/LGD walk-forward panel rows: {len(pan)}")
if len(pan):
    show = pan[["period_label_end", "pd_vintage", "m_quarter", "rwa_scale",
                "pred_dRWA_credit_eur", "risk_driven_dRWA_credit_eur", "sign_match"]].copy()
    show["pred_bn"] = show.pop("pred_dRWA_credit_eur") / 1e9
    show["realized_bn"] = show.pop("risk_driven_dRWA_credit_eur") / 1e9
    show["m_quarter"] = show["m_quarter"].round(2)
    show["rwa_scale"] = show["rwa_scale"].round(4)
    print(show.to_string(index=False))
    st = wf.walkforward_error_stats(pan)
    print("\nstats:")
    for k, v in st.items():
        print(f"  {k:22} {v:.4g}" if isinstance(v, float) else f"  {k:22} {v}")
