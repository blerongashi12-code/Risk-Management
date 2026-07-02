# -*- coding: utf-8 -*-
"""Vier didaktische Schaubilder für Modellannahmen.docx (Markenfarben)."""
import sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "backend"))
from two_factor_stress import SENSITIVITY_MATRIX

OUT = os.path.dirname(os.path.abspath(__file__)) + "/figs"
os.makedirs(OUT, exist_ok=True)

NAVY, MID, BLUE = "#051C2C", "#034B6F", "#2251FF"
CRIM, AMBER, TEAL = "#A52F4D", "#C9A227", "#00A9A5"
STONE, HAIR, CANVAS = "#6E6E6E", "#E6E6E6", "#F7F9FB"
plt.rcParams.update({"font.family": "Arial", "font.size": 11})


def box(ax, x, y, w, h, text, fc, tc="white", fs=11, weight="bold", ec=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec=ec or fc, lw=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight=weight, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=STONE, lw=2.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, color=color, lw=lw))


# ============================================================ 1 · Pipeline
fig, ax = plt.subplots(figsize=(10.6, 4.6))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

box(ax, 0.15, 3.35, 1.9, 0.95, "Ölpreis-Schock\nΔBrent (log)", AMBER, fs=10.5)
box(ax, 0.15, 1.95, 1.9, 0.95, "Zins-Schock\nΔr_10y (pp)", CRIM, fs=10.5)
box(ax, 2.75, 2.6, 2.1, 1.35,
    "β-Transmission\nje Kreditklasse\n(ΔPD, ΔLGD)", MID, fs=10.5)
box(ax, 5.55, 2.6, 1.95, 1.35,
    "IRB-Kapitalformel\n(Basel III)\n→ ΔRWA, ΔEL", NAVY, fs=10.5)
box(ax, 2.75, 0.45, 4.75, 1.15,
    "Sovereign-Kanal\nΔMtM = −Duration · Δr · Exposure\n"
    "(CET1-wirksam: nur FVOCI/FVTPL-Anteil)",
    "#5B7C93", fs=9)
box(ax, 8.15, 1.55, 1.7, 2.4, "CET1-Quote\nunter Stress\n\n(Zielgröße)", TEAL, fs=11)

arrow(ax, 2.05, 3.82, 2.75, 3.45)
arrow(ax, 2.05, 2.43, 2.75, 2.95)
arrow(ax, 4.85, 3.27, 5.55, 3.27)
arrow(ax, 7.5, 3.27, 8.15, 3.3)
arrow(ax, 2.05, 2.3, 2.6, 1.15)          # Zins → Sovereign
arrow(ax, 7.5, 1.05, 8.15, 2.0)
ax.text(5.0, 4.6, "Vom Makro-Schock zur Eigenkapitalquote — zwei Kanäle, eine Zielgröße",
        ha="center", fontsize=12.5, color=NAVY, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_pipeline.png", dpi=200,
                                facecolor="white"); plt.close()

# ============================================================ 2 · β-Heatmap
cls = ["corporate", "sme_corporate", "mortgage", "qrre",
       "other_retail", "bank", "sovereign"]
cols = [("pd_rate", "β PD · Zins"), ("pd_oil", "β PD · Öl"),
        ("lgd_rate", "γ LGD · Zins"), ("lgd_oil", "γ LGD · Öl")]
Z = np.array([[SENSITIVITY_MATRIX[c][k] for k, _ in cols] for c in cls])

fig, ax = plt.subplots(figsize=(8.4, 4.4))
im = ax.imshow(Z, cmap="RdBu_r", vmin=-1.6, vmax=1.6, aspect="auto")
ax.set_xticks(range(len(cols)), [l for _, l in cols], fontsize=10.5)
ax.set_yticks(range(len(cls)), cls, fontsize=10.5)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        v = Z[i, j]
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(v) > 0.9 else NAVY, fontweight="bold")
ax.set_title("Sensitivitäts-Matrix: Reaktion der Risikoparameter je Klasse\n"
             "(Prozentpunkte je Faktor-Einheit — rot = Risiko steigt, blau = sinkt)",
             fontsize=11.5, color=NAVY, pad=10)
fig.colorbar(im, ax=ax, shrink=0.85, label="β / γ")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_beta.png", dpi=200,
                                facecolor="white"); plt.close()

# ============================================================ 3 · CET1-Bridge
fig, ax = plt.subplots(figsize=(10.2, 4.0))
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")

box(ax, 0.2, 1.5, 2.3, 1.5, "CET1-Quote\nvor Stress\n= CET1 ÷ RWA", STONE, fs=11)
box(ax, 3.4, 2.6, 3.1, 1.15,
    "Zähler sinkt:\nCET1 − ΔEL (Kreditbuch)\n− ΔMtM (Sovereign)", CRIM, fs=10)
box(ax, 3.4, 0.65, 3.1, 1.15,
    "Nenner steigt:\nRWA + ΔRWA_credit", MID, fs=10.5)
box(ax, 7.3, 1.5, 2.4, 1.5,
    "CET1-Quote\nunter Stress\n(konservativ: kein\nNII-Gegeneffekt)", TEAL, fs=10)

arrow(ax, 2.5, 2.5, 3.4, 3.1)
arrow(ax, 2.5, 2.0, 3.4, 1.3)
arrow(ax, 6.5, 3.1, 7.3, 2.55)
arrow(ax, 6.5, 1.25, 7.3, 1.95)
ax.text(5.0, 4.15, "Die CET1-Bridge: beide Stress-Kanäle drücken die Quote von zwei Seiten",
        ha="center", fontsize=12.5, color=NAVY, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_bridge.png", dpi=200,
                                facecolor="white"); plt.close()

# ============================================================ 4 · Walk-Forward
fig, ax = plt.subplots(figsize=(10.6, 4.2))
ax.set_xlim(0, 10.6); ax.set_ylim(0, 4.6); ax.axis("off")

ax.annotate("", xy=(10.15, 1.1), xytext=(0.3, 1.1),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.4))
years = {"31.12.2021": 1.15, "31.12.2022": 4.15, "31.12.2023": 7.15}
for lbl, x in years.items():
    ax.plot([x, x], [0.95, 1.25], color=NAVY, lw=2.4)
    ax.text(x, 0.62, lbl, ha="center", fontsize=10.5, color=NAVY,
            fontweight="bold")
for i, (x0, yr) in enumerate([(1.15, "2022"), (4.15, "2023"), (7.15, "2024")]):
    box(ax, x0 - 0.92, 3.35, 1.84, 0.8,
        f"Portfolio einfrieren\n(PD/LGD/EAD {2021 + i})", MID, fs=8.6)
    ax.add_patch(FancyArrowPatch((x0, 3.33), (x0, 1.32), arrowstyle="-|>",
                                 mutation_scale=13, color=MID, lw=1.7))
    box(ax, x0 + 0.28, 2.15, 2.0, 0.78,
        f"realer Öl-/Zins-Schock\n{yr} einspeisen", AMBER, "black", fs=8.6)
    ax.add_patch(FancyArrowPatch((x0 + 2.3, 2.5), (x0 + 2.86, 1.7),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=TEAL, lw=1.7))
    box(ax, x0 + 2.32, 1.32, 1.35, 0.62,
        f"vs. gemeldete\nCET1 {yr}", TEAL, fs=8.0)
ax.text(5.3, 4.35, "Walk-Forward ohne Blick in die Zukunft: einfrieren → Schock → vergleichen → weiterrollen",
        ha="center", fontsize=12, color=NAVY, fontweight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_walkforward.png", dpi=200,
                                facecolor="white"); plt.close()

print("4 Schaubilder:", OUT)
