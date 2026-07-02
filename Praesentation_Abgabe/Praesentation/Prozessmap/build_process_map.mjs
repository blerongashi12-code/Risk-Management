import fs from "node:fs/promises";

const W = 3200;
const H = 1800;
const out = new URL("./Risk_Management_Prozessmap.svg", import.meta.url);

const esc = (s) => String(s)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;");

const lines = (items, x, y, opts = {}) => {
  const {
    size = 28,
    color = "#334E5C",
    weight = 400,
    lineHeight = 1.35,
    anchor = "start",
    family = "Segoe UI, Arial, sans-serif",
  } = opts;
  return `<text x="${x}" y="${y}" text-anchor="${anchor}" font-family="${family}" font-size="${size}" font-weight="${weight}" fill="${color}">` +
    items.map((t, i) => `<tspan x="${x}" dy="${i === 0 ? 0 : size * lineHeight}">${esc(t)}</tspan>`).join("") +
    `</text>`;
};

const rounded = (x, y, w, h, fill, stroke = "none", r = 22, sw = 2) =>
  `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}"/>`;

const pill = (x, y, w, text, fill, color = "#FFFFFF") =>
  `${rounded(x, y, w, 46, fill, "none", 23)}${lines([text], x + w / 2, y + 31, { size: 21, color, weight: 700, anchor: "middle" })}`;

const arrow = (x1, y1, x2, y2, color = "#607985", width = 5, dashed = false) =>
  `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${width}" ${dashed ? 'stroke-dasharray="14 12"' : ""} marker-end="url(#arrow)"/>`;

const phase = (n, title, x, y, w, color) =>
  `${rounded(x, y, w, 62, color, "none", 15)}
   ${rounded(x + 10, y + 9, 44, 44, "#FFFFFF", "none", 22)}
   ${lines([n], x + 32, y + 39, { size: 23, color, weight: 800, anchor: "middle" })}
   ${lines([title], x + 70, y + 40, { size: 26, color: "#FFFFFF", weight: 700 })}`;

const card = ({ x, y, w, h, title, subtitle, body = [], accent = "#034B6F", fill = "#FFFFFF", badge }) => {
  let s = `<g filter="url(#shadow)">${rounded(x, y, w, h, fill, "#D8E1E6", 22, 2)}</g>`;
  s += `<rect x="${x}" y="${y}" width="10" height="${h}" rx="5" fill="${accent}"/>`;
  if (badge) s += pill(x + 30, y + 24, badge.w, badge.text, badge.fill || accent, badge.color || "#FFFFFF");
  const titleY = badge ? y + 105 : y + 48;
  s += lines([title], x + 32, titleY, { size: 30, color: "#051C2C", weight: 750 });
  if (subtitle) s += lines([subtitle], x + 32, titleY + 38, { size: 21, color: accent, weight: 650 });
  if (body.length) s += lines(body, x + 32, titleY + 85, { size: 23, color: "#405965", lineHeight: 1.42 });
  return s;
};

let svg = `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<defs>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
    <feDropShadow dx="0" dy="10" stdDeviation="12" flood-color="#051C2C" flood-opacity="0.10"/>
  </filter>
  <marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,10 L12,5 z" fill="#607985"/>
  </marker>
  <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#F8FAFB"/>
    <stop offset="100%" stop-color="#EEF3F5"/>
  </linearGradient>
  <linearGradient id="final" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0%" stop-color="#051C2C"/>
    <stop offset="100%" stop-color="#123E55"/>
  </linearGradient>
</defs>

<rect width="${W}" height="${H}" fill="url(#bg)"/>
<rect x="0" y="0" width="${W}" height="16" fill="#A52F4D"/>

${lines(["EU BANKING CREDIT STRESS COCKPIT"], 120, 102, { size: 24, color: "#A52F4D", weight: 800 })}
${lines(["Prozessmap des Gesamtmodells"], 120, 170, { size: 55, color: "#051C2C", weight: 750 })}
${lines(["Von bankpublizierten Ausgangsdaten und Makroschocks bis zur gestressten CET1-Quote"], 120, 218, { size: 27, color: "#526A76" })}
${pill(2655, 92, 425, "2-Faktor · 2-Kanal · 10 IRB-Banken", "#051C2C")}

<!-- Phase headers -->
${phase("1", "Datenbasis", 120, 285, 430, "#4B6573")}
${phase("2", "Stressfaktoren", 610, 285, 430, "#034B6F")}
${phase("3", "Transmission", 1100, 285, 460, "#1B6B73")}
${phase("4", "Risikokanäle", 1620, 285, 760, "#A52F4D")}
${phase("5", "Aggregation & Ergebnis", 2440, 285, 640, "#051C2C")}

<!-- Phase 1 -->
${card({
  x: 120, y: 385, w: 430, h: 485,
  title: "Ausgangsbasis",
  subtitle: "Einheitlicher Modell-Stichtag",
  accent: "#4B6573",
  badge: { text: "BASELINE", w: 138, fill: "#4B6573" },
  body: [
    "Pillar 3 · EU-CR6",
    "PD & LGD je Bank × Klasse",
    "",
    "EBA Transparency",
    "EAD · CET1 · RWA · Sovereigns",
    "",
    "Marktdaten",
    "ICE Brent · Bundesbank 10Y",
    "",
    "Universe: 10 größte EU-IRB-Banken"
  ]
})}

<!-- Phase 2 -->
${card({
  x: 610, y: 385, w: 430, h: 222,
  title: "ΔBrent",
  subtitle: "Ölpreis-Schock · Log-Return",
  accent: "#034B6F",
  badge: { text: "FAKTOR A", w: 138, fill: "#034B6F" },
  body: ["Energie · Inflation · Inputkosten", "wirken auf Schuldner-Cashflows"]
})}
${card({
  x: 610, y: 648, w: 430, h: 222,
  title: "Δr 10Y",
  subtitle: "Zinsschock · Prozentpunkte",
  accent: "#A52F4D",
  badge: { text: "FAKTOR B", w: 138, fill: "#A52F4D" },
  body: ["Refinanzierung · Affordability", "und Bond-Bewertung"]
})}
${rounded(648, 900, 354, 86, "#E8F1F4", "#B8CDD5", 18, 2)}
${lines(["Empirisch nahezu unabhängig", "ρ ≈ +0,07 · separate Modellierung"], 825, 935, { size: 22, color: "#214C60", weight: 650, anchor: "middle", lineHeight: 1.35 })}

<!-- Phase 3 -->
${card({
  x: 1100, y: 385, w: 460, h: 485,
  title: "Sektorale Übersetzung",
  subtitle: "Gleicher Schock · andere Wirkung",
  accent: "#1B6B73",
  badge: { text: "β / γ-MATRIX", w: 178, fill: "#1B6B73" },
  body: [
    "ΔPD = βoil × ΔBrent",
    "          + βrate × Δr10Y",
    "",
    "ΔLGD = γoil × ΔBrent",
    "           + γrate × Δr10Y",
    "",
    "Corporate · SME · Mortgage",
    "QRRE · Other Retail · Bank",
    "Sovereign: kein PD/LGD-Zins-β",
    "(Vermeidung von Doppelzählung)"
  ]
})}

<!-- arrows phases 1-3 -->
${arrow(550, 628, 610, 628)}
${arrow(1040, 628, 1100, 628)}

<!-- Phase 4 two channels -->
${card({
  x: 1620, y: 385, w: 760, h: 340,
  title: "Kanal 1 · Kreditbuch",
  subtitle: "Ausfall- und Kapitalanforderungs-Kanal",
  accent: "#A52F4D",
  badge: { text: "LOAN BOOK", w: 150, fill: "#A52F4D" },
  body: [
    "Gestresste PD & LGD je Bank × Exposure-Klasse",
    "→ Expected Loss = PD × LGD × EAD",
    "→ Risiko-Vorsorge senkt CET1 (Zähler)",
    "→ Basel-III-IRB / Vasicek-Kapitalformel",
    "→ zusätzliche Kredit-RWA erhöhen den Nenner"
  ]
})}
${card({
  x: 1620, y: 770, w: 760, h: 340,
  title: "Kanal 2 · Sovereign-Buch",
  subtitle: "Zins- und Marktwert-Kanal",
  accent: "#D69E2E",
  badge: { text: "SOVEREIGNS", w: 180, fill: "#D69E2E", color: "#051C2C" },
  body: [
    "ΔFair Value ≈ − Duration × Δr × Exposure",
    "→ Laufzeit-Buckets bestimmen die Zinssensitivität",
    "→ IFRS-9-Filter je Bank: HfT + FVTPL + FVOCI",
    "→ nur marktwertwirksamer Anteil senkt CET1",
    "→ Amortised Cost bleibt bilanziell zunächst latent"
  ]
})}

<!-- split arrows: sector transmission feeds both risk channels -->
<path d="M1560 555 L1620 555" fill="none" stroke="#607985" stroke-width="5" marker-end="url(#arrow)"/>
<path d="M1560 700 C1590 700, 1580 940, 1620 940" fill="none" stroke="#607985" stroke-width="5" marker-end="url(#arrow)"/>

<!-- Phase 5 aggregation -->
${rounded(2440, 385, 640, 435, "url(#final)", "none", 24)}
${pill(2480, 420, 196, "CET1-BRIDGE", "#A52F4D")}
${lines(["Zusammenführung"], 2480, 525, { size: 36, color: "#FFFFFF", weight: 750 })}
${lines([
  "Neuer Zähler =",
  "CET1 vorher − Kreditvorsorge",
  "− Sovereign-Marktwertverlust",
  "",
  "Neuer Nenner =",
  "RWA vorher + zusätzliche Kredit-RWA",
  "",
  "Gestresste CET1-Quote = Zähler / Nenner"
], 2480, 580, { size: 25, color: "#EAF0F3", weight: 500, lineHeight: 1.42 })}

${card({
  x: 2440, y: 850, w: 640, h: 280,
  title: "Regulatorisches Ergebnis",
  subtitle: "Bankenvergleich & Schwellenanalyse",
  accent: "#051C2C",
  badge: { text: "OUTPUT", w: 118, fill: "#051C2C" },
  body: [
    "CET1 vor Stress → CET1 nach Stress",
    "Pillar 1: 4,5 % · inkl. CCB: 7,0 %",
    "SREP-Orientierung: 8,0 %"
  ]
})}

<!-- merge arrows -->
${arrow(2380, 555, 2440, 555)}
${arrow(2380, 940, 2440, 730)}
${arrow(2760, 820, 2760, 850)}

<!-- Bottom navigation / validation -->
${rounded(120, 1195, 2960, 420, "#FFFFFF", "#D7E0E5", 24, 2)}
${lines(["LESEREIHENFOLGE IM COCKPIT"], 165, 1255, { size: 22, color: "#A52F4D", weight: 800 })}

${pill(165, 1290, 420, "TAB 1 · FAKTOR-ANALYSE", "#034B6F")}
${lines(["Korrelation & Regression", "Schock → β/γ-Transmission"], 165, 1370, { size: 24, color: "#334E5C", weight: 600 })}

${arrow(600, 1375, 690, 1375)}

${pill(710, 1290, 360, "TAB 2 · KREDITBUCH", "#A52F4D")}
${lines(["PD/LGD-Stress · Expected Loss", "IRB-RWA · Bank-Drilldown"], 710, 1370, { size: 24, color: "#334E5C", weight: 600 })}

${arrow(1085, 1375, 1175, 1375)}

${pill(1195, 1290, 360, "TAB 3 · MARKTBUCH", "#D69E2E", "#051C2C")}
${lines(["Yield Curve · Duration-MtM", "IFRS-9-Sovereign-Split"], 1195, 1370, { size: 24, color: "#334E5C", weight: 600 })}

${arrow(1570, 1375, 1660, 1375)}

${pill(1680, 1290, 390, "TAB 4 · EIGENKAPITAL", "#051C2C")}
${lines(["2-Kanal-CET1-Waterfall", "Schwellen & Sensitivität"], 1680, 1370, { size: 24, color: "#334E5C", weight: 600 })}

${arrow(2085, 1375, 2175, 1375)}

${pill(2195, 1290, 360, "TAB 5 · VALIDIERUNG", "#1B6B73")}
${lines(["Walk-Forward-Backtest", "Plausibilisierung & Governance"], 2195, 1370, { size: 24, color: "#334E5C", weight: 600 })}

${rounded(2595, 1288, 430, 190, "#F3F7F8", "#C9D8DE", 18, 2)}
${lines(["MODELLGRENZEN"], 2625, 1330, { size: 20, color: "#4B6573", weight: 800 })}
${lines(["1-Jahres-Sicht · statische EAD", "Parallel-Shift · kein Hedging", "keine Investment-Empfehlung"], 2625, 1372, { size: 22, color: "#526A76", weight: 550, lineHeight: 1.45 })}

<!-- validation feedback -->
<path d="M2370 1515 C2050 1690, 1150 1690, 825 1515" fill="none" stroke="#1B6B73" stroke-width="5" stroke-dasharray="16 12" marker-end="url(#arrow)"/>
${lines(["Validierung und Sensitivitäts-Checks fließen als Feedback in Kalibrierung und Annahmen zurück"], 1600, 1690, { size: 23, color: "#1B6B73", weight: 700, anchor: "middle" })}

${lines(["Single Source of Truth: MODEL_ASSUMPTIONS.md  ·  Baseline 31.12.2024  ·  EBA Transparency 2025"], 1600, 1760, { size: 20, color: "#6A7D86", weight: 500, anchor: "middle" })}
</svg>`;

await fs.writeFile(out, svg, "utf8");
export default out.pathname;
