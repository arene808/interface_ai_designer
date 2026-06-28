# -*- coding: utf-8 -*-
"""
Nature-quality multi-panel figure: TPU-PLA interface tensile testing
=====================================================================
Figure archetype: quantitative grid (2×2 panels)
Core claim: TPU-PLA interface strength scales with interlock amplitude;
             sinusoidal N3 achieves +75% over flat baseline (2.38 MPa).
Export: 183 mm full-width, SVG + PDF (editable text) + TIFF (600 dpi)
"""

import os, sys, json, warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────
BASE = r"E:\Backup_WK\资料"
for d in os.listdir(BASE):
    full = os.path.join(BASE, d)
    if os.path.isdir(full) and "精益" in d:
        PROJECT = full
        break

PROJ = os.path.join(PROJECT, "interface_ai_designer")
OUT = os.path.join(PROJ, "analysis_output")
FIGDIR = os.path.join(OUT, "nature_figures")
os.makedirs(FIGDIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────
with open(os.path.join(OUT, "training_database.json"), "r", encoding="utf-8") as f:
    db = json.load(f)

# ── Nature style rcParams ─────────────────────────────
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "svg.fonttype": "none",      # editable text
    "pdf.fonttype": 42,           # editable TrueType
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.2,
    "legend.frameon": False,
    "figure.dpi": 300,
})

# ── NMI pastel colour family ──────────────────────────
C_FLAT   = "#B0BEC5"   # blue-grey light
C_SIN    = "#64B5F6"   # soft blue
C_SAW    = "#FF8A65"   # soft deep-orange
C_DOV    = "#81C784"   # soft green
C_SIN_D  = "#1565C0"   # dark blue (for emphasis)
C_SAW_D  = "#D84315"   # dark orange
C_BG     = "#F5F5F5"
C_GRID   = "#E0E0E0"

TYPE_ORDER = ["F0", "F1", "N1", "N2", "N3", "S1", "S2", "S3"]
TYPE_COLORS = {"F0": C_FLAT, "F1": C_FLAT, "N1": C_SIN, "N2": C_SIN, "N3": C_SIN,
               "S1": C_SAW, "S2": C_SAW, "S3": C_SAW}
TYPE_LABELS = {"F0": "Flat (0 mm)", "F1": "Flat (0 mm)", "N1": "Sin (1.2 mm)",
               "N2": "Sin (1.8 mm)", "N3": "Sin (2.4 mm)", "S1": "Saw (1.2 mm)",
               "S2": "Saw (1.8 mm)", "S3": "Saw (2.4 mm)"}

# ── Panel map ─────────────────────────────────────────
# a: strength_comparison  — bar chart with jittered points
# b: stress_strain        — representative curves per group
# c: amplitude_effect     — strength vs amplitude, type-coloured
# d: feature_scatter      — A1 & curvature vs strength

# ========================================================
# BUILD FIGURE
# ========================================================
fig = plt.figure(figsize=(7.2, 6.8), facecolor="white")  # 183 mm = 7.2 in

# ── Panel a: Strength comparison (top-left, 40% width, 50% height) ──
ax_a = fig.add_axes([0.06, 0.52, 0.44, 0.44])

groups = TYPE_ORDER
means = [db["group_stats"].get(g, {}).get("sigma_debond_mean", 0) for g in groups]
stds  = [db["group_stats"].get(g, {}).get("sigma_debond_std", 0) for g in groups]
colors = [TYPE_COLORS[g] for g in groups]
x_pos = np.arange(len(groups))

bars = ax_a.bar(x_pos, means, 0.55, color=colors, edgecolor="white", linewidth=0.3, zorder=2)

# Error bars
ax_a.errorbar(x_pos, means, yerr=stds, fmt="none", ecolor="#333333",
              elinewidth=0.8, capsize=3, capthick=0.8, zorder=3)

# Individual data points with jitter
np.random.seed(42)
for i, g in enumerate(groups):
    vals = db["group_stats"].get(g, {}).get("sigma_debond_values", [])
    if vals:
        jitter = np.random.normal(0, 0.06, len(vals))
        ax_a.scatter(x_pos[i] + jitter, vals, c="white", edgecolors="#333333",
                     linewidth=0.5, s=18, zorder=4, clip_on=False)

# Baseline horizontal
ax_a.axhline(y=means[0], color=C_FLAT, linestyle="--", linewidth=0.7, alpha=0.7, zorder=1)

# Annotations
for i, (m, s) in enumerate(zip(means, stds)):
    if m > 0:
        offset = 0.06 if i != 3 else 0.10
        ax_a.annotate(f"{m:.2f}", (x_pos[i], m + s + offset), ha="center",
                      fontsize=5.8, fontweight="bold", color="#333333")

# Highlight N3 (best) and N2 (anomaly)
ax_a.annotate("★ best", (x_pos[4], means[4] + stds[4] + 0.18), ha="center",
              fontsize=5.5, color=C_SIN_D, fontstyle="italic")
ax_a.annotate("⚠", (x_pos[3], means[3] + stds[3] + 0.10), ha="center",
              fontsize=6, color="#E53935")

ax_a.set_xticks(x_pos)
ax_a.set_xticklabels([TYPE_LABELS[g] for g in groups], rotation=35, ha="right", fontsize=5.8)
ax_a.set_ylabel("Tensile strength (MPa)", fontsize=7)
ax_a.set_ylim(0, 3.2)
ax_a.set_xlim(-0.7, len(groups) - 0.3)
ax_a.grid(axis="y", color=C_GRID, linewidth=0.4, alpha=0.6, zorder=0)
ax_a.text(-0.08, 1.02, "a", transform=ax_a.transAxes, fontsize=11, fontweight="bold",
          va="bottom", ha="left")

# ── Panel b: Stress-strain curves (top-right, 48% width, 50% height) ──
ax_b = fig.add_axes([0.56, 0.52, 0.40, 0.44])

curves_path = os.path.join(OUT, "stress_strain_curves.json")
if os.path.exists(curves_path):
    with open(curves_path, "r", encoding="utf-8") as f:
        curves_data = json.load(f)

    # Plot best and one random specimen per group
    np.random.seed(1)
    for g in ["F0", "N1", "N3", "S1", "S3"]:
        if g in curves_data:
            specimens = curves_data[g]
            color = {"F0": C_FLAT, "N1": "#90CAF9", "N3": C_SIN_D, "S1": "#FFAB91", "S3": C_SAW_D}[g]
            alpha_main = 0.85
            lw = 1.5

            # Best specimen (thick line)
            best = max(specimens, key=lambda x: max(x.get("stress", [0])))
            s_stress = np.array(best.get("stress", []))
            s_strain = np.array(best.get("strain", []))
            if len(s_stress) > 0:
                ax_b.plot(s_strain, s_stress, color=color, linewidth=lw, alpha=alpha_main, zorder=3)

            # Other specimens (thin, faint)
            for spec in specimens:
                if spec is best:
                    continue
                s_stress = np.array(spec.get("stress", []))
                s_strain = np.array(spec.get("strain", []))
                if len(s_stress) > 0:
                    ax_b.plot(s_strain, s_stress, color=color, linewidth=0.5, alpha=0.25, zorder=2)

    # Annotations
    ax_b.annotate("F0 (flat)", (4.5, 1.2), fontsize=5.5, color=C_FLAT, fontweight="bold")
    ax_b.annotate("N1", (7.5, 1.9), fontsize=5.5, color="#90CAF9", fontweight="bold")
    ax_b.annotate("N3 ★", (13, 2.4), fontsize=5.5, color=C_SIN_D, fontweight="bold")
    ax_b.annotate("S1", (8.5, 1.55), fontsize=5.5, color="#FFAB91", fontweight="bold")
    ax_b.annotate("S3", (10, 2.25), fontsize=5.5, color=C_SAW_D, fontweight="bold")

ax_b.set_xlabel("Strain (%)", fontsize=7)
ax_b.set_ylabel("Stress (MPa)", fontsize=7)
ax_b.set_xlim(0, None)
ax_b.set_ylim(0, None)
ax_b.grid(color=C_GRID, linewidth=0.4, alpha=0.6, zorder=0)
ax_b.text(-0.08, 1.02, "b", transform=ax_b.transAxes, fontsize=11, fontweight="bold",
          va="bottom", ha="left")

# ── Panel c: Amplitude effect (bottom-left, 44% width) ──
ax_c = fig.add_axes([0.06, 0.06, 0.44, 0.38])

amps = np.array([1.2, 1.8, 2.4])
n_strengths = np.array([
    db["group_stats"]["N1"]["sigma_debond_mean"],
    db["group_stats"]["N2"]["sigma_debond_mean"],
    db["group_stats"]["N3"]["sigma_debond_mean"],
])
n_stds = np.array([
    db["group_stats"]["N1"]["sigma_debond_std"],
    db["group_stats"]["N2"]["sigma_debond_std"],
    db["group_stats"]["N3"]["sigma_debond_std"],
])
s_strengths = np.array([
    db["group_stats"]["S1"]["sigma_debond_mean"],
    db["group_stats"]["S2"]["sigma_debond_mean"],
    db["group_stats"]["S3"]["sigma_debond_mean"],
])
s_stds = np.array([
    db["group_stats"]["S1"]["sigma_debond_std"],
    db["group_stats"]["S2"]["sigma_debond_std"],
    db["group_stats"]["S3"]["sigma_debond_std"],
])

# Trend lines
ax_c.errorbar(amps, n_strengths, yerr=n_stds, fmt="o-", color=C_SIN_D, linewidth=1.5,
              markersize=7, capsize=3, capthick=0.8, label="Sinusoidal", zorder=3,
              markerfacecolor="white", markeredgewidth=1.2)
ax_c.errorbar(amps, s_strengths, yerr=s_stds, fmt="s--", color=C_SAW_D, linewidth=1.5,
              markersize=7, capsize=3, capthick=0.8, label="Sawtooth", zorder=3,
              markerfacecolor="white", markeredgewidth=1.2)

# Flat baseline
flat_mean = db["group_stats"]["F0"]["sigma_debond_mean"]
ax_c.axhline(y=flat_mean, color=C_FLAT, linestyle=":", linewidth=1, alpha=0.8,
             label=f"Flat baseline ({flat_mean:.2f})", zorder=1)
ax_c.fill_between([0.8, 2.6], flat_mean - 0.22, flat_mean + 0.22,
                   color=C_FLAT, alpha=0.12, zorder=0)

# Annotate points
for a, ns, ss in zip(amps, n_strengths, s_strengths):
    ax_c.annotate(f"{ns:.2f}", (a + 0.04, ns + 0.05), fontsize=5.8, color=C_SIN_D)
    ax_c.annotate(f"{ss:.2f}", (a + 0.04, ss + 0.06), fontsize=5.8, color=C_SAW_D)

# Highlight N2 anomaly
ax_c.annotate("⚠ N2", (1.8, n_strengths[1] - 0.12), ha="center", fontsize=5.5,
              color="#E53935", fontweight="bold")

ax_c.set_xlabel("Amplitude (mm)", fontsize=7)
ax_c.set_ylabel("Tensile strength (MPa)", fontsize=7)
ax_c.set_xlim(0.9, 2.7)
ax_c.set_ylim(0.8, 3.0)
ax_c.legend(fontsize=5.8, loc="upper left", handlelength=1.5, borderpad=0.4)
ax_c.grid(color=C_GRID, linewidth=0.4, alpha=0.6, zorder=0)
ax_c.text(-0.08, 1.02, "c", transform=ax_c.transAxes, fontsize=11, fontweight="bold",
          va="bottom", ha="left")

# ── Panel d: Feature-strength scatter (bottom-right) ──
ax_d1 = fig.add_axes([0.56, 0.06, 0.19, 0.38])
ax_d2 = fig.add_axes([0.77, 0.06, 0.19, 0.38])

for ax_d, feat_idx, feat_name, feat_label in [
    (ax_d1, 0, "A₁", "Fundamental amplitude A₁ (mm)"),
    (ax_d2, 6, "κ_max", "Max curvature κ_max (mm⁻¹)"),
]:
    for g in TYPE_ORDER:
        gs = db["group_stats"].get(g, {})
        itype = gs.get("interface_type", "flat")
        feat = None
        for rec in db["records"]:
            if rec["group"] == g and rec["features"] is not None:
                feat = rec["features"]
                break
        if feat and gs.get("sigma_debond_mean", 0) > 0:
            marker = "s" if itype == "flat" else ("o" if itype == "sinusoidal" else "^")
            color = TYPE_COLORS[g]
            ax_d.scatter(feat[feat_idx], gs["sigma_debond_mean"], s=50, c=color,
                        edgecolors="#333333", linewidth=0.5, marker=marker, zorder=4, alpha=0.9)
            ax_d.annotate(g, (feat[feat_idx], gs["sigma_debond_mean"]),
                         textcoords="offset points", xytext=(4, 3), fontsize=5.5, fontweight="bold")

    ax_d.set_xlabel(feat_label, fontsize=6.5)
    if ax_d == ax_d1:
        ax_d.set_ylabel("Tensile strength (MPa)", fontsize=7)
    else:
        ax_d.tick_params(axis="y", labelleft=False)
    ax_d.grid(color=C_GRID, linewidth=0.4, alpha=0.6, zorder=0)

ax_d1.text(-0.10, 1.02, "d", transform=ax_d1.transAxes, fontsize=11, fontweight="bold",
           va="bottom", ha="left")

# ── Legend for markers ──
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=C_FLAT, markersize=8,
           markeredgecolor="#333333", markeredgewidth=0.5, label="Flat"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C_SIN, markersize=8,
           markeredgecolor="#333333", markeredgewidth=0.5, label="Sinusoidal"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=C_SAW, markersize=8,
           markeredgecolor="#333333", markeredgewidth=0.5, label="Sawtooth"),
]
ax_d2.legend(handles=legend_elements, fontsize=5.5, loc="lower right",
             handlelength=1.2, borderpad=0.3)

# ── Figure title ──
fig.text(0.02, 0.985, "TPU-PLA multi-material 3D-printed interface tensile behaviour",
         fontsize=9, fontweight="bold", ha="left", va="top")
fig.text(0.02, 0.972, "Test standard: GB/T 528-2009 | Speed: 50 mm/min | Date: 2026-06-27 | n = 35 specimens, 8 groups",
         fontsize=5.5, color="#666666", ha="left", va="top")

# ── Export ──
def save_pub(fig, filename, dpi=600):
    svg_path = os.path.join(FIGDIR, f"{filename}.svg")
    pdf_path = os.path.join(FIGDIR, f"{filename}.pdf")
    tiff_path = os.path.join(FIGDIR, f"{filename}.tiff")
    png_path = os.path.join(FIGDIR, f"{filename}.png")

    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)

    sizes = {}
    for p in [svg_path, pdf_path, tiff_path, png_path]:
        sizes[os.path.basename(p)] = f"{os.path.getsize(p)/1024:.0f} KB"
    return sizes

sizes = save_pub(fig, "Fig_TPU-PLA_interface_tensile")

print("=" * 60)
print("Nature-quality figure exported")
print("=" * 60)
for f, s in sizes.items():
    print(f"  {f}: {s}")
print(f"\nOutput directory: {FIGDIR}")
print("Formats: SVG (editable text) + PDF (editable TrueType) + TIFF (600 dpi) + PNG (preview)")
print("\nFigure specs:")
print("  Width: 183 mm (7.2 in) — Nature full page")
print("  Font: Arial 7pt / panel labels 11pt bold")
print("  Palette: NMI pastel (blue-grey / soft blue / soft orange)")
print("  Panels: (a) strength comparison (b) stress-strain curves")
print("          (c) amplitude effect  (d) feature-strength scatter")
