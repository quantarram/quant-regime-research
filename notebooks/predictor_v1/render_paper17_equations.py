"""
Renders Paper 16's display equations using the same offline matplotlib
mathtext + STIX pipeline established for Papers 12-15 -- real typeset
math, no LaTeX install needed. Separate output directory (eq17_figs/) so
earlier papers' already-finalized equations are never touched.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["svg.fonttype"] = "path"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq17_figs")
os.makedirs(OUT_DIR, exist_ok=True)

FONTSIZE = 13.0
MAX_WIDTH_PT = 455


def render(name, lines, fontsize=FONTSIZE):
    fig = plt.figure(figsize=(0.1, 0.1))
    renderer = fig.canvas.get_renderer()
    heights, max_w = [], 0
    for line in lines:
        t = fig.text(0, 0, line, fontsize=fontsize)
        bb = t.get_window_extent(renderer=renderer)
        max_w = max(max_w, bb.width)
        heights.append(bb.height * 1.35)
    plt.close(fig)

    total_h = sum(heights)
    pad_w = max_w * 1.06
    fig = plt.figure(figsize=(pad_w / 72, total_h / 72))
    y_top = total_h
    for line, h in zip(lines, heights):
        y_top -= h
        fig.text(0.0, (y_top + h / 2) / total_h, line, fontsize=fontsize, va="center", ha="left")
    fig.savefig(os.path.join(OUT_DIR, f"{name}.svg"), transparent=True)
    plt.close(fig)
    width_pt = pad_w
    n = len(lines)
    flag = "  *** TOO WIDE ***" if width_pt > MAX_WIDTH_PT else ""
    print(f"  {name}.svg  ({n} line{'s' if n != 1 else ''}, {width_pt:.0f}pt){flag}")


EQUATIONS = {
    "eq1_demand": [
        r"$D_{r,d} \;=\; \beta_{0} \,+\, \beta_{1}\,\dfrac{H_{r,d}}{10} \,+\, \beta_{2}\,\dfrac{C_{r,d}}{10} \,+\, "
        r"\beta_{3}\left(\dfrac{H_{r,d}}{10}\right)^{2} \,+\, \beta_{4}\left(\dfrac{C_{r,d}}{10}\right)^{2}$",
        r"$+\;\sum_{k=0}^{5} \gamma_{k}\,\mathbb{1}[\mathrm{dow}_{d}=k] \,+\, \delta\, y_{d} \,+\, \eta\, \mathrm{hol}_{d} \,+\, "
        r"\kappa\,\dfrac{C_{r,d}}{10}\,\mathbb{1}[\mathrm{May,Jun}]$",
        r"$H = \max(65 - T_{F},\, 0), \qquad C = \max(T_{F} - 65,\, 0)$",
    ],
    "eq2_wind": [
        r"$G_{r,d} \;=\; \left(1 + g\,y_{d}\right) \sum_{i \in r}\;\sum_{k=1}^{8} w_{i,k}\;\phi_{k}\!\left(v_{i,d}\right), \qquad w_{i,k} \geq 0$",
    ],
    "eq3_anchor": [
        r"$f_{r,d} \;=\; \dfrac{\sum_{s \in S_{d}} D^{\mathrm{obs}}_{r,s}}{\sum_{s \in S_{d}} D^{\mathrm{clim}}_{r,s}}$",
    ],
    "eq4_gas": [
        r"$A_{t} \;=\; \sum_{k=1}^{7}\left[\, \mathrm{DD}^{\,t}_{t+k} \,-\, \overline{\mathrm{DD}}_{\,t+k} \,\right]$",
        r"$R_{t} \;=\; \sum_{k=1}^{7}\left[\, \mathrm{DD}^{\,t}_{t+k} \,-\, \mathrm{DD}^{\,t-1}_{t+k} \,\right]$",
    ],
}

if __name__ == "__main__":
    print(f"Rendering {len(EQUATIONS)} equations to {OUT_DIR}/ ...")
    for name, lines in EQUATIONS.items():
        render(name, lines)
    print("Done.")
