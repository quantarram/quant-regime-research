"""
Renders Paper 13's display equations using the same offline matplotlib
mathtext + STIX pipeline established for Paper 12 (render_paper_equations.py)
-- real typeset math, no LaTeX install needed. Kept as a separate script and
separate output directory (eq13_figs/) so Paper 12's already-finalized
equations are never touched by this paper's work.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["svg.fonttype"] = "path"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq13_figs")
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
    "eq1_gap": [
        r"$G(\tau, q) \;=\; C(\tau, q) \,-\, D(\tau, q)$",
    ],
    "eq2_predictability_limit": [
        r"$\tau^{*} \;=\; \underset{\tau \,\in\, \mathrm{tradeable\ lags}}{\arg\max}\; G(\tau, q{=}2)$",
    ],
    "eq3_window_bound": [
        r"$\Delta_{\max} \;=\; w_{\mathrm{train}} + w_{\mathrm{test}} - 1$",
        r"$\mathrm{requiring}\;\; \Delta_{\max} \,\leq\, \tau^{*}, \;\; w_{\mathrm{train}} = w_{\mathrm{test}} = w:$",
        r"$\qquad w \;=\; \left\lfloor\, \tau^{*} / 2 \,\right\rfloor$",
    ],
    "eq4_stale_criterion": [
        r"$\ell_{\mathrm{stale}} \;\geq\; 2\,\tau^{*}$",
    ],
    "eq5_split_criterion": [
        r"$Q(j, s) \;=\!\! \sum_{i:\,x_{ij} \leq s} \!\! \left(y_i - \bar{y}_L\right)^{2}$",
        r"$\qquad\quad +\!\! \sum_{i:\,x_{ij} > s} \!\! \left(y_i - \bar{y}_R\right)^{2}$",
    ],
}

if __name__ == "__main__":
    print(f"Rendering {len(EQUATIONS)} equations to {OUT_DIR}/ ...")
    for name, lines in EQUATIONS.items():
        render(name, lines)
    print("Done.")
