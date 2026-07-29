"""
Renders Paper 14's display equations using the same offline matplotlib
mathtext + STIX pipeline established for Papers 12 and 13 -- real typeset
math, no LaTeX install needed. Separate output directory (eq14_figs/) so
Paper 13's already-finalized equations are never touched by this paper's
work.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["svg.fonttype"] = "path"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq14_figs")
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
    "eq1_rl_policy": [
        r"$\pi(a\,|\,x) \;=\; \mathcal{N}\!\left(a;\; w^{\top}\!x + b,\; \sigma^{2}\right),"
        r"\qquad r \;=\; -(a-y)^{2}$",
        r"$\Delta w \;\propto\; \left(r - \bar{r}\,\right)(a-\mu)\,x$",
    ],
    "eq2_gan_minimax": [
        r"$\min_{G}\,\max_{D}\;\; \mathbb{E}_{(x,y)}\!\left[\log D(x,y)\right]"
        r"\;+\; \mathbb{E}_{x,z}\!\left[\log\!\left(1 - D(x, G(x,z))\right)\right]$",
    ],
    "eq3_vae_elbo": [
        r"$\mathcal{L} \;=\; \mathbb{E}_{q(z\,|\,x,y)}\!\left[\log p(y\,|\,x,z)\right]"
        r"\;-\; \mathrm{KL}\!\left(q(z\,|\,x,y)\,\|\,p(z)\right)$",
    ],
    "eq4_window_sweep": [
        r"$w_{\mathrm{train}}(m) \;=\; \mathrm{round}\!\left(m \cdot \tau^{*}\right),"
        r"\qquad m \in \{0.5, 1, 1.5, 2, 3, 4, 6, 8\}$",
    ],
}

if __name__ == "__main__":
    print(f"Rendering {len(EQUATIONS)} equations to {OUT_DIR}/ ...")
    for name, lines in EQUATIONS.items():
        render(name, lines)
    print("Done.")
