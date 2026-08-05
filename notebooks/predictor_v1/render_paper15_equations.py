"""
Renders Paper 15's display equations using the same offline matplotlib
mathtext + STIX pipeline established for Papers 12-14 -- real typeset
math, no LaTeX install needed. Separate output directory (eq15_figs/) so
earlier papers' already-finalized equations are never touched.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["svg.fonttype"] = "path"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq15_figs")
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
    "eq3_pinball_loss": [
        r"$L_{\tau}(y,\hat{y}) \;=\; \max\!\left(\tau(y-\hat{y}),\;\; (\tau-1)(y-\hat{y})\right)$",
    ],
    "eq4_lq_loss": [
        r"$L_{q}(r) \;=\; |r|^{q}, \qquad r \;=\; \hat{y} - y$",
        r"$\dfrac{\partial L_{q}}{\partial \hat{y}} \;=\; q\,\mathrm{sign}(r)\,|r|^{q-1}$",
    ],
    "eq5_shape_vae": [
        r"$z \sim \mathcal{N}(0,I), \qquad x \,|\, z,c \;\sim\; \mathcal{N}\!\left(\mu_{d}(z,c),\;\sigma_{d}^{2}I\right)$",
        r"$\mu_{d}(z,c) \;=\; W_{dz}\,z \,+\, w_{dc}\,c \,+\, b_{d} \;\in\; \mathbb{R}^{h}$",
        r"$\mathcal{L} \;=\; \mathbb{E}_{q(z|x,c)}\!\left[\log p(x|z,c)\right] \,-\, \beta\,\mathrm{KL}\!\left(q(z|x,c)\,\|\,\mathcal{N}(0,I)\right)$",
    ],
    "eq6_variance_calibration": [
        r"$\mathrm{Var}(x) \;=\; \mathrm{Var}_{z}\!\left(\mu_{d}(z,c)\right) \,+\, \sigma_{d}^{2}$",
        r"$\sigma_{d}^{2} \;=\; \max\!\left(1 \,-\, \mathrm{Var}_{z}(\mu_{d}),\;\; \epsilon\right)$",
    ],
    "eq7_multiblock_chaining": [
        r"$n_{\mathrm{blocks}} \;=\; \left\lceil H / \tau^{*} \right\rceil, "
        r"\qquad c_{i} \;=\; r_{\mathrm{coarse}} \cdot \dfrac{L_{i}}{H}$",
        r"$R_{\mathrm{total}} \;=\; \sum_{i=1}^{n_{\mathrm{blocks}}} \sum_{t=1}^{L_{i}} x_{i,t}$",
    ],
    "eq8_crps_energy_score": [
        r"$\mathrm{CRPS}(\mathcal{E}, y) \;\approx\; \dfrac{1}{K}\sum_{i=1}^{K} |x_{i} - y| "
        r"\,-\, \dfrac{1}{2K^{2}}\sum_{i=1}^{K}\sum_{j=1}^{K} |x_{i} - x_{j}|$",
    ],
}

if __name__ == "__main__":
    for name, lines in EQUATIONS.items():
        render(name, lines)
    print(f"\nSaved {len(EQUATIONS)} equations to {OUT_DIR}/")
