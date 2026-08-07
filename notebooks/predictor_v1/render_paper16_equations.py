"""
Renders Paper 16's display equations using the same offline matplotlib
mathtext + STIX pipeline established for Papers 12-15 -- real typeset
math, no LaTeX install needed. Separate output directory (eq16_figs/) so
earlier papers' already-finalized equations are never touched.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["svg.fonttype"] = "path"

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq16_figs")
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
    "eq1_gap_taustar": [
        r"$G(\tau, q) \;=\; C(\tau, q) \,-\, D(\tau, q)$",
        r"$\tau^{*} \;=\; \underset{\tau \,\in\, \mathrm{tradeable\ lags}}{\arg\max}\; G(\tau, q{=}2)$",
    ],
    "eq2_cpe_definition": [
        r"$\mathrm{CPE}(y, h) \;=\; \Pr\!\left(\, |\Delta p_{y}(t\!:\!t+h)| > \theta_{y} "
        r"\;\mid\; \mathrm{cond}_{x}(t) \,\right)$",
    ],
    "eq3_ppo_reward": [
        r"$r_{t} \;=\; \left(a_{t} - 1\right) R_{t} \;-\; c\,\left|a_{t} - a_{t-1}\right|$",
    ],
    "eq4_gae": [
        r"$\delta_{t} \;=\; r_{t} \,+\, \gamma\, V(s_{t+1}) \,-\, V(s_{t})$",
        r"$\hat{A}_{t} \;=\; \sum_{l=0}^{\infty} \left(\gamma \lambda\right)^{l} \delta_{t+l}$",
    ],
    "eq5_ppo_objective": [
        r"$L^{\mathrm{CLIP}}(\theta) \;=\; \mathbb{E}_{t}\!\left[\, "
        r"\min\!\left( \rho_{t}(\theta)\,\hat{A}_{t},\;\; "
        r"\mathrm{clip}\!\left(\rho_{t}(\theta),\, 1{-}\epsilon,\, 1{+}\epsilon\right)\hat{A}_{t} \right) \right]$",
        r"$\mathrm{where}\quad \rho_{t}(\theta) \;=\; "
        r"\dfrac{\pi_{\theta}(a_{t}\,|\,s_{t})}{\pi_{\theta_{\mathrm{old}}}(a_{t}\,|\,s_{t})}$",
    ],
    "eq6_ga_fitness": [
        r"$\mathrm{fitness}(\pi) \;=\; "
        r"\dfrac{\overline{\mathrm{net\_ret}}(\pi)}{\overline{|a_{\pi}|}}$",
    ],
    "eq7_grinold": [
        r"$\mathrm{IR} \;\approx\; \mathrm{IC}\, \sqrt{\mathrm{breadth}}$",
    ],
    "eq8_gpd_tail": [
        r"$\Pr\!\left[\, X > u + y \;\middle|\; X > u \,\right] \;=\; "
        r"\left(1 \,+\, \dfrac{\xi\, y}{\sigma_{u}}\right)^{-1/\xi}$",
    ],
    "eq9_jensens_alpha": [
        r"$r_{\mathrm{strategy}}(t) \;=\; \alpha \,+\, \beta\, r_{\mathrm{benchmark}}(t) \,+\, \varepsilon(t)$",
        r"$\mathrm{SE}(\hat{\alpha}) \;\propto\; \dfrac{\sigma_{\varepsilon}}{\sqrt{n}}$",
    ],
    "eq10_kelly": [
        r"$f^{*}(t) \;=\; \dfrac{\hat{\mu}(t)}{\hat{\sigma}(t)^{2}}$",
        r"$w(t) \;=\; \mathrm{clip}\!\left(0.5\, f^{*}(t),\; 0,\; 2.0\right)$",
    ],
}

if __name__ == "__main__":
    print(f"Rendering {len(EQUATIONS)} equations to {OUT_DIR}/ ...")
    for name, lines in EQUATIONS.items():
        render(name, lines)
    print("Done.")
