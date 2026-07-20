"""
Renders the predictor_v1 preprint's display equations as vector (SVG) images
using matplotlib's built-in mathtext engine -- real typeset math (stacked
fractions, sized sums, proper sub/superscripts), no LaTeX install required,
fully offline. Run once; outputs land in predictor_v1/eq_figs/ and are
embedded via <img> in predictor_v1_paper_draft.md.

STIX font chosen to visually match the paper's Georgia serif body text
(paper_style.css) -- STIX was designed by AIP/APS/IEEE specifically to pair
with Times-family serif body text, which is what most of the author's actual
peer-reviewed venues (Scientific Reports, Earth and Space Science, HESS) use
for math typesetting.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["svg.fonttype"] = "path"  # embed as vector outlines, not font refs

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eq_figs")
os.makedirs(OUT_DIR, exist_ok=True)

FONTSIZE = 13.0
MAX_WIDTH_PT = 455  # printable column width at A4 w/ 2.2cm margins (~470pt), minus a safety margin


def render(name, lines, fontsize=FONTSIZE):
    """lines: list of mathtext strings, one per stacked row (each already
    wrapped in $...$ where needed), rendered left-aligned as one equation
    block."""
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
    "eq_structfunc": [
        r"$\left\langle |\Delta p(\tau)|^{q} \right\rangle \;\propto\; \tau^{\,\xi(q)}$",
    ],
    "eq1_credit_regime": [
        r"$R_{\mathrm{credit}}(t) \;=\; \mathrm{HYG}(t)\, /\, \mathrm{LQD}(t)$",
        r"$\mathrm{credit\_spread\_regime}(t) \;=$",
        r"$\qquad\dfrac{R_{\mathrm{credit}}(t) - \mathrm{MA}_{200}\!\left(R_{\mathrm{credit}}\right)\!(t)}"
        r"{\mathrm{SD}_{200}\!\left(R_{\mathrm{credit}}\right)\!(t)}$",
    ],
    "eq2_vix_regime": [
        r"$R_{\mathrm{vix}}(t) \;=\; \ln\!\left[\mathrm{VIXM}(t)\, /\, \mathrm{VIXY}(t)\right]$",
        r"$\mathrm{vix\_term\_slope}(t) \;=$",
        r"$\qquad\dfrac{R_{\mathrm{vix}}(t) - \mathrm{MA}_{200}\!\left(R_{\mathrm{vix}}\right)\!(t)}"
        r"{\mathrm{SD}_{200}\!\left(R_{\mathrm{vix}}\right)\!(t)}$",
    ],
    "eq3_interactions": [
        r"$\mathrm{interact}_{\mathrm{gap,credit}}(t) \;=\; "
        r"\mathrm{gap}_{\tau=21,\,q=4}^{\,z}(t)$",
        r"$\qquad\cdot\; \mathrm{credit\_spread\_regime}(t)$",
        r"$\mathrm{interact}_{\xi,\mathrm{credit}}(t) \;=\; "
        r"\xi_{4}^{\,z}(t) \,\cdot\, \mathrm{credit\_spread\_regime}(t)$",
    ],
    "eq4_pinball": [
        r"$L_{\alpha}(y, \hat{y}) \;=\; \max\!\left[\, \alpha (y - \hat{y}),\;\; (\alpha - 1)(y - \hat{y}) \,\right]$",
    ],
    "eq5_ensemble": [
        r"$F_{M}(x) \;=\; F_{0}(x) \,+\, \eta \sum_{m=1}^{M} h_{m}(x)$",
    ],
    "eq6_pseudoresidual": [
        r"$r_{i}^{(m)} \;=\; \alpha \,-\, \mathbb{1}\!\left[\, y_{i} < F_{m-1}(x_{i}) \,\right]$",
    ],
    "eq7_mape": [
        r"$\mathrm{MAPE} \;=\; \dfrac{1}{N}\sum_{t=1}^{N}"
        r"\left| \dfrac{\hat{P}(t+H)}{P(t+H)} - 1 \right| \times 100$",
        r"$\mathrm{where}\quad \hat{P}(t+H) \;=\; P(t)\cdot \exp\!\left(\hat{q}_{0.5}(t)\right)$",
    ],
    "eq8_jensens_alpha": [
        r"$r_{\mathrm{strategy}}(t) \;=\; \alpha \,+\, \beta\, r_{\mathrm{benchmark}}(t) \,+\, \varepsilon(t)$",
    ],
}

if __name__ == "__main__":
    print(f"Rendering {len(EQUATIONS)} equations to {OUT_DIR}/ ...")
    for name, lines in EQUATIONS.items():
        render(name, lines)
    print("Done.")
