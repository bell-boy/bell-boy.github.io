# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Rebuild the Atom-Bench figure with `uv run scripts/generate_atom_bench_figure.py`.

Values are the AS, PSR and CFS columns (%) of the "Atomic-to-compositional
transfer" table in ATOM-Bench, https://arxiv.org/pdf/2606.16826
AS (atomic score) is the mean PSR of a composite task's atoms; PSR is the
process success rate on the composite task itself. CFS = max(0, AS - PSR) /
(1 - PSR): the share of compositional failure that weak atomic skills can't
explain. Plotting AS next to CFS shows the confound: a policy with weak atoms
has little failure left to blame on composition.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "assets/images/vla"
BLUE = "#246ba0"
ORANGE = "#c57520"
INK = "#253343"
GRAY = "#7b8490"

MODELS = [r"$\pi_{0.5}$", "Motus", "LingBot-VLA", "GR00T N1.6", "SmolVLA"]
# platform: (atomic score, composite PSR, CFS) per model
DATA = {
    "Franka Panda (single-arm)": ([83.3, 69.3, 60.3, 66.3, 33.3],
                                  [30.4, 26.5, 12.1, 11.6, 6.6],
                                  [73.7, 49.4, 54.5, 61.2, 28.3]),
    "Cobot Magic (dual-arm)": ([79.5, 68.5, 47.2, 48.2, 32.5],
                               [42.6, 31.5, 24.4, 13.7, 6.3],
                               [56.8, 48.3, 29.3, 38.8, 27.3]),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "axes.edgecolor": "#c8ced5",
    "axes.spines.top": False,
    "axes.spines.right": False,
    # Embed glyph outlines so browsers without DejaVu Sans preserve the layout.
    "svg.fonttype": "path",
    "savefig.facecolor": "white",
})


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.4), sharey=True)
    fig.subplots_adjust(left=.14, right=.92, bottom=.13, top=.76, wspace=.32)
    y = np.arange(len(MODELS))[::-1]
    for ax, (platform, (atomic, composite, cfs)) in zip(axes, DATA.items()):
        ax.hlines(y, composite, atomic, color="#c8ced5", linewidth=3, zorder=1)
        ax.scatter(composite, y, s=90, color=ORANGE, edgecolor="white", linewidth=2, zorder=3)
        ax.scatter(atomic, y, s=90, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
        for yi, value in zip(y, cfs):
            ax.text(103, yi, f"{value:.0f}%", va="center", ha="left", fontsize=12, color=INK)
        ax.text(103, len(MODELS) - .35, "CFS", va="center", ha="left", fontsize=12, color=GRAY)
        ax.set_title(platform, loc="left", fontsize=13, color=INK, pad=10)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(-.6, len(MODELS) - .2)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e6e9ed", linewidth=.8, zorder=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(y, MODELS, color=INK)
    fig.supxlabel("Process success rate", x=.53, y=.02, fontsize=13, color=INK)
    fig.suptitle("Atomic skill vs. composite-task score, with compositional failure share",
                 x=.02, y=.97, ha="left", fontsize=15, color=INK)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=BLUE, label="Atomic score (atoms alone)"),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=ORANGE, label="Composite task"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.015, .91), ncol=2, frameon=False, fontsize=13)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "atom-bench-cfs.svg", metadata={"Date": None})
    fig.savefig(OUT / "atom-bench-cfs.png", dpi=180, metadata={"Software": "Matplotlib; ATOM-Bench Table 3"})
    plt.close(fig)


if __name__ == "__main__":
    main()
