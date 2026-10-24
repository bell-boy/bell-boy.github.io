# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3"]
# ///
"""Rebuild the RoboHiMan figure with `uv run scripts/generate_robohiman_figure.py`.

Values are the pi0-Vanilla and pi0.5-Vanilla rows of the appendix table
"Performance across different training levels (L1-L4) and test task (A&AP, C&CP)"
in RoboHiMan, https://arxiv.org/pdf/2510.13149
Vanilla policies run end to end, with no high-level planner.
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

# The post defines each level; L3 and L4 add composite-task demonstrations.
LEVELS = ["L1", "L2", "L3", "L4"]
RESULTS = {
    r"$\pi_0$": {"Atomic tasks": [15.3, 16.8, 12.6, 16.2], "Composite tasks": [0, 0, 0, 0.5]},
    r"$\pi_{0.5}$": {"Atomic tasks": [8.5, 8.9, 9.1, 9.9], "Composite tasks": [0, 0, 0, 0.6]},
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
    fig.subplots_adjust(left=.09, right=.98, bottom=.15, top=.77, wspace=.12)
    x = np.arange(len(LEVELS))
    width = .36
    for ax, (model, series) in zip(axes, RESULTS.items()):
        for i, ((task, values), color) in enumerate(zip(series.items(), [BLUE, ORANGE])):
            offset = (i - .5) * (width + .02)
            bars = ax.bar(x + offset, values, width, color=color, label=task, zorder=2)
            ax.bar_label(bars, labels=[f"{v:g}%" for v in values], padding=4, fontsize=12, color=INK)
        ax.set_title(model, fontsize=16, color=INK)
        ax.set_xticks(x, LEVELS, color=INK)
        ax.set_xlabel("Training dataset")
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 25)
        ax.set_yticks([0, 5, 10, 15, 20, 25], ["0%", "5%", "10%", "15%", "20%", "25%"])
        ax.grid(axis="y", color="#e6e9ed", linewidth=.8, zorder=0)
    axes[0].set_ylabel("Success rate")
    fig.suptitle("End-to-end policies never compose skills, even when trained on composites",
                 x=.09, y=.97, ha="left", fontsize=15, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.085, .91), ncol=2, frameon=False, fontsize=13)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "robohiman-results.svg", metadata={"Date": None})
    fig.savefig(OUT / "robohiman-results.png", dpi=180, metadata={"Software": "Matplotlib; RoboHiMan appendix table"})
    plt.close(fig)


if __name__ == "__main__":
    main()
