# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9,<4", "numpy>=2,<3", "scipy>=1.14,<2"]
# ///
"""Rebuild the BC figures with `uv run scripts/generate_bc_figures.py`.

These are analytic toy distributions, not generated artwork or model results.
The Gaussian maximum-likelihood fit matches the target's mean and variance;
the squared-error predictor is its mean.

For the flow illustration, A ~ .5 N(-2, .4²) + .5 N(2, .4²), Z ~ N(0, 1),
independently, and X_tau = (1-tau) Z + tau A. Its marginal is an exact Gaussian
mixture with component means tau*m and variance (1-tau)² + tau²*.4².
The plotted trajectories solve the marginal ODE v(x,tau)=E[A-Z | X_tau=x],
not straight lines between arbitrarily paired noise and action samples.
This is the ideal velocity field that conditional flow matching would learn.
Reference: https://arxiv.org/abs/2210.02747
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.special import softmax
from scipy.stats import norm


OUT = Path(__file__).resolve().parents[1] / "assets/images/vla"
BLUE = "#246ba0"
ORANGE = "#c57520"
INK = "#253343"
GRAY = "#7b8490"
MODES = np.array([-2.0, 2.0])
SIGMA = 0.4

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


def density(x, tau):
    sd = np.sqrt((1 - tau) ** 2 + (tau * SIGMA) ** 2)
    return norm.pdf(np.asarray(x)[..., None], tau * MODES, sd).mean(axis=-1)


def velocity(tau, x):
    variance = (1 - tau) ** 2 + (tau * SIGMA) ** 2
    posterior = softmax(norm.logpdf(x[:, None], tau * MODES, np.sqrt(variance)), axis=1)
    component_v = MODES + (tau * SIGMA ** 2 - (1 - tau)) / variance * (x[:, None] - tau * MODES)
    return (posterior * component_v).sum(axis=1)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", metadata={"Date": None})
    fig.savefig(OUT / f"{name}.png", dpi=180, metadata={"Software": "Matplotlib; analytic toy distributions"})
    plt.close(fig)


def gaussian_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1))
    fig.subplots_adjust(left=.08, right=.98, bottom=.14, top=.77, wspace=.25)
    x = np.linspace(-4, 4, 1600)
    mean = [1.0, 0.0]
    sd = [SIGMA, np.sqrt(4 + SIGMA ** 2)]
    titles = ["One cluster of good actions", "Two clusters of good actions"]
    for i, ax in enumerate(axes):
        target = norm.pdf(x, mean[i], SIGMA) if i == 0 else density(x, 1)
        gaussian = norm.pdf(x, mean[i], sd[i])
        ax.fill_between(x, target, color=BLUE, alpha=.13, linewidth=0)
        ax.plot(x, target, color=BLUE, linewidth=3)
        ax.plot(x, gaussian, color=ORANGE, linewidth=2.8, linestyle=(0, (5, 3)))
        ax.axvline(mean[i], color=ORANGE, alpha=.6, linewidth=1.5, linestyle=":")
        ax.plot(mean[i], 0, marker="D", color=ORANGE, markersize=8, clip_on=False, zorder=5)
        ax.set(xlim=(-3.6, 3.6), ylim=(0, 1.1), xticks=[-3, 0, 3], yticks=[0, .5, 1.0], xlabel="Action")
        ax.set_title(titles[i], fontsize=17, fontweight="bold", loc="left", pad=12)
        ax.grid(axis="y", color="#edf0f3")
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Probability density")
    axes[1].annotate("Mean prediction", xy=(0, .012), xytext=(0, .77), ha="center", color=ORANGE, fontsize=14,
                     arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.5})
    handles = [Line2D([], [], color=BLUE, lw=3, label="Demonstration distribution"),
               Line2D([], [], color=ORANGE, lw=2.8, ls="--", label="Best single Gaussian"),
               Line2D([], [], color=ORANGE, marker="D", ls="none", label="Mean / L₂ prediction")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97), ncol=2, frameon=False, fontsize=13)
    save(fig, "bc-gaussian-vs-multimodal")


def flow_evolution():
    times = np.linspace(0, 1, 301)
    probabilities = np.linspace(.035, .965, 18)
    initial = norm.ppf(probabilities)
    solution = solve_ivp(velocity, (0, 1), initial, t_eval=times, rtol=1e-9, atol=1e-10)
    if not solution.success:
        raise RuntimeError(solution.message)
    trajectories = solution.y

    # Density normalization and conserved quantiles verify that the ODE paths
    # and displayed marginals describe the same transport, including its end.
    full_x = np.linspace(-10, 10, 10001)
    for tau in [0, .33, .67, 1]:
        assert abs(np.trapezoid(density(full_x, tau), full_x) - 1) < 1e-8
    component_sd = np.sqrt((1 - times) ** 2 + (times * SIGMA) ** 2)
    cdf = norm.cdf(trajectories[..., None], times[None, :, None] * MODES,
                   component_sd[None, :, None]).mean(axis=-1)
    assert np.max(np.abs(cdf - probabilities[:, None])) < 1e-6
    assert np.all(np.diff(trajectories, axis=0) > 0)

    fig = plt.figure(figsize=(10.4, 7.1))
    grid = fig.add_gridspec(2, 4, left=.09, right=.98, bottom=.13, top=.83, hspace=.73,
                           wspace=.24, height_ratios=[1, 1.4])
    fig.text(.09, .94, "From Gaussian noise to two good choices", fontsize=21, fontweight="bold")
    fig.text(.09, .887, "The distribution changes shape during sampling.", color=GRAY, fontsize=15)
    x = np.linspace(-3.8, 3.8, 1600)
    for i, tau in enumerate([0, 1/3, 2/3, 1]):
        ax = fig.add_subplot(grid[0, i])
        ax.fill_between(x, density(x, tau), color=BLUE, alpha=.15, linewidth=0)
        ax.plot(x, density(x, tau), color=BLUE, lw=2.8)
        ax.set(xlim=(-3.6, 3.6), ylim=(0, .57), xticks=[-2, 0, 2], yticks=[], xlabel="Action")
        ax.spines["left"].set_visible(False)
        ax.set_title(f"τ = {tau:.2f}", fontsize=15, pad=12)
        ax.text(.5, .92, ["Noise", "Reshaping", "Two modes", "Actions"][i],
                transform=ax.transAxes, ha="center", fontsize=12, color=GRAY)

    ax = fig.add_subplot(grid[1, :])
    action_grid = np.linspace(-3.2, 3.2, 420)
    heatmap = np.stack([density(action_grid, tau) for tau in times], axis=1)
    cmap = LinearSegmentedColormap.from_list("density", ["#ffffff", "#e3eff7", "#a9cfe7", BLUE])
    ax.imshow(heatmap, origin="lower", extent=(0, 1, -3.2, 3.2), aspect="auto", cmap=cmap,
              vmin=0, vmax=.5, interpolation="bilinear", zorder=0)
    for path in trajectories:
        ax.plot(times, path, color=INK, lw=1, alpha=.48)
        ax.scatter([0, 1], path[[0, -1]], s=19, color=ORANGE, zorder=3, clip_on=False)
    ax.set(xlim=(0, 1), ylim=(-3.2, 3.2), xticks=[0, 1/3, 2/3, 1],
           xticklabels=["0", "⅓", "⅔", "1"], yticks=[-2, 0, 2],
           xlabel="Sampling time τ  →", ylabel="Action")
    ax.set_title("Follow individual samples through the flow", loc="left", fontsize=16, pad=14)
    ax.text(.985, .86, "Right mode", transform=ax.transAxes, ha="right", fontsize=12, color=BLUE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
    ax.text(.985, .12, "Left mode", transform=ax.transAxes, ha="right", fontsize=12, color=BLUE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 2})
    fig.text(.5, .035, "Darker blue = higher density. Each line follows one noise sample.", ha="center", fontsize=13, color=GRAY)
    save(fig, "bc-flow-evolution")
    print(f"Flow verified: normalized densities; maximum quantile error {np.max(np.abs(cdf - probabilities[:, None])):.2e}.")


if __name__ == "__main__":
    gaussian_comparison()
    flow_evolution()
    print(f"Saved SVG and PNG figures to {OUT}")
