# Build-up sequence for one slide of the defense talk: the cumulative
# distribution of the abiogenesis rate lambda under the log-uniform prior of
# Spiegel & Turner, PNAS 109(2), 395-400 (2012), built up over several
# images:
#
#   1. log-uniform prior
#   2. + posterior for Earth, optimistic scenario
#   2.5 same, with P(lambda <= 1 / Gyr) of that posterior marked by a gray line
#       from the y-axis to the curve
#   3. + posterior for Earth, artificial scenario with a modest Bayes
#      factor of 3
#   3.5 same, with P(lambda <= 1 / Gyr) of the conservative posterior marked
#       like in step 2.5
#   4. prior and optimistic posterior as in step 2, + posterior for Earth,
#      optimistic, with an independent second abiogenesis (on Mars)
#
# The axes are identical in all images, so the slides do not jump when
# switching between them. See recreate_spiegel_turner.md for the model and
# spiegel_turner.py for the recreation of the paper's figures.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation

# Fetch Montserrat from Google Fonts and use it for every text element. The
# font file is cached locally by pyfonts, so only the first run needs a
# network connection.
from pyfonts import load_google_font, set_default_font

set_default_font(load_google_font("Montserrat"))

# Font sizes (in points). Larger than in
# Chicxulub/Plotting/MatTmpPresentation.py (18 and 16), because the plot is
# shown small on the slide.
FONTSIZE_AXIS_LABELS = 24  # axis labels
FONTSIZE_AXIS_TICKS = 21  # tick labels
FONTSIZE_LEGEND = 19
FONTSIZE_PROBABILITY_LABEL = 24  # P(lambda <= 1 / Gyr) in steps 2.5 and 3.5

COLOR_TEXT = "#333333"
COLOR_AXIS = "#999999"
COLOR_PRIOR_LABEL = "#AAAAAA"

LINEWIDTH = 4.0  # curves
LINEWIDTH_AXIS = 1.5  # axis lines and tick marks

# Prior bounds in Gyr^-1
LAMBDA_MIN = 1e-3
LAMBDA_MAX = 1e3

# The grid is uniform in u = log10(lambda), not in lambda
u = np.linspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), 4001)
lam = 10.0**u

T_0 = 4.5  # current age of the Earth in Gyr


def earth_windows(t_min, t_emerge, t_max, dt_evolve):
    # Windows (in Gyr) within which life appeared (dt1) and within which it
    # had to appear for observers to exist now (dt2); paper's Table 1
    t_required = min(T_0 - dt_evolve, t_max)
    return t_emerge - t_min, t_required - t_min


DT1_OPTIMISTIC, DT2_OPTIMISTIC = earth_windows(0.5, 0.7, 10.0, 1.0)
# Artificial scenario, not in the paper: the optimistic one with life
# emerging later, so that the Bayes factor dt2/dt1 = 3.0 / 1.0 = 3
DT1_ARTIFICIAL, DT2_ARTIFICIAL = earth_windows(0.5, 1.5, 10.0, 1.0)

# Mars: a single window, not subject to the selection effect, so it has no
# partner to divide by
DT_MARS = 1.0 - 0.5


def p_life(dt):
    # 1 - exp(-lambda dt) without catastrophic cancellation at small lambda dt
    return -np.expm1(-lam * dt)


def likelihood_earth(dt1, dt2):
    # Paper's Eq. 3: life arose within dt1, given that it arose within dt2
    return p_life(dt1) / p_life(dt2)


def cdf(p_lambda):
    # Convert to a density per decade (Jacobian dlambda/du = lambda ln(10)),
    # normalise and integrate with the trapezoidal rule
    p_u = p_lambda * lam * np.log(10)
    increments = 0.5 * (p_u[1:] + p_u[:-1]) * np.diff(u)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    return cumulative / cumulative[-1]


prior = 1 / lam
L_optimistic = likelihood_earth(DT1_OPTIMISTIC, DT2_OPTIMISTIC)
L_artificial = likelihood_earth(DT1_ARTIFICIAL, DT2_ARTIFICIAL)
# Paper's Eq. 7: the second origin enters without a denominator
L_independent = p_life(DT_MARS) * L_optimistic

# One curve per build step: (label, CDF, colour, line style). The prior is
# labelled directly on its line, the posteriors in the legend.
CURVES = [
    ("log-uniform prior", cdf(prior), "#8DB9E5", "--"),
    ("posterior", cdf(prior * L_optimistic), "#00549F", "-"),
    ("posterior (cons.)", cdf(prior * L_artificial), "#E65173", "-."),
    ("posterior with\nindep. discovery", cdf(prior * L_independent), "#00549F", ":"),
]

# Position of the prior's label along its line, in log10(lambda)
PRIOR_LABEL_U = 0.8  # clear of the marker lines of steps 2.5 and 3.5

# Build steps: (file name suffix, indices of the visible curves in CURVES,
# index of the curve whose probability of lambda <= 1 / Gyr is marked, or None)
STEPS = [
    ("1", [0], None),
    ("2", [0, 1], None),
    ("2_5", [0, 1], 1),
    ("3", [0, 1, 2], None),
    ("3_5", [0, 1, 2], 2),
    ("4", [0, 1, 3], None),
]

print("P(lambda < 1 / Gyr):")
for label, values, *_ in CURVES:
    print(f"  {label.replace(chr(10), ' '):<34}{np.interp(0, u, values):.4f}")

for suffix, visible, marked in STEPS:
    fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.2, h_pad=0.15)

    # All curves are drawn in every step, but only some are visible
    lines = []
    for i, (label, values, color, linestyle) in enumerate(CURVES):
        (line,) = ax.plot(
            u, values, color=color, ls=linestyle, lw=LINEWIDTH, label=label,
            visible=i in visible,
            # The thick lines end on the edge of the axes at (3, 1)
            clip_on=False,
        )
        lines.append(line)

    # Gray horizontal line from the y-axis to the curve at log10(lambda) = 0,
    # at the height of the cumulative probability P(lambda <= 1 / Gyr)
    if marked is not None:
        probability = np.interp(0, u, CURVES[marked][1])
        ax.plot(
            [u[0], 0],
            [probability] * 2,
            color=COLOR_AXIS,
            lw=LINEWIDTH_AXIS * 2,
            solid_capstyle="butt",
            zorder=1.9,  # below the curves
            clip_on=False,
        )
        # Its value, placed like a y tick label (tick length plus padding to
        # the left of the axis). It is left out of the layout, so the axes
        # keep the same position as in the other steps.
        ax.annotate(
            f"{probability:.2f}",
            (0, probability),
            xycoords=("axes fraction", "data"),
            xytext=(-(7 + 6), 0),
            textcoords="offset points",
            ha="right",
            va="center_baseline",
            fontsize=FONTSIZE_AXIS_TICKS,
            color=COLOR_TEXT,
            annotation_clip=False,
            in_layout=False,
        )
        # What the value means, to the right of where the gray line meets the
        # curve, on a semi-transparent white box that keeps it readable where
        # it crosses other curves
        ax.annotate(
            r"P($\lambda \leq$ 1 Gyr$^{-1}$)",
            (0, probability),
            xytext=(12, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=FONTSIZE_PROBABILITY_LABEL,
            color=COLOR_TEXT,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.8,
            ),
            zorder=3,  # above the curves
        )

    # Label of the prior, rotated parallel to its line and shifted a little
    # above it. The angle is given in data coordinates and converted to the
    # on-screen angle when drawn (transform_rotates_text), which accounts for
    # the aspect ratio of the axes.
    prior_label, prior_cdf, *_ = CURVES[0]
    slope = np.gradient(prior_cdf, u)[0]
    ax.text(
        PRIOR_LABEL_U,
        np.interp(PRIOR_LABEL_U, u, prior_cdf),
        prior_label,
        rotation=np.degrees(np.arctan(slope)),
        transform_rotates_text=True,
        rotation_mode="anchor",
        ha="center",
        va="bottom",
        transform=ax.transData
        + ScaledTranslation(0, 8 / 72, fig.dpi_scale_trans),
        fontsize=FONTSIZE_LEGEND,
        color=COLOR_PRIOR_LABEL,
    )

    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(LINEWIDTH_AXIS)
    ax.tick_params(
        labelsize=FONTSIZE_AXIS_TICKS,
        labelcolor=COLOR_TEXT,
        color=COLOR_AXIS,
        width=LINEWIDTH_AXIS,
        length=7,
        pad=6,
    )
    ax.tick_params(which="minor", color=COLOR_AXIS, width=LINEWIDTH_AXIS, length=4)

    ax.set_xlim(u[0], u[-1])
    ax.set_xlabel(
        r"$\log_{10}[\lambda]$   ($\lambda$ in Gyr$^{-1}$)",
        fontsize=FONTSIZE_AXIS_LABELS,
        color=COLOR_TEXT,
    )
    ax.set_ylim(0, 1)
    # Labels at 0, 0.5 and 1 only, unlabelled ticks every 0.1 in between
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticks([y for y in np.arange(1, 10) / 10 if y != 0.5], minor=True)
    # Extra padding in every step leaves room for the wider probability value
    # of step 2.5 next to the tick labels
    ax.set_ylabel(
        "Cumulative Probability",
        fontsize=FONTSIZE_AXIS_LABELS,
        color=COLOR_TEXT,
        labelpad=20,
    )

    # Legend of the visible posteriors in the empty top-left corner, above the
    # prior. It lies inside the axes, so it does not affect the layout.
    ax.legend(
        handles=[lines[i] for i in visible if i > 0],
        loc="upper left",
        fontsize=FONTSIZE_LEGEND,
        labelcolor=COLOR_TEXT,
        handlelength=2.8,
        frameon=False,
    )

    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean
    name = f"cdf_build_step{suffix}"
    plt.rcParams["svg.hashsalt"] = name
    output = Path(__file__).resolve().parent / name
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})

    print(f"{name}: axes at {np.round(ax.get_position().bounds, 4)}")
    plt.close(fig)
