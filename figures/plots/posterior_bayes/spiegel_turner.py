# Recreation of Figures 1 and 3 of Spiegel & Turner, "Bayesian analysis of
# the astrobiological implications of life's early emergence on Earth", PNAS
# 109(2), 395-400 (2012), doi:10.1073/pnas.1111694108. Made for the defense
# talk; see recreate_spiegel_turner.md for the derivation and the validation
# targets that are printed to stdout.
#
# Figure 1: prior and posterior PDFs (left) and CDFs (right) of the
# abiogenesis rate lambda under a uniform, a logarithmic and an inverse-uniform
# prior.
# Figure 3: CDF of lambda under the logarithmic prior, with and without an
# independent second abiogenesis (on Mars).

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Fetch Montserrat from Google Fonts and use it for every text element. The
# font file is cached locally by pyfonts, so only the first run needs a
# network connection.
from pyfonts import load_google_font, set_default_font

set_default_font(load_google_font("Montserrat"))

# Font sizes (in points); same values as in
# Chicxulub/Plotting/MatTmpPresentation.py
FONTSIZE_AXIS_LABELS = 18  # axis labels and panel titles
FONTSIZE_AXIS_TICKS = 16  # tick labels, legends and annotations

COLOR_TEXT = "#333333"
COLOR_AXIS = "#999999"

# One colour per prior. The paper's colours (#3ddc4a, #b48ce8, #ff6a4d) are
# meant for a dark background; these darker shades keep the same hues
# readable on white.
COLOR_UNIFORM = "#2BA84A"
COLOR_LOG = "#8A5CC7"
COLOR_INVUNIF = "#E8553A"

LW_PRIOR = 1.6  # dashed
LW_POSTERIOR = 2.2  # solid

# Model parameters in Gyr: "optimistic" column of the paper's Table 1
T_MIN = 0.5  # planet becomes habitable
T_EMERGE = 0.7  # life has demonstrably appeared
T_MAX = 10.0  # last time abiogenesis is possible
DT_EVOLVE = 1.0  # from first life to intelligent observers
T_0 = 4.5  # current age of the planet

T_REQUIRED = min(T_0 - DT_EVOLVE, T_MAX)
DT1 = T_EMERGE - T_MIN  # window within which life appeared
DT2 = T_REQUIRED - T_MIN  # window within which life had to appear

# Mars (Figure 3): a single window, not subject to the selection effect, so
# it has no partner to divide by
DT_MARS = 1.0 - 0.5

# Prior bounds in Gyr^-1
LAMBDA_MIN = 1e-3
LAMBDA_MAX = 1e3

# The grid is uniform in u = log10(lambda), not in lambda
u = np.linspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), 4001)
lam = 10.0**u

U_LABEL = r"$\log_{10}[\lambda]$   ($\lambda$ in Gyr$^{-1}$)"


def p_life(lam, dt):
    # 1 - exp(-lam dt) without catastrophic cancellation at small lam dt
    return -np.expm1(-lam * dt)


def likelihood_earth(lam):
    # Paper's Eq. 3: life arose within dt1, given that it arose within dt2
    return p_life(lam, DT1) / p_life(lam, DT2)


def likelihood_independent(lam):
    # Paper's Eq. 7: the second origin is unconditioned (no denominator)
    return p_life(lam, DT_MARS) * likelihood_earth(lam)


def density_in_u(p_lambda):
    # Density per decade: the Jacobian dlambda/du = lambda ln(10) makes the
    # area under the plotted curve a genuine probability
    p_u = p_lambda * lam * np.log(10)
    return p_u / np.trapz(p_u, u)


def cdf(p_u):
    increments = 0.5 * (p_u[1:] + p_u[:-1]) * np.diff(u)
    return np.concatenate([[0.0], np.cumsum(increments)])


def at(values, u_value):
    return np.interp(u_value, u, values)


def percentile(cdf_values, probability):
    # u at which the CDF reaches the given probability
    return np.interp(probability, cdf_values, u)


def style_axis(ax):
    for side in ("right", "top"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
    ax.tick_params(
        labelsize=FONTSIZE_AXIS_TICKS,
        labelcolor=COLOR_TEXT,
        color=COLOR_AXIS,
        length=5,
        pad=6,
    )
    ax.set_xlim(u[0], u[-1])
    ax.set_xlabel(U_LABEL, fontsize=FONTSIZE_AXIS_LABELS, color=COLOR_TEXT)


def save(fig, name):
    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean
    plt.rcParams["svg.hashsalt"] = name
    output = Path(__file__).resolve().parent / name
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})


# --- Figure 1 ---------------------------------------------------------------

L_earth = likelihood_earth(lam)

# Priors as densities in lambda, up to proportionality
PRIORS = [
    ("Uniform", np.ones_like(lam), COLOR_UNIFORM),
    ("Log", 1 / lam, COLOR_LOG),
    ("InvUnif", 1 / lam**2, COLOR_INVUNIF),
]

curves = []  # (name, colour, prior density in u, posterior density in u)
for name, prior, color in PRIORS:
    curves.append((name, color, density_in_u(prior), density_in_u(prior * L_earth)))

print("Figure 1: likelihood plateaus")
print(f"  L(lambda=1e-3) = {at(L_earth, -3):.5f}   (dt1/dt2 = {DT1 / DT2:.6f})")
print(f"  L(lambda=1e3)  = {at(L_earth, 3):.6f}")
print(f"  Bayes factor dt2/dt1 = {DT2 / DT1:.1f}")

print("\nFigure 1, left panel: density per decade p(u)")
print(f"  {'curve':<20}{'u=-3':>12}{'u=0':>12}{'u=+3':>12}")
for name, _, prior_u, posterior_u in curves:
    for kind, p_u in (("prior", prior_u), ("posterior", posterior_u)):
        values = "".join(f"{at(p_u, x):>12.3e}" for x in (-3, 0, 3))
        print(f"  {name + ' ' + kind:<20}{values}")

print("\nFigure 1, right panel: CDF")
print(f"  {'curve':<20}{'u=-2':>9}{'u=0':>9}{'u=+2':>9}")
for name, _, prior_u, posterior_u in curves:
    for kind, p_u in (("prior", prior_u), ("posterior", posterior_u)):
        values = "".join(f"{at(cdf(p_u), x):>9.4f}" for x in (-2, 0, 2))
        print(f"  {name + ' ' + kind:<20}{values}")

fig, (ax_pdf, ax_cdf) = plt.subplots(
    1, 2, figsize=(13, 5.7), sharex=True, layout="constrained"
)
fig.get_layout_engine().set(w_pad=0.2, h_pad=0.15, wspace=0.08)

for name, color, prior_u, posterior_u in curves:
    ax_pdf.plot(u, prior_u, color=color, lw=LW_PRIOR, ls="--")
    ax_pdf.plot(u, posterior_u, color=color, lw=LW_POSTERIOR)
    ax_cdf.plot(u, cdf(prior_u), color=color, lw=LW_PRIOR, ls="--")
    ax_cdf.plot(u, cdf(posterior_u), color=color, lw=LW_POSTERIOR)

for ax in (ax_pdf, ax_cdf):
    style_axis(ax)
    ax.set_title("Optimistic", fontsize=FONTSIZE_AXIS_LABELS, color=COLOR_TEXT)

ax_pdf.set_yscale("log")
ax_pdf.set_ylim(1e-7, 1e1)
ax_pdf.set_ylabel(
    "Probability Density", fontsize=FONTSIZE_AXIS_LABELS, color=COLOR_TEXT
)

ax_cdf.set_ylim(0, 1)
ax_cdf.set_ylabel(
    "Cumulative Probability", fontsize=FONTSIZE_AXIS_LABELS, color=COLOR_TEXT
)

# Key in the empty strip at the top of the left panel: the series names in
# their own colours on one row, the line styles on the row below. Each name is
# placed to the right of the previous one.
KEY_LEFT = 0.25  # in axes coordinates
KEY_TOP = 0.98
key_style = dict(fontsize=FONTSIZE_AXIS_TICKS, ha="left", va="top")
previous = None
for name, color, *_ in curves:
    if previous is None:
        previous = ax_pdf.text(
            KEY_LEFT, KEY_TOP, name, transform=ax_pdf.transAxes, color=color,
            **key_style,
        )
    else:
        previous = ax_pdf.annotate(
            name, (1, 1), xycoords=previous, xytext=(12, 0),
            textcoords="offset points", color=color, **key_style,
        )
ax_pdf.annotate(
    "Posterior: Solid   Prior: Dashed",
    (KEY_LEFT, KEY_TOP),
    xycoords="axes fraction",
    xytext=(0, -1.4 * FONTSIZE_AXIS_TICKS),
    textcoords="offset points",
    color=COLOR_TEXT,
    **key_style,
)

save(fig, "spiegel_turner_fig1")

# --- Figure 3 ---------------------------------------------------------------

L_indep = likelihood_independent(lam)
prior_log = 1 / lam

cdf_prior = cdf(density_in_u(prior_log))
cdf_earth = cdf(density_in_u(prior_log * L_earth))
cdf_indep = cdf(density_in_u(prior_log * L_indep))

print("\nFigure 3: likelihoods")
print(f"  {'u':>4}{'L_earth':>12}{'L_indep':>12}{'lam*dt_mars':>13}")
for x in (-3, -2, -1, 0, 1, 3):
    print(
        f"  {x:>+4d}{at(L_earth, x):>12.3e}{at(L_indep, x):>12.3e}"
        f"{10.0**x * DT_MARS:>13.3e}"
    )
print(f"  L_indep/L_earth at u=-3: {at(L_indep, -3) / at(L_earth, -3):.3e}")

print("\nFigure 3: CDF")
print(f"  {'u':>4}{'prior':>9}{'Earth':>9}{'indep.':>9}")
for x in range(-3, 4):
    print(
        f"  {x:>+4d}{at(cdf_prior, x):>9.4f}{at(cdf_earth, x):>9.4f}"
        f"{at(cdf_indep, x):>9.4f}"
    )

print("\nFigure 3: percentiles in u")
print(f"  {'curve':<25}{'median':>8}{'2 sigma lower':>15}")
for name, values in (
    ("prior", cdf_prior),
    ("posterior, Earth only", cdf_earth),
    ("posterior, independent", cdf_indep),
):
    print(
        f"  {name:<25}{percentile(values, 0.5):>+8.3f}"
        f"{percentile(values, 1 - 0.9545):>+15.3f}"
    )

fig, ax = plt.subplots(figsize=(7.5, 6), layout="constrained")
fig.get_layout_engine().set(w_pad=0.2, h_pad=0.15)

ax.plot(u, cdf_prior, color=COLOR_LOG, lw=LW_PRIOR, ls="--", label="Prior")
ax.plot(
    u,
    cdf_indep,
    color=COLOR_UNIFORM,
    lw=LW_POSTERIOR,
    label="Posterior if independent abiogenesis",
)
ax.plot(
    u,
    cdf_earth,
    color=COLOR_INVUNIF,
    lw=LW_POSTERIOR,
    label="Posterior for Earth (Optimistic)",
)

style_axis(ax)
ax.set_ylim(0, 1)
ax.set_yticks(np.arange(0, 1.01, 0.1))
ax.set_ylabel(
    "Cumulative Probability", fontsize=FONTSIZE_AXIS_LABELS, color=COLOR_TEXT
)

legend = ax.legend(
    title="Independent Life, log. prior",
    loc="upper left",
    fontsize=FONTSIZE_AXIS_TICKS - 2,
    title_fontsize=FONTSIZE_AXIS_TICKS - 2,
    labelcolor=COLOR_TEXT,
    frameon=False,
)
legend.get_title().set_color(COLOR_TEXT)

save(fig, "spiegel_turner_fig3")
