# Liquid temperature ranges of methane, ammonia and water at one atmosphere of
# pressure. Made for the defense talk.

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle, FancyBboxPatch

# Fetch Montserrat from Google Fonts and use it for every text element
# (axis label, tick labels, substance names, temperature ranges). The font
# file is cached locally by pyfonts, so only the first run needs a network
# connection.
from pyfonts import load_google_font, set_default_font

set_default_font(load_google_font("Montserrat"))

# Font sizes (in points); same values as in
# Chicxulub/Plotting/MatTmpPresentation.py
FONTSIZE_AXIS_LABELS = 18  # x-axis label and substance names
FONTSIZE_AXIS_TICKS = 16  # x-axis ticks and temperature ranges above the bars

# Liquid range at 1 atm: (name, lowest and highest liquid temperature in
# degrees Celsius, bar colour). Listed from top to bottom.
LIQUIDS = [
    ("Methane", -182, -162, "#F19EB1"),
    ("Ammonia", -78, -33, "#FABE50"),
    ("Water", 0, 100, "#8DB9E5"),
]

XLIM = (-200, 100)  # in degrees Celsius
BAR_HEIGHT = 0.46  # in units of the distance between two bars
CORNER_RADIUS = 0.12  # as a fraction of the bar height

COLOR_TEXT = "#333333"
COLOR_NAMES = "#1A1A1A"
COLOR_AXIS = "#999999"


def with_minus(text):
    # Real minus signs (U+2212); matplotlib only substitutes them in the tick
    # labels of its own formatters
    return text.replace("-", "\N{MINUS SIGN}")


fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
# White space around the plot (in inches)
fig.get_layout_engine().set(w_pad=0.2, h_pad=0.15)

# One row per substance, the first one on top
rows = range(len(LIQUIDS))
ax.set_xlim(XLIM)
ax.set_ylim(len(LIQUIDS) - 0.25, -0.75)

# Only the x-axis line is drawn
for side in ("left", "right", "top"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(COLOR_AXIS)

# x-axis: temperature in degrees Celsius
ax.set_xticks(range(XLIM[0], XLIM[1] + 1, 100))
ax.xaxis.set_major_formatter(lambda value, _: with_minus(f"{value:.0f} °C"))
ax.set_xlabel(
    "liquid ranges at 1 [atm]",
    fontsize=FONTSIZE_AXIS_LABELS,
    color=COLOR_TEXT,
    labelpad=14,
)
ax.tick_params(
    axis="x",
    labelsize=FONTSIZE_AXIS_TICKS,
    labelcolor=COLOR_TEXT,
    color=COLOR_AXIS,
    length=5,
    pad=6,
)

# y-axis: substance names, without tick marks
ax.set_yticks(rows, [name for name, *_ in LIQUIDS])
ax.tick_params(
    axis="y",
    labelsize=FONTSIZE_AXIS_LABELS,
    labelcolor=COLOR_NAMES,
    length=0,
    pad=12,
)

# The size of the axes is only known once the layout has been computed. It is
# needed to make the rounded corners of the bars circular on screen, although
# one unit in x and one unit in y are different distances.
fig.canvas.draw()
bbox = ax.get_window_extent()
px_per_x = bbox.width / (XLIM[1] - XLIM[0])
px_per_y = bbox.height / abs(ax.get_ylim()[1] - ax.get_ylim()[0])
corner_radius_x = CORNER_RADIUS * BAR_HEIGHT * px_per_y / px_per_x

for row, (name, low, high, color) in zip(rows, LIQUIDS):
    bar_top = row - BAR_HEIGHT / 2  # the y-axis points downwards

    ax.add_patch(
        FancyBboxPatch(
            (low, bar_top),
            high - low,
            BAR_HEIGHT,
            boxstyle=BoxStyle("round", pad=0, rounding_size=corner_radius_x),
            mutation_aspect=px_per_x / px_per_y,
            facecolor=color,
            edgecolor="none",
        )
    )

    # Temperature range in degrees Celsius above the left end of the bar
    ax.annotate(
        with_minus(f"{low} to {high} °C"),
        (low, bar_top),
        xytext=(0, 4),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=FONTSIZE_AXIS_TICKS,
        color=COLOR_TEXT,
    )

# A fixed hash salt and no date make the SVG identical on every run, which
# keeps the version-control diffs clean
plt.rcParams["svg.hashsalt"] = "freezing_ranges"

output = Path(__file__).resolve().parent / "freezing_ranges"
fig.savefig(output.with_suffix(".png"), dpi=300)
fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
