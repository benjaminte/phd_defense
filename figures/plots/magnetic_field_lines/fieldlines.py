# The magnetic field lines around a planet, for the PhD defense talk: loops
# that leave the planet in one hemisphere and return in the other, like the
# field of a bar magnet. The planet itself is left out, so that any picture of
# a planet can be placed in the gap in the middle.
#
# Credit
# The figure script and this description is created
# with the help of AI, using the model Anthropic Claude Opus 5.5
# ------
# The sizes of the loops and the frame are measured from a reference sketch of
# a planet with its field lines, whose source is not recorded here.
#
# What the figure shows
# ---------------------
# On each side of the planet, a set of circles of increasing size, cut off
# where they enter the planet, the left side a mirror image of the right. The
# largest circle reaches beyond the frame: only the arcs next to the poles are
# seen, fading out into transparency towards the top and bottom, and hint at
# the larger loops further out.
#
# Running
# -------
# Use the virtual environment of the repository (see README-venv.md), from any
# directory:
#   .venv/bin/python other_plots/magnetic_field_lines/fieldlines.py
# The script writes one PNG (DPI dots per inch) and SVG per entry in VERSIONS
# next to itself: fieldlines.png/.svg in colour and fieldlines_bw.png/.svg in
# black. It prints the size of each image, of the planet and of the gap left
# for it, to size a picture of the planet on the slide. All images have a
# transparent background, with the centre of the planet in the middle. The SVGs
# are identical on every run (fixed hash salt, no date).
#
# Coordinates
# -----------
# x runs to the right and y upwards from the centre of the planet, in planet
# radii. Line widths are in points: at FIG_WIDTH = 8 inches and FRAME_X = 4.2,
# one planet radius is about 69 points.
#
# How to edit
# -----------
# Colours      VERSIONS: the file name and colour of the lines of each image.
# Field lines  FIELD_LOOPS: the centre and radius of each circle on the right
#              side (the left side follows), LINE_START (where the lines stop
#              short of the centre), LINEWIDTH_FIELD_LINES and FADE_* (the fade
#              at the top and bottom).
# Canvas       FRAME_X and FRAME_Y (the part of the drawing shown), FIG_WIDTH
#              (the height follows from the frame), DPI.
#
# The code follows the other plotting scripts of this repository: black
# formatting at 88 columns, settings as upper-case constants at the top,
# comments in full sentences.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

# One image per entry: its file name (without suffix) and the colour of its
# lines
VERSIONS = {
    "fieldlines": "#8DB9E5",
    "fieldlines_bw": "black",
}

# Field lines on the right side of the planet, from the inside out: the centre
# of each circle, on the equator, and its radius, in planet radii. Each circle
# passes close to the centre of the planet, so that it runs out of the planet
# steeply, near the poles for the larger ones. The last circle is larger than
# the frame.
FIELD_LOOPS = [
    (0.62, 0.79),
    (1.38, 1.13),
    (1.78, 1.62),
    (2.0, 2.03),
    (3.42, 3.54),
]
# The lines start at this distance from the centre of the planet, in planet
# radii: 1 at its surface, more to leave a gap around it. In the reference,
# the thick outline of the planet hides the ends of the lines; there, the two
# smallest loops, and the two arcs at each pole, nearly touch. A little above
# 1, they end apart.
LINE_START = 1.1
LINEWIDTH_FIELD_LINES = 3.5  # in points
# Above this height (in planet radii) and below its negative, the lines that run
# out of the frame fade out, until they are fully transparent at the top and
# bottom of the frame. Below FRAME_Y. The opacity falls in FADE_STEPS steps.
FADE_START = 1.6
FADE_STEPS = 40

# Canvas, centred on the planet
FRAME_X = 4.2  # half the width of the drawing, in planet radii
FRAME_Y = 2.73  # half its height
FIG_WIDTH = 8.0  # in inches; the height follows from the frame
DPI = 300  # of the PNG image


def loop(x_centre, radius):
    # Points of the circle around (x_centre, 0) with this radius, outside
    # LINE_START: the arc on the far side from the planet. The angle t runs
    # around the centre of the circle, 0 at its point furthest from the planet;
    # the arc ends where the circle crosses LINE_START, at cos(t) = cos_end.
    # Cutting it to the frame is left to faded.
    cos_end = (LINE_START**2 - x_centre**2 - radius**2) / (2 * abs(x_centre) * radius)
    if cos_end >= 1:  # inside LINE_START all the way round
        return None
    t_end = np.arccos(max(cos_end, -1))
    t = np.linspace(-t_end, t_end, 2000)
    return np.column_stack(
        [x_centre + np.sign(x_centre) * radius * np.cos(t), radius * np.sin(t)]
    )


def faded(line):
    # The line as strokes, with their opacities. A line inside the frame is one
    # opaque stroke. A line that runs out at the top or bottom is cut into its
    # parts inside the frame, each running from the planet outwards: opaque up
    # to FADE_START from the equator, then fading out towards the edge of the
    # frame. Strokes side by side, each with its own opacity, would leave faint
    # seams where they meet. The fade is therefore built from FADE_STEPS
    # overlapping strokes, which all start under the end of the opaque stroke
    # and end one after another towards the edge: wherever one ends, the
    # opacity of those left steps down.
    inside = (np.abs(line[:, 0]) <= FRAME_X) & (np.abs(line[:, 1]) <= FRAME_Y)
    if inside.all():
        return [line], [1.0]
    strokes, alphas = [], []
    for part in np.split(line, np.flatnonzero(np.diff(inside.astype(int))) + 1):
        x, y = np.abs(part[0])
        if x > FRAME_X or y > FRAME_Y:  # outside the frame
            continue
        if np.abs(part[0, 1]) > np.abs(part[-1, 1]):
            part = part[::-1]
        start = np.flatnonzero(np.abs(part[:, 1]) > FADE_START)[0]
        strokes.append(part[: start + 1])
        alphas.append(1.0)
        # Transparency wanted between neighbouring ends, from the mean height
        ends = np.linspace(start, len(part) - 1, FADE_STEPS + 1).round().astype(int)
        heights = [np.mean(np.abs(part[a : b + 1, 1])) for a, b in zip(ends, ends[1:])]
        clear = (np.array(heights) - FADE_START) / (FRAME_Y - FADE_START)
        clear = np.clip(clear, 1e-3, 1)
        # Between ends j - 1 and j, the strokes j and further overlap, so that
        # clear[j] is the product of (1 - alpha) over them
        beyond = np.append(clear[1:], 1.0)
        for end, here, there in zip(ends[1:], clear, beyond):
            strokes.append(part[max(start - 1, 0) : end + 1])
            alphas.append(1 - here / there)
    return strokes, alphas


lines = [loop(side * x, radius) for x, radius in FIELD_LOOPS for side in (1, -1)]
lines = [line for line in lines if line is not None]
strokes, alphas = zip(*(faded(line) for line in lines))
strokes = [stroke for line_strokes in strokes for stroke in line_strokes]
alphas = [alpha for line_alphas in alphas for alpha in line_alphas]

for name, color in VERSIONS.items():
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * FRAME_Y / FRAME_X))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-FRAME_X, FRAME_X)
    ax.set_ylim(-FRAME_Y, FRAME_Y)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Butt caps end the lines exactly at LINE_START
    ax.add_collection(
        LineCollection(
            strokes,
            colors=[(*to_rgb(color), alpha) for alpha in alphas],
            linewidths=LINEWIDTH_FIELD_LINES,
            capstyle="butt",
            joinstyle="round",
        )
    )

    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean
    plt.rcParams["svg.hashsalt"] = name
    output = Path(__file__).resolve().parent / name
    fig.savefig(output.with_suffix(".png"), dpi=DPI, transparent=True)
    fig.savefig(output.with_suffix(".svg"), transparent=True, metadata={"Date": None})

    # Whole pixels, as matplotlib cuts off fractions
    width, height = (fig.get_size_inches() * DPI).astype(int)
    planet = 2 * FIG_WIDTH * DPI / (2 * FRAME_X)
    print(
        f"{name}: {width} x {height} px, planet {planet:.0f} px wide, "
        f"gap for it {LINE_START * planet:.0f} px"
    )
    plt.close(fig)
