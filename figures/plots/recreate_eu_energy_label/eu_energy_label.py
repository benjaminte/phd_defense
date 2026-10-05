# The colour scale of the EU energy efficiency label, from A (green, most
# efficient) to G (red, least efficient), with black arrows that mark the
# class of a product, for the PhD defense talk. The scale and each arrow are
# saved as separate images, so that the arrows can be shown on their own steps
# of the slide.
#
# Credit
# The figure script and this description is created
# with the help of AI, using the model Anthropic Claude Opus 5.5
#
# What the figure shows
# ---------------------
# Seven bars, one per efficiency class, stacked from A at the top to G at the
# bottom. Each bar starts at the left edge of the image and ends in a point,
# and each is longer than the one above it. The letter of the class sits in
# white at the left end of its bar. An arrow is a black pointer that points
# left, towards the scale, at the height of a class, with a letter or sign in
# white. ARROWS lists them: one with A next to class A, and one with a
# question mark next to the middle class, D, for a class that is not yet known.
#
# Running
# -------
# Use the virtual environment of the repository (see README-venv.md), from any
# directory:
#   .venv/bin/python other_plots/recreate_eu_energy_label/eu_energy_label.py
# The script writes eu_energy_label_scale.png/.svg (the bars) and
# eu_energy_label_arrow_<name>.png/.svg (one pair per entry in ARROWS) next to
# itself, and prints the size of each image. The first run needs a network
# connection to fetch the font, which pyfonts then caches. The SVGs are
# identical on every run (fixed hash salt, no date), and the letters are drawn
# as the outlines of their glyphs, so they display without the font installed.
#
# All images share one canvas: the same size in pixels and points, with every
# element at the same place, and a transparent background. Placed on top of
# the scale on a slide, an arrow sits next to the bar of its class. The canvas
# has room for an arrow at every class, so it stays the same whichever class
# the arrows point at. To keep it that way, never save with
# bbox_inches="tight".
#
# Coordinates
# -----------
# Lengths are in units of the height of a bar: x runs from the left edge of
# the bars to the right, y upwards, with the middle of bar A at y = 0 and the
# middle of each further bar BAR_SPACING lower. The proportions follow the
# reference image.
#
# How to edit
# -----------
# Colours      CLASS_COLORS (one per class, from the top to the bottom),
#              COLOR_LETTERS, COLOR_ARROW and COLOR_ARROW_LETTER.
# Arrows       ARROWS: the name in the file name, the text in the arrow, and
#              the class it points at (one of the keys of CLASS_COLORS).
# Font         FONT_FAMILY and FONT_WEIGHT, any family on Google Fonts.
# Scale        BAR_SPACING, BAR_LENGTH (of bar A, to its point), BAR_LENGTH_STEP
#              (added per class), BAR_POINT (how far the point sticks out),
#              LETTER_HEIGHT and LETTER_INSET.
# Arrow shape  ARROW_TIP (x of its point), ARROW_LENGTH, ARROW_HEIGHT,
#              ARROW_POINT and ARROW_LETTER_HEIGHT.
# Canvas       FIG_WIDTH (the height follows from the drawing), DPI.

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

# Fetch the font from Google Fonts. The font file is cached locally by pyfonts,
# so only the first run needs a network connection.
from pyfonts import load_google_font

# --- Colours -----------------------------------------------------------------

# Fill of each bar, from the most efficient class at the top to the least
# efficient at the bottom, taken from the reference image
CLASS_COLORS = {
    "A": "#128440",
    "B": "#16A550",
    "C": "#54CC4E",
    "D": "#FFBC00",
    "E": "#FF5E0D",
    "F": "#FF321D",
    "G": "#DD2238",
}
COLOR_LETTERS = "#FFFFFF"  # letters on the bars
COLOR_ARROW = "#000000"
COLOR_ARROW_LETTER = "#FFFFFF"

# --- Content -----------------------------------------------------------------

# One image per arrow: (name in the file name, text in the arrow, class it
# points at). The text is drawn at ARROW_LETTER_HEIGHT like a capital letter.
ARROWS = [
    ("A", "A", "A"),
    ("question", "?", "D"),
]

FONT_FAMILY = "Montserrat"
FONT_WEIGHT = "bold"
FONT = load_google_font(FONT_FAMILY, weight=FONT_WEIGHT)

# --- Geometry, in bar heights ------------------------------------------------

BAR_SPACING = 1.2  # from the middle of one bar to the next: a gap of 0.2
BAR_LENGTH = 4.16  # of bar A, from its left edge to its point
BAR_LENGTH_STEP = 0.44  # each class is this much longer than the one above
BAR_POINT = 0.35  # length of the pointed end
LETTER_HEIGHT = 0.45  # height of the capital letters
LETTER_INSET = 0.43  # from the left edge of a bar to its letter

ARROW_TIP = 7.14  # x of the point of the arrow, to the right of all bars
ARROW_LENGTH = 3.0  # from the point to the right end
ARROW_HEIGHT = 1.5
ARROW_POINT = 0.69  # length of the pointed end
ARROW_LETTER_HEIGHT = 0.98  # centred in the rectangular part of the arrow

# --- Canvas ------------------------------------------------------------------

FIG_WIDTH = 8  # in inches; the height follows from the drawing
DPI = 300


def pointer(left, right, middle, height, point):
    # Corners of a bar of the given height, from x = left to x = right, with its
    # vertical middle at y = middle and its right end drawn out into a point
    # that sticks out by point. A negative point puts it on the left end.
    half = height / 2
    if point > 0:
        return [
            (left, middle - half),
            (right - point, middle - half),
            (right, middle),
            (right - point, middle + half),
            (left, middle + half),
        ]
    return [
        (left - point, middle - half),
        (right, middle - half),
        (right, middle + half),
        (left - point, middle + half),
        (left, middle),
    ]


def letter(text, height, x, y, align):
    # Outline of a text whose capital letters are height tall, with the middle
    # of the capitals at y. align "left" puts the left end of the glyphs at x,
    # "center" their middle.
    capitals = TextPath((0, 0), "H", size=1, prop=FONT).get_extents()
    glyphs = TextPath((0, 0), text, size=height / capitals.height, prop=FONT)
    extents = glyphs.get_extents()
    offset = x - (extents.x0 if align == "left" else (extents.x0 + extents.x1) / 2)
    # The capitals stand on the baseline, so their middle is at half their
    # height above it
    return MplPath(glyphs.vertices + (offset, y - height / 2), glyphs.codes)


def middle_of(label):
    # Vertical middle of the bar of a class
    return -list(CLASS_COLORS).index(label) * BAR_SPACING


# --- The drawing -------------------------------------------------------------

bars = []  # (colour, corners)
letters = []  # (colour, outline)
for index, (label, color) in enumerate(CLASS_COLORS.items()):
    right = BAR_LENGTH + index * BAR_LENGTH_STEP
    bars.append((color, pointer(0, right, middle_of(label), 1, BAR_POINT)))
    outline = letter(label, LETTER_HEIGHT, LETTER_INSET, middle_of(label), "left")
    letters.append((COLOR_LETTERS, outline))

# The patches of each image as (colour, patch), from the bottom to the top
IMAGES = {
    "scale": [(color, Polygon(corners)) for color, corners in bars]
    + [(color, PathPatch(outline)) for color, outline in letters],
}
arrow_right = ARROW_TIP + ARROW_LENGTH
for name, text, label in ARROWS:
    middle = middle_of(label)
    corners = pointer(ARROW_TIP, arrow_right, middle, ARROW_HEIGHT, -ARROW_POINT)
    center = (ARROW_TIP + ARROW_POINT + arrow_right) / 2
    outline = letter(text, ARROW_LETTER_HEIGHT, center, middle, "center")
    IMAGES[f"arrow_{name}"] = [
        (COLOR_ARROW, Polygon(corners)),
        (COLOR_ARROW_LETTER, PathPatch(outline)),
    ]

# The canvas spans from the left edge of the bars to the right end of the
# arrows, and has room for an arrow at the top and bottom class, as the arrows
# are taller than a bar
overhang = max(ARROW_HEIGHT, 1) / 2
x0, x1 = 0, arrow_right
y0 = middle_of(list(CLASS_COLORS)[-1]) - overhang
y1 = overhang
figsize = (FIG_WIDTH, FIG_WIDTH * (y1 - y0) / (x1 - x0))

for image, patches in IMAGES.items():
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    # Later patches (the letters) are drawn on top of earlier ones (the bars)
    for zorder, (color, patch) in enumerate(patches):
        patch.set(facecolor=color, edgecolor="none", zorder=zorder)
        ax.add_patch(patch)

    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean
    name = f"eu_energy_label_{image}"
    plt.rcParams["svg.hashsalt"] = name
    output = Path(__file__).resolve().parent / name
    fig.savefig(output.with_suffix(".png"), dpi=DPI, transparent=True)
    fig.savefig(
        output.with_suffix(".svg"), transparent=True, metadata={"Date": None}
    )

    # Whole pixels, as matplotlib cuts off fractions
    width, height = (fig.get_size_inches() * DPI).astype(int)
    print(f"{name}: {width} x {height} px")
    plt.close(fig)
