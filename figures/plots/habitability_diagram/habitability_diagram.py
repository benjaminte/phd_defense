# The requirements that make an environment habitable, as a Venn diagram for
# the PhD defense talk: four overlapping ellipses, one per requirement, and the
# region inside all four, where life can be active.
#
# Credit
# The figure script and this description is created
# with the help of AI, using the model Anthropic Claude Opus 5.5
# ------
# The figure is a recreation of Figure 3 ("Instantaneous habitability") of
#   C. S. Cockell, T. Bush, C. Bryce, S. Direito, M. Fox-Powell,
#   J. P. Harrison, H. Lammer, H. Landenmark, J. Martin-Torres, N. Nicholson,
#   L. Noack, J. O'Malley-James, S. J. Payler, A. Rushby, T. Samuels,
#   P. Schwendner, J. Wadsworth and M. P. Zorzano, "Habitability: A Review",
#   Astrobiology 16(1), 89-117 (2016), doi:10.1089/ast.2015.1295,
# which in turn is adapted from
#   T. M. Hoehler, "An energy balance concept for habitability",
#   Astrobiology 7(6), 824-838 (2007), doi:10.1089/ast.2006.0095.
# The article of Cockell et al. is not open access; their figure is (c) 2016
# Mary Ann Liebert, Inc.
# Changes: redrawn with matplotlib, with the geometry measured from the
# figure of Cockell et al.; for slides, the ellipses moved closer together and
# the text enlarged; in colour, set in Montserrat, with shortened labels.
# The images carry this credit, of both works, in their metadata (TITLE,
# CREDIT, SOURCE and COPYRIGHT). Credit both wherever the images are shown,
# e.g. on the slide:
#   Adapted from Cockell et al. (2016), Astrobiology 16, 89-117, Fig. 3,
#   after Hoehler (2007), Astrobiology 7, 824-838
# Before using the images in a publication, e.g. the thesis or a website,
# check the reuse permissions of the publisher.
#
# What the figure shows
# ---------------------
# Four ellipses of the same size, one per requirement: elemental requirements
# (top left), energy (top right), appropriate physicochemical conditions
# (bottom right) and a solvent, water (bottom left). Each is tilted by 45
# degrees, with its long axis pointing at the middle of the diagram, where all
# four overlap. That region is labelled "Habitable": an
# environment is habitable only where all four requirements come together.
# Each label sits, centred, in the part of its ellipse that overlaps no other.
# A second version asks "habitable?" instead, in white.
#
# Running
# -------
# Use the virtual environment of the repository (see README-venv.md), from any
# directory:
#   .venv/bin/python other_plots/habitability_diagram/habitability_diagram.py
# The script writes habitability_diagram.png (DPI dots per inch) and .svg next
# to itself, and the second version as habitability_diagram_question.png and
# .svg, and prints the size of the PNGs. The first run needs a network
# connection to fetch the font, which pyfonts then caches. The SVG is
# identical on every run (fixed hash salt, no date), and its text is drawn as
# the outlines of the glyphs, so it displays without the font installed. Both
# have a transparent background.
#
# Coordinates
# -----------
# x runs to the right and y upwards from the middle of the diagram, in units
# of the length of an ellipse (its long axis). The canvas is fitted to the
# ellipses, so keep the labels inside them. Font sizes and line widths are in
# points: at FIG_WIDTH = 7 inches, one unit is about 390 points.
#
# Sized for slides
# ----------------
# Against the original, the text is about 1.5 times as large relative to the
# ellipses, so it stays legible when the figure is scaled down on a slide. To
# make room, the ellipses sit closer together (ELLIPSE_DISTANCE 0.32 instead
# of 0.395), which widens the middle for HABITABLE. At this size, the longest
# lines ("physicochemical", "solvent (water?)") keep about half a font size of
# space to the nearest outline, and HABITABLE about a third. Larger text
# relative to FIG_WIDTH runs them into the outlines.
#
# How to edit
# -----------
# Colours  COLOR_ELEMENTS, COLOR_ENERGY, COLOR_CONDITIONS and COLOR_SOLVENT
#          (the outline of each ellipse), COLOR_TEXT, COLOR_HABITABLE (the fill
#          of the middle), COLOR_HABITABLE_TEXT and, for the second version,
#          COLOR_HABITABLE_QUESTION_TEXT.
# Labels   REQUIREMENTS: the text of each ellipse, with "\n" between its lines,
#          and the centre of its first line, on the baseline; HABITABLE, centred
#          in the middle, and HABITABLE_QUESTION in its place in the second
#          version. FONTSIZE, FONTSIZE_HABITABLE and LINE_SPACING.
# Shape    ELLIPSE_WIDTH, ELLIPSE_DISTANCE and the direction of each ellipse in
#          REQUIREMENTS; LINEWIDTH of the outlines.
# Credit   TITLE, CREDIT, SOURCE and COPYRIGHT, written to the metadata.
# Canvas   FIG_WIDTH (the height follows from the drawing), MARGIN, DPI.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon
from matplotlib.textpath import TextPath

# Fetch Montserrat from Google Fonts and use it for every text element. The
# font file is cached locally by pyfonts, so only the first run needs a
# network connection.
from pyfonts import load_google_font, set_default_font

FONT = load_google_font("Montserrat")
FONT_BOLD = load_google_font("Montserrat", weight="bold")  # for HABITABLE
set_default_font(FONT)

# --- Credit ------------------------------------------------------------------

# Written to the metadata of the PNG and the SVG
TITLE = "Instantaneous habitability"
CREDIT = (
    'Adapted from Figure 3 of C. S. Cockell et al., "Habitability: A Review", '
    "Astrobiology 16(1), 89-117 (2016), doi:10.1089/ast.2015.1295, which is "
    'adapted from T. M. Hoehler, "An energy balance concept for habitability", '
    "Astrobiology 7(6), 824-838 (2007), doi:10.1089/ast.2006.0095"
)
# Cockell et al., then Hoehler
SOURCE = "https://doi.org/10.1089/ast.2015.1295 https://doi.org/10.1089/ast.2006.0095"
COPYRIGHT = "Original figure \N{COPYRIGHT SIGN} 2016 Mary Ann Liebert, Inc."

# --- Colours -----------------------------------------------------------------

# Outlines of the ellipses, one colour per requirement
COLOR_ELEMENTS = "#8DB9E5"
COLOR_ENERGY = "#F19EB1"
COLOR_CONDITIONS = "#FABE50"
COLOR_SOLVENT = "#00549F"
COLOR_TEXT = "#39383A"  # labels of the ellipses, the dark grey of the original
# Fill of the region inside all four ellipses; "none" leaves it transparent
COLOR_HABITABLE = "none"
COLOR_HABITABLE_TEXT = "#000000"
COLOR_HABITABLE_QUESTION_TEXT = "#FFFFFF"  # in the second version

# --- Content -----------------------------------------------------------------

# One ellipse per requirement, clockwise from the top left: (direction from
# the middle of the diagram to the centre of the ellipse, in degrees
# anticlockwise from the right; label; (x, y) of the centre of its first line,
# on the baseline; colour of the outline). Further lines follow LINE_SPACING
# below, centred as well. The positions give each label the most space to the
# outlines around it.
REQUIREMENTS = [
    (135, "chemical\nelements", (-0.366, 0.408), COLOR_ELEMENTS),
    (45, "energy", (0.366, 0.379), COLOR_ENERGY),
    (315, "physicochemical\nconditions", (0.346, -0.434), COLOR_CONDITIONS),
    (225, "solvent (water?)", (-0.346, -0.434), COLOR_SOLVENT),
]
HABITABLE = "habitable"  # centred in the region inside all four ellipses
HABITABLE_QUESTION = "habitable?"  # in its place in the second version

# --- Geometry, in lengths of an ellipse --------------------------------------

ELLIPSE_WIDTH = 0.6  # the short axis; the long axis is 1
# From the middle of the diagram to each centre; 0.395 in the original
ELLIPSE_DISTANCE = 0.32
LINEWIDTH = 2.0  # in points

# --- Text --------------------------------------------------------------------

FONTSIZE = 24  # in points
FONTSIZE_HABITABLE = 24  # in bold
LINE_SPACING = 1.25  # from one baseline to the next, in font sizes

# --- Canvas ------------------------------------------------------------------

FIG_WIDTH = 7  # in inches; the height follows from the drawing
MARGIN = 0.01  # around the ellipses
DPI = 300


def center_of(direction):
    # Centre of the ellipse in the given direction from the middle
    angle = np.radians(direction)
    return ELLIPSE_DISTANCE * np.array([np.cos(angle), np.sin(angle)])


def distance_to_outline(direction, rays):
    # Distance from the middle of the diagram to the outline of the ellipse in
    # the given direction, along each ray (unit vectors, one per row). The
    # middle lies inside every ellipse, so each ray crosses the outline once.
    angle = np.radians(direction)
    # Turn into the frame of the ellipse, with its long axis along x, and
    # stretch the ellipse to the unit circle
    turn = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    semi_axes = np.array([0.5, ELLIPSE_WIDTH / 2])
    along = rays @ turn.T / semi_axes
    start = -turn @ center_of(direction) / semi_axes
    # The ray reaches the circle where |t along + start| = 1
    a = (along**2).sum(axis=1)
    b = along @ start
    c = start @ start - 1
    return (-b + np.sqrt(b**2 - a * c)) / a


# --- The drawing -------------------------------------------------------------

# The region inside all four ellipses: along each ray from the middle, it ends
# at the nearest outline. As the region is convex, 3600 rays trace it finely.
directions = [requirement[0] for requirement in REQUIREMENTS]
angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
rays = np.column_stack([np.cos(angles), np.sin(angles)])
reach = np.min([distance_to_outline(d, rays) for d in directions], axis=0)
habitable = rays * reach[:, None]

# The canvas spans the ellipses. For a tilted ellipse, the extent along x and
# y combines both of its semi-axes.
corners = []
for direction in directions:
    angle = np.radians(direction)
    long, short = 0.5, ELLIPSE_WIDTH / 2
    half = np.hypot(
        [long * np.cos(angle), long * np.sin(angle)],
        [short * np.sin(angle), short * np.cos(angle)],
    )
    corners += [center_of(direction) - half, center_of(direction) + half]
(x0, y0), (x1, y1) = np.min(corners, axis=0), np.max(corners, axis=0)
x0, y0, x1, y1 = x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN

fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * (y1 - y0) / (x1 - x0)))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")

# Fill of the middle below the outlines, the text on top
ax.add_patch(Polygon(habitable, facecolor=COLOR_HABITABLE, edgecolor="none"))
for direction, label, (x, y), color in REQUIREMENTS:
    ax.add_patch(
        Ellipse(
            center_of(direction),
            width=1,
            height=ELLIPSE_WIDTH,
            angle=direction,
            fill=False,
            edgecolor=color,
            linewidth=LINEWIDTH,
        )
    )
    # One text per line, so that the baselines are evenly spaced whatever the
    # height of the glyphs in a line
    for index, line in enumerate(label.split("\n")):
        ax.annotate(
            line,
            (x, y),
            xytext=(0, -index * LINE_SPACING * FONTSIZE),
            textcoords="offset points",
            ha="center",
            va="baseline",
            fontsize=FONTSIZE,
            color=COLOR_TEXT,
        )

# Centre the capitals in the middle: the baseline goes half their height
# below it
capital_height = TextPath((0, 0), "H", size=1, prop=FONT_BOLD).get_extents().height
habitable_label = ax.annotate(
    HABITABLE,
    (0, 0),
    xytext=(0, -capital_height * FONTSIZE_HABITABLE / 2),
    textcoords="offset points",
    ha="center",
    va="baseline",
    fontproperties=FONT_BOLD,
    fontsize=FONTSIZE_HABITABLE,
    color=COLOR_HABITABLE_TEXT,
)

plt.rcParams["savefig.transparent"] = True
folder = Path(__file__).resolve().parent
metadata = {"Title": TITLE, "Description": CREDIT, "Source": SOURCE}


def save(name):
    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean
    plt.rcParams["svg.hashsalt"] = name
    output = folder / name
    # PNG and SVG name the copyright differently
    fig.savefig(
        output.with_suffix(".png"),
        dpi=DPI,
        metadata=metadata | {"Copyright": COPYRIGHT},
    )
    fig.savefig(
        output.with_suffix(".svg"),
        metadata=metadata | {"Rights": COPYRIGHT, "Date": None},
    )
    # Whole pixels, as matplotlib cuts off fractions
    width, height = (fig.get_size_inches() * DPI).astype(int)
    print(f"{name}: {width} x {height} px")


save("habitability_diagram")

# The second version, identical but for the label in the middle
habitable_label.set_text(HABITABLE_QUESTION)
habitable_label.set_color(COLOR_HABITABLE_QUESTION_TEXT)
save("habitability_diagram_question")
