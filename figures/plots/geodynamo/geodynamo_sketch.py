# A sketch of the geodynamo in Earth's outer core, for the PhD defense talk:
# the convection of the liquid iron, organised by the Coriolis force into
# swirling columns along the rotation axis (Taylor columns), twists the
# magnetic field lines and so keeps up Earth's magnetic field.
#
# Credit
# The figure script and this description is created
# with the help of AI, using the model Anthropic Claude Opus 5.5
# ------
# The figure is a recreation of "Dynamo Theory - Outer core convection and
# magnetic field generation" by A. Z. Colvin, licensed under CC BY-SA 4.0
# (https://creativecommons.org/licenses/by-sa/4.0/). Original:
# https://commons.wikimedia.org/wiki/File:Dynamo_Theory_-_Outer_core_convection_and_magnetic_field_generation.svg
# Changes: redrawn with matplotlib from parametric curves; the mantle drawn as
# a shell around the core; new colours, font and labels; only the names of the
# three layers kept. As an adaptation, the images fall under the same licence
# (share alike). Credit the original wherever the images are shown, e.g. on
# the slide:
#   Adapted from "Dynamo Theory - Outer core convection and magnetic field
#   generation", A. Z. Colvin, CC BY-SA 4.0
#
# What the figure shows
# ---------------------
# A cut through the Earth along its rotation axis, north at the top: the
# mantle, the liquid outer core and the solid inner core. Four Taylor columns
# stand in the outer core, two above and two below the equator, to the left
# and right of the rotation axis. Each is a
# ribbon swirling around the column axis, seen from slightly above the
# equator: its outer side shows in the front half of every turn, its inner
# side in the back half. The arrow at the end of each ribbon, next to the
# equator, gives the sense of the swirl. The magnetic field lines run from
# north to south through the columns, twisted by the flow, and fan out
# through the mantle like the field of a bar magnet; above and below the
# Earth they fade into the background. Their arrows point in the direction of
# today's field: into the Earth in the north, out of it in the south.
#
# Running
# -------
# Use the virtual environment of the repository (see README-venv.md), from any
# directory:
#   .venv/bin/python other_plots/geodynamo/geodynamo_sketch.py
# The script writes geodynamo_step<suffix>.png (DPI dots per inch) and .svg
# next to itself, one pair per entry in LAYERS, and prints the size of each
# image. The first run needs a network connection to fetch the Montserrat
# font, which pyfonts then caches. The SVGs are identical on every run (fixed
# hash salt, no date, numbered clip paths), and the text is drawn as the
# outlines of its glyphs, so it displays without the font installed.
#
# Build-up steps
# --------------
# Each entry of LAYERS adds one layer; the image of a step shows its layer and
# those of all steps before it:
#   step0  earth: mantle, outer core and inner core, the bare annuli
#   step1  + the names of the three layers
#   step2  + the Taylor columns, swirling the liquid iron of the outer core
#   step3  + the magnetic field lines, with their arrows
# All images share one canvas, FIG_SIZE inches square around the centre of
# the Earth, showing FRAME Earth radii to each side: the same size in pixels
# and points, with every element at the same place, so that the images can be
# stacked on a slide and revealed one after another. To keep it that way,
# never save with bbox_inches="tight". The images have a transparent
# background (COLOR_BACKGROUND), so that they lie on any slide background, and
# each one contains the whole sketch up to its step: the field lines of step3
# pass between the back and front halves of the ribbons of step2. Replace one step
# with the next rather than laying it on top: where both are translucent (the
# fade of the field lines, the smoothed edges), they would add up and darken.
#
# Coordinates
# -----------
# Screen coordinates are the data coordinates of the axes, with the centre of
# the Earth at the origin and equal scales in x and y. The radii of the layers
# and everything outside the outer core are in Earth radii; everything inside
# it (columns, wiggles of the field lines) is in units of R_OUTER_CORE, so that
# it scales with the core.
#
# How to edit
# -----------
# Colours      COLOR_* below, one per part of the sketch, and the background.
# Layers       R_OUTER_CORE and R_INNER_CORE. Not to scale: the real radii are
#              0.55 and 0.19 Earth radii; the core is drawn larger to leave
#              room for the columns.
# Columns      COLUMN_* (position, size and shape of the ribbons) and
#              RIBBON_* (width, arrow heads). The lower columns are the upper
#              ones turned upside down, so both change together.
# Field lines  FIELD_LINE_X (where the lines cross the equator), FAN_RADIUS
#              (how quickly they spread outside the core), WIGGLE_* (how the
#              flow twists them; another WIGGLE_SEED tangles them anew),
#              FIELD_ARROW_* (arrow heads) and FIELD_FADE_* (the fade at the top
#              and bottom).
# Labels       LABELS: text, the circle and angle to bend it along, and its
#              direction;
#              LABEL_INNER_CORE sits straight in the centre. FONTSIZE_LABELS.
# Steps        LAYERS: (file name suffix, layer name), in the order of the
#              build-up. A new layer is drawn in the loop over the steps under
#              `if "<layer name>" in layers:`, with a zorder from the list
#              below.
# Canvas       FIG_SIZE, FRAME (the part of the drawing shown), DPI; the same
#              for all steps.
#
# Drawing order (zorder): 1 mantle, 2 outer core, 3 back halves of the
# ribbons, 4 field lines, 5 front halves of the ribbons, 6 arrows on the field
# lines, 7 inner core, 8 labels. The field lines thus pass between the two
# halves of each turn.
#
# The code follows the other plotting scripts of this repository: black
# formatting at 88 columns, settings as upper-case constants at the top,
# comments in full sentences.

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

# Fetch Montserrat from Google Fonts for the labels, and make it the default
# for any other text. The font file is cached locally by pyfonts, so only the
# first run needs a network connection.
from pyfonts import load_google_font, set_default_font

FONT = load_google_font("Montserrat")
set_default_font(FONT)

# Colours of the parts of the sketch
COLOR_MANTLE = "#E6E6E6"
COLOR_OUTER_CORE = "#F19EB1"
COLOR_INNER_CORE = "#E65173"
COLOR_FIELD_LINES = "#404040"
# Swirl ribbons of the Taylor columns: their outer side, seen in the front half
# of every turn, and their inner side, seen in the back half
COLOR_COLUMNS = "#FABE50"
COLOR_COLUMNS_INSIDE = "#FDDFA7"
COLOR_LABELS = "black"
# Background of the figure: "none" leaves it transparent. The field lines fade
# out into it at the top and bottom, whatever its colour.
COLOR_BACKGROUND = "none"

# Radii of the layers, in Earth radii (the Earth's surface is at 1)
R_OUTER_CORE = 0.68
R_INNER_CORE = 0.2

# Taylor columns, in units of R_OUTER_CORE. The values describe the two
# upper columns; the lower ones are the same, rotated upside down about the
# horizontal axis in the page. Each ribbon starts at the back of its column,
# at the top, and turns clockwise (seen from above) down towards the equator,
# where it ends in an arrow.
COLUMN_X = (-0.5, 0.5)  # positions of the column axes
COLUMN_RADIUS = 0.22  # of the turns of the ribbon
COLUMN_TOP = 0.62  # height of the start of the ribbon above the equator
COLUMN_TURNS = 2.55  # length of the ribbon, in turns
COLUMN_PITCH = 0.165  # drop of the ribbon per turn
# Apparent height of a turn relative to its width: the columns are seen from
# slightly above the equator, so that a turn looks like a flat ellipse
COLUMN_TILT = 0.38
RIBBON_WIDTH = 0.05  # vertical width of the ribbon
RIBBON_ARROW_LENGTH = 0.1
RIBBON_ARROW_WIDTH = 0.12

# Magnetic field lines, one for each place where a line crosses the equator,
# in units of R_OUTER_CORE (0: the rotation axis). Near the equator the lines
# run straight up; away from it they spread like the field of a current loop
# of radius FAN_RADIUS (in Earth radii) around the equator, near its axis.
FIELD_LINE_X = (-0.63, -0.47, -0.3, -0.13, 0.0, 0.14, 0.31, 0.48, 0.62)
FAN_RADIUS = 0.8
# Inside the outer core, the flow twists the field lines: each line wiggles
# sideways with its own phases, with this amplitude and wavelength (in units
# of R_OUTER_CORE). The wiggles fade out over WIGGLE_TAPER inside the
# core-mantle boundary, so that each line runs smoothly into the mantle.
WIGGLE_AMPLITUDE = 0.08
WIGGLE_WAVELENGTH = 0.6
WIGGLE_TAPER = 0.15
WIGGLE_SEED = 1  # another seed tangles the field lines differently
LINEWIDTH_FIELD_LINES = 1.2  # in points
# Arrow heads where the field lines cross this radius (in Earth radii), and
# their size in Earth radii
FIELD_ARROW_RADIUS = (1 + R_OUTER_CORE) / 2
FIELD_ARROW_LENGTH = 0.065
FIELD_ARROW_WIDTH = 0.05
# Above this height (in Earth radii) and below its negative, the field lines
# fade out, turning fully transparent at the top and bottom of the frame.
# At least 1, the poles of the Earth, so that the fade lies on the background
# and not on the mantle.
FIELD_FADE_START = 1.0

# Names of the outer layers: text, the radius (in Earth radii) and angle (in
# degrees, counter-clockwise from the right) of its middle, and whether it runs
# clockwise. Each name is bent along the circle through its middle. Clockwise
# text reads from the bottom up on the left side and from left to right along
# the top; counter-clockwise text reads from the bottom up on the right side
# and from left to right along the bottom.
LABELS = [
    ("Mantle", (1 + R_OUTER_CORE) / 2, 180, True),
    ("Outer core", 0.9 * R_OUTER_CORE, 0, False),
]
LABEL_INNER_CORE = "Inner\ncore"  # straight, in the centre
FONTSIZE_LABELS = 20  # in points

# Build-up sequence: (file name suffix, layer added in the step). The image of
# each step shows its layer and those of all steps before it.
LAYERS = [
    ("0", "earth"),  # the three layers of the Earth, without annotations
    ("1", "labels"),  # the names of the three layers
    ("2", "columns"),  # the Taylor columns, without the field lines
    ("3", "field lines"),
]

# Canvas, the same for all steps
FIG_SIZE = 6.0  # width and height, in inches
FRAME = 1.12  # half the width of the drawing, in Earth radii
DPI = 300  # of the PNG images


def smoothstep(s):
    # 0 below 0, 1 above 1, and a smooth S-curve in between
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def arrow_head(base, direction, length, width):
    # Triangle with the middle of its base at base, pointing in direction
    direction = direction / np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    half = width / 2 * normal
    return np.array([base + half, base + length * direction, base - half])


def ribbon(x_axis, upside_down):
    # The ribbon of one column in screen coordinates, as a list of
    # (is_front, outline) pieces, one per half turn, and the arrow head at its
    # end. The angle phi around the column axis is 90 degrees at the back and
    # falls along the ribbon (clockwise seen from above). Cutting at multiples
    # of 180 degrees, the left and right edges of the column, puts every
    # piece entirely in the front or in the back half.
    start = np.pi / 2
    end = start - 2 * np.pi * COLUMN_TURNS
    cuts = np.pi * np.arange(np.floor(start / np.pi), np.ceil(end / np.pi) - 1, -1)
    edges = np.concatenate([[start], cuts[(cuts < start) & (cuts > end)], [end]])

    pieces = []
    for a, b in zip(edges[:-1], edges[1:]):
        phi = np.linspace(a, b, 100)
        x = x_axis + COLUMN_RADIUS * np.cos(phi)
        depth = COLUMN_RADIUS * np.sin(phi)  # positive at the back
        z = COLUMN_TOP - COLUMN_PITCH * (start - phi) / (2 * np.pi)
        if upside_down:
            depth, z = -depth, -z
        y = z + COLUMN_TILT * depth
        top = np.column_stack([x, y + RIBBON_WIDTH / 2])
        bottom = np.column_stack([x, y - RIBBON_WIDTH / 2])
        is_front = np.mean(depth) < 0
        pieces.append((is_front, R_OUTER_CORE * np.concatenate([top, bottom[::-1]])))

    # The arrow continues the middle line of the last piece
    tip, before = np.column_stack([x, y])[[-1, -2]]
    head = arrow_head(tip, tip - before, RIBBON_ARROW_LENGTH, RIBBON_ARROW_WIDTH)
    return pieces, (is_front, R_OUTER_CORE * head)


def field_line(x_equator, rng):
    # Points of the field line through (x_equator, 0) (in Earth radii), from
    # the bottom of the frame to its top
    z = np.linspace(-FRAME, FRAME, 4000)
    x = x_equator * (1 + (z / FAN_RADIUS) ** 2) ** 0.75
    # Wiggles of two wavelengths with random phases, inside the outer core
    wavenumber = 2 * np.pi / (WIGGLE_WAVELENGTH * R_OUTER_CORE)
    phase_long, phase_short = rng.uniform(0, 2 * np.pi, 2)
    wiggle = (
        np.sin(wavenumber * z + phase_long)
        + 0.3 * np.sin(2.1 * wavenumber * z + phase_short)
    ) / 1.3
    depth_in_core = (R_OUTER_CORE - np.hypot(x, z)) / (WIGGLE_TAPER * R_OUTER_CORE)
    x = x + WIGGLE_AMPLITUDE * R_OUTER_CORE * wiggle * smoothstep(depth_in_core)
    return np.column_stack([x, z])


def field_arrows(line):
    # Arrow heads where the line crosses FIELD_ARROW_RADIUS in the north and in
    # the south, both pointing south (down the line), centred on the crossing
    r = np.hypot(*line.T)
    heads = []
    for hemisphere in (1, -1):
        outside = (r > FIELD_ARROW_RADIUS) & (hemisphere * line[:, 1] > 0)
        # The last point before the crossing, coming from the south
        i = np.flatnonzero(np.diff(outside.astype(int)) != 0)
        if len(i) == 0:
            continue
        i = i[0]
        s = (FIELD_ARROW_RADIUS - r[i]) / (r[i + 1] - r[i])
        crossing = line[i] + s * (line[i + 1] - line[i])
        south = line[i] - line[i + 1]  # the points run from south to north
        south /= np.linalg.norm(south)
        base = crossing - FIELD_ARROW_LENGTH / 2 * south
        heads.append(arrow_head(base, south, FIELD_ARROW_LENGTH, FIELD_ARROW_WIDTH))
    return heads


def outline(points, width):
    # Outline of the line through points drawn with the given width (in Earth
    # radii) and flat ends: both sides of the line, half the width away from
    # it. The field lines bend gently enough for the sides not to cross.
    tangent = np.gradient(points, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    side = width / 2 * np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return MplPath(np.concatenate([points + side, (points - side)[::-1]]))


def curved_text(text, radius, angle, clockwise, size):
    # Outline of a text with the given font size (in Earth radii), centred on
    # the point at radius and angle (degrees) and bent along the circle through
    # it: each point of the glyphs moves along the circle by its distance from
    # the middle of the text, and away from it by its height above the middle.
    # The tops of the letters point away from the centre of the circle for
    # clockwise text, towards it for counter-clockwise text.
    glyphs = TextPath((0, 0), text, size=size, prop=FONT)
    (x_min, _), (x_max, _) = glyphs.get_extents().get_points()
    # The middle lies halfway between the lowest and highest points of "lp",
    # as for centred matplotlib text
    reference = TextPath((0, 0), "lp", size=size, prop=FONT)
    (_, y_min), (_, y_max) = reference.get_extents().get_points()
    along = glyphs.vertices[:, 0] - (x_min + x_max) / 2
    up = glyphs.vertices[:, 1] - (y_min + y_max) / 2
    sense = -1 if clockwise else 1  # +1: the angle grows along the text
    theta = np.radians(angle) + sense * along / radius
    rho = radius - sense * up
    vertices = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
    return MplPath(vertices, glyphs.codes)


# The geometry of all layers, computed once, so that every step shows the same
# elements at the same place

# Ribbons of the Taylor columns, as (is_front, outline): the half turns and the
# arrow heads of all four columns
ribbons = []
for x_axis in COLUMN_X:
    for upside_down in (False, True):
        column_pieces, head = ribbon(x_axis, upside_down)
        ribbons += column_pieces + [head]

# Field lines and their arrow heads in the mantle. The middle of each line, up
# to FIELD_FADE_START from the equator, is drawn as a line. Its ends are drawn
# as a gradient that fades out towards the top and bottom of the frame, clipped
# to their outlines: one image with a smooth fade, where translucent pieces of
# line would show seams wherever they meet.
rng = np.random.default_rng(WIGGLE_SEED)
lines = [field_line(x * R_OUTER_CORE, rng) for x in FIELD_LINE_X]
field_heads = [head for line in lines for head in field_arrows(line)]
line_middles = [line[np.abs(line[:, 1]) <= FIELD_FADE_START] for line in lines]
line_width = LINEWIDTH_FIELD_LINES * 2 * FRAME / (FIG_SIZE * 72)  # in Earth radii
line_ends = MplPath.make_compound_path(
    *(
        outline(line[hemisphere * line[:, 1] >= FIELD_FADE_START], line_width)
        for line in lines
        for hemisphere in (1, -1)
    )
)
# The gradient, as an image of one column from the bottom of the frame to its
# top
heights = np.linspace(-FRAME, FRAME, 1000)
fade = (np.abs(heights) - FIELD_FADE_START) / (FRAME - FIELD_FADE_START)
line_gradient = np.tile(to_rgba(COLOR_FIELD_LINES), (len(heights), 1, 1))
line_gradient[:, 0, 3] *= 1 - np.clip(fade, 0, 1)

# Names of the outer layers, with the font size converted from points to Earth
# radii
size = FONTSIZE_LABELS * 2 * FRAME / (FIG_SIZE * 72)
label_paths = [
    curved_text(text, radius, angle, clockwise, size)
    for text, radius, angle, clockwise in LABELS
]

for step, (suffix, _) in enumerate(LAYERS):
    layers = [layer for _, layer in LAYERS[: step + 1]]
    fig = plt.figure(figsize=(FIG_SIZE, FIG_SIZE), facecolor=COLOR_BACKGROUND)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-FRAME, FRAME)
    ax.set_ylim(-FRAME, FRAME)
    ax.set_aspect("equal")
    ax.set_axis_off()

    if "earth" in layers:
        ax.add_patch(Circle((0, 0), 1, color=COLOR_MANTLE, zorder=1))
        ax.add_patch(Circle((0, 0), R_OUTER_CORE, color=COLOR_OUTER_CORE, zorder=2))
        ax.add_patch(Circle((0, 0), R_INNER_CORE, color=COLOR_INNER_CORE, zorder=7))

    if "labels" in layers:
        for path in label_paths:
            ax.add_patch(
                PathPatch(path, facecolor=COLOR_LABELS, edgecolor="none", zorder=8)
            )
        ax.text(
            0,
            0,
            LABEL_INNER_CORE,
            color=COLOR_LABELS,
            fontsize=FONTSIZE_LABELS,
            ha="center",
            va="center",
            linespacing=1.1,
            zorder=8,
        )

    # Field lines: the faded ends first, so that the round caps of the middles
    # cover the seams where the two meet
    if "field lines" in layers:
        ends = ax.imshow(
            line_gradient,
            extent=(-FRAME, FRAME, -FRAME, FRAME),
            origin="lower",
            interpolation="bilinear",
            zorder=4,
        )
        ends.set_clip_path(line_ends, ax.transData)
        ax.add_collection(
            LineCollection(
                line_middles,
                colors=COLOR_FIELD_LINES,
                linewidths=LINEWIDTH_FIELD_LINES,
                capstyle="round",
                joinstyle="round",
                zorder=4,
            )
        )
        ax.add_collection(
            PolyCollection(
                field_heads,
                facecolors=COLOR_FIELD_LINES,
                edgecolors="none",
                zorder=6,
            )
        )

    # Ribbons: the back halves of the turns below the field lines, the front
    # halves above them. Each piece is edged in its own colour, which hides the
    # seams between neighbouring pieces.
    if "columns" in layers:
        for is_front, zorder in ((False, 3), (True, 5)):
            color = COLOR_COLUMNS if is_front else COLOR_COLUMNS_INSIDE
            ax.add_collection(
                PolyCollection(
                    [outline for front, outline in ribbons if front == is_front],
                    facecolors=color,
                    edgecolors=color,
                    linewidths=0.3,
                    zorder=zorder,
                )
            )

    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean. The ids of clip paths other than
    # rectangles hash their place in memory, though, so they are numbered in
    # order instead.
    name = f"geodynamo_step{suffix}"
    plt.rcParams["svg.hashsalt"] = name
    output = Path(__file__).resolve().parent / name
    fig.savefig(output.with_suffix(".png"), dpi=DPI)
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", metadata={"Date": None})
    svg = buffer.getvalue()
    for number, oid in enumerate(re.findall(r'<clipPath id="(\w+)"', svg)):
        svg = svg.replace(oid, f"clip{number}")
    output.with_suffix(".svg").write_text(svg, encoding="utf-8")

    # Whole pixels, as matplotlib cuts off fractions
    width, height = (fig.get_size_inches() * DPI).astype(int)
    print(f"{name}: {', '.join(layers)} ({width} x {height} px)")
    plt.close(fig)
