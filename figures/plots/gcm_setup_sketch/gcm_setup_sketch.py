# The building blocks of a generic climate model (general circulation model,
# GCM), for the PhD defense talk: a sequence of images that builds up the
# sketch step by step on the slides.
#
# Credit
# The figure script and this description is created
# with the help of AI, using the model Anthropic Claude Opus 5.5
# ------
# The figure is a recreation of "Schematic structure of a General Circulation
# Model" by David Bice, © Penn State University, licensed under CC BY-NC-SA
# 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/). Original:
# https://courses.ems.psu.edu/earth103/node/524
# Changes: redrawn with matplotlib as a parametric mesh; new colours, font and
# labels; circulation, radiation and clouds placed anew; split into build-up
# steps. As an adaptation, the images fall under the same licence (share
# alike, non-commercial use only). Credit the original wherever the images
# are shown, e.g. on the slide:
#   Adapted from "Schematic structure of a General Circulation Model",
#   D. Bice, © Penn State University, CC BY-NC-SA 4.0
#
# What the figure shows
# ---------------------
# A cutout of a cylindrical shell through the top of the Earth: the
# lithosphere; the ocean, with thin layers near its surface, a shallow shelf
# and sea ice; the land surface with soil, vegetation, bare soil and an ice
# sheet; and the atmosphere. An oblique projection shows three faces: the
# front (a cross section), the top of the atmosphere and the right side. The
# sea surface shows through the atmosphere as a darker area. Circulation,
# radiation, clouds and the grid of the model are drawn on top in later steps.
#
# Running
# -------
# Use the virtual environment of the repository (see README-venv.md), from any
# directory:
#   .venv/bin/python other_plots/gcm_setup_sketch/gcm_setup_sketch.py
# The script writes gcm_setup_step<suffix>.png (DPI dots per inch) and .svg
# next to itself, one pair per entry in LAYERS, and prints the size of each
# image. The first run needs a network connection to fetch the Montserrat
# font, which pyfonts then caches. The SVGs are identical on every run (fixed
# hash salt, no date), and the labels are drawn as the outlines of their
# glyphs, so they display without the font installed.
#
# Build-up steps
# --------------
# Each entry of LAYERS adds one layer; the image of a step shows its layer and
# those of all steps before it:
#   step1    domain: the regimes, their outlines, the sea surface seen
#            through the atmosphere, and the names of the regimes; no grid
#   step1_5  + circulation: an overturning loop in the ocean and a cell in
#            the atmosphere above it
#   step2    + radiation: incoming sunlight, part of it reflected (short
#            wavelengths), and heat radiated to space (long wavelengths)
#   step3    + clouds in the front face of the atmosphere
#   step4    + the grid of the numerical model
# All images share one canvas, fitted once to everything drawn in the last
# step: the same size in pixels and points, with every element at the same
# place, so that the images can be stacked on a slide and revealed one after
# another. To keep it that way, never save with bbox_inches="tight", and add
# the screen geometry of every new element to `drawn`, so that the canvas
# includes it. The PNGs have an opaque white background.
#
# Coordinates
# -----------
# Everything is placed in cylindrical coordinates of the cutout:
#   phi    angle around the cylinder axis; 0 points straight up, positive
#          angles to the right. In degrees in the tables (LABELS, ARROWS,
#          CIRCULATION, CLOUDS), in radians in PHI_EDGES and in the code. The
#          cutout spans -22.5 to 22.5 degrees in 10 columns of 4.5 degrees,
#          centred at -20.25, -15.75, ..., 20.25; the coast between the ocean
#          (left) and the land (right) is at 0.
#   r      radius: R_BOTTOM = 6.7 (bottom of the cutout and ocean floor),
#          R_SEA = 8 (sea level and land surface), R_TOP = 9.1 (top of the
#          atmosphere). The layers of the atmosphere end at 8.22, 8.5, 8.8
#          and 9.1, the thin top layers of the ocean reach down to 7.74.
#   depth  distance into the page, from 0 (front face) to 1.75 (back), in 4
#          rows of cells.
# project(phi, r, depth) turns these into screen coordinates (phi in radians;
# to_screen() takes degrees). It is an oblique projection: the front face is
# seen face-on, stretched vertically by VERTICAL_EXAGGERATION, and the depth
# axis recedes up and to the right at DEPTH_ANGLE. Screen coordinates are the
# data coordinates of the axes, with equal scales in x and y. The canvas spans
# about 8.5 x 6.4 of these units, and one unit is about 68 points at
# FIG_WIDTH = 8 inches. Sizes "measured on screen" (wavelengths, cloud width,
# offsets of arrows) are in these units; line widths and font sizes are in
# points. Faces seen from behind are left out automatically.
#
# The mesh
# --------
# COLUMNS gives the ground and the surface cover of each column, from left to
# right; column_cells() turns an entry into its stack of cells from the bottom
# to the top, each belonging to a regime: lithosphere, ocean, soil, sea ice,
# vegetation, ice sheet or atmosphere. There must be len(PHI_EDGES) - 1
# columns. The layer thicknesses are set by OCEAN_LAYERS, SHELF_LAYERS,
# SOIL_THICKNESS, COVER_THICKNESS and ATMOSPHERE_LAYERS; every cover needs an
# entry in both COVER_THICKNESS and STYLES. The lithosphere is not part of the
# model grid and is drawn as one block.
#
# How to edit
# -----------
# Colours      STYLES (fill of each regime, and grid_color for the lines
#              between its cells, light in the dark ocean), SEA_SURFACE_STYLE,
#              RADIATION, the first entry of each CIRCULATION arrow,
#              COLOR_CLOUDS and COLOR_LINES.
# Labels       LABELS: text, colour and position. Each label is centred on
#              its position and bent along the arc of constant r through it
#              (curved_text()), so that it follows the curvature of the mesh
#              at its height. Font size FONTSIZE_LABELS. The labels are not
#              part of `drawn`, as their size follows from the canvas, so
#              keep them inside the domain.
# Radiation    ARROWS: wavy arrows between two positions, or between a
#              position and an offset on screen (used for the vertical rays of
#              sunlight). The kind of radiation sets colour, wavelength and
#              amplitude in RADIATION.
# Circulation  CIRCULATION: arrows along arcs of ellipses in (phi, r) in the
#              front face. They run from the start to the end angle: growing
#              angles turn counter-clockwise, falling ones clockwise.
# Clouds       CLOUDS (positions in the front face), CLOUD_WIDTH, and the
#              shape in CLOUD_BUMPS.
# Steps        To add a step, add (file name suffix, layer name) to LAYERS,
#              draw its elements in the loop over the steps under
#              `if "<layer name>" in layers:` with a zorder from the list
#              below, and add their screen geometry to `drawn`. The order of
#              LAYERS is the order of the build-up.
# Canvas       FIG_WIDTH (the height follows from the drawing), MARGIN, DPI.
# After an edit, run the script and check the last step, which contains
# everything, for overlaps.
#
# Drawing order (zorder): 1 atmosphere, 2 sea surface seen through it, 3 the
# other regimes, 4 grid, 5 outlines, 6 circulation and clouds, 7 radiation,
# 8 labels.
#
# How the mesh is drawn
# ---------------------
# For each visible face, face_labels() finds the cell behind each patch of the
# grid of all cell boundaries (EDGES). Where two neighbouring patches belong
# to different cells, add_face_lines() adds a line: an "outline" between two
# regimes and along the edges of the cutout, a "grid" line between two cells
# of one regime. Lines are keyed by their coordinates, not by their face, so
# that edges shared by two faces are drawn only once. face_fills() fills runs
# of patches of the same regime; each fill is edged in its own colour, which
# hides the seams between neighbouring fills.
#
# The code follows the other plotting scripts of this repository: black
# formatting at 88 columns, settings as upper-case constants at the top,
# comments in full sentences.

from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import ArrowStyle, FancyArrowPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

# Fetch Montserrat from Google Fonts for the labels, and make it the default
# for any other text. The font file is cached locally by pyfonts, so only the
# first run needs a network connection.
from pyfonts import load_google_font, set_default_font

FONT = load_google_font("Montserrat")
set_default_font(FONT)

# Points of the cutout are given in cylindrical coordinates: the angle phi
# around the cylinder axis (0 points straight up, positive angles to the
# right), the radius r from the axis and the depth into the page along the
# axis. The drawing is not to scale; lengths are in arbitrary units.
AXES = ("phi", "r", "depth")

R_SEA = 8.0  # sea level, also the land surface below vegetation and ice

# Boundaries of the 10 columns of cells side by side, and of the 4 rows of
# cells one behind the other
PHI_EDGES = np.radians(np.linspace(-22.5, 22.5, 11))
DEPTH_EDGES = np.linspace(0, 1.75, 5)

# Layer thicknesses from the sea surface downwards: thin layers near the
# surface, thick ones in the deep ocean. The ocean reaches the bottom of the
# cutout.
OCEAN_LAYERS = [0.065] * 4 + [0.26] * 4
SHELF_LAYERS = 4  # ocean layers above the continental shelf
SOIL_THICKNESS = 0.07  # a single layer below the land surface
# Layer thicknesses from sea level upwards; the lowest layer starts on top of
# whatever covers the surface of its column
ATMOSPHERE_LAYERS = [0.22, 0.28, 0.3, 0.3]

# Layers on top of the surface
COVER_THICKNESS = {
    "sea ice": 0.03,
    "vegetation": 0.04,
    "ice sheet": 0.17,
}

# One entry per column, from left to right: the ground ("ocean" down to the
# bottom of the cutout, "shelf" for a shallow sea above the lithosphere,
# "land" for soil above the lithosphere) and what covers it (None: nothing)
COLUMNS = [
    ("ocean", "sea ice"),
    ("ocean", None),
    ("ocean", None),
    ("ocean", None),
    ("shelf", None),
    ("land", "vegetation"),
    ("land", "vegetation"),
    ("land", None),
    ("land", "vegetation"),
    ("land", "ice sheet"),
]

R_BOTTOM = R_SEA - sum(OCEAN_LAYERS)
R_TOP = R_SEA + sum(ATMOSPHERE_LAYERS)

# Build-up sequence: (file name suffix, layer added in the step). The image
# of each step shows its layer and those of all steps before it.
LAYERS = [
    ("1", "domain"),
    ("1_5", "circulation"),
    ("2", "radiation"),
    ("3", "clouds"),
    ("4", "grid"),
]

COLOR_ICE = "#DCDDDE"

# Look of each regime: fill colour, whether its cells are outlined, and
# optionally the colour of the lines between its cells (default: COLOR_LINES;
# the boundaries between regimes always have that colour). The lithosphere is
# not part of the model grid and is drawn as one block.
STYLES = {
    "atmosphere": dict(color="#8DB9E5", grid=True),
    # Light grid lines, which remain visible on the dark blue
    "ocean": dict(color="#00549F", grid=True, grid_color="#8DB9E5"),
    "sea ice": dict(color=COLOR_ICE, grid=True),
    "soil": dict(color="#8B6A45", grid=True),
    "vegetation": dict(color="#ABCB91", grid=True),
    "ice sheet": dict(color=COLOR_ICE, grid=True),
    "lithosphere": dict(color="#9C9E9F", grid=False),
}
# The sea surface, seen through the atmosphere above the ocean columns
SEA_SURFACE_STYLE = dict(color="#00549F", alpha=0.25)

COLOR_LINES = "#333333"
LINEWIDTH = 0.8  # grid lines and outlines, also of the clouds

# Names of the regimes: text, colour, and position as (phi in degrees, r,
# depth). Each name is centred on its position and bent along the arc of
# constant r through it.
LABELS = [
    ("atmosphere", "black", (7, R_TOP, 1.05)),
    ("ocean", "white", (-13.5, R_SEA - 0.72, 0)),
    ("lithosphere", "black", (9, R_SEA - 0.68, 0)),
]
FONTSIZE_LABELS = 24  # in points

# Radiation, drawn as wavy arrows: colour, and wavelength and amplitude of
# the waves in the length units of the drawing, measured on screen
RADIATION = {
    "shortwave": dict(color="#FABE50", wavelength=0.25, amplitude=0.07),
    "longwave": dict(color="#E65173", wavelength=0.6, amplitude=0.09),
}
# Each arrow: the kind of radiation, and its start and end, each either a
# position (phi in degrees, r, depth) or, for one of the two, an offset
# (dx, dy) on screen from the other end
ARROWS = [
    # Sunlight onto the top of the atmosphere, in parallel rays from above ...
    ("shortwave", (0, 1.4), (-14, R_TOP, 0.9)),
    ("shortwave", (0, 1.4), (-2, R_TOP, 0.9)),
    ("shortwave", (0, 1.4), (17, R_TOP, 0.9)),
    # ... part of which is reflected back to space
    ("shortwave", (18.5, R_TOP, 0.9), (0.75, 0.9)),
    # Heat radiated to space, from the lower atmosphere above the ocean and
    # from the bare soil
    ("longwave", (-15.5, R_SEA + 0.3, 0), (-15.5, R_TOP + 0.45, 0)),
    ("longwave", (11.25, R_SEA + 0.12, 0), (11.25, R_TOP + 0.45, 0)),
]
LINEWIDTH_ARROWS = 3.0  # in points
# Size of the arrow heads, in units of ARROW_HEAD_SCALE points
ARROW_HEAD = dict(head_length=0.7, head_width=0.4)
ARROW_HEAD_SCALE = 20
# The waves end in a straight piece of this length, below the arrow heads
STRAIGHT_END = 0.25

# Circulation, drawn as arrows in the front face along arcs of ellipses in
# (phi, r): colour, centre (phi in degrees, r), half axes (in degrees, in r),
# and the angles on the ellipse (in degrees, counter-clockwise from the
# direction of growing phi) at which the arrow starts and ends. A flow along
# the layers has a half axis of 0 in r.
CIRCULATION = [
    # Overturning of the ocean, around its name: sinking near the sea ice and
    # rising further out
    ("white", (-13.5, R_SEA - 0.78), (7.5, 0.38), 100, 250),
    ("white", (-13.5, R_SEA - 0.78), (7.5, 0.38), 280, 430),
    # A cell in the atmosphere above, sinking towards the coast
    ("black", (-7.5, R_SEA + 0.36), (5, 0.18), 80, -70),
    ("black", (-7.5, R_SEA + 0.36), (5, 0.18), 260, 110),
]
LINEWIDTH_CIRCULATION = 2.0  # in points
CIRCULATION_HEAD_SCALE = 14  # size of the arrow heads, see ARROW_HEAD

# Clouds in the front face of the atmosphere: (phi in degrees, r) of the
# middle of their flat bottom
CLOUDS = [
    (-10, R_SEA + 0.65),
    (-2.5, R_SEA + 0.8),
    (6, R_SEA + 0.45),
    (16.5, R_SEA + 0.6),
]
CLOUD_WIDTH = 0.6  # in the length units of the drawing, measured on screen
# Outline of a cloud: its bumps from left to right, as (x, y, radius) in
# units of its width, with the flat bottom at y = 0 from the lowest point of
# the first bump to that of the last
CLOUD_BUMPS = [
    (-0.33, 0.17, 0.17),
    (-0.12, 0.3, 0.23),
    (0.14, 0.3, 0.2),
    (0.34, 0.16, 0.16),
]
COLOR_CLOUDS = COLOR_ICE

# Oblique projection, as in the original sketch: the front face is seen
# face-on, and the depth axis recedes up and to the right at this angle above
# the horizontal, at full length
DEPTH_ANGLE = np.radians(55)
# The front face is stretched vertically by this factor, which makes the
# layers look thicker and the arcs more curved, like in the original
VERTICAL_EXAGGERATION = 1.3

ARC_STEP = np.radians(0.25)  # sampling of curved lines
FIG_WIDTH = 8.0  # in inches; the height follows from the drawing
DPI = 300  # of the PNG images
MARGIN = 0.03  # white space around the drawing, as a fraction of its width


def column_cells(ground, cover):
    # Radial cells of one column, from the bottom to the top, as
    # (regime, r_bottom, r_top)
    ocean_levels = R_SEA - np.concatenate([[0], np.cumsum(OCEAN_LAYERS)])
    cells = []
    if ground == "land":
        cells.append(("lithosphere", R_BOTTOM, R_SEA - SOIL_THICKNESS))
        cells.append(("soil", R_SEA - SOIL_THICKNESS, R_SEA))
    else:
        n_ocean = len(OCEAN_LAYERS) if ground == "ocean" else SHELF_LAYERS
        if n_ocean < len(OCEAN_LAYERS):
            cells.append(("lithosphere", R_BOTTOM, ocean_levels[n_ocean]))
        for layer in reversed(range(n_ocean)):
            cells.append(("ocean", ocean_levels[layer + 1], ocean_levels[layer]))

    surface = R_SEA
    if cover is not None:
        cells.append((cover, surface, surface + COVER_THICKNESS[cover]))
        surface += COVER_THICKNESS[cover]
    for level in R_SEA + np.cumsum(ATMOSPHERE_LAYERS):
        cells.append(("atmosphere", surface, level))
        surface = level
    return cells


CELLS = [column_cells(ground, cover) for ground, cover in COLUMNS]

# Every cell boundary along each coordinate. Rounding merges boundaries that
# differ by floating-point noise only.
EDGES = {
    "phi": PHI_EDGES,
    "r": np.unique(np.round([r for col in CELLS for _, *rs in col for r in rs], 9)),
    "depth": DEPTH_EDGES,
}

# Outer faces of the cutout: the fixed coordinate, the index of its value in
# EDGES, and the two free coordinates u and v, ordered so that u x v points
# out of the cutout. A face is seen from the outside, and so drawn, when its
# outline runs counter-clockwise on screen.
FACES = [
    ("depth", 0, "phi", "r"),  # front
    ("depth", -1, "r", "phi"),  # back
    ("r", -1, "phi", "depth"),  # top
    ("r", 0, "depth", "phi"),  # bottom
    ("phi", -1, "depth", "r"),  # right
    ("phi", 0, "r", "depth"),  # left
]


def project(phi, r, depth):
    # Screen coordinates of points given in cylindrical coordinates
    x = r * np.sin(phi) + depth * np.cos(DEPTH_ANGLE)
    y = VERTICAL_EXAGGERATION * r * np.cos(phi) + depth * np.sin(DEPTH_ANGLE)
    return np.broadcast_arrays(x, y)


def sample(axis, start, end):
    # Points along one coordinate; only lines along phi are curved
    if axis != "phi":
        return np.array([start, end])
    n = max(2, int(np.ceil(abs(end - start) / ARC_STEP)) + 1)
    return np.linspace(start, end, n)


def to_cylindrical(surface, u, v):
    # Cylindrical coordinates of the points (u, v) on a surface of constant
    # coordinate, given as (fixed axis, its value, u axis, v axis)
    fixed_axis, fixed_value, u_axis, v_axis = surface
    coords = {fixed_axis: fixed_value, u_axis: u, v_axis: v}
    return coords["phi"], coords["r"], coords["depth"]


def outline(surface, u_range, v_range):
    # Projected outline of the rectangle u_range x v_range on a surface of
    # constant coordinate, counter-clockwise in (u, v)
    _, _, u_axis, v_axis = surface
    (u0, u1), (v0, v1) = u_range, v_range
    us, vs = sample(u_axis, u0, u1), sample(v_axis, v0, v1)
    u = np.concatenate([us, np.full(len(vs), u1), us[::-1], np.full(len(vs), u0)])
    v = np.concatenate([np.full(len(us), v0), vs, np.full(len(us), v1), vs[::-1]])
    return np.column_stack(project(*to_cylindrical(surface, u, v)))


def face_surface(face):
    # The surface of an outer face, with the value of its fixed coordinate
    fixed_axis, index, u_axis, v_axis = face
    return fixed_axis, EDGES[fixed_axis][index], u_axis, v_axis


def is_visible(face):
    _, _, u_axis, v_axis = face
    x, y = outline(face_surface(face), EDGES[u_axis][[0, -1]], EDGES[v_axis][[0, -1]]).T
    return np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) > 0


def cell_label(phi, r, depth):
    # The cell containing a point inside the cutout: (regime, column, layer,
    # row) for regimes with a grid, and (regime,) for the others, whose cells
    # merge into one block. None outside the cutout.
    col = np.searchsorted(PHI_EDGES, phi) - 1
    row = np.searchsorted(DEPTH_EDGES, depth) - 1
    if not (0 <= col < len(COLUMNS) and 0 <= row < len(DEPTH_EDGES) - 1):
        return None
    for layer, (regime, r_bottom, r_top) in enumerate(CELLS[col]):
        if r_bottom < r < r_top:
            return (regime, col, layer, row) if STYLES[regime]["grid"] else (regime,)
    return None


def face_labels(face):
    # Labels of the cells just behind a face, on the grid of all cell
    # boundaries, padded with None on all sides
    fixed_axis, index, u_axis, v_axis = face
    edges = EDGES[fixed_axis]
    inside = (edges[index] + edges[1 if index == 0 else -2]) / 2
    u_mid = (EDGES[u_axis][1:] + EDGES[u_axis][:-1]) / 2
    v_mid = (EDGES[v_axis][1:] + EDGES[v_axis][:-1]) / 2
    labels = np.full((len(u_mid) + 2, len(v_mid) + 2), None, dtype=object)
    for a, u in enumerate(u_mid):
        for b, v in enumerate(v_mid):
            point = to_cylindrical((fixed_axis, inside, u_axis, v_axis), u, v)
            labels[a + 1, b + 1] = cell_label(*point)
    return labels


def add_face_lines(lines, face, labels):
    # Add the boundaries between differing cells of a face to lines, which
    # maps (kind, colour) -> (coordinate along the line, values of the other
    # two coordinates) -> the intervals covered. Lines of kind "outline" are
    # the edges of the cutout and the boundaries between regimes, those of
    # kind "grid" separate the cells of one regime. Keying by the
    # coordinates, not by the face, draws edges shared by two faces only once.
    fixed_axis, fixed_value, u_axis, v_axis = face_surface(face)
    for along, across in ((v_axis, u_axis), (u_axis, v_axis)):
        # Put the labels in (across, along) order
        grid = labels if across == u_axis else labels.T
        for a, position in enumerate(EDGES[across]):
            for b, (start, end) in enumerate(zip(EDGES[along][:-1], EDGES[along][1:])):
                before, after = grid[a, b + 1], grid[a + 1, b + 1]
                if before == after:
                    continue
                if before and after and before[0] == after[0]:
                    kind = ("grid", STYLES[before[0]].get("grid_color", COLOR_LINES))
                else:
                    kind = ("outline", COLOR_LINES)
                fixed = {fixed_axis: fixed_value, across: position}
                others = [round(fixed[ax], 9) for ax in AXES if ax != along]
                key = (along, *others)
                lines.setdefault(kind, {}).setdefault(key, []).append((start, end))


def face_fills(face, labels):
    # Fill polygons of a face as (regime, polygon); consecutive cells of the
    # same regime along v share one polygon
    surface = face_surface(face)
    _, _, u_axis, v_axis = surface
    u_edges, v_edges = EDGES[u_axis], EDGES[v_axis]
    fills = []
    for a in range(len(u_edges) - 1):
        column = labels[a + 1, 1:-1]
        b = 0
        for regime, run in groupby(column, key=lambda label: label and label[0]):
            n = len(list(run))
            if regime is not None:
                v_range = (v_edges[b], v_edges[b + n])
                fills.append((regime, outline(surface, u_edges[a : a + 2], v_range)))
            b += n
    return fills


def merge(intervals):
    # Merge overlapping and touching intervals
    runs = []
    for start, end in sorted(intervals):
        if runs and start <= runs[-1][1] + 1e-9:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    return runs


def polyline(key, start, end):
    # Projected line along one coordinate, the other two fixed
    along, *fixed_values = key
    coords = dict(zip([ax for ax in AXES if ax != along], fixed_values))
    coords[along] = sample(along, start, end)
    return np.column_stack(project(coords["phi"], coords["r"], coords["depth"]))


def to_screen(phi, r, depth):
    # Screen point of a position given with phi in degrees
    return np.array(project(np.radians(phi), r, depth), dtype=float)


def wave(start, end, wavelength, amplitude):
    # Screen points of a wave from start to end. It fades out smoothly over
    # half a wavelength and ends straight, so that the arrow head sitting on
    # the straight end points along the arrow.
    length = np.linalg.norm(end - start)
    along = (end - start) / length
    across = np.array([-along[1], along[0]])
    s = np.linspace(0, length, 400)
    envelope = np.clip((length - STRAIGHT_END - s) / (wavelength / 2), 0, 1)
    envelope = envelope**2 * (3 - 2 * envelope)
    offset = amplitude * envelope * np.sin(2 * np.pi * s / wavelength)
    return start + np.outer(s, along) + np.outer(offset, across)


def cloud(bottom_middle, width):
    # Screen outline of a cloud: arcs over its bumps from right to left, and
    # back along its flat bottom, which is centred on bottom_middle
    bumps = [
        (bottom_middle + width * np.array([x, y]), width * radius)
        for x, y, radius in CLOUD_BUMPS
    ]
    # The outline passes from one bump to the next where their circles cross
    # in the upper half of the cloud
    joints = []
    for (center_1, radius_1), (center_2, radius_2) in zip(bumps, bumps[1:]):
        distance = np.linalg.norm(center_2 - center_1)
        along = (center_2 - center_1) / distance
        a = (radius_1**2 - radius_2**2 + distance**2) / (2 * distance)
        h = np.sqrt(radius_1**2 - a**2)
        crossings = center_1 + a * along + np.outer([h, -h], [-along[1], along[0]])
        joints.append(max(crossings, key=lambda point: point[1]))
    # Joints of each bump with its left and right neighbours; None at the
    # ends of the cloud, whose arcs start and end at their lowest points
    joints = [None, *joints, None]
    arcs = []
    for i in reversed(range(len(bumps))):
        center, radius = bumps[i]
        start, stop = joints[i + 1], joints[i]
        start = -np.pi / 2 if start is None else np.arctan2(*(start - center)[::-1])
        stop = 3 * np.pi / 2 if stop is None else np.arctan2(*(stop - center)[::-1])
        angles = np.linspace(start, stop + 2 * np.pi * (stop < start), 60)
        arcs.append(center + radius * np.column_stack([np.cos(angles), np.sin(angles)]))
    return np.concatenate(arcs)


def curved_text(text, position, size):
    # Screen outline of a text with the given font size (in the length units
    # of the drawing), centred on position (phi in degrees, r, depth) and bent
    # along the arc of constant r through it: each point of the glyphs moves
    # along the arc by its distance from the middle of the text, and away from
    # the arc by its height above the middle.
    phi, r, depth = position
    glyphs = TextPath((0, 0), text, size=size, prop=FONT)
    # The middle lies halfway between the lowest and highest points of "lp",
    # as for centred matplotlib text, so that names with and without
    # ascenders or descenders sit alike
    (x_min, _), (x_max, _) = glyphs.get_extents().get_points()
    reference = TextPath((0, 0), "lp", size=size, prop=FONT)
    (_, y_min), (_, y_max) = reference.get_extents().get_points()
    along = glyphs.vertices[:, 0] - (x_min + x_max) / 2
    across = glyphs.vertices[:, 1] - (y_min + y_max) / 2
    # The arc on screen, far beyond the ends of the text, with its length
    # measured from position, which is its middle point
    arc = to_screen(phi + np.linspace(-45, 45, 1801), r, depth).T
    steps = np.linalg.norm(np.diff(arc, axis=0), axis=1)
    length = np.concatenate([[0], np.cumsum(steps)])
    length -= length[len(arc) // 2]
    tangent = np.gradient(arc, axis=0)
    points, directions = (
        np.column_stack([np.interp(along, length, xy[:, i]) for i in range(2)])
        for xy in (arc, tangent)
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    normals = np.column_stack([-directions[:, 1], directions[:, 0]])
    return MplPath(points + across[:, None] * normals, glyphs.codes)


fills, lines = [], {}
for face in filter(is_visible, FACES):
    labels = face_labels(face)
    fills += face_fills(face, labels)
    add_face_lines(lines, face, labels)
polylines = {
    kind: [
        polyline(key, start, end)
        for key, intervals in lines_of_kind.items()
        for start, end in merge(intervals)
    ]
    for kind, lines_of_kind in lines.items()
}

# The sea surface above each run of ocean columns
sea_level = ("r", R_SEA, "phi", "depth")
sea_surfaces = []
col = 0
for is_sea, run in groupby(COLUMNS, key=lambda column: column[0] != "land"):
    n = len(list(run))
    if is_sea:
        phi_range = PHI_EDGES[[col, col + n]]
        sea_surfaces.append(outline(sea_level, phi_range, DEPTH_EDGES[[0, -1]]))
    col += n

# Radiation arrows as (colour, screen path), and cloud outlines
arrows = []
for kind, start, end in ARROWS:
    if len(start) == 2:
        end = to_screen(*end)
        start = end + start
    elif len(end) == 2:
        start = to_screen(*start)
        end = start + end
    else:
        start, end = to_screen(*start), to_screen(*end)
    style = RADIATION[kind]
    path = wave(start, end, style["wavelength"], style["amplitude"])
    arrows.append((style["color"], path))
clouds = [cloud(to_screen(phi, r, 0), CLOUD_WIDTH) for phi, r in CLOUDS]

# Circulation arrows as (colour, screen path)
circulation = []
for color, (phi, r), (half_phi, half_r), start, end in CIRCULATION:
    angles = np.radians(np.linspace(start, end, 100))
    path = np.column_stack(
        project(
            np.radians(phi + half_phi * np.cos(angles)),
            r + half_r * np.sin(angles),
            0,
        )
    )
    circulation.append((color, path))

# One canvas for all steps, just large enough for everything drawn in the
# last one, with equal scales in x and y. Its size and position are the same
# in every image, so that the images can be stacked on the slides.
drawn = [polygon for _, polygon in fills] + clouds
drawn += [path for _, path in arrows + circulation]
points = np.concatenate(drawn)
pad = MARGIN * np.ptp(points[:, 0])
(x0, y0), (x1, y1) = points.min(axis=0) - pad, points.max(axis=0) + pad
figsize = (FIG_WIDTH, FIG_WIDTH * (y1 - y0) / (x1 - x0))

# The names of the regimes as (colour, screen outline of the glyphs). Their
# size in the length units of the drawing follows from the canvas, so they are
# not part of `drawn`; they lie inside the domain.
size = FONTSIZE_LABELS * (x1 - x0) / (FIG_WIDTH * 72)
texts = [
    (color, curved_text(text, position, size)) for text, color, position in LABELS
]

# Drawing order, from the bottom: the atmosphere, the sea surface seen
# through it, the other regimes, the grid, the outlines, the circulation and
# the clouds, the radiation and the names of the regimes
for step, (suffix, _) in enumerate(LAYERS):
    layers = [layer for _, layer in LAYERS[: step + 1]]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Each fill gets an outline in its own colour, which hides the seams
    # between neighbouring polygons
    for regime, style in STYLES.items():
        ax.add_collection(
            PolyCollection(
                [polygon for name, polygon in fills if name == regime],
                facecolors=style["color"],
                edgecolors=style["color"],
                linewidths=0.3,
                zorder=1 if regime == "atmosphere" else 3,
            )
        )
    ax.add_collection(
        PolyCollection(
            sea_surfaces,
            facecolors=SEA_SURFACE_STYLE["color"],
            alpha=SEA_SURFACE_STYLE["alpha"],
            edgecolors="none",
            zorder=2,
        )
    )

    # The grid below the outlines, so that the outlines stay unbroken where
    # the two meet
    for (kind, color), lines_of_kind in polylines.items():
        if kind == "outline" or "grid" in layers:
            ax.add_collection(
                LineCollection(
                    lines_of_kind,
                    colors=color,
                    linewidths=LINEWIDTH,
                    capstyle="round",
                    joinstyle="round",
                    zorder=5 if kind == "outline" else 4,
                )
            )

    if "circulation" in layers:
        for color, path in circulation:
            ax.add_patch(
                FancyArrowPatch(
                    path=MplPath(path),
                    arrowstyle=ArrowStyle("-|>", **ARROW_HEAD),
                    mutation_scale=CIRCULATION_HEAD_SCALE,
                    color=color,
                    linewidth=LINEWIDTH_CIRCULATION,
                    capstyle="round",
                    joinstyle="round",
                    zorder=6,
                )
            )

    if "clouds" in layers:
        for outline_xy in clouds:
            ax.add_patch(
                Polygon(
                    outline_xy,
                    facecolor=COLOR_CLOUDS,
                    edgecolor=COLOR_LINES,
                    linewidth=LINEWIDTH,
                    joinstyle="round",
                    zorder=6,
                )
            )

    if "radiation" in layers:
        for color, path in arrows:
            ax.add_patch(
                FancyArrowPatch(
                    path=MplPath(path),
                    arrowstyle=ArrowStyle("-|>", **ARROW_HEAD),
                    mutation_scale=ARROW_HEAD_SCALE,
                    color=color,
                    linewidth=LINEWIDTH_ARROWS,
                    capstyle="round",
                    joinstyle="round",
                    zorder=7,
                )
            )

    for color, path in texts:
        ax.add_patch(PathPatch(path, facecolor=color, edgecolor="none", zorder=8))

    # A fixed hash salt and no date make the SVG identical on every run, which
    # keeps the version-control diffs clean
    name = f"gcm_setup_step{suffix}"
    plt.rcParams["svg.hashsalt"] = name
    output = Path(__file__).resolve().parent / name
    fig.savefig(output.with_suffix(".png"), dpi=DPI)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})

    # Whole pixels, as matplotlib cuts off fractions
    width, height = (fig.get_size_inches() * DPI).astype(int)
    print(f"{name}: {', '.join(layers)} ({width} x {height} px)")
    plt.close(fig)
