# Abundance of the most common species in the universe. Made for the defense
# talk.

from pathlib import Path

import matplotlib.pyplot as plt

# Fetch Montserrat from Google Fonts and use it for every text element
# (title, species names, abundances). The font file is cached locally by
# pyfonts, so only the first run needs a network connection.
from pyfonts import load_google_font, set_default_font

set_default_font(load_google_font("Montserrat"))

# Font sizes (in points); same values as in
# Chicxulub/Plotting/MatTmpPresentation.py
FONTSIZE_TITLE = 20
FONTSIZE_AXIS_LABELS = 18  # species names and abundances

# Abundance of the species (in per cent, the values add up to 100) and bar
# colour. Species with the same abundance keep the order of this list.
SPECIES = [
    ("Hydrogen", 75, "#729FCF"),
    ("Helium", 23, "#2A6099"),
    ("Oxygen", 1, "#FF0000"),
    ("All others", 1, "#650953"),
]

XLIM = (0, 100)  # leaves room for the label of the longest bar
BAR_HEIGHT = 0.5  # in units of the distance between two bars

# Most abundant species on top
species = sorted(SPECIES, key=lambda entry: entry[1], reverse=True)

fig, ax = plt.subplots(figsize=(5, 4))
# White space around the plot (in inches)
fig.set_layout_engine("constrained", w_pad=0.2, h_pad=0.15)

ax.set_title("species abundance in %", fontsize=FONTSIZE_TITLE)

# One row per species, the first one on top
rows = range(len(species))
ax.set_xlim(XLIM)
ax.set_ylim(len(species) - 0.5, -0.5)

bars = ax.barh(
    rows,
    [abundance for _, abundance, _ in species],
    height=BAR_HEIGHT,
    color=[color for _, _, color in species],
)
ax.bar_label(bars, padding=5, fontsize=FONTSIZE_AXIS_LABELS)

# No frame and no tick marks, only the species names remain next to the bars
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks([])
ax.set_yticks(rows, [name for name, _, _ in species])
ax.tick_params(axis="y", labelsize=FONTSIZE_AXIS_LABELS, length=0, pad=8)

# A fixed hash salt and no date make the SVG identical on every run, which
# keeps the version-control diffs clean
plt.rcParams["svg.hashsalt"] = "species_abundance"

output = Path(__file__).resolve().parent / "species_abundance"
fig.savefig(output.with_suffix(".png"), dpi=300)
fig.savefig(output.with_suffix(".svg"), metadata={"Date": None})
