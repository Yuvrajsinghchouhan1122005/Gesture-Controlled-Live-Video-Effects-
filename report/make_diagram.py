"""make_diagram.py - processing pipeline flowchart for the report (matplotlib)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 110); ax.set_ylim(0, 62); ax.axis("off")


def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=color, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, wrap=True)


def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.3))
    if label:
        ax.text((x1 + x2) / 2 + 0.8, (y1 + y2) / 2, label, fontsize=7.5, color="#555555")


G, B, Y, P = "#dff0d8", "#d9e8f6", "#fcf3cf", "#f3e0f0"
# gesture path (left)
box(2, 52, 24, 7, "Webcam frame\n1280x720 @ 30 fps", G)
box(2, 41, 24, 7, "Mirror + downscale\n(640 px), BGR -> RGB", G)
box(2, 30, 24, 7, "MediaPipe Hands\n21 landmarks / hand", G)
box(2, 19, 24, 7, "Moving-average smoothing\n+ finger-state tests", G)
box(2, 8, 24, 7, "Hand-shape classifier +\ndebounce + hold timers", G)
for y in (52, 41, 30, 19):
    arrow(14, y, 14, y - 4)
box(31, 6, 24, 11, "Gesture events\nclick, next, prev, ok, back\n+ pinch offset, hand angle", Y)
arrow(26, 11.5, 31, 11.5)

# image path (middle)
box(42, 52, 26, 7, "Input photo\n(webcam capture / upload)", B)
box(42, 41, 26, 7, "Resize to <= 640 px\n(keep aspect ratio)", B)
box(42, 30, 26, 7, "Rotate by hand tilt\n(affine warp, border fill)", B)
box(42, 19, 26, 7, "Selected operation\n(registry: Units 1, 2, 4, 5, 6)", B)
for y in (52, 41, 30):
    arrow(55, y, 55, y - 4)
arrow(38, 17, 41.6, 22.5)
ax.text(28.5, 18.6, "next / prev,\npinch", fontsize=7.5, color="#555555", ha="left")
arrow(34, 17, 41.6, 33.5)
ax.text(28.5, 29.0, "hand angle\n(open hand)", fontsize=7.5, color="#555555", ha="left")

# output (right)
box(76, 41, 30, 18, "Display (web page / window)\nOriginal | Processed\nSTEPS of the algorithm\nhistogram, kernel, curve, values", P)
box(76, 19, 30, 14, "Background worker thread\n+ result cache (LRU, 40 entries)", P)
box(76, 4, 30, 9, "Save result -> results/*.png", P)
arrow(68, 22.5, 76, 26)
arrow(91, 33, 91, 41)
arrow(55, 6, 76, 8.5, "thumbs up")
ax.set_title("Gesture IVP Lab - processing pipeline", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig_pipeline.png"), dpi=140)
print("saved fig_pipeline.png")
