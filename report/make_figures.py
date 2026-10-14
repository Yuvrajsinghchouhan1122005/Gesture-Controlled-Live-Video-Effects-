"""
make_figures.py - produce every figure / measurement used in the project report.

Runs the SAME registered operations the app uses (registry.run) on the test
image, saves figure PNGs into report/figs and measurements into report/figs/data.json.

Run from the project folder:  venv\\Scripts\\python.exe report\\make_figures.py
"""

import json
import os
import sys
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import registry as R          # noqa: E402
import operations             # noqa: E402,F401
from operations.segmentation import otsu  # noqa: E402

OUT = os.path.join(ROOT, "report", "figs")
os.makedirs(OUT, exist_ok=True)
data = {}

img = cv2.imdecode(np.fromfile("images/sample_shapes.png", np.uint8), cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# low-contrast version (intensities squeezed into 70..170) for enhancement demos
low = cv2.cvtColor((70 + gray.astype(np.float32) * (100 / 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def run(name, image, value=None, option=None):
    op = R.find(name)
    v = op.param_default if value is None else value
    return R.run(op, image, v, {"option": option} if option else {})


def show(im):
    if im.ndim == 2:
        return im, "gray"
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB), None


def grid(items, fname, cols=4, w=3.2):
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * w, rows * w * 0.8))
    for ax in np.array(axes).ravel():
        ax.axis("off")
    for ax, (im, title) in zip(np.array(axes).ravel(), items):
        a, cmap = show(im)
        ax.imshow(a, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)


# 1. input images --------------------------------------------------------------
grid([(img, "Test image (640x480, colour)"), (gray, "Grayscale"),
      (low, "Low-contrast version"), (run("Grayscale", img).image, "Grayscale (op output)")],
     "fig_inputs.png", cols=4)

# 2. point processing ----------------------------------------------------------
grid([(cv2.cvtColor(low, cv2.COLOR_BGR2GRAY), "Input (low contrast)"),
      (run("Digital Negative", low).image, "Negative  s = 255 - r"),
      (run("Contrast Stretching", low, 70).image, "Contrast stretching (k=70)"),
      (run("Histogram Equalization", low).image, "Histogram equalization"),
      (run("Log Transformation", low, 46).image, "Log  s = 46 log(1+r)"),
      (run("Power-law (Gamma)", low, 0.5).image, "Gamma = 0.5"),
      (run("Power-law (Gamma)", low, 2.0).image, "Gamma = 2.0"),
      (run("Thresholding", img, 128).image, "Thresholding T = 128")],
     "fig_point.png", cols=4)

# histogram equalisation: before / after histograms
lg = cv2.cvtColor(low, cv2.COLOR_BGR2GRAY)
eq = run("Histogram Equalization", low).image
fig, ax = plt.subplots(1, 2, figsize=(10, 2.8))
for a, im, t in ((ax[0], lg, "Before equalization"), (ax[1], eq, "After equalization")):
    a.bar(range(256), np.bincount(im.ravel(), minlength=256), width=1.0, color="#3a6ea5")
    a.set_title(t, fontsize=10); a.set_xlim(0, 255)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_hist.png"), dpi=130); plt.close(fig)
data["hist_eq"] = {"in_range": [int(lg.min()), int(lg.max())], "in_std": round(float(lg.std()), 1),
                   "out_range": [int(eq.min()), int(eq.max())], "out_std": round(float(eq.std()), 1)}

# 3. filtering ------------------------------------------------------------------
med = run("Median Filter", img, 5)
noisy = med.input_override
grid([(noisy, "Input + 6% salt & pepper"),
      (cv2.medianBlur(noisy, 5), "Median 5x5"),
      (cv2.blur(noisy, (5, 5)), "Mean 5x5 (same input)"),
      (run("Gaussian Filter", img, 2.0).image, "Gaussian sigma = 2"),
      (run("Mean (Box) Filter", img, 9).image, "Mean (box) 9x9"),
      (run("Laplacian Sharpening", img, 1.0).image, "Laplacian sharpening c = 1"),
      (run("Unsharp Masking / High-boost", img, 2.0).image, "High-boost k = 2"),
      (img, "Original")], "fig_filters.png", cols=4)
clean = gray.astype(np.float32)
psnr = lambda a: round(float(10 * np.log10(255 ** 2 / np.mean((clean - a.astype(np.float32)) ** 2))), 2)
data["denoise_psnr"] = {"noisy": psnr(noisy), "median5": psnr(cv2.medianBlur(noisy, 5)),
                        "mean5": psnr(cv2.blur(noisy, (5, 5))),
                        "gauss": psnr(cv2.GaussianBlur(noisy, (0, 0), 1.0))}

# 4. edges ----------------------------------------------------------------------
cmp_items = []
for n in ("Prewitt", "Sobel", "Laplacian of Gaussian", "Canny"):
    op = R.find(n)
    cmp_items.append((R.run(op, img, op.param_default, {"compare": True}).image, n))
sob = R.run(R.find("Sobel"), img, 100, {})
grid([(gray, "Input")] + cmp_items + [(run("Laplacian", img).image, "Laplacian |response| x2")],
     "fig_edges.png", cols=3)
grid([(sob.image, "Sobel: |Gx|, |Gy|, magnitude, |G| > 100")], "fig_sobel.png", cols=1, w=7)
data["edge_pixels"] = {n: round(float((im > 0).mean() * 100), 2) for im, n in cmp_items}

# 5. morphology -----------------------------------------------------------------
binary = R.otsu_binarize(gray)[0]
items = [(binary, "Binary A (Otsu)")]
morph_stats = {}
for n, v, opt in (("Erosion", 9, "rect"), ("Dilation", 9, "rect"), ("Opening", 11, "ellipse"),
                  ("Closing", 11, "ellipse"), ("Boundary Extraction", 3, None),
                  ("Hit-or-Miss Transform", None, "corners"), ("Thinning (Zhang-Suen)", 60, None)):
    r = run(n, img, v, opt)
    items.append((r.image, f"{n}" + (f" ({opt} {v}x{v})" if v and opt else "")))
    morph_stats[n] = round(float((cv2.cvtColor(r.image, cv2.COLOR_BGR2GRAY) if r.image.ndim == 3
                                  else r.image).astype(bool).mean() * 100), 1)
grid(items, "fig_morph.png", cols=4)
data["white_pct_binary"] = round(float((binary > 0).mean() * 100), 1)
data["morph_white_pct"] = morph_stats
grid([(run("Grayscale Morphology", img, 15, o).image, f"Gray {o} (15x15)")
      for o in ("erosion", "dilation", "top-hat", "black-hat", "gradient")] + [(gray, "Input")],
     "fig_graymorph.png", cols=3)

# 6. segmentation ---------------------------------------------------------------
T, sb = otsu(gray)
T_cv = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)[0]
fig, ax = plt.subplots(1, 2, figsize=(10, 2.8))
ax[0].bar(range(256), np.bincount(gray.ravel(), minlength=256), width=1.0, color="#3a6ea5")
ax[0].axvline(T, color="red"); ax[0].set_title(f"Histogram, Otsu T = {T}", fontsize=10)
ax[1].plot(sb / sb.max(), color="#e0a000"); ax[1].axvline(T, color="red")
ax[1].set_title("Between-class variance  w0 w1 (mu0 - mu1)^2  vs T", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_otsu_curve.png"), dpi=130); plt.close(fig)
grid([(gray, "Input"), (run("Global Thresholding", img, 90).image, "Global T = 90"),
      (run("Otsu Thresholding", img).image, f"Otsu (T = {T})"),
      (run("Adaptive Thresholding", img, 31, "gaussian").image, "Adaptive Gaussian, block 31")],
     "fig_seg.png", cols=4)
otsu_cmp = []
for name, im in (("sample (colour)", gray), ("low-contrast", lg), ("equalized", eq),
                 ("Gaussian-blurred", cv2.GaussianBlur(gray, (0, 0), 3))):
    otsu_cmp.append([name, otsu(im)[0], int(cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)[0])])
data["otsu_compare"] = otsu_cmp

# 7. rotation + operation ---------------------------------------------------------
sys.argv = ["x"]
import main  # noqa: E402
rot = main._fit_max(main.rotate_image(img, 30), 640)
grid([(img, "Input"), (rot, "Rotated 30 deg (hand tilt)"),
      (R.run(R.find("Erosion"), rot, 7, {"option": "rect"}).image, "Erosion on rotated image"),
      (R.run(R.find("Canny"), rot, 50, {}).image, "Canny on rotated image")],
     "fig_rotation.png", cols=4)

# 8. timing of every operation (ms, 640x480, median of 3 runs) -------------------
timing = []
for op in R.OPERATIONS:
    ts = []
    for _ in range(3):
        t = time.perf_counter()
        R.run(op, img, op.param_default, {"option": op.options[0] if op.options else None})
        ts.append((time.perf_counter() - t) * 1000)
    timing.append([op.name, R.UNITS[op.unit], round(float(np.median(ts)), 1)])
data["timing"] = timing
data["n_ops"] = len(R.OPERATIONS)
data["ops_by_unit"] = {R.UNITS[u]: [o.name for o in R.ops_in_unit(u)] for u in R.units_with_ops()}

json.dump(data, open(os.path.join(OUT, "data.json"), "w"), indent=1)
print("figures:", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
print(json.dumps({k: v for k, v in data.items() if k not in ("timing", "ops_by_unit")}, indent=1))
