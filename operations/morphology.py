"""
Unit 4 - Morphological Image Processing.

Binary morphology works on a 0/255 image (auto-binarized with Otsu) using a
small STRUCTURING ELEMENT (SE) B:
    Erosion   A (-) B : pixel stays white only if B fits completely inside A   -> shrinks
    Dilation  A (+) B : pixel becomes white if B touches A anywhere            -> grows
    Opening   (A (-) B) (+) B : removes small white specks, smooths outline
    Closing   (A (+) B) (-) B : fills small holes and gaps
"""

import cv2
import numpy as np

from registry import register, Result
from operations.common import structuring_element, mosaic, kernel_panel, text_panel

SE_SHAPES = ["rect", "ellipse", "cross"]
BASIC = ["Erosion", "Dilation", "Opening", "Closing"]


def _se_panel(se, shape, size, extra_lines=()):
    show = se if size <= 9 else se[:9, :9]
    lines = [f"SE: {shape} {size}x{size}"] + list(extra_lines)
    return kernel_panel([show], [f"structuring element ({shape})" + (" - top-left 9x9" if size > 9 else "")]), lines


def erode(binary, se):
    """Binary erosion written out: a pixel survives only if EVERY SE position
    (shifted copy of the image) is white  -> logical AND (minimum)."""
    k = se.shape[0] // 2
    padded = np.pad(binary, k, constant_values=0)
    out = np.full(binary.shape, 255, np.uint8)
    for dy, dx in zip(*np.nonzero(se)):
        out = np.minimum(out, padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]])
    return out


def dilate(binary, se):
    """Binary dilation: a pixel becomes white if ANY SE position hits white
    -> logical OR (maximum). (SE is reflected; our SEs are symmetric.)"""
    k = se.shape[0] // 2
    padded = np.pad(binary, k, constant_values=0)
    out = np.zeros(binary.shape, np.uint8)
    for dy, dx in zip(*np.nonzero(se)):
        out = np.maximum(out, padded[dy:dy + binary.shape[0], dx:dx + binary.shape[1]])
    return out


def _pct(before, after):
    return f"white pixels: {(before > 0).mean() * 100:.1f}% -> {(after > 0).mean() * 100:.1f}%"


@register("Erosion", unit=4, param=("SE size", 3, 31, 5, "odd"), options=SE_SHAPES,
          needs_binary=True, compare=BASIC,
          info="A erode B = {z | B shifted to z fits inside A} : shrinks objects, removes thin parts")
def erosion(binary, p, extra):
    """
    1. Binarize with Otsu (white = object A)
    2. Build structuring element B (shape + size)
    3. Place the centre of B on every pixel
    4. Keep white only if ALL pixels of B lie on white (minimum)
    5. Objects shrink; thin parts and small specks vanish
    """
    se = structuring_element(extra.get("option", "rect"), p)
    out = erode(binary, se)
    panel, lines = _se_panel(se, extra.get("option", "rect"), p, [_pct(binary, out)])
    return Result(out, panel=panel, lines=lines, input_override=binary)


@register("Dilation", unit=4, param=("SE size", 3, 31, 5, "odd"), options=SE_SHAPES,
          needs_binary=True, compare=BASIC,
          info="A dilate B = {z | B shifted to z overlaps A} : grows objects, bridges small gaps")
def dilation(binary, p, extra):
    """
    1. Binarize with Otsu (white = object A)
    2. Build structuring element B (shape + size)
    3. Place the centre of B on every pixel
    4. Make white if ANY pixel of B touches white (maximum)
    5. Objects grow; small holes and gaps are filled
    """
    se = structuring_element(extra.get("option", "rect"), p)
    out = dilate(binary, se)
    panel, lines = _se_panel(se, extra.get("option", "rect"), p, [_pct(binary, out)])
    return Result(out, panel=panel, lines=lines, input_override=binary)


@register("Opening", unit=4, param=("SE size", 3, 31, 7, "odd"), options=SE_SHAPES,
          needs_binary=True, compare=BASIC,
          info="Opening = erosion then dilation : removes small white noise, keeps object size")
def opening(binary, p, extra):
    """
    1. Binarize with Otsu (object A)
    2. Erode A with B -> small white specks removed
    3. Dilate the result with the same B -> size restored
    4. Opening = (A erode B) dilate B: smooth, noise-free objects
    """
    se = structuring_element(extra.get("option", "rect"), p)
    out = dilate(erode(binary, se), se)
    panel, lines = _se_panel(se, extra.get("option", "rect"), p, ["erode -> dilate"])
    return Result(out, panel=panel, lines=lines, input_override=binary)


@register("Closing", unit=4, param=("SE size", 3, 31, 7, "odd"), options=SE_SHAPES,
          needs_binary=True, compare=BASIC,
          info="Closing = dilation then erosion : fills small black holes and gaps, keeps object size")
def closing(binary, p, extra):
    """
    1. Binarize with Otsu (object A)
    2. Dilate A with B -> small holes and gaps filled
    3. Erode the result with the same B -> size restored
    4. Closing = (A dilate B) erode B: holes closed
    """
    se = structuring_element(extra.get("option", "rect"), p)
    out = erode(dilate(binary, se), se)
    panel, lines = _se_panel(se, extra.get("option", "rect"), p, ["dilate -> erode"])
    return Result(out, panel=panel, lines=lines, input_override=binary)


@register("Boundary Extraction", unit=4, param=("SE size", 3, 15, 3, "odd"),
          needs_binary=True,
          info="beta(A) = A - (A erode B) : the pixels removed by erosion form the object boundary")
def boundary(binary, p, extra):
    """
    1. Binarize with Otsu (object A)
    2. Erode A with B
    3. Boundary = A - (A erode B)
    4. Only the outline pixels remain
    """
    se = structuring_element("rect", p)
    out = cv2.subtract(binary, erode(binary, se))
    return Result(out, input_override=binary,
                  lines=[f"SE rect {p}x{p}", "thicker SE -> thicker boundary"])


# Hit-or-miss kernels: 1 = must be foreground, -1 = must be background, 0 = don't care
HMT_KERNELS = {
    "isolated points": [np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]])],
    "corners": [np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]]),     # top-right corner
                np.array([[-1, -1, 0], [-1, 1, 1], [0, 1, 0]]),     # top-left
                np.array([[0, 1, 0], [1, 1, -1], [0, -1, -1]]),     # bottom-right
                np.array([[0, 1, 0], [-1, 1, 1], [-1, -1, 0]])],    # bottom-left
    "line end points": [np.rot90(np.array([[0, 0, 0], [-1, 1, -1], [-1, -1, -1]]), k) for k in range(4)],
}


@register("Hit-or-Miss Transform", unit=4, options=list(HMT_KERNELS),
          needs_binary=True,
          info="A (*) B = (A erode B1) AND (A^c erode B2) : finds pixels whose neighbourhood matches a pattern")
def hit_or_miss(binary, p, extra):
    """
    1. Binarize with Otsu (object A)
    2. Pattern: B1 = must be white, B2 = must be black
    3. Erode A with B1 (foreground fits)
    4. Erode the complement A^c with B2 (background fits)
    5. AND both -> exact pattern locations (red dots)
    """
    choice = extra.get("option", "isolated points")
    kernels = HMT_KERNELS[choice]
    src = binary
    if choice == "isolated points":
        # add a few isolated white pixels so there is something to find
        rng = np.random.default_rng(3)
        src = binary.copy()
        ys, xs = rng.integers(0, binary.shape[0], 150), rng.integers(0, binary.shape[1], 150)
        src[ys, xs] = 255
    hits = np.zeros_like(src)
    for k in kernels:
        # B1 = foreground part (k == 1), B2 = background part (k == -1)
        fg_fit = erode(src, (k == 1).astype(np.uint8))
        bg_fit = erode(255 - src, (k == -1).astype(np.uint8))
        hits |= cv2.bitwise_and(fg_fit, bg_fit)
    ys, xs = np.nonzero(hits)
    out = cv2.cvtColor(src // 3, cv2.COLOR_GRAY2BGR)
    for y, x in zip(ys[:3000], xs[:3000]):
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), -1)
    return Result(out, input_override=src,
                  panel=kernel_panel(kernels[:4], [f"B{i + 1}" for i in range(len(kernels[:4]))]),
                  lines=[f"pattern: {choice}", f"matches found: {len(ys)}",
                         "1 = foreground, -1 = background, 0 = don't care"],
                  note="red = pixels where the pattern matched")


def zhang_suen(binary, max_iter=100):
    """Zhang-Suen thinning (vectorised with NumPy).
    Each iteration has two sub-steps. A white pixel P1 with neighbours P2..P9
    (clockwise from the top) is deleted if:
      (a) 2 <= B(P1) <= 6      B = number of white neighbours
      (b) A(P1) == 1           A = number of 0->1 transitions in P2,P3,...,P9,P2
      (c) step 1: P2*P4*P6 == 0 and P4*P6*P8 == 0
          step 2: P2*P4*P8 == 0 and P2*P6*P8 == 0
    Repeat until nothing changes -> 1-pixel-wide skeleton."""
    img = (binary > 0).astype(np.uint8)
    img = np.pad(img, 1)
    iters = 0
    for iters in range(1, max_iter + 1):
        changed = False
        for step in (0, 1):
            P = img
            p2, p3, p4 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:]
            p5, p6, p7 = P[2:, 2:], P[2:, 1:-1], P[2:, :-2]
            p8, p9 = P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(nb[:8])
            A = sum(((nb[i] == 0) & (nb[i + 1] == 1)).astype(np.uint8) for i in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            delete = (P[1:-1, 1:-1] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if delete.any():
                changed = True
                img[1:-1, 1:-1][delete] = 0
        if not changed:
            break
    return (img[1:-1, 1:-1] * 255).astype(np.uint8), iters


@register("Thinning (Zhang-Suen)", unit=4, param=("max iterations", 1, 60, 60, "int"),
          needs_binary=True,
          info="Repeatedly delete boundary pixels that are not needed for connectivity -> 1-pixel skeleton")
def thinning(binary, p, extra):
    """
    1. Binarize with Otsu and remove small specks
    2. Find white pixels with 2..6 white neighbours
    3. Delete them only if exactly one 0->1 transition (keeps connectivity)
    4. Two sub-steps per pass: south-east, then north-west
    5. Repeat until nothing changes -> 1-pixel skeleton
    """
    # erase small specks first so the skeleton is readable
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    thin, n = zhang_suen(clean, int(p))
    out = cv2.cvtColor(clean // 4, cv2.COLOR_GRAY2BGR)
    out[thin > 0] = (255, 255, 255)
    return Result(out, input_override=clean,
                  lines=[f"iterations run: {n}", "lower the parameter to watch",
                         "the object get thinner step by step"],
                  note="white = skeleton, dark = original object")


@register("Thickening", unit=4, param=("iterations", 1, 30, 5, "int"),
          needs_binary=True,
          info="Thickening = complement of thinning the background : objects grow but never merge")
def thickening(binary, p, extra):
    """
    1. Binarize with Otsu (object A)
    2. Take the complement A^c (the background)
    3. Thin the background (Zhang-Suen)
    4. Complement again -> objects thickened
    5. Objects grow but never merge
    """
    # Thicken A  ==  thin the BACKGROUND (A^c) then take the complement.
    bg = 255 - binary
    thin_bg, n = zhang_suen(bg, int(p))
    out = 255 - thin_bg
    return Result(out, input_override=binary,
                  lines=[f"iterations: {n}", "computed as (thin(A^c))^c"])


GRAY_OPS = ["erosion", "dilation", "top-hat", "black-hat", "gradient"]


@register("Grayscale Morphology", unit=4, param=("SE size", 3, 31, 9, "odd"), options=GRAY_OPS,
          info="Gray erosion = local MIN, dilation = local MAX; top-hat = f - open(f), black-hat = close(f) - f")
def gray_morph(img, p, extra):
    """
    1. Gray image, elliptical structuring element
    2. Erosion = local MIN, Dilation = local MAX
    3. Top-hat = f - opening(f) (small bright details)
    4. Black-hat = closing(f) - f (small dark details)
    5. Gradient = dilation - erosion (outlines)
    """
    g = extra["gray"]
    se = structuring_element("ellipse", p)
    choice = extra.get("option", "erosion")
    ops = {"erosion": lambda: cv2.erode(g, se),
           "dilation": lambda: cv2.dilate(g, se),
           "top-hat": lambda: cv2.morphologyEx(g, cv2.MORPH_TOPHAT, se),
           "black-hat": lambda: cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, se),
           "gradient": lambda: cv2.morphologyEx(g, cv2.MORPH_GRADIENT, se)}
    out = ops[choice]()
    if choice in ("top-hat", "black-hat"):
        out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    explain = {"erosion": "local minimum -> darker, bright details shrink",
               "dilation": "local maximum -> brighter, dark details shrink",
               "top-hat": "f - opening(f): small BRIGHT details (stretched)",
               "black-hat": "closing(f) - f: small DARK details (stretched)",
               "gradient": "dilation - erosion: outlines of objects"}[choice]
    return Result(out, lines=[f"{choice}, ellipse {p}x{p}", explain])
