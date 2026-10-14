"""
Unit 2 - Image Enhancement in the spatial domain.

Point processing   : output pixel depends only on the SAME input pixel, s = T(r).
                     Implemented with a 256-entry look-up table (LUT).
Neighbourhood      : output pixel depends on a small window around the pixel
                     (convolution with a kernel / mask).
Histogram          : equalization spreads the gray levels using the CDF.

(Frequency-domain filters are added in stage 7.)
"""

import cv2
import numpy as np

from registry import register, Result
from operations.common import (point_panel, hist_panel, kernel_panel, mosaic,
                               salt_pepper, text_panel)

L = 256                         # number of gray levels
r = np.arange(L, dtype=np.float32)   # every possible input intensity 0..255


def _apply_lut(gray, lut):
    """Point processing: replace every pixel value r by lut[r]."""
    lut8 = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    out = cv2.LUT(gray, lut8)
    return out, lut8


def _point_result(gray, lut, title, lines=()):
    out, lut8 = _apply_lut(gray, lut)
    return Result(out, panel=point_panel(lut8, gray, out, title), lines=list(lines))


# --------------------------------------------------------------------------
# Point processing
# --------------------------------------------------------------------------
@register("Digital Negative", unit=2,
          info="s = (L-1) - r : reverses intensities, useful to see detail in dark regions")
def negative(img, p, extra):
    """
    1. Convert to gray (L = 256 levels, 0..255)
    2. Build the table s = (L-1) - r = 255 - r
    3. Replace every pixel r by s (look-up table)
    4. Dark areas become bright and bright become dark
    """
    lut = (L - 1) - r
    return _point_result(extra["gray"], lut, "s = 255 - r")


@register("Contrast Stretching", unit=2, param=("stretch", 0, 120, 60, "int"),
          info="Piecewise linear: compress dark (<r1) and bright (>r2) levels, stretch the middle")
def contrast_stretch(img, p, extra):
    """
    1. Choose breakpoints r1 = p and r2 = 255 - p
    2. Below r1: slope < 1 -> dark levels compressed
    3. Between r1 and r2: slope > 1 -> mid-tones stretched
    4. Above r2: slope < 1 -> bright levels compressed
    5. Apply this piecewise-linear table to every pixel
    """
    # Breakpoints (r1, s1) and (r2, s2). Between them the slope is > 1,
    # so mid-gray differences are amplified (more contrast).
    r1, r2 = p, 255 - p
    s1, s2 = 0.25 * p, 255 - 0.25 * p
    lut = np.piecewise(
        r, [r < r1, (r >= r1) & (r <= r2), r > r2],
        [lambda x: x * (s1 / max(r1, 1e-6)),
         lambda x: s1 + (x - r1) * (s2 - s1) / max(r2 - r1, 1e-6),
         lambda x: s2 + (x - r2) * (255 - s2) / max(255 - r2, 1e-6)])
    slope = (s2 - s1) / max(r2 - r1, 1e-6)
    return _point_result(extra["gray"], lut, "contrast stretch",
                         [f"(r1,s1) = ({r1:.0f},{s1:.0f})", f"(r2,s2) = ({r2:.0f},{s2:.0f})",
                          f"middle slope = {slope:.2f}"])


@register("Thresholding", unit=2, param=("T", 0, 255, 128, "int"),
          info="s = 255 if r > T else 0 : the extreme case of contrast stretching (binary image)")
def threshold(img, p, extra):
    """
    1. Convert the image to gray
    2. Choose the threshold T (pinch / mouse wheel)
    3. If pixel value r > T  ->  output 255 (white)
    4. Otherwise             ->  output 0 (black)
    5. Result: a binary (black & white) image
    """
    lut = np.where(r > p, 255, 0)
    res = _point_result(extra["gray"], lut, f"threshold T={p}")
    white = int((res.image > 0).mean() * 100)
    res.lines = [f"T = {p}", f"white pixels: {white}%"]
    return res


@register("Intensity Level Slicing", unit=2, param=("band centre", 0, 255, 150, "int"),
          options=["with background", "without background"],
          info="Highlight a band of gray levels [c-w, c+w]; keep or remove the rest")
def level_slicing(img, p, extra):
    """
    1. Choose a band of gray levels [c - 25, c + 25]
    2. Pixels inside the band -> 255 (highlighted)
    3. Outside: keep the value (with background)
    4. ...or set to 0 (without background)
    """
    w = 25
    lo, hi = max(0, p - w), min(255, p + w)
    band = (r >= lo) & (r <= hi)
    if extra.get("option") == "without background":
        lut = np.where(band, 255, 0)            # band -> white, rest -> black
    else:
        lut = np.where(band, 255, r)            # band -> white, rest unchanged
    return _point_result(extra["gray"], lut, f"slice [{lo},{hi}]",
                         [f"band = [{lo}, {hi}]", extra.get("option", "")])


@register("Log Transformation", unit=2, param=("c", 10, 100, 46, "float"),
          info="s = c log(1 + r) : expands dark levels, compresses bright ones (c=46 maps 255->255)")
def log_transform(img, p, extra):
    """
    1. For every level r compute s = c * log(1 + r)
    2. c = 255 / log(256) = 46 keeps the full 0..255 range
    3. Dark levels are expanded, bright ones compressed
    4. Apply the table to every pixel
    """
    lut = p * np.log1p(r)
    return _point_result(extra["gray"], lut, f"s = {p:.0f} log(1+r)")


@register("Power-law (Gamma)", unit=2, param=("gamma", 0.1, 5.0, 1.0, "float"),
          info="s = 255 (r/255)^gamma : gamma<1 brightens, gamma>1 darkens (display gamma correction)")
def gamma(img, p, extra):
    """
    1. Normalise every level: r / 255 (0..1)
    2. Raise to the power gamma: (r/255)^gamma
    3. Scale back to 0..255
    4. gamma < 1 -> brighter, gamma > 1 -> darker
    """
    lut = 255.0 * (r / 255.0) ** p
    return _point_result(extra["gray"], lut, f"gamma = {p:.2f}")


# --------------------------------------------------------------------------
# Neighbourhood processing (spatial filtering)
# --------------------------------------------------------------------------
SMOOTHING = ["Mean (Box) Filter", "Gaussian Filter", "Median Filter", "Laplacian Sharpening"]


@register("Mean (Box) Filter", unit=2, param=("kernel size", 1, 31, 5, "odd"), compare=SMOOTHING,
          info="Each pixel = average of its k x k neighbours (all weights 1/k^2) -> blurs, removes noise")
def mean_filter(img, p, extra):
    """
    1. Make a k x k kernel with all weights = 1/k^2
    2. Slide the kernel over every pixel
    3. Output = sum(weight x neighbour) = local average
    4. Bigger k -> stronger blur, noise reduced
    """
    k = int(p)
    kernel = np.ones((k, k), np.float32) / (k * k)   # box kernel
    out = cv2.filter2D(img, -1, kernel)               # 2-D convolution
    show = kernel[:5, :5] if k > 5 else kernel
    return Result(out, panel=kernel_panel([show], [f"kernel ({k}x{k}, each = 1/{k * k})"]),
                  lines=[f"kernel {k} x {k}", f"weight = 1/{k * k} = {1 / (k * k):.4f}"])


@register("Gaussian Filter", unit=2, param=("sigma", 0.5, 10.0, 2.0, "float"), compare=SMOOTHING,
          info="Weights follow a 2-D Gaussian: near pixels count more -> smooth blur without box artefacts")
def gaussian_filter(img, p, extra):
    """
    1. Kernel size = 6 x sigma + 1
    2. Weight w(x,y) = exp(-(x^2 + y^2) / 2 sigma^2), normalised
    3. Convolve rows, then columns (separable filter)
    4. Near pixels count more -> smooth, natural blur
    """
    k = int(2 * round(3 * p) + 1)                     # kernel covers +-3 sigma
    g1 = cv2.getGaussianKernel(k, p)                  # 1-D Gaussian (separable)
    out = cv2.sepFilter2D(img, -1, g1, g1)            # rows then columns
    g2 = (g1 @ g1.T)
    c = k // 2
    show = g2[c - 2:c + 3, c - 2:c + 3] if k > 5 else g2
    return Result(out, panel=kernel_panel([show], [f"centre of {k}x{k} Gaussian kernel"]),
                  lines=[f"sigma = {p:.2f}", f"kernel size = {k} (6 sigma + 1)"])


@register("Median Filter", unit=2, param=("kernel size", 3, 15, 3, "odd"), compare=SMOOTHING,
          info="Each pixel = MEDIAN of its neighbours: removes salt & pepper noise but keeps edges")
def median_filter(img, p, extra):
    """
    1. Add 6% salt & pepper noise (left image)
    2. For each pixel take its k x k neighbours
    3. Sort them and pick the MIDDLE value
    4. Extreme 0/255 outliers are ignored -> noise gone
    5. Mean filter shown for comparison (only smears noise)
    """
    # Salt & pepper noise is added first so the median filter's effect is visible.
    noisy = salt_pepper(extra["gray"], 0.06)
    out = cv2.medianBlur(noisy, int(p))
    mean = cv2.blur(noisy, (int(p), int(p)))
    return Result(mosaic([(out, f"median {p}x{p}"), (mean, f"mean {p}x{p} (compare)")], cols=2),
                  input_override=noisy, note="left = input with 6% salt & pepper noise added",
                  lines=["median ignores the extreme 0/255 outliers",
                         "mean only spreads them out"])


LAPLACIAN_K = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)


@register("Laplacian Sharpening", unit=2, param=("strength", 0.0, 3.0, 1.0, "float"), compare=SMOOTHING,
          info="g = f - c * Laplacian(f) : adds back the 2nd derivative so edges/detail become sharper")
def laplacian_sharpen(img, p, extra):
    """
    1. Laplacian kernel [0 1 0; 1 -4 1; 0 1 0]
    2. Convolve -> 2nd derivative (large at edges)
    3. g = f - c x Laplacian(f)
    4. Clip to 0..255 -> edges and details sharper
    """
    f = img.astype(np.float32)
    lap = cv2.filter2D(f, -1, LAPLACIAN_K)            # second derivative (centre -4)
    g = np.clip(f - p * lap, 0, 255).astype(np.uint8)  # centre is negative -> subtract
    return Result(g, panel=kernel_panel([LAPLACIAN_K], ["Laplacian kernel"]),
                  lines=[f"c = {p:.2f}", "g = f - c * (f * kernel)"])


@register("Unsharp Masking / High-boost", unit=2, param=("boost k", 0.0, 5.0, 1.5, "float"),
          info="mask = f - blur(f);  g = f + k * mask.  k = 1 unsharp masking, k > 1 high-boost")
def unsharp(img, p, extra):
    """
    1. Blur the image (Gaussian, sigma = 3)
    2. mask = original - blurred (the fine details)
    3. g = original + k x mask
    4. k = 1: unsharp masking, k > 1: high-boost filtering
    """
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), 3)
    mask = f - blur                                    # the detail that blurring removed
    g = np.clip(f + p * mask, 0, 255).astype(np.uint8)
    mask_show = np.clip(mask * 3 + 128, 0, 255).astype(np.uint8)
    panel = cv2.resize(mask_show, (int(200 * mask.shape[1] / mask.shape[0]), 200))
    panel = np.hstack([panel, text_panel(["unsharp mask = f - blur(f)",
                                          "(gray = 0, shown x3)", f"k = {p:.2f}"],
                                         w=760 - panel.shape[1])])[:, :760]
    return Result(g, panel=panel, lines=["k=1: unsharp masking", "k>1: high-boost filtering"])


# --------------------------------------------------------------------------
# Histogram processing
# --------------------------------------------------------------------------
@register("Histogram Equalization", unit=2,
          info="s_k = (L-1) * CDF(r_k) : maps gray levels through the cumulative histogram -> flat histogram")
def hist_equalize(img, p, extra):
    """
    1. Count pixels at each gray level: histogram n_k
    2. Cumulative sum of the histogram -> CDF
    3. s_k = round((CDF_k - CDF_min) / (N - CDF_min) x 255)
    4. Replace every pixel r_k by s_k
    5. Output histogram is spread over 0..255 (more contrast)
    """
    gray = extra["gray"]
    hist = np.bincount(gray.ravel(), minlength=256)      # n_k for each level
    cdf = np.cumsum(hist)                                 # cumulative count
    cdf_min = cdf[cdf > 0][0]
    # normalised CDF scaled to 0..255 (the standard formula)
    lut = np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255)
    out = lut.clip(0, 255).astype(np.uint8)[gray]
    return Result(out, panel=hist_panel([(gray, "before"), (out, "after (equalized)")]),
                  lines=[f"input range  {gray.min()}-{gray.max()}, std {gray.std():.1f}",
                         f"output range {out.min()}-{out.max()}, std {out.std():.1f}"])
