"""
Unit 6 - Image Segmentation.
(Thresholding methods now; labeling, region growing, split & merge, EM and
K-means follow in stage 5.)
"""

import cv2
import numpy as np

from registry import register, Result
from operations.common import hist_plot, hist_panel, PANEL_W, PANEL_H

COMPARE = ["Global Thresholding", "Otsu Thresholding", "Adaptive Thresholding"]


@register("Global Thresholding", unit=6, param=("T", 0, 255, 128, "int"), compare=COMPARE,
          info="One threshold T for the whole image: pixel -> object if f(x,y) > T, else background")
def global_threshold(img, p, extra):
    """
    1. Convert to gray and draw the histogram
    2. Choose one threshold T for the whole image (red line)
    3. f(x,y) > T -> object (255)
    4. otherwise  -> background (0)
    """
    g = extra["gray"]
    out = np.where(g > p, 255, 0).astype(np.uint8)
    return Result(out, panel=hist_plot(g, PANEL_W, PANEL_H, title="histogram + threshold", vline=p),
                  lines=[f"T = {p}", f"object pixels: {(out > 0).mean() * 100:.1f}%"])


def otsu(gray):
    """Otsu's method written out: try every T and pick the one that maximises
    the BETWEEN-CLASS variance  sigma_B^2 = w0 * w1 * (mu0 - mu1)^2."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / hist.sum()
    levels = np.arange(256)
    w0 = np.cumsum(prob)                       # class 0 probability (levels <= T)
    w1 = 1.0 - w0
    cum_mean = np.cumsum(prob * levels)
    mu_total = cum_mean[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu0 = cum_mean / w0
        mu1 = (mu_total - cum_mean) / w1
        sigma_b = w0 * w1 * (mu0 - mu1) ** 2
    sigma_b = np.nan_to_num(sigma_b)
    T = int(np.argmax(sigma_b))
    return T, sigma_b


@register("Otsu Thresholding", unit=6, compare=COMPARE,
          info="Otsu: automatically choose T that maximises between-class variance w0*w1*(mu0-mu1)^2")
def otsu_threshold(img, p, extra):
    """
    1. Histogram -> probability p(i) of each level
    2. For every T compute w0, w1 (class weights) and mu0, mu1 (means)
    3. Between-class variance = w0 x w1 x (mu0 - mu1)^2
    4. Pick T with the MAXIMUM variance (peak of yellow curve)
    5. Threshold the image with that T
    """
    g = extra["gray"]
    T, sigma_b = otsu(g)
    out = np.where(g > T, 255, 0).astype(np.uint8)
    # panel: histogram with T line + the sigma_B^2(T) curve
    h = hist_plot(g, PANEL_W // 2 - 4, PANEL_H, title="histogram", vline=T)
    curve = np.full((PANEL_H, PANEL_W // 2, 3), 22, np.uint8)
    s = sigma_b / (sigma_b.max() + 1e-9)
    pts = np.array([[8 + i * (PANEL_W // 2 - 16) / 255, PANEL_H - 20 - s[i] * (PANEL_H - 45)]
                    for i in range(256)], np.int32)
    cv2.polylines(curve, [pts], False, (60, 220, 255), 2, cv2.LINE_AA)
    xT = int(8 + T * (PANEL_W // 2 - 16) / 255)
    cv2.line(curve, (xT, 22), (xT, PANEL_H - 20), (60, 60, 255), 2)
    cv2.putText(curve, "between-class variance vs T", (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 220, 120), 1, cv2.LINE_AA)
    panel = np.hstack([h, np.zeros((PANEL_H, 4, 3), np.uint8), curve])
    return Result(out, panel=panel, lines=[f"Otsu threshold T = {T}",
                                           f"(OpenCV gives {cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)[0]:.0f})",
                                           f"object pixels: {(out > 0).mean() * 100:.1f}%"])


@register("Adaptive Thresholding", unit=6, param=("block size", 3, 101, 31, "odd"),
          options=["mean", "gaussian"], compare=COMPARE,
          info="T(x,y) = local mean (or Gaussian-weighted mean) of a block - C : handles uneven lighting")
def adaptive_threshold(img, p, extra):
    """
    1. For every pixel take a (block x block) window
    2. Local threshold = mean (or Gaussian mean) - C
    3. Pixel > local threshold -> white, else black
    4. Works even with uneven lighting / shadows
    """
    g = extra["gray"]
    method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if extra.get("option") == "gaussian" \
        else cv2.ADAPTIVE_THRESH_MEAN_C
    C = 5
    out = cv2.adaptiveThreshold(g, 255, method, cv2.THRESH_BINARY, int(p), C)
    return Result(out, lines=[f"{extra.get('option', 'mean')}, block {p}x{p}, C = {C}",
                              "each pixel gets its own threshold",
                              "from its neighbourhood"])
