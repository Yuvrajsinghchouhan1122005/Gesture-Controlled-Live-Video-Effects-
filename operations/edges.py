"""
Unit 5 - Edge Detection.

An edge is a place where intensity changes sharply.
First-derivative operators (Prewitt, Sobel) measure the gradient:
    Gx = df/dx, Gy = df/dy,   magnitude |G| = sqrt(Gx^2 + Gy^2)
Second-derivative operators (Laplacian, LoG) give ZERO-CROSSINGS at edges.
Canny = Gaussian smoothing + Sobel gradient + non-maximum suppression +
        hysteresis thresholding (two thresholds) -> thin, connected edges.
"""

import cv2
import numpy as np

from registry import register, Result
from operations.common import mosaic, kernel_panel, abs_u8, to_u8

COMPARE = ["Prewitt", "Sobel", "Laplacian of Gaussian", "Canny"]

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)
PREWITT_Y = PREWITT_X.T.copy()
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)   # centre row weighted 2
SOBEL_Y = SOBEL_X.T.copy()
LAPLACE_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
LAPLACE_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)


def _gradient_result(gray, kx, ky, name, T, compare):
    f = gray.astype(np.float32)
    gx = cv2.filter2D(f, -1, kx)          # response to vertical edges
    gy = cv2.filter2D(f, -1, ky)          # response to horizontal edges
    mag = np.sqrt(gx ** 2 + gy ** 2)      # gradient magnitude
    edges = np.where(mag > T, 255, 0).astype(np.uint8)
    if compare:
        return Result(edges)
    img = mosaic([(abs_u8(gx), f"|Gx|"), (abs_u8(gy), f"|Gy|"),
                  (to_u8(mag), "magnitude"), (edges, f"|G| > {T}")], cols=2)
    return Result(img, panel=kernel_panel([kx, ky], [f"{name} Gx", f"{name} Gy"]),
                  lines=[f"|G| = sqrt(Gx^2 + Gy^2)", f"max |G| = {mag.max():.0f}",
                         f"edge pixels (|G|>{T}): {(edges > 0).mean() * 100:.1f}%"])


@register("Prewitt", unit=5, param=("threshold", 10, 300, 80, "int"), compare=COMPARE,
          info="Prewitt: difference of neighbouring columns/rows with equal weights 1 -> gradient Gx, Gy")
def prewitt(img, p, extra):
    """
    1. Convert to gray
    2. Convolve with Gx = [-1 0 1] on each row (vertical edges)
    3. Convolve with Gy = its transpose (horizontal edges)
    4. Magnitude |G| = sqrt(Gx^2 + Gy^2)
    5. Edge where |G| > threshold
    """
    return _gradient_result(extra["gray"], PREWITT_X, PREWITT_Y, "Prewitt", p, extra.get("compare"))


@register("Sobel", unit=5, param=("threshold", 10, 400, 100, "int"), compare=COMPARE,
          info="Sobel: like Prewitt but centre row/column weighted 2 -> some smoothing, less noise")
def sobel(img, p, extra):
    """
    1. Convert to gray
    2. Convolve with Gx = [-1 0 1; -2 0 2; -1 0 1]
    3. Convolve with Gy = its transpose
    4. Magnitude |G| = sqrt(Gx^2 + Gy^2)
    5. Edge where |G| > threshold (centre weight 2 = less noise)
    """
    return _gradient_result(extra["gray"], SOBEL_X, SOBEL_Y, "Sobel", p, extra.get("compare"))


@register("Laplacian", unit=5, options=["4-neighbour", "8-neighbour"], compare=COMPARE,
          info="Laplacian = d2f/dx2 + d2f/dy2 : 2nd derivative, isotropic, very sensitive to noise")
def laplacian(img, p, extra):
    """
    1. Convert to gray
    2. Convolve with the 4- or 8-neighbour Laplacian kernel
    3. 2nd derivative: strong response on both sides of an edge
    4. Show |response| (very sensitive to noise)
    """
    k = LAPLACE_8 if extra.get("option") == "8-neighbour" else LAPLACE_4
    lap = cv2.filter2D(extra["gray"].astype(np.float32), -1, k)
    out = abs_u8(lap * 2)
    return Result(out, panel=kernel_panel([k], [extra.get("option", "4-neighbour") + " kernel"]),
                  lines=["shown: |Laplacian| x 2", "double edges + noise:", "why we smooth first (LoG)"])


def zero_crossings(resp, min_slope):
    """Edge where the response changes sign between neighbours with a big enough jump."""
    zc = np.zeros(resp.shape, bool)
    for a, b in ((resp[:, :-1], resp[:, 1:]), (resp[:-1, :], resp[1:, :])):
        cross = (np.sign(a) != np.sign(b)) & (np.abs(a - b) > min_slope)
        if a.shape[1] != resp.shape[1]:
            zc[:, :-1] |= cross
        else:
            zc[:-1, :] |= cross
    return (zc * 255).astype(np.uint8)


@register("Laplacian of Gaussian", unit=5, param=("sigma", 0.5, 6.0, 2.0, "float"), compare=COMPARE,
          info="LoG (Marr-Hildreth): Gaussian smoothing then Laplacian; edges = zero-crossings")
def log_edges(img, p, extra):
    """
    1. Gaussian blur with sigma (removes noise)
    2. Laplacian of the smoothed image (2nd derivative)
    3. Find zero-crossings (sign changes)
    4. Keep strong crossings -> edges
    """
    f = extra["gray"].astype(np.float32)
    smooth = cv2.GaussianBlur(f, (0, 0), p)          # 1) suppress noise
    resp = cv2.Laplacian(smooth, cv2.CV_32F, ksize=3)  # 2) second derivative
    edges = zero_crossings(resp, 0.15 * np.abs(resp).max())   # 3) zero-crossings
    if extra.get("compare"):
        return Result(edges)
    resp_show = np.clip(resp / (np.abs(resp).max() + 1e-6) * 127 + 128, 0, 255).astype(np.uint8)
    return Result(mosaic([(resp_show, "LoG response (gray = 0)"), (edges, "zero-crossings")], cols=2),
                  lines=[f"sigma = {p:.2f}", "bigger sigma -> fewer,", "smoother edges"])


@register("Canny", unit=5, param=("low threshold", 5, 150, 50, "int"), compare=COMPARE,
          info="Canny: Gaussian -> Sobel gradient -> non-max suppression -> hysteresis (low, high = 3 x low)")
def canny(img, p, extra):
    """
    1. Gaussian blur 5x5 (sigma 1.4)
    2. Sobel gradient: magnitude and direction
    3. Non-maximum suppression -> 1-pixel-thin edges
    4. Double threshold: low (pinch) and high = 3 x low
    5. Hysteresis: weak edges kept only if connected to strong ones
    """
    low, high = int(p), int(min(3 * p, 255 * 3))
    blur = cv2.GaussianBlur(extra["gray"], (5, 5), 1.4)
    edges = cv2.Canny(blur, low, high)
    return Result(edges, lines=[f"low = {low}, high = {high}",
                                "strong (>high) edges kept, weak",
                                "(low..high) kept if connected",
                                f"edge pixels: {(edges > 0).mean() * 100:.1f}%"])
