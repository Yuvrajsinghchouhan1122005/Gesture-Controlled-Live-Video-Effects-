"""
operations/common.py - small helpers shared by the operation modules:
plots drawn with OpenCV (histograms, transfer curves), image mosaics, labels.
"""

import cv2
import numpy as np

PANEL_W, PANEL_H = 760, 200
FONT = cv2.FONT_HERSHEY_SIMPLEX


def to_bgr(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def to_u8(img):
    """Scale any float/int image to 0..255 uint8 for display (min-max)."""
    img = np.asarray(img, dtype=np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-9:
        return np.zeros(img.shape, np.uint8)
    return ((img - mn) * (255.0 / (mx - mn))).astype(np.uint8)


def abs_u8(img):
    """|response| clipped to 0..255 (for derivative filters)."""
    return np.clip(np.abs(img), 0, 255).astype(np.uint8)


def label(img, s, scale=0.6):
    """Write a caption in the top-left corner of an image (returns a BGR copy)."""
    out = to_bgr(img).copy()
    (w, h), _ = cv2.getTextSize(s, FONT, scale, 1)
    cv2.rectangle(out, (0, 0), (w + 12, h + 12), (20, 20, 20), -1)
    cv2.putText(out, s, (6, h + 6), FONT, scale, (255, 220, 120), 1, cv2.LINE_AA)
    return out


def mosaic(items, cols=2, gap=4):
    """Grid of (image, caption) pairs, all resized to the first image's size."""
    h, w = items[0][0].shape[:2]
    # caption size grows with the tile so it stays readable after the mosaic is shrunk to fit
    tiles = [label(cv2.resize(to_bgr(im), (w, h), interpolation=cv2.INTER_NEAREST), cap,
                   scale=max(0.6, w * cols / 900)) for im, cap in items]
    rows = (len(tiles) + cols - 1) // cols
    out = np.full((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 3), 40, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        out[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w] = t
    return out


def _blank(w=PANEL_W, h=PANEL_H):
    return np.full((h, w, 3), 22, np.uint8)


def hist_plot(gray, w, h, color=(220, 220, 220), title="", vline=None, mask=None):
    """Histogram of a gray image as a bar plot (OpenCV only)."""
    img = np.full((h, w, 3), 22, np.uint8)
    hist = cv2.calcHist([gray], [0], mask, [256], [0, 256]).ravel()
    pad_l, pad_b, pad_t = 8, 18, 22
    pw, ph = w - 2 * pad_l, h - pad_b - pad_t
    if hist.max() > 0:
        hist = hist / hist.max()
    xs = pad_l + np.arange(256) * pw / 256.0
    for i in range(256):
        x = int(xs[i])
        y = int(pad_t + ph * (1 - hist[i]))
        cv2.line(img, (x, pad_t + ph), (x, y), color, 1)
    cv2.line(img, (pad_l, pad_t + ph), (pad_l + pw, pad_t + ph), (120, 120, 120), 1)
    for v in (0, 128, 255):
        x = int(pad_l + v * pw / 256.0) - (22 if v == 255 else 0)
        cv2.putText(img, str(v), (x, h - 4), FONT, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
    if vline is not None:
        x = int(pad_l + float(vline) * pw / 256.0)
        cv2.line(img, (x, pad_t), (x, pad_t + ph), (60, 60, 255), 2)
        cv2.putText(img, f"T={float(vline):.0f}", (min(x + 4, w - 60), pad_t + 14), FONT, 0.45,
                    (60, 60, 255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (pad_l, 15), FONT, 0.45, (255, 220, 120), 1, cv2.LINE_AA)
    return img


def hist_panel(items, vline=None):
    """Side-by-side histograms: items = [(gray, title), ...]."""
    n = len(items)
    w = PANEL_W // n
    plots = [hist_plot(g, w - 4, PANEL_H, title=t, vline=vline) for g, t in items]
    return np.hstack([np.pad(p, ((0, 0), (0, 4), (0, 0))) for p in plots])[:, :PANEL_W]


def curve_plot(lut, w=200, h=PANEL_H, title="s = T(r)"):
    """Plot an intensity transfer function s = T(r) given as a 256-entry LUT."""
    img = np.full((h, w, 3), 22, np.uint8)
    pad, top = 22, 22
    size = min(w - 2 * pad, h - top - pad)
    x0, y0 = pad, top + size
    cv2.rectangle(img, (x0, top), (x0 + size, y0), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x0 + size, top), (70, 70, 70), 1)    # identity line
    lut = np.asarray(lut, dtype=np.float32).ravel()
    pts = np.array([[x0 + i * size / 255.0, y0 - np.clip(lut[i], 0, 255) * size / 255.0]
                    for i in range(256)], np.int32)
    cv2.polylines(img, [pts], False, (60, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(img, title, (4, 15), FONT, 0.42, (255, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "r", (x0 + size - 8, y0 + 15), FONT, 0.4, (150, 150, 150), 1)
    cv2.putText(img, "s", (4, top + 12), FONT, 0.4, (150, 150, 150), 1)
    return img


def point_panel(lut, gray_in, gray_out, title="s = T(r)"):
    """Transfer curve + input histogram + output histogram (for point operations)."""
    c = curve_plot(lut, 200, PANEL_H, title)
    rest = PANEL_W - 200
    h1 = hist_plot(gray_in, rest // 2 - 4, PANEL_H, title="input histogram")
    h2 = hist_plot(gray_out, rest // 2 - 4, PANEL_H, title="output histogram")
    gap = np.full((PANEL_H, 4, 3), 0, np.uint8)
    return np.hstack([c, gap, h1, gap, h2])


def text_panel(lines, w=PANEL_W, h=PANEL_H, scale=0.5):
    img = _blank(w, h)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (10, 24 + i * 22), FONT, scale, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def kernel_panel(kernels, titles, w=PANEL_W, h=PANEL_H):
    """Show small convolution kernels as number grids (for the viva)."""
    img = _blank(w, h)
    x = 10
    for k, t in zip(kernels, titles):
        k = np.asarray(k)
        cv2.putText(img, t, (x, 18), FONT, 0.45, (255, 220, 120), 1, cv2.LINE_AA)
        cell = min(34, (h - 40) // k.shape[0])
        for r in range(k.shape[0]):
            for c in range(k.shape[1]):
                v = k[r, c]
                s = f"{v:g}" if abs(v - round(v)) < 1e-6 else f"{v:.2f}"
                p = (x + c * cell, 30 + r * cell)
                cv2.rectangle(img, p, (p[0] + cell, p[1] + cell), (80, 80, 80), 1)
                cv2.putText(img, s, (p[0] + 4, p[1] + cell // 2 + 5), FONT, 0.38,
                            (230, 230, 230), 1, cv2.LINE_AA)
        x += k.shape[1] * cell + 30
    return img


def salt_pepper(gray, amount=0.05, seed=7):
    """Add salt (255) and pepper (0) noise to a fraction `amount` of the pixels."""
    rng = np.random.default_rng(seed)
    noisy = gray.copy()
    r = rng.random(gray.shape)
    noisy[r < amount / 2] = 0
    noisy[r > 1 - amount / 2] = 255
    return noisy


def structuring_element(shape, size):
    shapes = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS}
    return cv2.getStructuringElement(shapes[shape], (int(size), int(size)))
