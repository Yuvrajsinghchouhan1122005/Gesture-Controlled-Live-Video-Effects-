"""
Unit 1 - Image Fundamentals.
(More operations - pixel matrix, RGB split, resolution, distances, formats - in stage 6.)
"""

import numpy as np

from registry import register, Result
from operations.common import hist_panel


@register("Grayscale", unit=1,
          info="Gray = 0.299 R + 0.587 G + 0.114 B  (weights follow the eye's sensitivity: green most)")
def grayscale(img, p, extra):
    """
    1. Split the colour image into B, G, R channels
    2. For every pixel: Gray = 0.299 R + 0.587 G + 0.114 B
    3. Round and clip to 0..255 -> one 8-bit gray image
    4. Show the gray-level histogram
    """
    # Weighted sum of the three channels (ITU-R BT.601 luminance).
    # OpenCV stores pixels as B, G, R, so channel 2 is red.
    b, g, r = [img[:, :, i].astype(np.float32) for i in range(3)]
    gray = (0.299 * r + 0.587 * g + 0.114 * b).round().clip(0, 255).astype(np.uint8)
    return Result(gray, panel=hist_panel([(gray, "gray-level histogram")]),
                  lines=[f"size: {gray.shape[1]} x {gray.shape[0]} pixels",
                         f"min {gray.min()}  max {gray.max()}  mean {gray.mean():.1f}"])
