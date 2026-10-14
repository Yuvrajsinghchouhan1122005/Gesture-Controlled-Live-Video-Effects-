"""
ui.py - drawing helpers (OpenCV only): text, panels, progress rings, slider,
edge zones, cursor, hand skeleton, swipe arrows, help overlay.

Everything draws IN PLACE on a BGR canvas. Normalised coordinates (0..1)
are converted to pixels with `px()`.
"""

import time

import cv2
import numpy as np

import config as C

FONT = cv2.FONT_HERSHEY_SIMPLEX


def px(pt, w=C.CANVAS_W, h=C.CANVAS_H):
    """Normalised (x, y) -> integer pixel (x, y)."""
    return int(pt[0] * w), int(pt[1] * h)


# --------------------------------------------------------------------------
# Basic primitives
# --------------------------------------------------------------------------
def blend_rect(img, x1, y1, x2, y2, color=C.COL_BAR, alpha=0.6):
    """Semi-transparent filled rectangle."""
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def text(img, s, org, scale=0.6, color=C.COL_TEXT, thick=1, shadow=True):
    """Text with a dark shadow so it is readable on any background."""
    org = (int(org[0]), int(org[1]))
    if shadow:
        cv2.putText(img, s, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def text_size(s, scale=0.6, thick=1):
    (w, h), _ = cv2.getTextSize(s, FONT, scale, thick)
    return w, h


def text_center(img, s, center, scale=0.8, color=C.COL_TEXT, thick=2):
    w, h = text_size(s, scale, thick)
    text(img, s, (center[0] - w // 2, center[1] + h // 2), scale, color, thick)


def progress_ring(img, center, radius, progress, color=C.COL_RING, thick=4, bg=True):
    """Circular progress indicator used for every hold gesture."""
    center = (int(center[0]), int(center[1]))
    if bg:
        cv2.circle(img, center, radius, (60, 60, 60), thick, cv2.LINE_AA)
    if progress > 0:
        cv2.ellipse(img, center, (radius, radius), -90, 0, 360 * min(progress, 1.0),
                    color, thick, cv2.LINE_AA)


def border(img, color, thick=8):
    h, w = img.shape[:2]
    cv2.rectangle(img, (thick // 2, thick // 2), (w - thick // 2 - 1, h - thick // 2 - 1), color, thick)


def fit_image(img, box_w, box_h, bg=(18, 18, 18)):
    """Resize `img` to fit inside box_w x box_h keeping aspect ratio, centred
    on a background canvas. Returns (canvas, (x0, y0, scale))."""
    canvas = np.empty((box_h, box_w, 3), np.uint8)
    canvas[:] = bg
    if img is None:
        return canvas, (0, 0, 1.0)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    small = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (box_w - nw) // 2, (box_h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = small
    return canvas, (x0, y0, s)


# --------------------------------------------------------------------------
# Gesture feedback
# --------------------------------------------------------------------------
def draw_hand(img, hand, connections, color=(200, 200, 200)):
    """Thin hand skeleton (landmarks are normalised to the full canvas)."""
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in hand.lm]
    for a, b in connections:
        cv2.line(img, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for i, p in enumerate(pts):
        up = i in (8, 12, 16, 20) and hand.fingers[[8, 12, 16, 20].index(i) + 1]
        up = up or (i == 4 and hand.fingers[0])
        cv2.circle(img, p, 4 if up else 2, (80, 255, 80) if up else color, -1, cv2.LINE_AA)


def draw_cursor(img, pos, progress=0.0):
    p = px(pos, img.shape[1], img.shape[0])
    cv2.circle(img, p, 9, C.COL_CURSOR, -1, cv2.LINE_AA)
    cv2.circle(img, p, 10, (0, 0, 0), 1, cv2.LINE_AA)
    if progress > 0:
        progress_ring(img, p, 22, progress, C.COL_RING, 3)


# What the camera sees -> what will happen (shown at the bottom of the screen)
SHAPE_HINTS = {
    "point": "move the pointer - hold still 1.5 s = click",
    "pinch": "move your hand UP / DOWN = stronger / weaker",
    "palm": "tilt your hand = rotate the image",
}


def draw_gesture_hud(img, gf, actions, cx=640, top=638, max_w=900):
    """Bottom-centre box: detected hand shape -> action, with a fill bar while
    the shape is being held. Also a progress ring around the hand."""
    import config as _C
    from gestures import POSE_NAMES
    h, w = img.shape[:2]
    if not gf.hands:
        msg, col = "Show your hand to the camera", C.COL_DIM
    else:
        pose = gf.pose
        name = POSE_NAMES.get(pose, pose)
        if pose in _C.POSE_ACTIONS:
            event = _C.POSE_ACTIONS[pose][0]
            what = actions.get(event)
            secs = _C.POSE_ACTIONS[pose][1]
            msg = f"{name}  ->  hold {secs:g} s = {what}" if what else f"{name}  (not used here)"
        elif pose in SHAPE_HINTS:
            msg = f"{name}  ->  {SHAPE_HINTS[pose]}"
        else:
            msg = f"{name}"
        col = C.COL_TEXT
    scale = 0.55
    while text_size(msg, scale, 1)[0] + 30 > max_w and scale > 0.38:
        scale -= 0.03                                   # shrink long messages to fit
    tw, _ = text_size(msg, scale, 1)
    bw = min(max_w, max(tw + 30, 300))
    x1, y1 = cx - bw // 2, top
    blend_rect(img, x1, y1, x1 + bw, y1 + 44, (10, 10, 10), 0.85)
    text(img, msg, (x1 + 15, y1 + 22), scale, col)
    if gf.hold_progress > 0 and gf.hold_event in actions:
        cv2.rectangle(img, (x1 + 15, y1 + 32), (x1 + bw - 15, y1 + 38), (70, 70, 70), -1)
        cv2.rectangle(img, (x1 + 15, y1 + 32), (x1 + 15 + int((bw - 30) * gf.hold_progress), y1 + 38),
                      C.COL_OK, -1)
        if gf.primary is not None:
            c = px(gf.primary.palm_center, w, h)
            progress_ring(img, c, 46, gf.hold_progress, C.COL_OK, 6)
            label = actions[gf.hold_event].upper()
            lw, _ = text_size(label, 0.7, 2)
            text(img, label, (c[0] - lw // 2, c[1] - 58), 0.7, C.COL_OK, 2)


def draw_slider(img, value, label, value_text, y=None):
    """Horizontal slider shown while pinch controls a parameter."""
    h, w = img.shape[:2]
    y = int(h * 0.86) if y is None else y
    x1, x2 = int(w * 0.25), int(w * 0.75)
    blend_rect(img, x1 - 20, y - 38, x2 + 20, y + 22, C.COL_BAR, 0.7)
    cv2.line(img, (x1, y), (x2, y), (90, 90, 90), 6, cv2.LINE_AA)
    xv = int(x1 + (x2 - x1) * float(np.clip(value, 0, 1)))
    cv2.line(img, (x1, y), (xv, y), C.COL_SLIDER, 6, cv2.LINE_AA)
    cv2.circle(img, (xv, y), 11, C.COL_SLIDER, -1, cv2.LINE_AA)
    cv2.circle(img, (xv, y), 12, (0, 0, 0), 1, cv2.LINE_AA)
    text(img, f"{label}: {value_text}", (x1, y - 16), 0.55, C.COL_TEXT)


def draw_countdown(img, number):
    h, w = img.shape[:2]
    blend_rect(img, w // 2 - 90, h // 2 - 100, w // 2 + 90, h // 2 + 80, (0, 0, 0), 0.4)
    text_center(img, str(number), (w // 2, h // 2 - 10), 4.0, C.COL_ACCENT, 8)


def top_bar(img, fields, height=40):
    """Semi-transparent bar with ' | '-separated fields [(text, colour), ...]."""
    w = img.shape[1]
    blend_rect(img, 0, 0, w, height, C.COL_BAR, 0.65)
    x = 12
    for i, (s, col) in enumerate(fields):
        if i:
            text(img, "|", (x, 26), 0.6, C.COL_DIM)
            x += 18
        text(img, s, (x, 26), 0.55, col)
        x += text_size(s, 0.55)[0] + 14


def bottom_line(img, s, height=30, color=C.COL_DIM):
    h, w = img.shape[:2]
    blend_rect(img, 0, h - height, w, h, C.COL_BAR, 0.65)
    text(img, s, (12, h - 10), 0.5, color)


HELP_LINES = [
    ("1 finger (point)", "move pointer - hold still 1.5 s = click"),
    ("2 fingers, hold 1 s", "NEXT operation / image"),
    ("3 fingers, hold 1 s", "PREVIOUS operation / image"),
    ("Pinch + move up/down", "effect stronger / weaker"),
    ("Open hand + tilt", "rotate the image"),
    ("Thumbs up, hold 1 s", "SAVE result  /  TAKE photo"),
    ("Fist, hold 1.5 s", "go BACK"),
    ("Two index fingers", "two points (distance measures)"),
    ("Keyboard backup", "arrows  +/-  [ ]  S save  Esc back  C compare"),
]


_help_panel = None


def draw_help(img):
    """Help overlay. The panel is rendered once and cached (text drawing is slow)."""
    global _help_panel
    bw, bh = 740, 60 + 34 * len(HELP_LINES)
    if _help_panel is None:
        p = np.full((bh, bw, 3), 15, np.uint8)
        cv2.rectangle(p, (0, 0), (bw - 1, bh - 1), C.COL_ACCENT, 1)
        text(p, "GESTURES", (20, 36), 0.8, C.COL_ACCENT, 2)
        for i, (g, a) in enumerate(HELP_LINES):
            y = 72 + 34 * i
            text(p, g, (20, y), 0.55, C.COL_TEXT)
            text(p, a, (290, y), 0.55, C.COL_DIM)
        _help_panel = p
    h, w = img.shape[:2]
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    roi = img[y0:y0 + bh, x0:x0 + bw]
    cv2.addWeighted(_help_panel, 0.88, roi, 0.12, 0, dst=roi)
