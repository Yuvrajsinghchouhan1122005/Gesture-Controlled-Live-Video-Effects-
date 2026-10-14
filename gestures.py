"""
gestures.py - hand gesture recognition with MediaPipe Hands.

Every action is a simple HAND SHAPE (no fast movements, no screen corners):

    1 finger  (point)       -> pointer; hold still 1.5 s = click
    2 fingers (peace)       -> hold 1 s = NEXT
    3 fingers               -> hold 1 s = PREVIOUS
    pinch (thumb + index    -> move the hand up / down = parameter up / down
           tips touching)
    open hand (5 fingers)   -> tilt it = rotate the image
    thumbs up               -> hold 1 s = OK (save result / take photo)
    fist                    -> hold 1.5 s = BACK

Pipeline for every frame
------------------------
1. MediaPipe Hands finds up to 2 hands and returns 21 landmarks per hand
   (normalised x, y in 0..1) on a downscaled copy of the mirrored frame.
2. Landmarks are smoothed with a moving average over the last few frames.
3. For each hand we decide which fingers are "up" using distances measured
   from the wrist (rotation invariant) and classify the hand shape.
4. A shape only counts once it has been seen for several frames in a row
   (debouncing), and actions need the shape to be HELD, so one gesture
   produces exactly one action.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import mediapipe as mp

import config as C

# MediaPipe landmark indices
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
PALM_POINTS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

HAND_CONNECTIONS = list(mp.solutions.hands.HAND_CONNECTIONS)

# Names shown on screen for each hand shape
POSE_NAMES = {
    "point": "1 finger", "peace": "2 fingers", "three": "3 fingers", "pinch": "pinch",
    "palm": "open hand", "thumbs_up": "thumbs up", "fist": "fist", "other": "?", "none": "no hand",
}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass
class Hand:
    label: str                 # "Left"/"Right" as reported by MediaPipe
    lm: np.ndarray             # (21, 2) smoothed landmarks, normalised to the frame
    lm_raw: np.ndarray         # (21, 2) unsmoothed landmarks
    size: float                # hand size = |wrist -> middle MCP| (aspect corrected)
    fingers: list              # [thumb, index, middle, ring, pinky] -> True if up
    raw_pose: str              # shape in this frame
    pose: str = "none"         # debounced (stable) shape
    pinch_ratio: float = 1.0   # |thumb tip - index tip| / hand size
    angle: float = 0.0         # tilt of the hand in degrees (0 = fingers straight up, + = clockwise)

    @property
    def index_tip(self):
        return self.lm[INDEX_TIP]

    @property
    def palm_center(self):
        return self.lm[PALM_POINTS].mean(axis=0)


@dataclass
class GestureFrame:
    hands: list = field(default_factory=list)
    primary: Hand = None
    pose: str = "none"               # stable shape of the main hand
    cursor: tuple = None             # (x, y) normalised, while pointing
    dwell: float = 0.0               # 0..1 progress of point-and-hold (click)
    two_points: tuple = None         # ((x1,y1),(x2,y2)) when both index fingers point
    two_dwell: float = 0.0
    pinch_offset: float = None       # while pinching: hand moved up (+) / down (-), 1.0 = full range
    hold_event: str = None           # action that is being held (e.g. "next")
    hold_progress: float = 0.0       # 0..1 progress of that hold
    events: list = field(default_factory=list)   # [(name, data), ...]

    def has(self, name):
        return any(e[0] == name for e in self.events)


# --------------------------------------------------------------------------
# Hold timer
# --------------------------------------------------------------------------
class HoldTimer:
    """Fires once when a condition (optionally at a steady position) has been
    true for `hold` seconds. Re-arms only after the condition goes away or the
    position moves outside `radius`. Short drop-outs (< grace) are tolerated."""

    def __init__(self, hold, radius=None, grace=C.HOLD_GRACE):
        self.hold, self.radius, self.grace = hold, radius, grace
        self.start = None
        self.anchor = None
        self.last_true = 0.0
        self.armed = True

    def reset(self):
        self.start, self.anchor, self.armed = None, None, True

    def update(self, active, now, pos=None, aspect=1.0):
        """Returns (progress 0..1, fired_this_frame)."""
        if not active:
            if self.start is not None and now - self.last_true < self.grace:
                return self._progress(now), False      # tolerate a short drop-out
            self.reset()
            return 0.0, False
        self.last_true = now
        if pos is not None and self.radius is not None:
            p = np.asarray(pos, dtype=float).reshape(-1, 2)
            if self.anchor is None or _max_dist(p, self.anchor, aspect) > self.radius:
                self.anchor = p          # moved: restart the timer and re-arm
                self.start = now
                self.armed = True
        if self.start is None:
            self.start = now
        if not self.armed:
            return 0.0, False
        prog = self._progress(now)
        if prog >= 1.0:
            self.armed = False
            return 1.0, True
        return prog, False

    def _progress(self, now):
        if self.start is None or not self.armed:
            return 0.0
        return min(1.0, (now - self.start) / self.hold)


def _max_dist(p, q, aspect):
    d = (p - q) * np.array([aspect, 1.0])
    return float(np.max(np.linalg.norm(d, axis=1)))


# --------------------------------------------------------------------------
# Gesture engine
# --------------------------------------------------------------------------
class GestureEngine:
    def __init__(self):
        self.hands_model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=C.MAX_HANDS,
            model_complexity=C.MODEL_COMPLEXITY,
            min_detection_confidence=C.MIN_DETECTION_CONF,
            min_tracking_confidence=C.MIN_TRACKING_CONF,
        )
        self.aspect = 16 / 9
        self.history = {}           # label -> deque of raw landmark arrays (smoothing)
        self.pose_hist = {}         # label -> deque of raw poses (debouncing)
        self.stable_pose = {}       # label -> last debounced pose
        self.primary_label = None

        self.dwell = HoldTimer(C.DWELL_TIME, radius=C.DWELL_RADIUS)
        self.two_dwell = HoldTimer(C.DWELL_TIME, radius=C.DWELL_RADIUS)
        self.holds = {pose: (event, HoldTimer(secs)) for pose, (event, secs) in C.POSE_ACTIONS.items()}
        self.pinch_y0 = None        # hand height when the pinch started

    def close(self):
        self.hands_model.close()

    # ---------------------------------------------------------------- detect
    def _detect(self, frame_bgr):
        """Run MediaPipe on a downscaled RGB copy. Returns [(label, lm(21,2))]."""
        h, w = frame_bgr.shape[:2]
        self.aspect = w / h
        scale = C.DETECT_WIDTH / w
        small = cv2.resize(frame_bgr, (C.DETECT_WIDTH, int(h * scale))) if scale < 1 else frame_bgr
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.hands_model.process(rgb)
        out = []
        if res.multi_hand_landmarks:
            for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hd.classification[0].label
                if any(label == o[0] for o in out):      # two hands with same label
                    label += "2"
                arr = np.array([[p.x, p.y] for p in lms.landmark], dtype=np.float32)
                out.append((label, arr))
        return out

    # ------------------------------------------------------------- geometry
    def _d(self, a, b):
        """Euclidean distance between two normalised points, aspect corrected
        (x is stretched by width/height so x and y use the same unit)."""
        return math.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1])

    def _analyse(self, label, lm):
        """Finger states + hand shape for one (smoothed) hand."""
        d = self._d
        size = max(d(lm[WRIST], lm[MIDDLE_MCP]), 1e-6)

        # A finger is "up" when its tip is clearly farther from the wrist than
        # its PIP joint. Using distances (not y-coordinates) makes the test
        # work for a tilted or rotated hand.
        up = [d(lm[WRIST], lm[t]) > C.FINGER_EXTEND_RATIO * d(lm[WRIST], lm[p])
              for t, p in zip(FINGER_TIPS, FINGER_PIPS)]
        # Thumb is "out" when its tip is away from the index knuckle and
        # farther from the pinky knuckle than the thumb's IP joint is.
        thumb = (d(lm[THUMB_TIP], lm[INDEX_MCP]) > C.THUMB_OUT_RATIO * size and
                 d(lm[THUMB_TIP], lm[PINKY_MCP]) > d(lm[THUMB_IP], lm[PINKY_MCP]))
        fingers = [thumb] + up
        i, m, r, p = up

        pinch_ratio = d(lm[THUMB_TIP], lm[INDEX_TIP]) / size
        index_reach = d(lm[WRIST], lm[INDEX_TIP]) / size
        # pinch: thumb tip touches index tip (index not curled into a fist)
        pinch = pinch_ratio < C.PINCH_TOUCH_RATIO and index_reach > C.PINCH_INDEX_REACH
        # thumbs up: four fingers folded and the thumb tip clearly ABOVE the highest
        # knuckle (in a normal fist the thumb lies across the fingers, below it)
        knuckles_y = min(lm[INDEX_MCP][1], lm[MIDDLE_MCP][1])
        thumbs_up = not any(up) and lm[THUMB_TIP][1] < knuckles_y - C.THUMB_UP_RISE * size

        if pinch:
            pose = "pinch"
        elif i and m and r and p:
            pose = "palm"
        elif thumbs_up:
            pose = "thumbs_up"
        elif not (i or m or r or p):
            pose = "fist"
        elif i and not m and not r and not p:
            pose = "point"
        elif i and m and not r and not p:
            pose = "peace"
        elif i and m and r and not p:
            pose = "three"
        else:
            pose = "other"

        # Hand tilt: direction of the wrist -> middle-finger knuckle vector.
        # atan2(dx, -dy) is 0 when the hand points up and grows clockwise.
        vx = (lm[MIDDLE_MCP][0] - lm[WRIST][0]) * self.aspect
        vy = lm[MIDDLE_MCP][1] - lm[WRIST][1]
        angle = math.degrees(math.atan2(vx, -vy))
        return Hand(label=label, lm=lm, lm_raw=None, size=size, fingers=fingers,
                    raw_pose=pose, pinch_ratio=pinch_ratio, angle=angle)

    # --------------------------------------------------------------- process
    def process(self, frame_bgr, now=None):
        """Analyse one MIRRORED BGR frame and return a GestureFrame."""
        now = time.time() if now is None else now
        gf = GestureFrame()
        detections = self._detect(frame_bgr)

        # --- smoothing + per-hand analysis
        seen = set()
        for label, raw in detections:
            seen.add(label)
            hist = self.history.setdefault(label, deque(maxlen=C.SMOOTH_FRAMES))
            hist.append(raw)
            smooth = np.mean(np.stack(hist), axis=0)          # moving average
            hand = self._analyse(label, smooth)
            hand.lm_raw = raw
            ph = self.pose_hist.setdefault(label, deque(maxlen=C.POSE_STABLE_FRAMES))
            ph.append(hand.raw_pose)
            # debounce: the stable shape only changes when the last N raw shapes agree
            if len(ph) == ph.maxlen and len(set(ph)) == 1:
                self.stable_pose[label] = ph[0]
            hand.pose = self.stable_pose.get(label, "none")
            gf.hands.append(hand)
        for label in list(self.history):
            if label not in seen:                              # hand left the frame
                self.history.pop(label, None)
                self.pose_hist.pop(label, None)
                self.stable_pose.pop(label, None)

        # --- main hand: keep the same one while it is visible, else the biggest
        prim = next((h for h in gf.hands if h.label == self.primary_label), None)
        if prim is None and gf.hands:
            prim = max(gf.hands, key=lambda h: h.size)
        self.primary_label = prim.label if prim else None
        gf.primary = prim
        pose = prim.pose if prim else "none"
        gf.pose = pose

        # --- hold a shape -> action (2 fingers = next, 3 = previous, ...)
        for shape, (event, timer) in self.holds.items():
            prog, fired = timer.update(pose == shape, now)
            if prog > gf.hold_progress:
                gf.hold_event, gf.hold_progress = event, prog
            if fired:
                gf.events.append((event, None))

        # --- pinch + move up/down -> parameter (like grabbing a slider knob)
        if prim is not None and pose == "pinch":
            y = float(prim.palm_center[1])
            if self.pinch_y0 is None:
                self.pinch_y0 = y
                gf.events.append(("pinch_start", None))
            gf.pinch_offset = (self.pinch_y0 - y) / C.PINCH_TRAVEL     # up = positive
        else:
            self.pinch_y0 = None

        # --- two index fingers pointing => two points (distance measures)
        pointers = [h for h in gf.hands if h.pose == "point"]
        if len(pointers) == 2:
            pts = sorted([tuple(h.index_tip) for h in pointers])
            gf.two_points = (pts[0], pts[1])
            gf.two_dwell, fired = self.two_dwell.update(True, now, pts, self.aspect)
            if fired:
                gf.events.append(("two_select", gf.two_points))
        else:
            self.two_dwell.update(False, now)

        # --- pointer + hold still = click
        if prim is not None and pose == "point" and gf.two_points is None:
            gf.cursor = tuple(float(v) for v in prim.index_tip)
            gf.dwell, fired = self.dwell.update(True, now, gf.cursor, self.aspect)
            if fired:
                gf.events.append(("select", gf.cursor))
        else:
            self.dwell.update(False, now)
        return gf
