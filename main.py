"""
main.py - Gesture IVP Lab: app loop, modes and UI.

Modes:  Home  ->  Capture / Gallery / Upload  ->  Photo Lab
Every mode receives the same list of input EVENTS each frame. Events come
from hand gestures (gestures.py) and, as a backup, from keyboard and mouse:

    hand gesture                         keyboard        mouse
    1 finger, hold still 1.5 s (click)   -               left click
    2 fingers, hold 1 s   (next)         right arrow     -
    3 fingers, hold 1 s   (previous)     left arrow      -
    pinch + move up/down  (parameter)    + / -           wheel
    open hand + tilt      (rotate)       [ / ]           -
    thumbs up, hold 1 s   (save / photo) S or Space      -
    fist, hold 1.5 s      (back)         Esc/Backspace   -
    -                     (category)     up/down arrows  click < >
    -                     (compare)      C               click COMPARE

Run:   venv\\Scripts\\python.exe main.py        (Q = quit)
"""

import os
import sys


# --------------------------------------------------------------------------
# Always run from this folder with the project's venv, however we were started
# (e.g. VS Code opened on the parent folder picks a global Python without cv2)
# --------------------------------------------------------------------------
def _ensure_venv():
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)                                    # config.py uses relative folders
    if here not in sys.path:
        sys.path.insert(0, here)
    venv = os.path.join(here, "venv")
    venv_py = os.path.join(venv, "Scripts", "python.exe")
    if not os.path.exists(venv_py):
        venv_py = os.path.join(venv, "bin", "python")
    in_venv = os.path.normcase(os.path.abspath(sys.prefix)) == os.path.normcase(venv)
    if in_venv or not os.path.exists(venv_py) or os.environ.get("IVP_RELAUNCHED"):
        return
    print(f"[setup] restarting with the project venv: {venv_py}")
    import subprocess
    env = dict(os.environ, IVP_RELAUNCHED="1")
    proc = subprocess.Popen([venv_py, os.path.abspath(__file__)] + sys.argv[1:], cwd=here, env=env)
    while True:
        try:
            sys.exit(proc.wait())
        except KeyboardInterrupt:                     # Ctrl+C reaches the child too; wait for it
            pass


if __name__ == "__main__":
    _ensure_venv()

import copy
import math
import re
import shutil
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

import config as C
import registry as R
import operations  # noqa: F401  (registers all operations)
import ui
from gestures import GestureEngine, GestureFrame, HAND_CONNECTIONS, POSE_NAMES


# --------------------------------------------------------------------------
# File helpers (np.fromfile/imdecode also work with non-ASCII Windows paths)
# --------------------------------------------------------------------------
def imread_any(path):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except (OSError, ValueError):
        return None


def imwrite_any(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def timestamp():
    return time.strftime("%Y%m%d_%H%M%S")


def list_images():
    files = []
    for d in (C.IMAGES_DIR, C.CAPTURES_DIR):
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(C.IMAGE_EXTS):
                    files.append(os.path.join(d, f))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)   # newest first
    return files


# --------------------------------------------------------------------------
# Camera (grabs frames in a background thread so the UI never waits on it)
# --------------------------------------------------------------------------
class Camera:
    """Opens the webcam (retrying for a few seconds - another program may still
    be releasing it) and reconnects automatically if it stops delivering frames."""

    def __init__(self, startup_timeout=15.0):      # some webcams need ~8 s to start
        self.cap = None
        self.frame = None
        self.connected = False
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        t0 = time.time()
        while not self.connected and time.time() - t0 < startup_timeout:
            time.sleep(0.1)
        if not self.connected:
            print("[camera] no working camera found - mouse/keyboard still work (will keep retrying)")

    def _open(self):
        # MSMF is much faster than DirectShow on most Windows laptops; fall back if needed.
        for idx in C.CAMERA_INDICES:
            for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW):
                cap = cv2.VideoCapture(idx, api)
                if not cap.isOpened():
                    cap.release()
                    continue
                if C.CAMERA_USE_MJPG:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.CAMERA_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.CAMERA_H)
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[camera] index {idx}, backend {cap.getBackendName()}, "
                          f"{frame.shape[1]}x{frame.shape[0]}")
                    return cap, frame
                cap.release()
        return None, None

    @property
    def ok(self):
        return self.connected

    def _loop(self):
        fails = 0
        while self.running:
            if self.cap is None:
                cap, frame = self._open()
                if cap is None:
                    time.sleep(1.0)            # try again in a second
                    continue
                with self.lock:
                    self.cap, self.frame = cap, frame
                self.connected, fails = True, 0
            ok, frame = self.cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 50:                 # ~0.5 s without frames: reconnect
                    print("[camera] lost camera - reconnecting")
                    self.connected = False
                    self.cap.release()
                    self.cap = None
                time.sleep(0.01)
                continue
            fails = 0
            with self.lock:
                self.frame = frame
            self.new_frame.set()

    def read(self, timeout=0.1):
        """Latest frame, mirrored like a selfie view (None if no camera)."""
        if not self.connected:
            return None
        self.new_frame.wait(timeout)
        self.new_frame.clear()
        with self.lock:
            return None if self.frame is None else cv2.flip(self.frame, 1)

    def release(self):
        self.running = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------
# Hand detection in its own thread (the screen never waits for MediaPipe)
# --------------------------------------------------------------------------
class GestureThread:
    """Runs GestureEngine.process() on the newest camera frame in the background.
    `latest()` returns the most recent GestureFrame plus ALL events produced
    since the previous call, so no gesture action is ever lost."""

    def __init__(self, engine):
        self.engine = engine
        self.cv = threading.Condition()
        self.frame = None
        self.gf = GestureFrame()
        self.events = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, frame):
        with self.cv:
            self.frame = frame
            self.cv.notify()

    def latest(self):
        with self.cv:
            gf = copy.copy(self.gf)
            gf.events, self.events = self.events, []
        return gf

    def _loop(self):
        while self.running:
            with self.cv:
                self.cv.wait_for(lambda: self.frame is not None or not self.running, timeout=0.5)
                frame, self.frame = self.frame, None
            if frame is None:
                continue
            gf = self.engine.process(frame, time.time())
            with self.cv:
                self.gf = gf
                self.events.extend(gf.events)

    def stop(self):
        self.running = False
        with self.cv:
            self.cv.notify()
        self.thread.join(timeout=2)
        self.engine.close()


# --------------------------------------------------------------------------
# Background worker for image processing (keeps the UI smooth)
# --------------------------------------------------------------------------
class Worker:
    """Runs one job at a time; if several are submitted, only the newest runs."""

    def __init__(self):
        self.cv = threading.Condition()
        self.job = None
        self.done = []
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, key, fn):
        with self.cv:
            self.job = (key, fn)
            self.cv.notify()

    def collect(self):
        with self.cv:
            out, self.done = self.done, []
        return out

    def _loop(self):
        while True:
            with self.cv:
                while self.job is None:
                    self.cv.wait()
                key, fn = self.job
                self.job = None
            try:
                res = fn()
            except Exception as e:          # show the error instead of crashing
                res = e
            with self.cv:
                self.done.append((key, res))


# --------------------------------------------------------------------------
# Base class for modes
# --------------------------------------------------------------------------
class Mode:
    title = ""
    actions = {"back": "go back"}     # what each held gesture does in this mode (shown on screen)
    hud = (640, 638, 900)             # gesture indicator position: centre x, top y, max width

    def __init__(self, app):
        self.app = app

    def enter(self, **kw):
        pass

    def resume(self):
        pass

    def update(self, frame, gf, events, now):
        pass

    def draw(self, frame, gf, now):
        return cv2.resize(frame, (C.CANVAS_W, C.CANVAS_H))

    def top_fields(self):
        return []

    def info(self):
        return ""

    def handle_back(self):
        """Return True if the mode used the BACK gesture itself."""
        return False


def _clicks(events):
    """Positions of 'select' (point + hold) and mouse 'click' events."""
    return [e[1] for e in events if e[0] in ("select", "click") and e[1] is not None]


def _inside(pt, rect):
    x, y = pt[0] * C.CANVAS_W, pt[1] * C.CANVAS_H
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


# --------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------
class HomeMode(Mode):
    title = "Home"
    actions = {}
    TILES = [("capture", "CAMERA", "take your photo", "with the webcam"),
             ("upload", "UPLOAD IMAGE", "pick a photo", "from your laptop"),
             ("gallery", "GALLERY", "browse images/", "and captures/"),
             ("live", "LIVE VIDEO", "motion & effects", "(coming soon)")]

    def __init__(self, app):
        super().__init__(app)
        tw, th, gap = 270, 250, 24
        x0 = (C.CANVAS_W - (4 * tw + 3 * gap)) // 2
        y0 = 240
        self.rects = [(x0 + i * (tw + gap), y0, x0 + i * (tw + gap) + tw, y0 + th) for i in range(4)]

    def update(self, frame, gf, events, now):
        for pos in _clicks(events):
            for (key, *_), rect in zip(self.TILES, self.rects):
                if _inside(pos, rect):
                    self.activate(key)
                    return

    def activate(self, key):
        if key == "capture":
            if not self.app.cam.ok:
                self.app.toast("No camera available", C.COL_WARN)
            else:
                self.app.push(CaptureMode)
        elif key == "gallery":
            self.app.push(GalleryMode)
        elif key == "upload":
            self.app.upload()
        else:
            self.app.toast("Live Video is coming soon")

    def draw(self, frame, gf, now):
        canvas = cv2.resize(frame, (C.CANVAS_W, C.CANVAS_H))
        canvas //= 3                                         # dim the camera background
        ui.text_center(canvas, "GESTURE IVP LAB", (C.CANVAS_W // 2, 120), 1.8, C.COL_ACCENT, 3)
        ui.text_center(canvas, "Image & Video Processing with hand gestures", (C.CANVAS_W // 2, 175),
                       0.7, C.COL_TEXT, 1)
        for (key, name, l1, l2), (x1, y1, x2, y2) in zip(self.TILES, self.rects):
            hover = gf.cursor is not None and _inside(gf.cursor, (x1, y1, x2, y2))
            ui.blend_rect(canvas, x1, y1, x2, y2, (60, 60, 60) if hover else (35, 35, 35), 0.85)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), C.COL_ACCENT if hover else (110, 110, 110),
                          3 if hover else 1, cv2.LINE_AA)
            cx = (x1 + x2) // 2
            _tile_icon(canvas, key, (cx, y1 + 80), C.COL_ACCENT if hover else C.COL_TEXT)
            ui.text_center(canvas, name, (cx, y1 + 160), 0.75, C.COL_TEXT, 2)
            ui.text_center(canvas, l1, (cx, y1 + 196), 0.5, C.COL_DIM, 1)
            ui.text_center(canvas, l2, (cx, y1 + 220), 0.5, C.COL_DIM, 1)
        ui.text_center(canvas, "Point at a tile with your index finger and hold still (or click it)",
                       (C.CANVAS_W // 2, 545), 0.6, C.COL_TEXT, 1)
        return canvas

    def info(self):
        return "Show 1 finger, point at a tile and hold still 1.5 s to open it (or click it)."


def _tile_icon(img, key, c, col):
    x, y = c
    if key == "capture":
        cv2.rectangle(img, (x - 40, y - 22), (x + 40, y + 30), col, 3, cv2.LINE_AA)
        cv2.rectangle(img, (x - 15, y - 34), (x + 15, y - 22), col, -1)
        cv2.circle(img, (x, y + 4), 16, col, 3, cv2.LINE_AA)
    elif key == "upload":
        cv2.rectangle(img, (x - 40, y + 5), (x + 40, y + 32), col, 3, cv2.LINE_AA)
        cv2.arrowedLine(img, (x, y + 20), (x, y - 38), col, 4, cv2.LINE_AA, tipLength=0.4)
    elif key == "gallery":
        for i, (dx, dy) in enumerate([(-12, -12), (0, 0), (12, 12)]):
            cv2.rectangle(img, (x - 34 + dx, y - 26 + dy), (x + 22 + dx, y + 16 + dy), col,
                          2 if i < 2 else 3, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x - 42, y - 24), (x + 18, y + 24), col, 3, cv2.LINE_AA)
        pts = np.array([[x + 18, y - 4], [x + 42, y - 20], [x + 42, y + 20], [x + 18, y + 4]])
        cv2.polylines(img, [pts], True, col, 3, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
class CaptureMode(Mode):
    title = "Capture"
    actions = {"ok": "take photo", "back": "go back"}

    def enter(self, **kw):
        self.countdown_end = None
        self.flash_until = 0

    BUTTON = (490, 560, 790, 630)          # big "TAKE PHOTO" button (pixels)

    def update(self, frame, gf, events, now):
        start = any(e[0] == "ok" for e in events) or \
            any(_inside(p, self.BUTTON) for p in _clicks(events))
        if self.countdown_end is None and start:
            self.countdown_end = now + C.CAPTURE_COUNTDOWN
        if self.countdown_end is not None and now >= self.countdown_end:
            self.countdown_end = None
            self.flash_until = now + 0.25
            os.makedirs(C.CAPTURES_DIR, exist_ok=True)
            path = os.path.join(C.CAPTURES_DIR, f"capture_{timestamp()}.png")
            if imwrite_any(path, frame):
                self.app.toast(f"Saved {path}")
                self.app.push(PhotoLabMode, path=path)
            else:
                self.app.toast("Could not save the photo", C.COL_WARN)

    def draw(self, frame, gf, now):
        canvas = cv2.resize(frame, (C.CANVAS_W, C.CANVAS_H))
        if self.countdown_end is not None:
            ui.draw_countdown(canvas, int(np.ceil(self.countdown_end - now)))
        elif now < self.flash_until:
            canvas[:] = 255
        else:
            x1, y1, x2, y2 = self.BUTTON
            hover = gf.cursor is not None and _inside(gf.cursor, self.BUTTON)
            ui.blend_rect(canvas, x1, y1, x2, y2, (40, 140, 40) if hover else (30, 90, 30), 0.85)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (120, 255, 120), 2, cv2.LINE_AA)
            ui.text_center(canvas, "TAKE PHOTO", ((x1 + x2) // 2, (y1 + y2) // 2), 0.9, C.COL_TEXT, 2)
            ui.text_center(canvas, "click it, point at it and hold, or show THUMBS UP for 1 second",
                           (C.CANVAS_W // 2, y1 - 18), 0.55, C.COL_TEXT, 1)
        return canvas

    def info(self):
        return "Thumbs up 1 s (or TAKE PHOTO button) = 3-2-1 photo.   Fist 1.5 s = back."


# --------------------------------------------------------------------------
# Gallery
# --------------------------------------------------------------------------
class GalleryMode(Mode):
    title = "Gallery"
    actions = {"next": "next image", "prev": "previous image", "back": "go back"}
    MAIN = (200, 55, 1080, 545)          # main image area (pixels)
    STRIP_Y, THUMB_W, THUMB_H = 565, 150, 100

    def enter(self, **kw):
        self.files = list_images()
        self.idx = 0
        self.thumbs = {}
        self.big = {}
        self.thumb_rects = []

    def resume(self):
        current = self.files[self.idx] if self.files else None
        self.files = list_images()
        self.idx = self.files.index(current) if current in self.files else 0

    def _load(self, path, cache, w, h):
        if path not in cache:
            img = imread_any(path)
            cache[path] = None if img is None else ui.fit_image(img, w, h)[0]
            if len(cache) > 60:
                cache.pop(next(iter(cache)))
        return cache[path]

    def update(self, frame, gf, events, now):
        if not self.files:
            return
        n = len(self.files)
        for name, data in events:
            if name == "prev":
                self.idx = (self.idx - 1) % n
            elif name == "next":
                self.idx = (self.idx + 1) % n
        for pos in _clicks(events):
            if _inside(pos, self.MAIN):
                # PhotoLabMode shows a message and returns here if the file is unreadable
                self.app.push(PhotoLabMode, path=self.files[self.idx])
                return
            for i, rect in self.thumb_rects:
                if _inside(pos, rect):
                    self.idx = i

    def draw(self, frame, gf, now):
        canvas = np.full((C.CANVAS_H, C.CANVAS_W, 3), 24, np.uint8)
        if not self.files:
            ui.text_center(canvas, "No images yet", (C.CANVAS_W // 2, 300), 1.2, C.COL_ACCENT, 2)
            ui.text_center(canvas, f"Copy photos into the '{C.IMAGES_DIR}' folder, use UPLOAD IMAGE or CAPTURE",
                           (C.CANVAS_W // 2, 360), 0.6, C.COL_TEXT, 1)
            return canvas
        x1, y1, x2, y2 = self.MAIN
        path = self.files[self.idx]
        big = self._load(path, self.big, x2 - x1, y2 - y1)
        if big is None:
            ui.text_center(canvas, "unreadable file", ((x1 + x2) // 2, (y1 + y2) // 2), 0.9, C.COL_WARN, 2)
        else:
            canvas[y1:y2, x1:x2] = big
        hover = gf.cursor is not None and _inside(gf.cursor, self.MAIN)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C.COL_ACCENT if hover else (80, 80, 80), 2)
        ui.text(canvas, f"{self.idx + 1}/{len(self.files)}  {os.path.basename(path)}", (x1, y2 + 0 - 8),
                0.55, C.COL_TEXT)

        # thumbnail strip centred on the current image
        n = len(self.files)
        k = min(n, 7)
        start = max(0, min(self.idx - k // 2, n - k))
        total = k * (self.THUMB_W + 10) - 10
        x = (C.CANVAS_W - total) // 2
        self.thumb_rects = []
        for i in range(start, start + k):
            th = self._load(self.files[i], self.thumbs, self.THUMB_W, self.THUMB_H)
            y = self.STRIP_Y
            if th is not None:
                canvas[y:y + self.THUMB_H, x:x + self.THUMB_W] = th
            col = C.COL_ACCENT if i == self.idx else (70, 70, 70)
            cv2.rectangle(canvas, (x - 2, y - 2), (x + self.THUMB_W + 1, y + self.THUMB_H + 1), col,
                          3 if i == self.idx else 1)
            self.thumb_rects.append((i, (x, y, x + self.THUMB_W, y + self.THUMB_H)))
            x += self.THUMB_W + 10
        return canvas

    def info(self):
        return "2 fingers = next image, 3 fingers = previous.  Point at the photo and hold (or click) to open it."


# --------------------------------------------------------------------------
# Photo Lab
# --------------------------------------------------------------------------
class PhotoLabMode(Mode):
    """Main screen.

    +-----------+-----------------------+-----------------------+
    | OPERATIONS|  ORIGINAL (rotated)   |  PROCESSED            |
    |  list     |                       |                       |
    |-----------+-----------------------+-----------------------+
    | GESTURES  |  STEPS of the         |  histogram / kernel   |
    | webcam    |  algorithm            |  values, option btn   |
    +-----------+-----------------------+-----------------------+
    """
    title = "Photo Lab"
    actions = {"next": "next operation", "prev": "previous operation", "ok": "save result",
               "back": "go back"}
    GUIDE = [("1 finger", "hold still = click"),
             ("2 fingers", "hold = next"),
             ("3 fingers", "hold = previous"),
             ("pinch+up/dn", "stronger / weaker"),
             ("open hand", "tilt = rotate"),
             ("thumbs up", "hold = save"),
             ("fist", "hold = go back")]

    def __init__(self, app):
        super().__init__(app)
        # sidebar
        self.r_unit_prev = (10, 48, 50, 84)
        self.r_unit_next = (250, 48, 290, 84)
        self.op_y0, self.op_h = 92, 25
        self.guide_y0 = 410
        self.r_pip = (36, 572, 264, 688)
        # main area
        self.r_orig = (310, 48, 785, 398)
        self.r_proc = (795, 48, 1270, 398)
        self.r_steps = (310, 406, 785, 688)
        self.r_graph = (795, 406, 1270, 566)
        self.r_stats = (795, 570, 1270, 640)
        self.r_option = (795, 646, 948, 688)
        self.r_compare_btn = (953, 646, 1110, 688)
        self.r_reset = (1115, 646, 1270, 688)
        self.r_compare = (310, 48, 1270, 640)
        self.hud = (547, 642, 470)            # gesture indicator: centre x, top y, max width

    def enter(self, path=None, **kw):
        img = imread_any(path) if path else None
        self.path = path
        if img is None:
            self.app.toast(f"Cannot read image {path}", C.COL_WARN)
            self.app.back()
            return
        self.img = _fit_max(img, C.PROCESS_MAX)                   # process at <= 640 px
        self.img_id = f"{path}:{os.path.getmtime(path) if os.path.exists(path) else 0}"
        self.units = R.units_with_ops()
        self.unit_i = 0
        self.op_i = {u: 0 for u in self.units}
        self.values = {op.name: op.param_default for op in R.OPERATIONS}
        self.option_i = {op.name: 0 for op in R.OPERATIONS}
        self.point = None
        self.points = None
        self.compare = False
        self.rotation = 0                 # degrees, clockwise
        self.rot_engage = None            # hand angle when the palm started rotating
        self.rot_base = 0
        self.rot_shown_until = 0
        self.rotated = {0: self.img}      # cache of rotated inputs
        self.cache = OrderedDict()
        self.pending = None
        self.shown_key = None
        self.base = None
        self.result = None
        self.error = None
        self.worker = app_worker(self.app)

    # ---------------------------------------------------------------- state
    @property
    def unit(self):
        return self.units[self.unit_i]

    @property
    def op(self):
        ops = R.ops_in_unit(self.unit)
        return ops[self.op_i[self.unit] % len(ops)]

    @property
    def input(self):
        """The image the operation works on: the photo rotated by the hand angle."""
        if self.rotation not in self.rotated:
            if len(self.rotated) > 12:
                self.rotated = {0: self.img}
            self.rotated[self.rotation] = _fit_max(rotate_image(self.img, self.rotation), C.PROCESS_MAX)
        return self.rotated[self.rotation]

    def option(self, op):
        return op.options[self.option_i[op.name] % len(op.options)] if op.options else None

    def compare_ops(self):
        op = self.op
        if op.compare:
            names = op.compare
        else:
            ops = R.ops_in_unit(self.unit)
            i = ops.index(op)
            names = [ops[(i + k) % len(ops)].name for k in range(min(4, len(ops)))]
        return [o for o in (R.find(n) for n in names) if o is not None][:4]

    def key(self):
        if self.compare:
            return ("compare", self.img_id, self.rotation,
                    tuple((o.name, self.values[o.name], self.option(o)) for o in self.compare_ops()))
        op = self.op
        return ("single", self.img_id, self.rotation, op.name, self.values[op.name], self.option(op),
                self.point if op.needs_point else None, self.points if op.needs_point else None)

    def op_rects(self):
        ops = R.ops_in_unit(self.unit)
        return [(i, (10, self.op_y0 + i * self.op_h, 290, self.op_y0 + i * self.op_h + self.op_h - 2))
                for i in range(len(ops))]

    # --------------------------------------------------------------- events
    def select_op(self, i):
        self.op_i[self.unit] = i % len(R.ops_in_unit(self.unit))

    def change_unit(self, d):
        self.unit_i = (self.unit_i + d) % len(self.units)

    def update(self, frame, gf, events, now):
        for name, data in events:
            if name in ("prev", "next"):
                self.select_op(self.op_i[self.unit] + (-1 if name == "prev" else 1))
            elif name in ("cat_prev", "cat_next"):
                self.change_unit(-1 if name == "cat_prev" else 1)
            elif name == "pinch_start" and self.op.has_param:
                self.pinch_base = self.op.to_unit(self.values[self.op.name])
            elif name == "param" and self.op.has_param:
                self.values[self.op.name] = self.op.step(self.values[self.op.name], data)
            elif name == "option":
                self.cycle_option()
            elif name == "rotate":
                self.set_rotation(self.rotation + data)
            elif name == "rotate_reset":
                self.set_rotation(0)
            elif name == "compare":
                self.toggle_compare()
            elif name == "ok":
                self.save()
            elif name == "two_select":
                pts = [self.to_image(p) for p in data]
                if all(pts):
                    self.points = tuple(pts)
        for pos in _clicks(events):
            self.click(pos)

        op = self.op
        # pinch -> parameter (absolute mapping of the pinch distance)
        # pinch + move the hand up/down -> parameter (relative to where the pinch started)
        if gf.pinch_offset is not None and op.has_param and not self.compare:
            base = getattr(self, "pinch_base", op.to_unit(self.values[op.name]))
            self.values[op.name] = op.from_unit(base + gf.pinch_offset)
        self._hand_rotation(gf, now)
        self._process()

    def click(self, pos):
        """A point-and-hold selection or a mouse click at `pos`."""
        if _inside(pos, self.r_unit_prev):
            return self.change_unit(-1)
        if _inside(pos, self.r_unit_next):
            return self.change_unit(1)
        for i, rect in self.op_rects():
            if _inside(pos, rect):
                return self.select_op(i)
        if _inside(pos, self.r_compare_btn):
            return self.toggle_compare()
        if not self.compare:
            if self.op.options and _inside(pos, self.r_option):
                return self.cycle_option()
            if _inside(pos, self.r_reset):
                return self.set_rotation(0)
        p = self.to_image(pos)
        if p is not None:
            self.point = p
            if self.op.needs_point:
                self.app.toast(f"Point selected: x={p[0]}, y={p[1]}")

    def _hand_rotation(self, gf, now):
        """Tilt the open palm (Control Mode) to rotate the image.
        Works like a clutch: the angle when the palm appears is the reference,
        tilting beyond a small dead-band rotates the image relative to it."""
        hand = gf.primary
        if not (hand is not None and hand.pose == "palm" and not self.compare):
            self.rot_engage = None
            return
        if self.rot_engage is None:
            self.rot_engage, self.rot_base = hand.angle, self.rotation
            return
        d = hand.angle - self.rot_engage
        if abs(d) <= C.ROT_DEADBAND:
            return
        d -= math.copysign(C.ROT_DEADBAND, d)
        self.set_rotation(self.rot_base + d * C.ROT_GAIN, toast=False)
        self.rot_shown_until = now + 0.6

    def set_rotation(self, angle, toast=True):
        a = int(round(angle / C.ROT_STEP) * C.ROT_STEP)
        a = (a + 180) % 360 - 180                    # keep in -180..175
        if a != self.rotation:
            self.rotation = a
            if toast:
                self.app.toast(f"Rotation: {a} deg")

    def toggle_compare(self):
        self.compare = not self.compare
        self.app.toast("Compare view ON - 4 methods side by side" if self.compare else "Compare view OFF")

    def handle_back(self):
        """Fist / Esc in compare view first closes the compare view."""
        if self.compare:
            self.toggle_compare()
            return True
        return False

    def cycle_option(self):
        op = self.op
        if op.options:
            self.option_i[op.name] += 1
            self.app.toast(f"{op.name}: {self.option(op)}")

    def to_image(self, pos):
        """Canvas position (normalised) -> pixel (x, y) in the working image,
        if it lies on the Original panel (or the Processed one when same size)."""
        if self.compare:
            return None
        x, y = pos[0] * C.CANVAS_W, pos[1] * C.CANVAS_H
        h, w = self.input.shape[:2]
        for rect in (self.r_orig, self.r_proc):
            if not (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]):
                continue
            pw, ph = rect[2] - rect[0], rect[3] - rect[1]
            s = min(pw / w, ph / h)
            x0, y0 = rect[0] + (pw - w * s) / 2, rect[1] + (ph - h * s) / 2
            ix, iy = int((x - x0) / s), int((y - y0) / s)
            if 0 <= ix < w and 0 <= iy < h:
                return ix, iy
        return None

    # ----------------------------------------------------------- processing
    def _process(self):
        for key, res in self.worker.collect():
            if self.pending == key:
                self.pending = None
            self.cache[key] = res
            while len(self.cache) > C.CACHE_SIZE:
                self.cache.popitem(last=False)
        key = self.key()
        if key in self.cache:
            if key != self.shown_key:
                self.shown_key = key
                self._build_base(self.cache[key])
            return
        if self.pending != key:
            self.pending = key
            self.worker.submit(key, self._job())
        if self.base is None or self.base_view != (self.unit, self.op.name, self.compare):
            # another operation was chosen: show its steps at once, result follows
            self.shown_key = None
            self._build_base(None)

    def _job(self):
        img, point, points = self.input, self.point, self.points
        if self.compare:
            ops = [(o, self.values[o.name], self.option(o)) for o in self.compare_ops()]
            return lambda: [(o, R.run(o, img, v, {"option": opt, "compare": True})) for o, v, opt in ops]
        op, v, opt = self.op, self.values[self.op.name], self.option(self.op)
        extra = {"option": opt, "point": point, "points": points}
        return lambda: R.run(op, img, v, extra)

    # -------------------------------------------------------------- drawing
    def _build_base(self, res):
        """Static part of the screen; rebuilt only when the result changes."""
        base = np.full((C.CANVAS_H, C.CANVAS_W, 3), 24, np.uint8)
        if isinstance(res, Exception):
            self.error = f"{type(res).__name__}: {res}"
            res = None
        else:
            self.error = None
        ui.blend_rect(base, 0, 40, 300, C.CANVAS_H - 30, (34, 34, 34), 1.0)    # sidebar
        self._draw_sidebar_static(base)
        if self.compare:
            self._draw_compare(base, res)
        else:
            self._draw_single(base, res)
        self.base = base
        self.result = res
        self.base_view = (self.unit, self.op.name, self.compare)

    def _paste(self, base, img, rect, caption):
        x1, y1, x2, y2 = rect
        tile, _ = ui.fit_image(img, x2 - x1, y2 - y1)
        base[y1:y2, x1:x2] = tile
        cv2.rectangle(base, (x1, y1), (x2 - 1, y2 - 1), (70, 70, 70), 1)
        if caption:
            ui.blend_rect(base, x1, y1, x1 + ui.text_size(caption, 0.5)[0] + 16, y1 + 24, (0, 0, 0), 0.65)
            ui.text(base, caption, (x1 + 8, y1 + 17), 0.5, C.COL_ACCENT)

    def _draw_single(self, base, res):
        op = self.op
        inp = self.input
        rot = f"  (rotated {self.rotation} deg)" if self.rotation else ""
        orig = res.input_override if (res is not None and res.input_override is not None) else inp
        self._paste(base, orig, self.r_orig, ("ORIGINAL" if orig is inp else "INPUT") + rot)
        if res is not None:
            self._paste(base, res.image, self.r_proc, f"PROCESSED: {op.name}")
        else:
            x1, y1, x2, y2 = self.r_proc
            cv2.rectangle(base, (x1, y1), (x2 - 1, y2 - 1), (70, 70, 70), 1)
            msg = self.error or "processing..."
            ui.text_center(base, msg[:60], ((x1 + x2) // 2, (y1 + y2) // 2), 0.55,
                           C.COL_WARN if self.error else C.COL_DIM, 1)

        # STEPS of the algorithm
        x1, y1, x2, y2 = self.r_steps
        ui.blend_rect(base, x1, y1, x2, y2, (32, 32, 32), 1.0)
        ui.text(base, f"STEPS: {op.name}", (x1 + 10, y1 + 24), 0.6, C.COL_ACCENT)
        y = y1 + 52
        for i, step in enumerate(op.steps, 1):
            for j, line in enumerate(_wrap(step, 52)):
                ui.text(base, (f"{i}. " if j == 0 else "    ") + line, (x1 + 12, y), 0.5,
                        C.COL_TEXT if j == 0 else C.COL_TEXT)
                y += 22
            y += 4
            if y > y2 - 80:
                break
        if op.has_param:
            ui.text(base, f"parameter: {op.param_name}  ({op.fmt(op.param_min)} .. {op.fmt(op.param_max)})"
                          "  - pinch / mouse wheel", (x1 + 12, y2 - 56), 0.42, C.COL_SLIDER)

        # graph panel (histogram / kernel / curve)
        x1, y1, x2, y2 = self.r_graph
        if res is not None and res.panel is not None:
            self._paste(base, res.panel, self.r_graph, None)
        else:
            ui.blend_rect(base, x1, y1, x2, y2, (32, 32, 32), 1.0)
            ui.text(base, "(no graph for this operation)", (x1 + 12, y1 + 26), 0.45, (110, 110, 110))

        # values / features of the result
        x1, y1, x2, y2 = self.r_stats
        ui.blend_rect(base, x1, y1, x2, y2, (32, 32, 32), 1.0)
        lines = list(res.lines) if res is not None else []
        if res is not None and res.note:
            lines = [res.note] + lines
        if not lines:
            lines = ["(no values for this operation)"]
        for i, s in enumerate(lines[:3]):
            ui.text(base, str(s)[:62], (x1 + 10, y1 + 20 + i * 22), 0.45, C.COL_TEXT)

    def _draw_compare(self, base, res):
        x1, y1, x2, y2 = self.r_compare
        w2, h2 = (x2 - x1 - 8) // 2, (y2 - y1 - 8) // 2
        cells = [(x1, y1), (x1 + w2 + 8, y1), (x1, y1 + h2 + 8), (x1 + w2 + 8, y1 + h2 + 8)]
        if res is None:
            ui.text_center(base, self.error or "processing compare view...",
                           ((x1 + x2) // 2, (y1 + y2) // 2), 0.7, C.COL_DIM, 1)
            return
        for (op, r), (cx, cy) in zip(res, cells):
            cap = f"{op.name}" + (f"  {op.param_name}={op.fmt(self.values[op.name])}" if op.has_param else "")
            self._paste(base, r.image, (cx, cy, cx + w2, cy + h2), cap)

    def _draw_sidebar_static(self, base):
        """Operation menu + gesture guide. Drawn once into the cached base image
        (text drawing is slow); only hover highlights are drawn every frame."""
        for rect, sym in ((self.r_unit_prev, "<"), (self.r_unit_next, ">")):
            ui.blend_rect(base, *rect, (55, 55, 55), 1.0)
            ui.text_center(base, sym, ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2), 0.8, C.COL_TEXT, 2)
        ui.text_center(base, R.UNITS[self.unit], (150, 66), 0.55, C.COL_ACCENT, 1)
        cur_i = self.op_i[self.unit] % len(R.ops_in_unit(self.unit))
        for (i, rect), op in zip(self.op_rects(), R.ops_in_unit(self.unit)):
            if i == cur_i:
                ui.blend_rect(base, *rect, (110, 80, 20), 1.0)
                cv2.rectangle(base, (rect[0], rect[1]), (rect[0] + 4, rect[3]), C.COL_ACCENT, -1)
            ui.text(base, op.name[:30], (rect[0] + 12, rect[3] - 7), 0.47,
                    C.COL_TEXT if i == cur_i else C.COL_DIM)
        y = self.guide_y0
        ui.text(base, "GESTURES", (12, y), 0.5, C.COL_ACCENT)
        for g, a in self.GUIDE:
            y += 17
            ui.text(base, g, (12, y), 0.4, C.COL_TEXT)
            ui.text(base, a, (110, y), 0.4, C.COL_DIM)

    def _draw_sidebar_live(self, canvas, frame, gf):
        """Per-frame sidebar parts: hover highlight under the cursor + webcam."""
        cur = gf.cursor
        if cur is not None:
            for rect, sym in ((self.r_unit_prev, "<"), (self.r_unit_next, ">")):
                if _inside(cur, rect):
                    ui.blend_rect(canvas, *rect, (90, 70, 30), 1.0)
                    ui.text_center(canvas, sym, ((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                                   0.8, C.COL_TEXT, 2)
            cur_i = self.op_i[self.unit] % len(R.ops_in_unit(self.unit))
            for (i, rect), op in zip(self.op_rects(), R.ops_in_unit(self.unit)):
                if i != cur_i and _inside(cur, rect):
                    ui.blend_rect(canvas, *rect, (60, 60, 60), 1.0)
                    ui.text(canvas, op.name[:30], (rect[0] + 12, rect[3] - 7), 0.47, C.COL_TEXT)
        # webcam picture-in-picture so the user can see their hand
        x1, y1, x2, y2 = self.r_pip
        canvas[y1:y2, x1:x2] = cv2.resize(frame, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(canvas, (x1, y1), (x2 - 1, y2 - 1), (90, 90, 90), 1)

    def draw(self, frame, gf, now):
        canvas = self.base.copy()
        self._draw_sidebar_live(canvas, frame, gf)
        op = self.op
        cur = gf.cursor
        self._button(canvas, cur, self.r_compare_btn, "COMPARE: " + ("ON" if self.compare else "OFF"),
                     "4 methods side by side", active=self.compare)
        if not self.compare:
            if op.options:
                self._button(canvas, cur, self.r_option, f"option: {self.option(op)}", "click = change")
            self._button(canvas, cur, self.r_reset, f"rotate: {self.rotation} deg", "click = reset")
            for p in ([self.point] if self.point and op.needs_point else []):
                self._mark(canvas, p)
        if self.pending is not None:
            ui.text(canvas, "processing...", (self.r_proc[2] - 140, self.r_proc[3] - 10), 0.5, C.COL_ACCENT)
        if now < self.rot_shown_until:
            x1, y1, x2, y2 = self.r_orig
            ui.text_center(canvas, f"ROTATE {self.rotation} deg", ((x1 + x2) // 2, y2 - 30), 1.0,
                           C.COL_SLIDER, 2)
        if gf.pinch_offset is not None and op.has_param and not self.compare:
            ui.draw_slider(canvas, op.to_unit(self.values[op.name]), op.param_name,
                           op.fmt(self.values[op.name]), y=self.r_proc[3] - 40)
        return canvas

    @staticmethod
    def _button(canvas, cur, rect, title, sub, active=False):
        """Small clickable box (mouse click or point + hold with 1 finger)."""
        x1, y1, x2, y2 = rect
        hover = cur is not None and _inside(cur, rect)
        bg = (90, 70, 30) if hover else ((40, 90, 40) if active else (50, 50, 50))
        ui.blend_rect(canvas, x1, y1, x2, y2, bg, 1.0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), C.COL_ACCENT if hover else (100, 100, 100), 1)
        ui.text(canvas, title, (x1 + 8, y1 + 18), 0.43, C.COL_TEXT)
        ui.text(canvas, sub, (x1 + 8, y1 + 35), 0.36, C.COL_DIM)

    def _mark(self, canvas, p):
        h, w = self.input.shape[:2]
        for rect in (self.r_orig, self.r_proc):
            pw, ph = rect[2] - rect[0], rect[3] - rect[1]
            s = min(pw / w, ph / h)
            x = int(rect[0] + (pw - w * s) / 2 + p[0] * s)
            y = int(rect[1] + (ph - h * s) / 2 + p[1] * s)
            cv2.drawMarker(canvas, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 18, 2)

    def save(self):
        res = self.result
        if res is None:
            self.app.toast("Nothing to save yet", C.COL_WARN)
            return
        os.makedirs(C.RESULTS_DIR, exist_ok=True)
        if self.compare:
            x1, y1, x2, y2 = self.r_compare
            img, name = self.base[y1:y2, x1:x2], "compare_" + self.op.name
        else:
            img = res.image if res.image.ndim == 3 else cv2.cvtColor(res.image, cv2.COLOR_GRAY2BGR)
            name = self.op.name
            if self.op.has_param:
                name += f"_{self.op.param_name}{self.op.fmt(self.values[self.op.name])}"
            if self.rotation:
                name += f"_rot{self.rotation}"
        safe = re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_")
        path = os.path.join(C.RESULTS_DIR, f"{safe}_{timestamp()}.png")
        if imwrite_any(path, img):
            self.app.toast(f"Saved {path}", C.COL_OK)
        else:
            self.app.toast("Save failed", C.COL_WARN)

    # ------------------------------------------------------------- top bar
    def top_fields(self):
        op = self.op
        f = [(R.UNITS[self.unit], C.COL_TEXT), (op.name, C.COL_ACCENT)]
        if self.compare:
            f.append(("COMPARE VIEW", C.COL_SLIDER))
        elif op.has_param:
            f.append((f"{op.param_name} = {op.fmt(self.values[op.name])}", C.COL_SLIDER))
        if self.rotation:
            f.append((f"rot {self.rotation}", C.COL_SLIDER))
        return f

    def info(self):
        return self.op.info


def _fit_max(img, m):
    """Shrink so that width and height are <= m (keeps aspect ratio)."""
    h, w = img.shape[:2]
    s = min(1.0, m / w, m / h)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img


def rotate_image(img, angle_cw):
    """Rotate by `angle_cw` degrees clockwise around the centre.
    The canvas is enlarged so no corner is cut off. The new empty corners are
    filled with the photo's typical border colour (median of the edge pixels),
    so they do not disturb histograms / Otsu thresholds like black corners would.
    Rotation matrix: [x'; y'] = [cos -sin; sin cos] [x; y] (OpenCV angle = counter-clockwise)."""
    if angle_cw == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle_cw, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2                    # shift so the result is centred
    M[1, 2] += nh / 2 - h / 2
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    fill = tuple(int(v) for v in np.median(edge, axis=0))
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=fill)


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def app_worker(app):
    if app.worker is None:
        app.worker = Worker()
    return app.worker


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440


class App:
    def __init__(self, web=None):
        self.web = web                     # WebUI -> show in the browser instead of a window
        if web is None:
            cv2.namedWindow(C.WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(C.WINDOW_NAME, C.CANVAS_W, C.CANVAS_H)
            cv2.setMouseCallback(C.WINDOW_NAME, self._on_mouse)
        self._splash("Starting camera... (this can take ~10 seconds)")
        self.cam = Camera()
        self.engine = GestureEngine()
        self.gestures = GestureThread(self.engine)
        self.worker = None
        self.stack = []
        self.mouse_events = []
        self.toast_msg, self.toast_col, self.toast_until = "", C.COL_TEXT, 0
        self.help_until = time.time() + C.HELP_SHOW_TIME
        self.fps, self.last = 0.0, time.time()
        self.running = True
        for d in (C.IMAGES_DIR, C.CAPTURES_DIR, C.RESULTS_DIR, C.RECORDINGS_DIR):
            os.makedirs(d, exist_ok=True)
        self.push(HomeMode)
        if not self.cam.ok:
            self.toast("No camera found - use mouse and keyboard", C.COL_WARN, 6)

    def _splash(self, msg):
        canvas = np.full((C.CANVAS_H, C.CANVAS_W, 3), 24, np.uint8)
        ui.text_center(canvas, "GESTURE IVP LAB", (C.CANVAS_W // 2, 300), 1.6, C.COL_ACCENT, 3)
        ui.text_center(canvas, msg, (C.CANVAS_W // 2, 380), 0.7, C.COL_TEXT, 1)
        self.show(canvas)

    def show(self, canvas):
        if self.web is not None:
            self.web.publish(canvas)
        else:
            cv2.imshow(C.WINDOW_NAME, canvas)
            cv2.waitKey(1)

    # --------------------------------------------------------------- modes
    @property
    def mode(self):
        return self.stack[-1]

    def push(self, cls, **kw):
        m = cls(self)
        self.stack.append(m)
        m.enter(**kw)

    def back(self):
        if len(self.stack) > 1:
            self.stack.pop()
            self.mode.resume()

    def show_help(self):
        self.help_until = time.time() + C.HELP_SHOW_TIME

    def toast(self, msg, col=C.COL_TEXT, secs=C.TOAST_TIME):
        self.toast_msg, self.toast_col, self.toast_until = msg, col, time.time() + secs
        print("[info]", msg)

    def upload(self):
        """Pick an image with the normal Windows file dialog, copy it into images/."""
        if self.web is not None:
            # browser mode: the web page opens its own file picker
            self.web.ask_upload = True
            self.toast("Choose a photo in the browser (or press 'Upload image' at the top)")
            return
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", " ".join("*" + e for e in C.IMAGE_EXTS)), ("All files", "*.*")])
        root.destroy()
        if not path:
            return
        if imread_any(path) is None:
            self.toast("That file is not a readable image", C.COL_WARN)
            return
        dest = os.path.join(C.IMAGES_DIR, os.path.basename(path))
        if os.path.abspath(dest) != os.path.abspath(path):
            if os.path.exists(dest):
                b, e = os.path.splitext(os.path.basename(path))
                dest = os.path.join(C.IMAGES_DIR, f"{b}_{timestamp()}{e}")
            shutil.copy2(path, dest)
        self.push(PhotoLabMode, path=dest)

    # -------------------------------------------------------------- input
    def _on_mouse(self, event, x, y, flags, param):
        pos = (x / C.CANVAS_W, y / C.CANVAS_H)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_events.append(("click", pos))
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.mouse_events.append(("param", 1 if flags > 0 else -1))

    def _key_events(self, k):
        if k == -1:
            return []
        keymap = {KEY_LEFT: "prev", KEY_RIGHT: "next", KEY_UP: "cat_prev",
                  KEY_DOWN: "cat_next", 27: "back", 8: "back", ord("s"): "ok",
                  ord("S"): "ok", 32: "ok", ord("c"): "compare", ord("C"): "compare",
                  ord("o"): "option", ord("O"): "option", ord("h"): "help", ord("H"): "help",
                  ord("r"): "rotate_reset", ord("R"): "rotate_reset"}
        if k in keymap:
            return [(keymap[k], None)]
        if k in (ord("+"), ord("=")):
            return [("param", 1)]
        if k in (ord("-"), ord("_")):
            return [("param", -1)]
        if k in (ord("["), ord("]")):
            return [("rotate", -15 if k == ord("[") else 15)]
        return []

    WEB_KEYS = {"ArrowLeft": KEY_LEFT, "ArrowRight": KEY_RIGHT, "ArrowUp": KEY_UP, "ArrowDown": KEY_DOWN,
                "Escape": 27, "Backspace": 8, " ": 32}

    def _web_events(self):
        """Clicks / keys / wheel / uploads coming from the browser page."""
        events = []
        for kind, data in self.web.get_events():
            if kind == "key":
                code = self.WEB_KEYS.get(data, ord(data) if len(data) == 1 else -1)
                if code not in (ord("q"), ord("Q")):          # closing the tab is enough
                    events += self._key_events(code)
            else:
                events.append((kind, data))
        return events

    # --------------------------------------------------------------- loop
    def run(self):
        blank = np.zeros((C.CANVAS_H, C.CANVAS_W, 3), np.uint8)
        try:
            while self.running:
                frame = self.cam.read()
                now = time.time()
                if frame is None:                  # no camera (yet): mouse/keyboard still work
                    frame = blank
                    gf = GestureFrame()
                    time.sleep(0.02)
                else:
                    self.gestures.submit(frame)    # detected in the background thread
                    gf = self.gestures.latest()

                if self.web is not None:
                    events = list(gf.events) + self._web_events()
                else:
                    k = cv2.waitKeyEx(1)
                    if k in (ord("q"), ord("Q")):
                        break
                    events = list(gf.events) + self._key_events(k) + self.mouse_events
                    self.mouse_events = []
                self._handle(frame, gf, events, now)
                if self.web is None and cv2.getWindowProperty(C.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            self.gestures.stop()
            self.cam.release()
            if self.web is None:
                cv2.destroyAllWindows()

    def _handle(self, frame, gf, events, now):
        mode = self.mode
        # global events
        for name, data in events:
            if name == "back" and "back" in mode.actions:
                if not mode.handle_back():
                    self.back()
                return
            if name == "help":
                self.show_help()
            if name == "open":                 # photo uploaded from the browser
                self.push(PhotoLabMode, path=data)
                return
        mode.update(frame, gf, events, now)
        if self.mode is not mode:          # mode changed during update
            return

        canvas = mode.draw(frame, gf, now)
        self._overlay(canvas, gf, now)
        if self.web is not None:
            self.web.publish(canvas)
            # live gesture state for the guide cards on the web page
            fingers = gf.primary.fingers if gf.primary is not None else []
            self.web.state = {"hand": bool(gf.hands), "pose": gf.pose, "hold": gf.hold_event,
                              "fingers": [bool(f) for f in fingers], "dwell": round(gf.dwell, 2),
                              "progress": round(gf.hold_progress, 2), "actions": mode.actions,
                              "mode": mode.title}
        else:
            cv2.imshow(C.WINDOW_NAME, canvas)

    def _overlay(self, canvas, gf, now):
        mode = self.mode
        for hand in gf.hands:
            ui.draw_hand(canvas, hand, HAND_CONNECTIONS, (150, 150, 150))
        if gf.two_points:
            a, b = ui.px(gf.two_points[0]), ui.px(gf.two_points[1])
            cv2.line(canvas, a, b, C.COL_CURSOR, 1, cv2.LINE_AA)
            for p in (a, b):
                cv2.circle(canvas, p, 8, C.COL_CURSOR, -1, cv2.LINE_AA)
        if gf.cursor:
            ui.draw_cursor(canvas, gf.cursor, gf.dwell)
        ui.draw_gesture_hud(canvas, gf, mode.actions, *mode.hud)

        dt = now - self.last
        self.last = now
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 / dt
        fields = [(mode.title, C.COL_ACCENT)] + mode.top_fields()
        fields += [(f"Hand: {POSE_NAMES.get(gf.pose, gf.pose)}", C.COL_OK if gf.hands else C.COL_DIM),
                   (f"FPS {self.fps:4.1f}", C.COL_DIM)]
        ui.top_bar(canvas, fields)
        ui.bottom_line(canvas, mode.info())

        if now < self.toast_until:
            w, _ = ui.text_size(self.toast_msg, 0.65, 2)
            x = (C.CANVAS_W - w) // 2
            y = C.CANVAS_H - 100
            ui.blend_rect(canvas, x - 16, y - 28, x + w + 16, y + 12, (10, 10, 10), 0.85)
            ui.text(canvas, self.toast_msg, (x, y), 0.65, self.toast_col, 2)
        if now < self.help_until:
            ui.draw_help(canvas)


def already_running():
    """Only one copy of the app may run: a second copy could not get the
    webcam (it would only see a black image). A local port works as a lock."""
    import socket
    global _lock_socket
    _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _lock_socket.bind(("127.0.0.1", 47653))
        return False
    except OSError:
        return True


def main():
    """Default: run in the web browser (prints and opens an http:// link).
    `python main.py --window` uses a normal OpenCV window instead."""
    import sys
    import webbrowser
    use_window = "--window" in sys.argv
    if already_running():
        print("Gesture IVP Lab is already running.")
        if not use_window:
            print(f"Open it here:  http://127.0.0.1:{C.WEB_PORT}")
            webbrowser.open(f"http://127.0.0.1:{C.WEB_PORT}")
        return
    if use_window:
        App().run()
        return
    from web import WebUI
    web = WebUI(C.IMAGES_DIR, port=C.WEB_PORT)
    print("\n" + "=" * 60)
    print(f"  Gesture IVP Lab is running.  Open this link in your browser:")
    print(f"      {web.url}")
    print("  (Stop the app with Ctrl+C in this terminal.)")
    print("=" * 60 + "\n")
    webbrowser.open(web.url)
    try:
        App(web=web).run()
    except KeyboardInterrupt:
        pass
    finally:
        web.close()


if __name__ == "__main__":
    main()
