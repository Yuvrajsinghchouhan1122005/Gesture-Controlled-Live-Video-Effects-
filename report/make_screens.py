"""
make_screens.py - render clean screenshots of the app screens for the report
(no camera needed; a synthetic hand illustrates the pointer gesture).
Run from the project folder:  venv\\Scripts\\python.exe report\\make_screens.py
"""

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
sys.argv = ["x"]

import config as C                                 # noqa: E402
import main                                        # noqa: E402
from gestures import GestureFrame, GestureEngine   # noqa: E402

OUT = os.path.join(ROOT, "report", "figs")
os.makedirs(OUT, exist_ok=True)


class FakeWeb:                       # captures what the app would send to the browser
    def __init__(self):
        self.last = None
        self.state = {}
        self.ask_upload = False

    def publish(self, canvas):
        self.last = canvas.copy()

    def get_events(self):
        return []


class NoCam:
    ok = False
    connected = False

    def read(self, timeout=0.1):
        return None

    def release(self):
        pass


main.Camera = lambda *a, **k: NoCam()
web = FakeWeb()
app = main.App(web=web)
app.help_until = 0
app.toast_until = 0
blank = np.full((720, 1280, 3), 40, np.uint8)
eng = GestureEngine()
eng.aspect = 16 / 9


def synthetic_point_hand(tip_xy, size=2.6):
    """Upright hand, only the index finger up, index tip at tip_xy (normalised)."""
    A = 16 / 9
    P = np.zeros((21, 2))
    xs = [-0.03, 0, 0.03, 0.055]; mcpy = [-0.15, -0.15, -0.145, -0.13]
    for f, (x, my) in enumerate(zip(xs, mcpy)):
        b = 5 + 4 * f; P[b] = (x, my)
        if f == 0:
            P[b + 1] = (x, my - 0.05); P[b + 2] = (x, my - 0.08); P[b + 3] = (x, my - 0.11)
        else:
            P[b + 1] = (x, my - 0.04); P[b + 2] = (x, my - 0.02); P[b + 3] = (x * 0.8, my + 0.01)
    P[1] = (-0.04, -0.03); P[2] = (-0.07, -0.06); P[3] = (-0.05, -0.09); P[4] = (-0.01, -0.10)
    P *= size / 6.5
    P[:, 0] /= A
    P += np.array(tip_xy) - P[8]
    return eng._analyse("Right", P.astype(np.float32))


def frame(gf=None, events=()):
    app.fps, app.last = 29.0, time.time() - 1 / 29.0     # show the measured live rate
    app._handle(blank, gf or GestureFrame(), list(events), time.time())
    return web.last


def settle(t=8):
    end = time.time() + t
    frame()
    while time.time() < end and getattr(app.mode, "pending", None) is not None:
        time.sleep(0.05)
        frame()
    frame()


def save(name, img):
    cv2.imwrite(os.path.join(OUT, name), img)


# Home with a pointing hand on the UPLOAD tile (dwell ring 60 %)
tip = (492 / 1280, 340 / 720)
h = synthetic_point_hand(tip)
h.pose = "point"
gf = GestureFrame(hands=[h], primary=h, pose="point", cursor=tip, dwell=0.6)
save("scr_home.png", frame(gf))

# Gallery
app.push(main.GalleryMode)
app.mode.idx = [os.path.basename(p) for p in app.mode.files].index("sample_shapes.png")
save("scr_gallery.png", frame())

# Photo Lab screens
app.push(main.PhotoLabMode, path="images/sample_shapes.png")
lab = app.mode


def goto(unit, name, value=None, option_steps=0):
    lab.unit_i = lab.units.index(unit)
    names = [o.name for o in main.R.ops_in_unit(unit)]
    lab.select_op(names.index(name))
    if value is not None:
        lab.values[name] = value
    lab.option_i[name] = option_steps
    settle()


goto(4, "Erosion", 9)
save("scr_lab_erosion.png", frame())
goto(2, "Histogram Equalization")
save("scr_lab_histeq.png", frame())
goto(6, "Otsu Thresholding")
save("scr_lab_otsu.png", frame())
goto(5, "Canny", 50)
lab.compare = True
settle(15)
save("scr_compare_edges.png", frame())
lab.compare = False
goto(4, "Dilation", 7)
lab.set_rotation(30, toast=False)
settle()
save("scr_lab_rotated.png", frame())
print("screens:", sorted(f for f in os.listdir(OUT) if f.startswith("scr_")))
os._exit(0)
