"""
registry.py - the operation registry.

Every image-processing operation registers itself with ONE decorator line:

    @register("Negative", unit=2, info="s = (L-1) - r : dark becomes bright")
    def negative(img, p, extra):
        '''
        1. Build the table s = 255 - r
        2. Replace every pixel r by s
        '''
        return 255 - img

The numbered lines of the docstring are shown on screen as the algorithm STEPS.

Optional decorator arguments:
    param=("name", min, max, default, type)   type = "int" | "odd" | "float"
    options=["rect", "ellipse", "cross"]      a choice the user can cycle
    needs_point=True                          operation uses a selected pixel
    needs_binary=True                         input is auto-binarized with Otsu first
    compare=["Sobel", "Prewitt", ...]         ops shown together in Compare View

The function receives:
    img   : BGR uint8 image (or the binary 0/255 image if needs_binary)
    p     : current parameter value (already mapped into min..max)
    extra : dict -> "gray", "original", "option", "point", "points", ...
and returns either an image or a `Result`.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

UNITS = {
    1: "Image Fundamentals",
    2: "Image Enhancement",
    3: "Image Transforms",
    4: "Morphology",
    5: "Edge Detection",
    6: "Segmentation",
    7: "Colour Processing",
    8: "Digital Video",
    9: "Creative Effects",
}


@dataclass
class Result:
    image: np.ndarray                    # main processed image (gray or BGR)
    panel: np.ndarray = None             # optional bottom panel (histogram, curve, ...)
    lines: list = field(default_factory=list)   # short text lines (stats, values)
    note: str = ""                       # e.g. "auto-binarized with Otsu"
    input_override: np.ndarray = None    # shown as "original" (e.g. noisy input for median)


@dataclass
class Operation:
    name: str
    unit: int
    func: callable
    info: str = ""
    param_name: str = None
    param_min: float = 0
    param_max: float = 1
    param_default: float = 0
    param_type: str = "float"
    options: list = None
    needs_point: bool = False
    needs_binary: bool = False
    compare: list = None
    steps: list = field(default_factory=list)   # algorithm steps (from the docstring)

    @property
    def has_param(self):
        return self.param_name is not None

    def from_unit(self, t):
        """Map a 0..1 value (pinch distance) into the parameter range."""
        t = float(np.clip(t, 0.0, 1.0))
        v = self.param_min + t * (self.param_max - self.param_min)
        return self.clean(v)

    def to_unit(self, v):
        if self.param_max == self.param_min:
            return 0.0
        return (v - self.param_min) / (self.param_max - self.param_min)

    def clean(self, v):
        v = float(np.clip(v, self.param_min, self.param_max))
        if self.param_type == "int":
            return int(round(v))
        if self.param_type == "odd":
            v = int(round(v))
            if v % 2 == 0:
                v = v + 1 if v + 1 <= self.param_max else v - 1
            return v
        return round(v, 2)

    def step(self, v, direction):
        """One keyboard / mouse-wheel step up or down."""
        if self.param_type == "int":
            d = 1
        elif self.param_type == "odd":
            d = 2
        else:
            d = (self.param_max - self.param_min) / 40.0
        return self.clean(v + direction * d)

    def fmt(self, v):
        if not self.has_param:
            return "-"
        return str(v) if self.param_type in ("int", "odd") else f"{v:.2f}"


OPERATIONS = []


def steps_from_doc(func):
    """The numbered lines ("1. ...") of a function's docstring = algorithm steps."""
    import re
    lines = (func.__doc__ or "").splitlines()
    return [m.group(1).strip() for m in (re.match(r"\s*\d+\.\s+(.*)", s) for s in lines) if m]


def register(name, unit, info="", param=None, options=None, needs_point=False,
             needs_binary=False, compare=None):
    """Decorator: add a function to the registry (one line per operation)."""
    def deco(func):
        op = Operation(name=name, unit=unit, func=func, info=info, options=options,
                       needs_point=needs_point, needs_binary=needs_binary, compare=compare,
                       steps=steps_from_doc(func))
        if param:
            op.param_name, op.param_min, op.param_max, op.param_default = param[:4]
            op.param_type = param[4] if len(param) > 4 else "float"
            op.param_default = op.clean(op.param_default)
        OPERATIONS.append(op)
        return func
    return deco


def ops_in_unit(unit):
    return [op for op in OPERATIONS if op.unit == unit]


def units_with_ops(exclude=(8,)):
    """Units that currently have operations (Unit 8 lives in Live Video mode)."""
    return [u for u in UNITS if u not in exclude and ops_in_unit(u)]


def find(name):
    return next((op for op in OPERATIONS if op.name == name), None)


def otsu_binarize(gray):
    """Auto-binarize with Otsu's threshold (used by needs_binary operations)."""
    t, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, t


def run(op, img_bgr, value, extra):
    """Run one operation and always return a Result."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    extra = dict(extra)
    extra.setdefault("gray", gray)
    extra.setdefault("original", img_bgr)
    note = ""
    inp = img_bgr
    if op.needs_binary:
        binary, t = otsu_binarize(gray)
        extra["binary"] = binary
        inp = binary
        note = f"auto-binarized with Otsu (T = {t:.0f})"
    out = op.func(inp, value, extra)
    res = out if isinstance(out, Result) else Result(image=out)
    if note:
        res.note = (note + "   " + res.note).strip()
    return res
