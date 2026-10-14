"""
config.py - every tunable number of Gesture IVP Lab lives here.

Change values here instead of hunting through the code.
All gesture positions/distances are NORMALISED (0..1 of the frame width/height)
unless stated otherwise, so they do not depend on the camera resolution.
"""

# --------------------------------------------------------------------------
# Window / camera
# --------------------------------------------------------------------------
WINDOW_NAME = "Gesture IVP Lab"
CANVAS_W, CANVAS_H = 1280, 720        # size of the drawn UI canvas
CAMERA_INDICES = [0, 1, 2]            # webcam indices to try, in order
CAMERA_W, CAMERA_H = 1280, 720        # requested camera resolution
CAMERA_USE_MJPG = True                # MJPG gives higher FPS on most webcams

# --------------------------------------------------------------------------
# MediaPipe Hands
# --------------------------------------------------------------------------
DETECT_WIDTH = 640                    # hand detection runs on a downscaled frame (speed)
MAX_HANDS = 2
MODEL_COMPLEXITY = 1                  # 0 = fast, 1 = more accurate (runs in its own thread)
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF = 0.5

# --------------------------------------------------------------------------
# Gesture recognition
# --------------------------------------------------------------------------
SMOOTH_FRAMES = 5                     # moving-average window for landmarks
POSE_STABLE_FRAMES = 3                # a hand shape must repeat this many frames to count

# finger state thresholds (ratios relative to distances inside the hand)
FINGER_EXTEND_RATIO = 1.15            # |wrist->tip| > ratio * |wrist->PIP|  => finger up
THUMB_OUT_RATIO = 0.60                # |thumb tip -> index MCP| > ratio * hand size => thumb out
THUMB_UP_RISE = 0.30                  # thumbs up: thumb tip this much (x hand size) above the knuckles

# pinch = thumb tip touching index tip (distance / hand size)
PINCH_TOUCH_RATIO = 0.35
PINCH_INDEX_REACH = 0.95              # index tip must be this far from the wrist (not a fist)
PINCH_TRAVEL = 0.35                   # moving the pinched hand this much (fraction of frame
                                      # height) up/down = the full parameter range

# Hand shapes that trigger an action when HELD STILL:  pose -> (event, seconds)
POSE_ACTIONS = {
    "peace":     ("next", 1.0),       # 2 fingers  -> next operation / image
    "three":     ("prev", 1.0),       # 3 fingers  -> previous operation / image
    "thumbs_up": ("ok", 1.0),         # thumbs up  -> save result / take photo
    "fist":      ("back", 1.5),       # fist       -> go back
}

# point / dwell selection (pointer = tip of the index finger)
DWELL_TIME = 1.5                      # hold the pointing finger still this long => click
DWELL_RADIUS = 0.035                  # "still" = cursor stays inside this radius
HOLD_GRACE = 0.25                     # brief detection drop-outs (s) don't reset a hold

# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
HELP_SHOW_TIME = 6.0                  # help overlay shown at startup (s)
FONT = 0                              # cv2.FONT_HERSHEY_SIMPLEX

# colours are BGR
COL_ACCENT = (255, 200, 60)           # cyan-ish accent
COL_TEXT = (240, 240, 240)
COL_DIM = (170, 170, 170)
COL_BAR = (25, 25, 25)
COL_CURSOR = (60, 200, 255)           # orange-yellow cursor
COL_RING = (60, 220, 255)
COL_SLIDER = (255, 170, 60)
COL_OK = (80, 220, 80)
COL_WARN = (60, 80, 255)

# --------------------------------------------------------------------------
# Folders / files
# --------------------------------------------------------------------------
IMAGES_DIR = "images"                 # your own photos (Gallery)
CAPTURES_DIR = "captures"             # webcam photos
RESULTS_DIR = "results"               # saved processed images
RECORDINGS_DIR = "recordings"         # saved videos
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------
PROCESS_MAX = 640                     # photos are processed at most this wide/high (speed)
CAPTURE_COUNTDOWN = 3                 # seconds: 3-2-1 before taking the photo
TOAST_TIME = 2.5                      # seconds a status message stays on screen
CACHE_SIZE = 40                       # processed results kept in memory

# Hand rotation (Photo Lab): tilt the OPEN HAND to rotate the image
ROT_DEADBAND = 12.0                   # degrees of tilt ignored (so a normal open hand doesn't rotate)
ROT_GAIN = 1.5                        # image degrees per degree of hand tilt
ROT_STEP = 5                          # rotation is rounded to multiples of this (fewer recomputes)

# Web browser mode (default): the app is shown at http://127.0.0.1:WEB_PORT
WEB_PORT = 8000
