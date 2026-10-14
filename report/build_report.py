"""
build_report.py - builds the IVP mini-project report (PDF) with ReportLab.

1. Fill in the title-page details in DETAILS below.
2. Run (from the project folder):
       venv\\Scripts\\python.exe report\\make_figures.py      (figures + measurements)
       venv\\Scripts\\python.exe report\\make_screens.py      (app screenshots)
       venv\\Scripts\\python.exe report\\make_diagram.py      (pipeline diagram)
       venv\\Scripts\\python.exe report\\build_report.py      (the PDF)
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

# ---------------------------------------------------------------------------
# TITLE PAGE DETAILS - edit these
# ---------------------------------------------------------------------------
DETAILS = {
    "students": [("[Student Name 1]", "[Roll No. 1]"),
                 ("[Student Name 2]", "[Roll No. 2]"),
                 ("[Student Name 3]", "[Roll No. 3]")],
    "programme": "[Programme] / Semester [ ] / Division [ ]",
    "faculty": "[Faculty Name]",
    "institute": "SVKM's NMIMS Mukesh Patel School of Technology Management & Engineering (MPSTME), "
                 "[Department]",
    "year": "2026-27",
}

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs")
OUT = os.path.join(HERE, "IVP_Mini_Project_Report.pdf")
D = json.load(open(os.path.join(FIG, "data.json")))

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10.5,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=6)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=14, bulletIndent=4, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16,
                    textColor=colors.HexColor("#1f3b63"), spaceBefore=4, spaceAfter=10)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
                    textColor=colors.HexColor("#1f3b63"), spaceBefore=8, spaceAfter=5)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=9, leading=11.5, alignment=TA_CENTER,
                     textColor=colors.HexColor("#444444"), spaceBefore=3, spaceAfter=10)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=9, leading=11.5, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")
CODE = ParagraphStyle("code", parent=BODY, fontName="Courier", fontSize=8.8, leading=11.5,
                      alignment=0, leftIndent=10, backColor=colors.HexColor("#f4f4f4"),
                      borderPadding=6, spaceBefore=4, spaceAfter=10)

story = []
fig_no = [0]
tab_no = [0]


def P(text, style=BODY):
    story.append(Paragraph(text, style))


def bullets(items):
    for it in items:
        story.append(Paragraph(it, BUL, bulletText="•"))


def h1(t):
    story.append(Paragraph(t, H1))


def h2(t):
    story.append(Paragraph(t, H2))


def figure(fname, caption, width=17 * cm):
    from reportlab.lib.utils import ImageReader
    path = os.path.join(FIG, fname)
    iw, ih = ImageReader(path).getSize()
    fig_no[0] += 1
    story.append(KeepTogether([Image(path, width=width, height=width * ih / iw),
                               Paragraph(f"<b>Figure {fig_no[0]}.</b> {caption}", CAP)]))


def table(rows, widths, caption, header=True):
    tab_no[0] += 1
    data = [[Paragraph(str(c), CELLB if (header and r == 0) else CELL) for c in row]
            for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3")))
    t.setStyle(TableStyle(st))
    story.append(KeepTogether([Paragraph(f"<b>Table {tab_no[0]}.</b> {caption}", CAP), t,
                               Spacer(1, 10)]))


# ===========================================================================
# TITLE PAGE
# ===========================================================================
T_BIG = ParagraphStyle("tb", parent=BODY, fontName="Helvetica-Bold", fontSize=24, leading=30,
                       alignment=TA_CENTER, textColor=colors.HexColor("#1f3b63"))
T_SUB = ParagraphStyle("ts", parent=BODY, fontSize=13, leading=18, alignment=TA_CENTER)
T_MID = ParagraphStyle("tm", parent=BODY, fontSize=12, leading=17, alignment=TA_CENTER)
story += [Spacer(1, 2.2 * cm),
          Paragraph("Gesture IVP Lab", T_BIG), Spacer(1, 6),
          Paragraph("A Hand-Gesture-Controlled Image Processing Laboratory", T_SUB),
          Spacer(1, 1.2 * cm),
          Paragraph("Mini-Project Report", T_MID),
          Paragraph("<b>Course:</b> Image and Video Processing", T_MID),
          Spacer(1, 1.2 * cm)]
st_rows = [["Student Name", "Roll Number"]] + [list(s) for s in DETAILS["students"]]
t = Table([[Paragraph(c, CELLB if i == 0 else CELL) for c in r] for i, r in enumerate(st_rows)],
          colWidths=[8 * cm, 5 * cm])
t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                       ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe8f3"))]))
story += [t, Spacer(1, 1.0 * cm),
          Paragraph(f"<b>Programme / Semester / Division:</b> {DETAILS['programme']}", T_MID),
          Paragraph(f"<b>Faculty:</b> {DETAILS['faculty']}", T_MID), Spacer(1, 8),
          Paragraph(DETAILS["institute"], T_MID), Spacer(1, 8),
          Paragraph(f"<b>Academic Year:</b> {DETAILS['year']}", T_MID),
          PageBreak()]

# ===========================================================================
# ABSTRACT
# ===========================================================================
h1("Abstract")
P("Classical image-processing algorithms are usually taught and demonstrated through static code "
  "and parameter values typed by hand, which makes it hard to see how a parameter changes the result. "
  "This project, <b>Gesture IVP Lab</b>, is an interactive desktop/web application in which an image is "
  "captured with the webcam or uploaded, and syllabus operations are applied to it while the user "
  "controls the application with hand gestures. MediaPipe Hands provides 21 hand landmarks per frame; "
  "the live video is smoothed, classified into seven hand shapes (pointer, two fingers, three fingers, "
  "pinch, open hand, thumbs up, fist) and turned into actions such as selecting an operation, changing "
  f"its parameter or rotating the image. {D['n_ops']} operations are implemented: grayscale conversion, "
  "point processing, spatial filtering, histogram equalization, binary and grayscale morphology "
  "(including Zhang-Suen thinning and hit-or-miss), Prewitt/Sobel/LoG/Canny edge detection, and "
  "global, Otsu and adaptive thresholding. Original and processed images are shown side by side with "
  "the algorithm's steps, histograms and kernels. The interface runs at about 29 frames per second on "
  "an Intel Core i3 laptop, our NumPy Otsu implementation matches OpenCV's threshold on all test images, "
  f"and the median filter raises PSNR on salt-and-pepper noise from {D['denoise_psnr']['noisy']} dB to "
  f"{D['denoise_psnr']['median5']} dB.")
story.append(PageBreak())

# ===========================================================================
# INTRODUCTION
# ===========================================================================
h1("1. Introduction")
h2("1.1 Background")
P("Digital image processing transforms an image, a two-dimensional array of intensities f(x, y), "
  "to enhance it, extract information or prepare it for analysis. The Image and Video Processing "
  "syllabus covers image fundamentals, spatial and frequency-domain enhancement, transforms, "
  "morphology, edge detection, segmentation, colour processing and digital video. Each topic has "
  "parameters (a threshold T, a kernel size k, a gamma value, a structuring element) whose effect is "
  "best understood by changing them and watching the result immediately.")
P("At the same time, computer vision has made touch-free human-computer interaction practical: "
  "a single webcam and a hand-landmark model can recognise hand shapes in real time, without gloves "
  "or special sensors. Touch-free control is useful in classrooms and labs, in hygienic environments "
  "(hospitals, kitchens), at kiosks and in presentations.")
h2("1.2 Relevance of the problem")
bullets(["Students often see an algorithm as a single before/after pair; they rarely see "
         "<i>how</i> the output changes as a parameter is swept, or how two methods compare on the same input.",
         "Running each algorithm separately in scripts is slow; a single tool that holds many operations, "
         "explains each one and compares them side by side shortens the learning loop.",
         "Gesture interfaces are themselves an application of image and video processing, so the "
         "control layer of the tool also demonstrates the subject."])
h2("1.3 Role of image and video processing")
P("Image and video processing appears in two places in the project. (1) <b>The control layer</b> "
  "processes the live webcam video: each frame is mirrored, down-scaled and converted to RGB, hand "
  "landmarks are detected, smoothed with a temporal moving average, and turned into gestures with "
  "geometric tests and temporal debouncing. (2) <b>The processing layer</b> applies the syllabus "
  "operations to the selected photo; most are written with NumPy so the algorithm is visible "
  "(e.g. Otsu's between-class variance, histogram equalization through the CDF, binary erosion and "
  "dilation as minimum and maximum over shifted copies, Zhang-Suen thinning).")
h2("1.4 Motivation")
P("We wanted a project that covers a large part of the syllabus in one working system, can be "
  "demonstrated live in the viva, and is interesting to use. Combining a gesture interface with a "
  "library of classical operations gave us both: every concept is explained on screen with its steps, "
  "and the whole lab is driven by the same kind of real-time video analysis that the course teaches.")
story.append(PageBreak())

# ===========================================================================
# PROBLEM STATEMENT & OBJECTIVES
# ===========================================================================
h1("2. Problem Statement and Objectives")
h2("2.1 Problem statement")
P("To build an interactive, touch-free image-processing laboratory in which a user captures or "
  "uploads a photo and then selects, tunes and compares classical image-processing operations "
  "using hand gestures recognised from a live webcam video, with the original and processed images, "
  "the algorithm steps and quantitative values shown together in real time.")
h2("2.2 Objectives")
bullets(["<b>O1 - Gesture recognition:</b> detect the hand in live webcam video and reliably "
         "recognise seven hand shapes (1, 2 and 3 fingers, pinch, open hand, thumbs up, fist), "
         "using smoothing and hold timers so that one gesture gives exactly one action.",
         f"<b>O2 - Operation library:</b> implement {D['n_ops']} image-processing operations from the "
         "syllabus (enhancement, morphology, edge detection, segmentation), with the core algorithms "
         "written in NumPy and each operation registered with its parameter range and explanation.",
         "<b>O3 - Interactive comparison:</b> show Original vs Processed side by side, the step-by-step "
         "algorithm, histograms / kernels / transfer curves, and a Compare View with four related "
         "methods on the same input.",
         "<b>O4 - Gesture-driven parameters and geometry:</b> change the active parameter by pinching "
         "and moving the hand, and rotate the image by tilting the open hand (affine rotation).",
         "<b>O5 - Real-time performance:</b> keep the interface at 20+ frames per second on a normal "
         "laptop by running hand detection and image processing in background threads with caching."])
story.append(PageBreak())

# ===========================================================================
# SCOPE
# ===========================================================================
h1("3. Scope of the Project")
h2("3.1 What the project covers")
bullets(["Image input from the webcam (3-2-1 countdown capture), from a file upload, or from a gallery "
         "of saved images.",
         "Thirty operations in five categories (Table 1), each with a parameter, explanation and steps.",
         "Gesture control with seven hand shapes, plus mouse and keyboard as a backup.",
         "Rotation of the input by hand tilt; every operation then works on the rotated image.",
         "Compare View, saving of results, and a browser-based interface (local web server) as well as "
         "a native OpenCV window."])
h2("3.2 What the project does not cover")
bullets(["Frequency-domain filtering and image transforms (DFT, DCT, Walsh-Hadamard), colour-model "
         "processing, and Unit 8 video algorithms (block-matching motion estimation, I/P/B frames) are "
         "planned but <b>not yet implemented</b>; the Live Video tile is a placeholder.",
         "No machine-learning model was trained by us; hand landmarks come from the pre-trained "
         "MediaPipe Hands model.",
         "Gesture recognition was tested with synthetic landmark sequences and live use; a formal "
         "accuracy study on a labelled gesture dataset was not carried out.",
         "The application is single-user and runs locally (http://127.0.0.1); it is not a hosted "
         "web service."])
h2("3.3 Type of images / videos")
P("Still images: colour photographs and test images in JPG, PNG, BMP or TIFF format, processed at "
  "a maximum of 640 pixels on the longer side. Video: the live webcam stream (1280x720, about 30 "
  "frames per second) used for gesture recognition.")
h2("3.4 Application domain")
P("Education (teaching and demonstrating image processing), touch-free interaction for presentations "
  "and kiosks, and quick visual experimentation with classical operations.")
story.append(PageBreak())

# ===========================================================================
# DATASET / INPUT
# ===========================================================================
h1("4. Dataset / Input Description")
P("The project does not use a public dataset, because it is an interactive tool that works on any "
  "photo. The inputs used during development and in this report are:")
table([["Input", "Source", "Number", "Resolution", "Format"],
       ["Synthetic test image (shapes, text, holes, gradient, Gaussian noise sigma = 6)",
        "Generated with OpenCV/NumPy (images/sample_shapes.png)", "1", "640 x 480, 24-bit colour", "PNG"],
       ["Low-contrast variant (intensities squeezed to 70-170)", "Derived from the test image", "1",
        "640 x 480, 8-bit gray", "in memory"],
       ["Webcam photographs", "Captured with the app (captures/)", "3", "1280 x 720, 24-bit colour", "PNG"],
       ["Uploaded user photos", "Laptop files via the Upload button (images/)", "any", "any (resized to <= 640 px)",
        "JPG / PNG / BMP / TIFF"],
       ["Live video", "Laptop webcam (USB2.0 HD UVC WebCam)", "continuous stream",
        "1280 x 720 @ ~30 fps (MJPG)", "camera stream"]],
      [4.6, 4.4, 1.6, 3.4, 2.4], "Inputs used by the project.")
P("The synthetic test image was designed for the syllabus operations: it contains a ring with a hole "
  "(closing / boundary), a rectangle with six small holes (dilation and closing fill them), a triangle "
  "with sharp corners (hit-or-miss corner detection), text with thin strokes (erosion and thinning), "
  "a speckled ellipse (opening removes the specks) and a thin line (erosion removes it). A smooth colour "
  "gradient and additive Gaussian noise make thresholding and edge detection non-trivial. Webcam "
  "photographs of people are not reproduced in this report.")
figure("fig_inputs.png", "Sample inputs: test image, its grayscale version and the low-contrast "
       "version used for enhancement experiments.")
story.append(PageBreak())

# ===========================================================================
# METHODOLOGY
# ===========================================================================
h1("5. Methodology / Proposed Approach")
P("The system has two pipelines running in parallel (Figure 2): a <b>gesture pipeline</b> that "
  "analyses every webcam frame and produces events, and an <b>image pipeline</b> that applies the "
  "selected operation to the chosen photo. A third, display stage combines both into the screen that "
  "is shown in the browser (as an MJPEG stream) or in an OpenCV window.")
figure("fig_pipeline.png", "Processing pipeline. Green: gesture (video) path; blue: image path; "
       "yellow: gesture events; purple: processing back-end and output.")
h2("5.1 Gesture pipeline (per video frame)")
story.append(Paragraph(
    "1. grab newest frame (camera thread); mirror horizontally<br/>"
    "2. downscale to 640 px wide; BGR -> RGB; MediaPipe Hands -> 21 (x, y) landmarks per hand<br/>"
    "3. smooth landmarks: moving average over the last 5 frames<br/>"
    "4. finger i is UP if |wrist-tip| &gt; 1.15 x |wrist-PIP|  (rotation invariant)<br/>"
    "5. classify shape: pinch (|thumb tip-index tip| &lt; 0.35 x hand size), open hand, thumbs up,<br/>"
    "&nbsp;&nbsp;&nbsp;fist, 1 / 2 / 3 fingers; accept a shape only after 3 identical frames (debounce)<br/>"
    "6. hold timers: 2 fingers 1 s -&gt; NEXT, 3 fingers 1 s -&gt; PREV, thumbs up 1 s -&gt; OK,<br/>"
    "&nbsp;&nbsp;&nbsp;fist 1.5 s -&gt; BACK, pointer still 1.5 s -&gt; CLICK<br/>"
    "7. continuous values: pinch offset (hand height change), hand angle atan2(dx, -dy)", CODE))
h2("5.2 Image pipeline (per change of operation / parameter)")
story.append(Paragraph(
    "1. load photo; resize so that max(width, height) &lt;= 640 (keep aspect ratio)<br/>"
    "2. rotate by the hand angle (affine warp, canvas enlarged, corners filled with border median)<br/>"
    "3. if the operation needs a binary image: Otsu threshold first<br/>"
    "4. run operation f(image, parameter, option) in a background worker thread<br/>"
    "5. cache the result (key = image, rotation, operation, parameter, option)<br/>"
    "6. draw Original | Processed, STEPS, histogram / kernel / curve, numeric values", CODE))
P("Every operation is registered with one decorator line giving its name, category, parameter "
  "range, options and a one-line explanation; the numbered lines of its docstring are shown on "
  "screen as the algorithm's steps. Adding a new operation therefore needs one function and one "
  "registry line.")
story.append(PageBreak())

# ===========================================================================
# TECHNIQUES
# ===========================================================================
h1("6. Techniques / Algorithms Used")
units = D["ops_by_unit"]
table([["Category (syllabus unit)", "Operations implemented"]] +
      [[u, ", ".join(ops)] for u, ops in units.items()],
      [4.2, 12.2], f"The {D['n_ops']} implemented operations.")
h2("6.1 Image processing")
bullets(["<b>Grayscale conversion:</b> Y = 0.299 R + 0.587 G + 0.114 B (ITU-R BT.601 weights).",
         "<b>Point processing</b> with a 256-entry look-up table s = T(r): negative s = 255 - r, piecewise-"
         "linear contrast stretching, thresholding, intensity-level slicing, log s = c log(1 + r), "
         "power-law s = 255 (r/255)<super>gamma</super>.",
         "<b>Histogram equalization</b> (own implementation): s<sub>k</sub> = round((CDF<sub>k</sub> - "
         "CDF<sub>min</sub>) / (N - CDF<sub>min</sub>) x 255).",
         "<b>Spatial filtering:</b> mean (box) filter, separable Gaussian (kernel 6 sigma + 1), median filter "
         "on salt-and-pepper noise, Laplacian sharpening g = f - c Laplacian(f), unsharp masking / "
         "high-boost g = f + k (f - blur(f)).",
         "<b>Edge detection:</b> Prewitt and Sobel gradients |G| = sqrt(G<sub>x</sub><super>2</super> + "
         "G<sub>y</sub><super>2</super>), 4/8-neighbour Laplacian, Laplacian of Gaussian with zero-crossing "
         "detection, Canny (Gaussian, Sobel, non-maximum suppression, hysteresis with high = 3 x low).",
         "<b>Binary morphology</b> (NumPy): erosion = minimum and dilation = maximum over the image shifted "
         "by every structuring-element offset (rect / ellipse / cross); opening, closing, boundary "
         "extraction A - (A erode B), hit-or-miss (isolated points, corners, line ends), Zhang-Suen "
         "thinning (vectorised) and thickening as the complement of thinning the background.",
         "<b>Grayscale morphology:</b> erosion, dilation, top-hat, black-hat, morphological gradient.",
         "<b>Segmentation:</b> global thresholding; Otsu (own implementation maximising "
         "sigma<sub>B</sub><super>2</super> = w<sub>0</sub> w<sub>1</sub> (mu<sub>0</sub> - "
         "mu<sub>1</sub>)<super>2</super> over all T); adaptive mean / Gaussian thresholding.",
         "<b>Geometric transform:</b> rotation with the 2x3 affine matrix [cos -sin; sin cos] about the "
         "centre, with the canvas enlarged so no corner is lost."])
h2("6.2 Video processing")
bullets(["<b>Frame acquisition and pre-processing:</b> continuous capture in a dedicated thread (only the "
         "newest frame is kept), horizontal flip (mirror view), down-scaling and colour-space conversion.",
         "<b>Hand tracking:</b> MediaPipe Hands (palm detector + landmark model) in tracking mode.",
         "<b>Temporal filtering:</b> moving-average smoothing of landmarks over 5 frames, pose "
         "debouncing over 3 frames, hold timers with a 0.25 s tolerance for missed detections.",
         "<b>Pose features:</b> finger extension from landmark distances, hand tilt angle, pinch "
         "displacement and dwell detection (pointer still within 3.5 % for 1.5 s)."])
story.append(PageBreak())

# ===========================================================================
# IMPLEMENTATION
# ===========================================================================
h1("7. Implementation")
table([["Item", "Details"],
       ["Language", "Python 3.11.9"],
       ["Libraries", "OpenCV 4.11 (opencv-contrib-python), NumPy 1.26.4, MediaPipe 0.10.14 (Hands); "
                     "Matplotlib and ReportLab for the report figures and PDF"],
       ["User interface", "Local web page served by Python's http.server (MJPEG video stream, clicks, keys "
                          "and uploads via HTTP POST); optional native OpenCV window (--window)"],
       ["Tools", "Visual Studio Code, Windows 11, virtual environment (venv)"],
       ["Hardware", "13th Gen Intel Core i3-1315U, 7.6 GB RAM, built-in USB2.0 HD UVC webcam; no GPU used"],
       ["Code size", "config.py, gestures.py, ui.py, registry.py, web.py, main.py and five operation "
                     "modules (fundamentals, enhancement, morphology, edges, segmentation)"]],
      [3.2, 13.2], "Implementation environment.")
h2("7.1 Important implementation steps")
bullets(["<b>Camera:</b> opened with the Media Foundation (MSMF) back-end and MJPG format, which gave "
         "30 fps against about 10 fps with DirectShow on this laptop; the camera thread reconnects "
         "automatically if the stream is lost, and only one copy of the app may run (socket lock), "
         "because a second copy received only black frames.",
         "<b>Gesture engine:</b> hand shapes are tested on distances between landmarks rather than on y "
         "coordinates, so tilted hands are still recognised. The first design (swipes, a 'control mode' "
         "and screen-edge zones) was hard to use, so it was replaced by shapes that are simply held still.",
         "<b>Operation registry:</b> a decorator <font face='Courier'>@register(name, unit, param, "
         "options, needs_binary, info)</font> adds each function to a list; the docstring's numbered "
         "lines become the on-screen STEPS.",
         "<b>Responsiveness:</b> hand detection runs in its own thread and image operations in a worker "
         "thread with a 40-entry result cache; static parts of the screen are drawn once and reused.",
         "<b>Browser front-end:</b> each rendered frame is JPEG-encoded and streamed; a small JavaScript "
         "page sends clicks and keys back, opens the file picker for uploads and shows live gesture "
         "guide cards from a /state endpoint."])
P("Example of how an operation is added:")
story.append(Paragraph(
    "@register(\"Erosion\", unit=4, param=(\"SE size\", 3, 31, 5, \"odd\"),<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;options=[\"rect\", \"ellipse\", \"cross\"], needs_binary=True,<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;info=\"A erode B = {z | B shifted to z fits inside A}\")<br/>"
    "def erosion(binary, p, extra):<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;se = structuring_element(extra[\"option\"], p)<br/>"
    "&nbsp;&nbsp;&nbsp;&nbsp;out = 255 * ones;&nbsp;&nbsp;# for every SE offset (dy, dx): out = min(out, shifted image)",
    CODE))
story.append(PageBreak())

# ===========================================================================
# RESULTS
# ===========================================================================
h1("8. Results and Discussion")
h2("8.1 Application screens")
figure("scr_home.png", "Home screen. A synthetic 'one finger' hand is shown pointing at UPLOAD IMAGE; "
       "the ring around the yellow pointer fills while the finger is held still (60 % here).", 15 * cm)
figure("scr_lab_erosion.png", "Photo Lab: menu of operations (left), Original (binarized input) and "
       "Processed (erosion, 9x9 rectangle), the STEPS of the algorithm, the structuring element and the "
       "measured values. The webcam view (bottom-left) is blank because the camera was in use.", 15 * cm)
figure("scr_lab_otsu.png", "Otsu thresholding in the app: histogram with the selected threshold "
       "and the between-class-variance curve.", 15 * cm)
figure("scr_compare_edges.png", "Compare View for edge detection: Prewitt, Sobel, LoG and Canny on "
       "the same input, each with its current parameter.", 15 * cm)
figure("scr_lab_rotated.png", "Image rotated by 30 degrees (open-hand tilt) with dilation applied "
       "to the rotated input.", 15 * cm)

h2("8.2 Point processing and histogram equalization")
figure("fig_point.png", "Point operations on the low-contrast input (top row) and on the original "
       "(thresholding).")
figure("fig_hist.png", "Histograms of the low-contrast input before and after equalization.", 15 * cm)
he = D["hist_eq"]
P(f"The low-contrast input uses only the gray levels {he['in_range'][0]}-{he['in_range'][1]} "
  f"(standard deviation {he['in_std']}). After histogram equalization the output covers "
  f"{he['out_range'][0]}-{he['out_range'][1]} and its standard deviation rises to {he['out_std']}, i.e. "
  f"the contrast is about {he['out_std'] / he['in_std']:.1f} times higher. Equalization also amplifies "
  "the background noise, which is visible in the gradient region; gamma &lt; 1 brightens and gamma &gt; 1 "
  "darkens without changing the order of gray levels.")

h2("8.3 Spatial filtering")
figure("fig_filters.png", "Spatial filters. Top row: salt-and-pepper noise removed by median and mean "
       "filters; Gaussian and box smoothing. Bottom: Laplacian sharpening, high-boost and the original.")
ps = D["denoise_psnr"]
table([["Image (vs clean grayscale)", "PSNR (dB)"],
       ["Noisy input (6 % salt & pepper)", ps["noisy"]],
       ["Mean filter 5x5", ps["mean5"]],
       ["Gaussian filter, sigma = 1", ps["gauss"]],
       ["Median filter 5x5", ps["median5"]]],
      [9, 4], "Denoising quality, PSNR = 10 log<sub>10</sub>(255<super>2</super> / MSE).")
P(f"The median filter gives the highest PSNR ({ps['median5']} dB, an improvement of "
  f"{ps['median5'] - ps['noisy']:.1f} dB) because the 0 and 255 outliers never become the median of a "
  "neighbourhood, whereas the mean and Gaussian filters only spread them out and blur the edges.")

h2("8.4 Edge detection")
figure("fig_edges.png", "Edge detectors on the same input (Prewitt T = 80, Sobel T = 100, "
       "LoG sigma = 2, Canny low = 50 / high = 150, Laplacian).")
figure("fig_sobel.png", "Intermediate results of the Sobel operator: |Gx| (vertical edges), "
       "|Gy| (horizontal edges), gradient magnitude and thresholded edges.", 13 * cm)
ep = D["edge_pixels"]
table([["Detector", "Edge pixels (%)", "Observation"],
       ["Prewitt", ep["Prewitt"], "thick (2-3 px) edges, some noise response"],
       ["Sobel", ep["Sobel"], "similar to Prewitt, centre weight 2 smooths slightly"],
       ["Laplacian of Gaussian", ep["Laplacian of Gaussian"], "thin closed contours from zero-crossings"],
       ["Canny", ep["Canny"], "thin, continuous 1-px edges; least noise"]],
      [4, 3, 9.4], "Edge pixel percentage on the test image.")

h2("8.5 Morphological processing")
figure("fig_morph.png", "Binary morphology on the Otsu-binarized test image. Red dots in the "
       "hit-or-miss result mark detected corner pixels; white lines in the last image are the "
       "Zhang-Suen skeleton.")
mp = D["morph_white_pct"]
table([["Operation", "White pixels (%)", "Effect seen"],
       ["Binary input A", D["white_pct_binary"], "-"],
       ["Erosion (rect 9x9)", mp["Erosion"], "objects shrink, thin line and specks removed, small holes grow"],
       ["Dilation (rect 9x9)", mp["Dilation"], "objects grow, the six small holes are filled"],
       ["Opening (ellipse 11x11)", mp["Opening"], "speckled ellipse removed, shapes keep their size"],
       ["Closing (ellipse 11x11)", mp["Closing"], "gaps in the speckled ellipse closed, size kept"],
       ["Boundary A - (A erode B)", mp["Boundary Extraction"], "1-pixel outlines"]],
      [4.6, 3, 8.8], "Area change produced by the morphological operations.")
figure("fig_graymorph.png", "Grayscale morphology with a 15x15 elliptical structuring element.", 14 * cm)

h2("8.6 Segmentation")
figure("fig_seg.png", "Thresholding methods.")
figure("fig_otsu_curve.png", "Otsu's method: histogram with the chosen threshold and the "
       "between-class variance as a function of T (maximum at the chosen T).", 15 * cm)
table([["Test image", "Our Otsu (NumPy)", "OpenCV Otsu"]] + D["otsu_compare"],
      [6, 4, 4], "Our Otsu implementation compared with cv2.threshold(THRESH_OTSU).")
P("Our implementation returns exactly the same threshold as OpenCV on all four images. Global "
  "thresholding at a fixed T fails on the gradient background, while adaptive thresholding separates "
  "the shapes from the slowly varying background but also outlines interior texture.")

h2("8.7 Gesture-driven rotation")
figure("fig_rotation.png", "Rotation by hand tilt (30 degrees) and operations applied to the "
       "rotated image.")
P("Filling the new corners with the median border colour instead of black was important: with black "
  "corners Otsu's threshold dropped from 128 to 53 and almost the whole rotated photo became "
  "foreground; with the border fill the threshold stayed at 128.")

h2("8.8 Performance and gesture tests")
tm = sorted(D["timing"], key=lambda r: -r[2])
table([["Operation", "Category", "Time (ms)"]] + [[a, b, c] for a, b, c in tm],
      [6.5, 5, 3], "Processing time of every operation on the 640x480 test image (median of 3 runs, "
                   "Intel Core i3-1315U).")
table([["Measurement", "Result"],
       ["Camera capture rate, MSMF + MJPG vs DirectShow", "30.3 fps vs 10.0 fps"],
       ["Camera start-up time", "about 8.3 s"],
       ["MediaPipe Hands inline in the render loop", "~65 ms per frame -> UI 8.1 fps"],
       ["After moving detection to a thread + caching static UI", "UI 28.8-30.2 fps"],
       ["Gesture thread rate (model complexity 0, 480 px)", "23.8 fps"],
       ["Gesture thread rate (model complexity 1, 640 px)", "18.6 fps"],
       ["Synthetic hand-shape classification tests", "9 / 9 correct"],
       ["Synthetic hold / event timing tests", "10 / 10 correct (no early or double firing)"],
       ["Automated app tests (menus, rotation, pinch, compare, save, all 30 ops on rotated input)",
        "all passed"]],
      [10, 6.4], "Real-time performance and functional test results.")
P("Most operations take only a few milliseconds and update immediately while the parameter is "
  "changed. Zhang-Suen thinning and thickening are the slowest because they iterate until no pixel "
  "changes; they run in the background thread, so the interface does not freeze.")

# ===========================================================================
# CHALLENGES
# ===========================================================================
story.append(PageBreak())
h1("9. Challenges / Limitations")
bullets(["<b>Gesture design:</b> the first design (fast swipes, a separate control mode and holding the "
         "palm at the screen edge) was hard for a new user; hands near the frame edge are also lost by the "
         "detector. It was replaced by seven held hand shapes with on-screen feedback.",
         "<b>Lighting:</b> a bright window or white wall behind the user darkens the hand and reduces "
         "detection reliability; the hand must be fully visible at about 40-60 cm.",
         "<b>Ambiguous shapes:</b> a relaxed thumb can turn 'one finger' into a pinch-like shape, and a "
         "loose fist can look like 'thumbs up'; thresholds in config.py had to be tuned.",
         "<b>Camera access on Windows:</b> several copies of the app started at the same time competed for "
         "the webcam and showed black frames; solved with a single-instance lock and automatic reconnection.",
         "<b>Computation:</b> MediaPipe takes about 50-65 ms per frame on the i3 CPU, and thinning about "
         "0.4-0.6 s per image; threads and caching keep the UI responsive, but the result can lag the "
         "pinch slightly for slow operations.",
         "<b>Rotation side-effects:</b> rotating enlarges the canvas; the filled corners still slightly "
         "change histograms.",
         "<b>Evaluation:</b> no labelled gesture dataset was used, so recognition accuracy on many different "
         "users and backgrounds was not measured quantitatively; tests used synthetic landmarks and live trials.",
         "<b>Not yet implemented:</b> frequency-domain filters, DCT/DFT/Hadamard transforms, colour models "
         "and the video-coding demonstrations (block matching, I/P/B frames)."])

# ===========================================================================
# CONCLUSION
# ===========================================================================
h1("10. Conclusion")
P(f"We built <b>Gesture IVP Lab</b>, a touch-free image-processing laboratory that captures or uploads "
  f"a photo and applies {D['n_ops']} syllabus operations to it under hand-gesture control. The video side "
  "of the project processes the live webcam stream with MediaPipe Hands, temporal smoothing, geometric "
  "finger tests and hold timers to recognise seven hand shapes; the image side implements point "
  "processing, histogram equalization, spatial filtering, binary and grayscale morphology, four edge "
  "detectors and three thresholding methods, many of them written directly in NumPy.")
P(f"The results show the expected behaviour of each method: equalization raised the contrast "
  f"(standard deviation {D['hist_eq']['in_std']} -> {D['hist_eq']['out_std']}), the median filter gave the "
  f"best denoising ({D['denoise_psnr']['median5']} dB PSNR), Canny produced the thinnest and cleanest edges, "
  "and our Otsu implementation matched OpenCV exactly. With background threads the interface runs at "
  "about 29 fps on an ordinary laptop.")
P("<b>Learning.</b> We learned how the classical algorithms work at the pixel level (look-up tables, "
  "convolution, order statistics, set operations with structuring elements, between-class variance), "
  "how parameters change their output, and how real-time video analysis needs smoothing, debouncing "
  "and multithreading to feel reliable. We also learned that usability matters: simple held gestures "
  "with clear feedback worked far better than fast swipes. Future work is to add the frequency-domain, "
  "transform, colour and video-coding units and to evaluate gesture accuracy on more users.")

# ===========================================================================
# REFERENCES
# ===========================================================================
story.append(PageBreak())
h1("References")
refs = [
    "R. C. Gonzalez and R. E. Woods, <i>Digital Image Processing</i>, 4th ed., Pearson, 2018.",
    "A. K. Jain, <i>Fundamentals of Digital Image Processing</i>, Prentice Hall, 1989.",
    "N. Otsu, \"A threshold selection method from gray-level histograms,\" <i>IEEE Trans. Systems, Man, "
    "and Cybernetics</i>, vol. 9, no. 1, pp. 62-66, 1979.",
    "J. Canny, \"A computational approach to edge detection,\" <i>IEEE Trans. Pattern Analysis and "
    "Machine Intelligence</i>, vol. PAMI-8, no. 6, pp. 679-698, 1986.",
    "D. Marr and E. Hildreth, \"Theory of edge detection,\" <i>Proc. Royal Society of London B</i>, "
    "vol. 207, pp. 187-217, 1980.",
    "T. Y. Zhang and C. Y. Suen, \"A fast parallel algorithm for thinning digital patterns,\" "
    "<i>Communications of the ACM</i>, vol. 27, no. 3, pp. 236-239, 1984.",
    "J. Serra, <i>Image Analysis and Mathematical Morphology</i>, Academic Press, 1982.",
    "F. Zhang et al., \"MediaPipe Hands: On-device real-time hand tracking,\" arXiv:2006.10214, 2020.",
    "C. Lugaresi et al., \"MediaPipe: A framework for building perception pipelines,\" "
    "arXiv:1906.08172, 2019.",
    "G. Bradski, \"The OpenCV Library,\" <i>Dr. Dobb's Journal of Software Tools</i>, 2000; "
    "OpenCV documentation, https://docs.opencv.org/4.x/",
    "C. R. Harris et al., \"Array programming with NumPy,\" <i>Nature</i>, vol. 585, pp. 357-362, 2020.",
    "Python Software Foundation, Python 3.11 documentation (http.server, threading), https://docs.python.org/3.11/",
]
for i, r in enumerate(refs, 1):
    story.append(Paragraph(f"[{i}] {r}", ParagraphStyle("ref", parent=BODY, leftIndent=18,
                                                       firstLineIndent=-18, alignment=0)))


# ---------------------------------------------------------------------------
def footer(canvas, doc):
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 8.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(2 * cm, 1.2 * cm, "Gesture IVP Lab - Image and Video Processing mini-project")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                        bottomMargin=2 * cm, title="Gesture IVP Lab - IVP Mini-Project Report",
                        author=", ".join(s[0] for s in DETAILS["students"]))
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print("saved", OUT)
