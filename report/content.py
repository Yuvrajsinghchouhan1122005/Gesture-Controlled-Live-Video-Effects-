"""
content.py - the TEXT of the IVP mini-project report, in the exact order and headings of the
report guideline. Used by build_docx.py (Word) - run:  venv\\Scripts\\python.exe report\\build_docx.py

Helper functions provided by the builder:
    title_page(), h1(text), h2(text), P(text), bullets([..]), numbered([..]),
    figure(file, caption, width_cm), table(rows, widths_cm, caption), code(text), pagebreak(), ref(text)
Text may contain simple tags: <b> <i> <sub> <super> <br/> and &lt; &gt; &amp;.
D = measurements from figs/data.json.
"""

title_page()

ps = D["denoise_psnr"]
he = D["hist_eq"]

# =====================================================================================
h1("Abstract")
P(f"<b>Problem addressed:</b> Classical image-processing algorithms are usually demonstrated with "
  "static code and hand-typed parameter values, so it is difficult to see how a parameter changes the "
  "result or how different methods compare on the same image. "
  "<b>Proposed solution:</b> Gesture IVP Lab is an interactive application in which a photo is "
  "captured with the webcam or uploaded, and syllabus operations are applied to it while the user "
  "controls everything with hand gestures recognised from the live webcam video. "
  "<b>Techniques used:</b> hand-landmark detection and tracking (MediaPipe Hands), temporal smoothing, "
  "finger-state and hand-shape classification; grayscale conversion, point processing, histogram "
  "equalization, spatial filtering, binary and grayscale morphology, Prewitt/Sobel/LoG/Canny edge "
  f"detection, and global, Otsu and adaptive thresholding ({D['n_ops']} operations in total). "
  "<b>Outcome:</b> the application shows the original and processed images side by side together "
  "with the algorithm steps, histograms and kernels, runs at about 29 frames per second on an Intel "
  f"Core i3 laptop, our Otsu implementation matches OpenCV exactly, and the median filter improves "
  f"PSNR on salt-and-pepper noise from {ps['noisy']} dB to {ps['median5']} dB.")
pagebreak()

# =====================================================================================
h1("Introduction")
h2("Background of the problem")
P("A digital image is a two-dimensional array of intensities f(x, y). Image processing changes this "
  "array to enhance it, to extract information from it or to prepare it for analysis. The Image and "
  "Video Processing course covers image fundamentals, enhancement, transforms, morphology, edge "
  "detection, segmentation, colour processing and digital video. Every topic has parameters - a "
  "threshold T, a kernel size k, a gamma value, a structuring element - whose effect is best understood "
  "by changing them and watching the result immediately. In practice students usually see only one "
  "before/after pair produced by a script.")
h2("Why the problem is relevant")
bullets(["Seeing how the output changes while a parameter is varied, and comparing several methods "
         "on the same input, makes the concepts much easier to understand.",
         "Running each algorithm in a separate script is slow; one tool that holds many operations, "
         "explains them and compares them side by side shortens the learning loop.",
         "Touch-free (gesture) control is useful in classrooms and labs, presentations, kiosks and "
         "hygienic environments such as hospitals."])
h2("Role of image/video processing in solving it")
P("Image and video processing is used in two places. (1) <b>Video processing</b> drives the "
  "interface: every webcam frame is mirrored, down-scaled and converted to RGB, the hand is detected "
  "and tracked (21 landmarks), the landmarks are smoothed over time, and geometric tests turn them into "
  "hand shapes and actions. (2) <b>Image processing</b> is the content of the lab: 30 syllabus "
  "operations are applied to the chosen photo, most of them written in NumPy so the algorithm is "
  "visible (Otsu, histogram equalization, erosion/dilation, Zhang-Suen thinning).")
h2("Motivation for selecting the project")
P("We wanted a project that covers a large part of the syllabus in one working system, can be "
  "demonstrated live, and is interesting to use. Combining a gesture interface (a real-time video "
  "processing problem) with a library of classical image-processing operations gave us both: every "
  "operation is explained on screen with its steps, and the whole lab is controlled by the hand.")
pagebreak()

# =====================================================================================
h1("Problem Statement and Objectives")
h2("Which real-world problem are you solving?")
bullets(["Classical image-processing operations are hard to understand from static before/after "
         "images and code alone.",
         "There is no single, easy tool to apply many syllabus operations to one's own photo, change "
         "their parameters live and compare methods side by side.",
         "Controlling a computer without touching it (by hand gestures from a normal webcam) is useful "
         "for teaching, presentations and hygienic settings."])
h2("What are your objectives?")
P("The objective of this project is to build an interactive, touch-free image-processing laboratory "
  "that recognises hand gestures from a live webcam video and uses them to select, tune and compare "
  "classical image-processing operations on a captured or uploaded photo, showing the original and "
  "processed images, the algorithm steps and quantitative values in real time.")
h2("Specific objectives")
numbered(["To detect the hand in live webcam video and recognise seven hand shapes (1, 2 and 3 "
          "fingers, pinch, open hand, thumbs up, fist), using smoothing and hold timers so that one "
          "gesture gives exactly one action.",
          f"To implement {D['n_ops']} image-processing operations from the syllabus (enhancement, "
          "morphology, edge detection, segmentation), with the core algorithms written in NumPy.",
          "To show Original vs Processed side by side with the algorithm steps, histograms, kernels and "
          "a Compare View of four related methods.",
          "To change the parameter by pinching and moving the hand, and to rotate the image by tilting "
          "the open hand.",
          "To keep the interface real-time (20+ frames per second) on a normal laptop."])
pagebreak()

# =====================================================================================
h1("Scope of the Project")
h2("What the project covers")
bullets(["Image input from the webcam (3-2-1 countdown capture), from a file upload, or from a gallery "
         "of saved images.",
         f"{D['n_ops']} operations in five categories: image fundamentals, image enhancement, "
         "morphology, edge detection and segmentation (listed in the Techniques section).",
         "Gesture control with seven hand shapes, with mouse and keyboard as a backup.",
         "Rotation of the image by hand tilt; every operation then works on the rotated image.",
         "Compare View, saving of results, and a browser-based interface as well as a desktop window."])
h2("What it does not cover")
bullets(["Frequency-domain filtering and image transforms (DFT, DCT, Walsh-Hadamard), colour-model "
         "processing, and video-coding topics (motion estimation, I/P/B frames) are not implemented "
         "in this version.",
         "No machine-learning model was trained by us; the hand landmarks come from the pre-trained "
         "MediaPipe Hands model.",
         "Gesture accuracy was not measured on a large labelled dataset of many users.",
         "The application runs locally on one computer; it is not an online service."])
h2("Type of images/videos used")
P("Still images: colour photographs and test images in JPG, PNG, BMP or TIFF format, processed at a "
  "maximum of 640 pixels on the longer side. Video: the live webcam stream (1280 x 720, about 30 frames "
  "per second), used for gesture recognition.")
h2("Application domain")
P("Education (teaching and demonstrating image processing), touch-free human-computer interaction "
  "(presentations, kiosks, labs) and quick visual experimentation with classical operations.")
pagebreak()

# =====================================================================================
h1("Dataset / Input Description")
P("The project is an interactive tool that works on any photo, so no public dataset is used. "
  "The inputs are described below.")
h2("Source of images/videos")
bullets(["A synthetic test image generated with OpenCV/NumPy (images/sample_shapes.png).",
         "Photographs captured with the laptop webcam inside the app (captures/ folder).",
         "Photographs uploaded by the user from the laptop (images/ folder).",
         "Live video from the built-in webcam (USB2.0 HD UVC WebCam)."])
h2("Dataset name, if applicable")
P("Not applicable - no public dataset was used.")
h2("Number of images/videos")
table([["Input", "Number"],
       ["Synthetic test image (+ a low-contrast variant derived from it)", "1 (+1)"],
       ["Webcam photographs captured during testing", "3"],
       ["Uploaded photographs", "any number (user's choice)"],
       ["Live video", "1 continuous webcam stream"]],
      [11, 5.4], "Number of inputs.")
h2("Image resolution")
P("Test image: 640 x 480 pixels. Webcam photographs: 1280 x 720 pixels. Uploaded photographs: any "
  "size. All photos are resized so that the longer side is at most 640 pixels before processing "
  "(aspect ratio kept).")
h2("Image format")
P("PNG for the test image and webcam captures; JPG, JPEG, PNG, BMP and TIFF are accepted for uploads. "
  "Images are 24-bit colour (8 bits per channel, BGR in OpenCV).")
h2("Video resolution and duration, if applicable")
P("1280 x 720 pixels at about 30 frames per second (MJPG), continuous for as long as the app runs. "
  "For hand detection each frame is down-scaled to 640 x 360.")
h2("Sample input images/frames")
P("The synthetic test image was designed for the syllabus operations: a ring with a hole, a rectangle "
  "with six small holes, a triangle with sharp corners, text with thin strokes, a speckled ellipse, a thin "
  "line, a colour gradient and Gaussian noise (sigma = 6). Webcam photographs of people are not "
  "reproduced in this report.")
figure("fig_inputs.png", "Sample inputs: test image, grayscale version and the low-contrast version "
       "used for enhancement experiments.")
pagebreak()

# =====================================================================================
h1("Methodology / Proposed Approach")
h2("Complete processing pipeline")
numbered(["<b>Frame capture:</b> a camera thread reads 1280 x 720 frames at 30 fps; only the newest "
          "frame is kept and it is mirrored.",
          "<b>Hand detection:</b> the frame is down-scaled to 640 px and converted BGR -> RGB; MediaPipe "
          "Hands returns 21 landmarks per hand.",
          "<b>Smoothing:</b> landmarks are averaged over the last 5 frames to remove jitter.",
          "<b>Gesture recognition:</b> finger states and hand shape are computed, debounced over 3 frames, "
          "and held shapes become events (next, previous, save, back, click, pinch, rotate).",
          "<b>Image input:</b> a photo is captured or uploaded and resized to at most 640 px.",
          "<b>Rotation:</b> the photo is rotated by the hand-tilt angle (affine warp).",
          "<b>Processing:</b> the selected operation runs in a background thread (Otsu binarization first "
          "for morphology); results are cached.",
          "<b>Display:</b> Original | Processed, the algorithm steps, histogram / kernel / curve and values "
          "are drawn and streamed to the browser; a thumbs-up saves the result."])
h2("Block diagram / Flowchart")
figure("fig_pipeline.png", "Block diagram. Green: gesture (video) path; blue: image path; yellow: gesture "
       "events; purple: processing back-end and output.", 12)
h2("Algorithm pseudocode")
code("FOR every webcam frame:<br/>"
     "&nbsp;&nbsp;frame = mirror(newest_frame); small = resize(frame, 640); rgb = BGR2RGB(small)<br/>"
     "&nbsp;&nbsp;landmarks = MediaPipeHands(rgb); smooth = mean(last 5 landmark sets)<br/>"
     "&nbsp;&nbsp;finger_up[i] = |wrist - tip_i| &gt; 1.15 * |wrist - pip_i|<br/>"
     "&nbsp;&nbsp;shape = classify(finger_up, thumb, pinch distance)&nbsp;&nbsp;# accepted after 3 equal frames<br/>"
     "&nbsp;&nbsp;IF shape held long enough: emit event (next / prev / ok / back / click)<br/>"
     "&nbsp;&nbsp;IF pinch: parameter = start_value + (start_height - hand_height) / 0.35<br/>"
     "&nbsp;&nbsp;IF open hand: rotation = 1.5 * (angle - start_angle)&nbsp;&nbsp;# 12 degree dead-band<br/>"
     "ON change of photo / rotation / operation / parameter:<br/>"
     "&nbsp;&nbsp;img = rotate(resize(photo, 640), rotation)<br/>"
     "&nbsp;&nbsp;IF operation needs binary: img = otsu_threshold(gray(img))<br/>"
     "&nbsp;&nbsp;result = cache.get(key) or worker.run(operation, img, parameter)<br/>"
     "&nbsp;&nbsp;draw(original, result, steps, histogram/kernel, values)")
pagebreak()

# =====================================================================================
h1("Techniques / Algorithms Used")
h2("Image Processing")
table([["Technique", "Used?", "How it is used in the project"],
       ["Grayscale conversion", "Yes", "Y = 0.299 R + 0.587 G + 0.114 B (own NumPy code)."],
       ["Histogram equalization", "Yes", "Own implementation: s<sub>k</sub> = round((CDF<sub>k</sub> - CDF<sub>min</sub>) "
                                          "/ (N - CDF<sub>min</sub>) x 255); before/after histograms shown."],
       ["Thresholding", "Yes", "Point operation s = 255 if r &gt; T else 0; also negative, contrast stretching, "
                               "level slicing, log and gamma (look-up tables)."],
       ["Filtering", "Yes", "Mean (box), separable Gaussian, median (on salt & pepper noise), Laplacian "
                            "sharpening g = f - c Laplacian(f), unsharp masking / high-boost."],
       ["Edge detection", "Yes", "Prewitt and Sobel |G| = sqrt(G<sub>x</sub><super>2</super> + G<sub>y</sub><super>2</super>), "
                                 "Laplacian, Laplacian of Gaussian (zero-crossings), Canny (high = 3 x low)."],
       ["Morphological operations", "Yes", "Own erosion/dilation (min/max over shifted images; rect, ellipse, "
                                           "cross SE), opening, closing, boundary, hit-or-miss, Zhang-Suen "
                                           "thinning, thickening, grayscale top-hat/black-hat/gradient."],
       ["DFT/DCT", "No", "Not implemented in this version (planned future work)."],
       ["Image segmentation", "Yes", "Global threshold; Otsu (own code maximising w<sub>0</sub> w<sub>1</sub> "
                                     "(mu<sub>0</sub> - mu<sub>1</sub>)<super>2</super>); adaptive mean / Gaussian."],
       ["Feature extraction", "Yes", "Hand features from landmarks (finger extension, tilt angle, pinch "
                                     "distance); gradient magnitude; region statistics (white-pixel %, mean, std)."],
       ["Geometric transform", "Yes", "Rotation with the affine matrix [cos -sin; sin cos], canvas enlarged."]],
      [3.6, 1.4, 11.4], "Image-processing techniques.")
h2("Video Processing")
table([["Technique", "Used?", "How it is used in the project"],
       ["Frame extraction", "Yes", "A camera thread grabs frames continuously (1280 x 720, 30 fps); the newest "
                                   "frame is mirrored, down-scaled and converted to RGB."],
       ["Background subtraction", "No", "Not needed: the hand is found by the MediaPipe palm detector."],
       ["Optical flow", "No", "Not used."],
       ["Motion detection", "Partly", "Hand movement is measured from landmarks: vertical movement during a "
                                      "pinch and 'pointer is still' (dwell) detection."],
       ["Object tracking", "Yes", "MediaPipe Hands tracks the hand between frames; the app keeps the same main "
                                  "hand while it is visible; landmarks smoothed over 5 frames."],
       ["Frame differencing", "No", "Not used in this version (planned for the Live Video mode)."],
       ["Temporal filtering", "Yes", "Moving average of landmarks, debouncing of shapes over 3 frames, hold "
                                     "timers with a 0.25 s tolerance for missed detections."]],
      [3.6, 1.4, 11.4], "Video-processing techniques.")
pagebreak()

# =====================================================================================
h1("Implementation")
h2("Programming language")
P("Python 3.11.9.")
h2("Software/tools")
P("Visual Studio Code, Windows 11, Python virtual environment (venv), web browser (the app is shown at "
  "http://127.0.0.1:8000), Microsoft Word for the report.")
h2("Libraries/frameworks")
P("Python 3.11, OpenCV 4.11 (opencv-contrib-python), NumPy 1.26.4 and MediaPipe 0.10.14 (Hands) were "
  "used for implementation; Python's built-in http.server and threading for the browser interface; "
  "Matplotlib for the report graphs.")
h2("Hardware, if relevant")
P("13th Gen Intel Core i3-1315U processor, 7.6 GB RAM, built-in USB2.0 HD UVC webcam. No GPU was used.")
h2("Important implementation steps")
numbered(["<b>Project structure:</b> config.py (all settings), gestures.py (gesture recognition), "
          "registry.py (list of operations), operations/ (one file per topic), ui.py (drawing), "
          "web.py (browser interface), main.py (application).",
          "<b>Camera:</b> opened with Media Foundation (MSMF) and MJPG, which gave 30 fps instead of 10 fps "
          "with DirectShow; automatic reconnection; only one copy of the app may run.",
          "<b>Gesture engine:</b> finger tests use distances between landmarks, so tilted hands work. A first "
          "design with swipes and screen-edge zones was hard to use and was replaced by held hand shapes.",
          "<b>Operation registry:</b> each operation is one function plus one @register(...) line with its "
          "parameter range and explanation; the numbered docstring lines are shown on screen as STEPS.",
          "<b>Speed:</b> hand detection runs in its own thread and image operations in a worker thread with "
          "a 40-entry result cache; static parts of the screen are drawn once.",
          "<b>Browser interface:</b> frames are JPEG-encoded and streamed (MJPEG); clicks, keys and uploads "
          "are sent back by HTTP; gesture guide cards light up for the detected hand shape."])
pagebreak()

# =====================================================================================
h1("Results and Discussion")
h2("Input vs. output images")
figure("fig_point.png", "Point processing on the low-contrast input (top row) and thresholding of the "
       "original.")
figure("fig_filters.png", "Spatial filtering: salt-and-pepper noise removed by median and mean filters, "
       "smoothing, sharpening and high-boost.")
figure("fig_edges.png", "Edge detectors on the same input (Prewitt T = 80, Sobel T = 100, LoG sigma = 2, "
       "Canny 50/150, Laplacian).")
figure("fig_morph.png", "Binary morphology on the Otsu-binarized test image (red dots: hit-or-miss corners; "
       "white lines: Zhang-Suen skeleton).")
figure("fig_graymorph.png", "Grayscale morphology with a 15 x 15 elliptical structuring element.", 14)
figure("fig_seg.png", "Segmentation: global, Otsu and adaptive thresholding.")
figure("fig_rotation.png", "Image rotated by 30 degrees (open-hand tilt) and operations applied to it.")

h2("Intermediate processing results")
figure("fig_sobel.png", "Intermediate results of the Sobel operator: |Gx|, |Gy|, gradient magnitude and "
       "thresholded edges.", 13)
P("For morphology, the photo is first binarized with Otsu's threshold; for the median filter, salt and "
  "pepper noise is added first so that the effect is visible; for LoG, the smoothed Laplacian response "
  "is shown before the zero-crossings are extracted. The app shows these intermediate images as the "
  "'INPUT' image next to the result.")

h2("Screenshots")
figure("scr_home.png", "Home screen with the pointer gesture selecting UPLOAD IMAGE (the ring fills while "
       "the finger is held still).", 15)
figure("scr_lab_erosion.png", "Photo Lab: operation menu, Original and Processed (erosion 9 x 9), STEPS of "
       "the algorithm, structuring element and values.", 15)
figure("scr_compare_edges.png", "Compare View: Prewitt, Sobel, LoG and Canny side by side.", 15)
figure("scr_lab_otsu.png", "Otsu thresholding in the app with histogram and variance curve.", 15)

h2("Graphs/charts")
figure("fig_hist.png", "Histogram of the low-contrast input before and after histogram equalization.", 15)
figure("fig_otsu_curve.png", "Otsu's method: histogram with the chosen threshold, and between-class "
       "variance as a function of T.", 15)

h2("Tables")
mp = D["morph_white_pct"]
table([["Operation", "White pixels (%)", "Effect seen"],
       ["Binary input A", D["white_pct_binary"], "-"],
       ["Erosion (rect 9x9)", mp["Erosion"], "objects shrink, thin line and specks removed"],
       ["Dilation (rect 9x9)", mp["Dilation"], "objects grow, the six small holes are filled"],
       ["Opening (ellipse 11x11)", mp["Opening"], "speckled ellipse removed, shapes keep their size"],
       ["Closing (ellipse 11x11)", mp["Closing"], "gaps in the speckled ellipse closed"],
       ["Boundary A - (A erode B)", mp["Boundary Extraction"], "1-pixel outlines"]],
      [4.6, 3, 8.8], "Area change produced by morphological operations.")
ep = D["edge_pixels"]
table([["Detector", "Edge pixels (%)", "Observation"],
       ["Prewitt", ep["Prewitt"], "thick (2-3 px) edges, some noise"],
       ["Sobel", ep["Sobel"], "similar to Prewitt, slightly smoother"],
       ["Laplacian of Gaussian", ep["Laplacian of Gaussian"], "thin closed contours"],
       ["Canny", ep["Canny"], "thin, continuous edges, least noise"]],
      [4, 3, 9.4], "Edge detectors compared on the test image.")
table([["Test image", "Our Otsu (NumPy)", "OpenCV Otsu"]] + D["otsu_compare"],
      [6, 4, 4], "Our Otsu implementation vs OpenCV.")

h2("Quantitative performance measures")
table([["Image (vs clean grayscale)", "PSNR (dB)"],
       ["Noisy input (6 % salt & pepper)", ps["noisy"]],
       ["Mean filter 5x5", ps["mean5"]],
       ["Gaussian filter, sigma = 1", ps["gauss"]],
       ["Median filter 5x5", ps["median5"]]],
      [9, 4], "Denoising quality, PSNR = 10 log<sub>10</sub>(255<super>2</super> / MSE).")
table([["Measure", "Before", "After"],
       ["Histogram equalization: gray-level range", f"{he['in_range'][0]}-{he['in_range'][1]}",
        f"{he['out_range'][0]}-{he['out_range'][1]}"],
       ["Histogram equalization: standard deviation", he["in_std"], he["out_std"]],
       ["Otsu threshold after rotation (black corners vs border fill)", "53", "128"],
       ["Interface speed (detection inline vs in its own thread)", "8.1 fps", "28.8-30.2 fps"],
       ["Camera capture (DirectShow vs MSMF + MJPG)", "10.0 fps", "30.3 fps"]],
      [9, 3.6, 3.6], "Before / after measurements.")
tm = sorted(D["timing"], key=lambda r: -r[2])
table([["Operation", "Category", "Time (ms)"]] + [[a, b, c] for a, b, c in tm],
      [6.5, 5, 3], "Processing time of every operation on the 640 x 480 test image (median of 3 runs).")
table([["Test", "Result"],
       ["Gesture thread rate (model complexity 1, 640 px)", "18.6 fps"],
       ["Synthetic hand-shape classification tests", "9 / 9 correct"],
       ["Synthetic hold / event timing tests", "10 / 10 correct (no early or double firing)"],
       ["Automated app tests (menus, rotation, pinch, compare, save, all 30 operations)", "all passed"]],
      [10, 6.4], "Gesture and functional tests.")
P(f"<b>Discussion.</b> The median filter gives the best denoising ({ps['median5']} dB) because the 0 and "
  "255 outliers are never the median of a neighbourhood, while the mean and Gaussian filters only spread "
  f"them. Histogram equalization raised the contrast about {he['out_std'] / he['in_std']:.1f} times but "
  "also amplified background noise. Canny produced the thinnest and cleanest edges. Our Otsu "
  "implementation matched OpenCV on all images. Most operations take a few milliseconds; thinning and "
  "thickening are slowest because they iterate until no pixel changes, so they run in the background.")
pagebreak()

# =====================================================================================
h1("Challenges / Limitations")
h2("Problems encountered")
bullets(["The first gesture design (fast swipes, a control mode, holding the palm at the screen edge) was "
         "hard to use; it was replaced by hand shapes that are held still.",
         "Several copies of the app running together competed for the webcam and showed black images; "
         "solved with a single-instance lock and automatic camera reconnection.",
         "The default DirectShow camera driver gave only 10 fps; switching to MSMF + MJPG gave 30 fps."])
h2("Difficult cases")
bullets(["A bright window or white wall behind the user darkens the hand and makes detection harder.",
         "Hands that are partly outside the frame, too far (&gt; 1 m) or too close are not detected.",
         "A relaxed thumb can make '1 finger' look like a pinch; a loose fist can look like thumbs up."])
h2("Computational limitations")
bullets(["MediaPipe needs about 50-65 ms per frame on the Core i3 CPU (no GPU).",
         "Zhang-Suen thinning takes about 0.6-1.8 s per 640 x 480 image (depending on the load on the "
         "laptop); the result appears slightly after the pinch.",
         "The webcam takes about 8 seconds to start."])
h2("Dataset limitations")
bullets(["No public or labelled dataset was used; tests used a synthetic image, a few webcam photos and "
         "synthetic landmark sequences.",
         "Gesture accuracy for many different users, skin tones, backgrounds and lighting was not measured."])
h2("Cases where the method failed")
bullets(["Rotating the image with black corners made Otsu choose T = 53 instead of 128 (fixed by filling "
         "the corners with the border colour).",
         "Global thresholding with a fixed T fails on the gradient background; adaptive thresholding also "
         "outlines interior texture.",
         "Histogram equalization amplifies noise in flat regions."])
pagebreak()

# =====================================================================================
h1("Conclusion")
h2("What was implemented?")
P(f"Gesture IVP Lab: a touch-free image-processing laboratory that captures or uploads a photo and "
  f"applies {D['n_ops']} syllabus operations to it under hand-gesture control, showing Original vs "
  "Processed, the algorithm steps, graphs and values, with a Compare View and saving of results.")
h2("What techniques were used?")
P("Video processing: frame capture, hand detection and tracking (MediaPipe Hands), temporal smoothing, "
  "finger-state features, shape classification and hold timers. Image processing: grayscale conversion, "
  "point processing, histogram equalization, spatial filtering, morphology, edge detection, "
  "thresholding-based segmentation and affine rotation.")
h2("What was achieved?")
P(f"All five objectives were met for the implemented units: seven gestures, {D['n_ops']} operations, "
  f"side-by-side comparison with steps, pinch and rotation control, and about 29 fps. Measured results "
  f"match theory: median filtering gave {ps['median5']} dB PSNR, equalization raised contrast from "
  f"{he['in_std']} to {he['out_std']}, and our Otsu matched OpenCV exactly.")
h2("What is your learning?")
P("We learned how the classical algorithms work at the pixel level (look-up tables, convolution, order "
  "statistics, structuring elements, between-class variance) and how their parameters change the "
  "output. We also learned that real-time video analysis needs smoothing, debouncing and multithreading "
  "to feel reliable, and that simple gestures with clear feedback work better than complex ones. Future "
  "work: frequency-domain filters, DCT/DFT, colour processing and video motion estimation.")
pagebreak()

# =====================================================================================
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
    "C. Lugaresi et al., \"MediaPipe: A framework for building perception pipelines,\" arXiv:1906.08172, 2019.",
    "G. Bradski, \"The OpenCV Library,\" <i>Dr. Dobb's Journal of Software Tools</i>, 2000; "
    "OpenCV documentation, https://docs.opencv.org/4.x/",
    "C. R. Harris et al., \"Array programming with NumPy,\" <i>Nature</i>, vol. 585, pp. 357-362, 2020.",
    "Python Software Foundation, Python 3.11 documentation, https://docs.python.org/3.11/",
]
for i, r in enumerate(refs, 1):
    ref(f"[{i}] {r}")
