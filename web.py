"""
web.py - show the app in a web browser (http://127.0.0.1:8000).

The app still runs in Python (webcam, MediaPipe, image processing). Every
frame it draws is JPEG-encoded and streamed to the browser as MJPEG
(multipart/x-mixed-replace). Clicks, mouse-wheel and key presses on the page
are sent back to Python with small HTTP POST requests, and photos are uploaded
through the browser's file picker. Only Python's standard library is used.
"""

import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gesture IVP Lab</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; --accent: #3cc8ff; --ok: #4fdc6a; }
  body { margin: 0; background: #111; color: #ddd; font-family: Segoe UI, Arial, sans-serif; }
  header { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; padding: 8px 14px; background: #1b1b1b; }
  header h1 { font-size: 18px; margin: 0 12px 0 0; color: var(--accent); }
  button { background: #2a2a2a; color: #eee; border: 1px solid #444; border-radius: 6px;
           padding: 7px 14px; font-size: 14px; cursor: pointer; }
  button:hover { background: #3a3a3a; }
  button.main { background: #1f6f3a; border-color: #2c9; }
  #wrap { display: flex; justify-content: center; padding: 8px 8px 0; }
  #view { width: 100%; max-width: 1280px; aspect-ratio: 16 / 9; background: #000;
          cursor: crosshair; user-select: none; border-radius: 4px; }
  #guide { max-width: 1280px; margin: 8px auto 16px; padding: 0 8px; }
  #guide h2 { font-size: 15px; margin: 4px 0 8px; color: #aaa; font-weight: 600; }
  #guide h2 b { color: var(--accent); }
  #cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 8px; }
  .card { background: #1c1c1c; border: 2px solid #2c2c2c; border-radius: 10px; padding: 10px;
          position: relative; overflow: hidden; transition: border-color .15s, background .15s; }
  .card .emo { font-size: 34px; line-height: 1.1; }
  .card .name { font-weight: 700; color: #fff; margin-top: 4px; }
  .card .how { font-size: 12px; color: #999; margin-top: 2px; }
  .card .what { font-size: 14px; color: var(--accent); margin-top: 6px; font-weight: 600; }
  .card.unused .what { color: #666; font-weight: 400; }
  .card.now { border-color: var(--accent); background: #13283a; }
  .card .bar { position: absolute; left: 0; bottom: 0; height: 5px; width: 0; background: var(--ok); }
  #status { font-size: 13px; color: #888; margin-left: auto; }
  #live { max-width: 1280px; margin: 8px auto 0; padding: 10px 16px; border-radius: 10px;
          font-size: 18px; font-weight: 600; background: #3a1d1d; color: #ffb3b3; }
  #live.ok { background: #16321f; color: #a8f0b8; }
  #live small { display: block; font-size: 13px; font-weight: 400; color: #bbb; margin-top: 3px; }
</style></head>
<body>
<header>
  <h1>Gesture IVP Lab</h1>
  <button class="main" onclick="pick()">Upload image</button>
  <button onclick="key('Escape')">Back</button>
  <button onclick="key('s')">Save / Take photo</button>
  <button onclick="key('c')">Compare</button>
  <button onclick="key('h')">Help</button>
  <span id="status">starting...</span>
</header>
<div id="wrap"><img id="view" src="/stream" alt="starting..."></div>
<div id="live">starting...</div>
<section id="guide">
  <h2>Hand gestures - <b id="modename">Home</b> &nbsp;(the card that lights up is what the camera sees)</h2>
  <div id="cards"></div>
</section>
<input type="file" id="file" accept="image/*" style="display:none">
<script>
const GESTURES = [
  {pose: 'point',     emo: '☝️', name: '1 finger',  how: 'only the index finger up',
   fixed: 'pointer - hold still 1.5 s = CLICK'},
  {pose: 'peace',     emo: '✌️', name: '2 fingers', how: 'index + middle up, hold 1 s', event: 'next'},
  {pose: 'three',     emo: '3️⃣',  name: '3 fingers', how: 'index + middle + ring up, hold 1 s', event: 'prev'},
  {pose: 'pinch',     emo: '👌', name: 'pinch',     how: 'touch thumb and index tips',
   fixed: 'move hand UP = stronger, DOWN = weaker'},
  {pose: 'palm',      emo: '🖐️', name: 'open hand', how: 'all 5 fingers open',
   fixed: 'tilt left / right = rotate image'},
  {pose: 'thumbs_up', emo: '👍', name: 'thumbs up', how: 'thumb up, fingers closed, hold 1 s', event: 'ok'},
  {pose: 'fist',      emo: '✊',       name: 'fist',      how: 'close your hand, hold 1.5 s', event: 'back'},
];
const cards = document.getElementById('cards');
for (const g of GESTURES) {
  g.el = document.createElement('div'); g.el.className = 'card';
  g.el.innerHTML = `<div class="emo">${g.emo}</div><div class="name">${g.name}</div>
    <div class="how">${g.how}</div><div class="what"></div><div class="bar"></div>`;
  cards.appendChild(g.el);
}
const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'little'];
const HOWTO = {
  'Home': 'Show 1 finger: an orange dot appears. Move it onto a tile and keep still until the circle fills.',
  'Capture': 'Thumbs up for 1 second (or point at TAKE PHOTO and hold) to take your photo.',
  'Gallery': '2 fingers = next photo, 3 fingers = previous. Point at the photo and hold to open it.',
  'Photo Lab': '2 fingers = next operation. Pinch + move up/down = strength. Open hand + tilt = rotate.',
};
function render(st) {
  const live = document.getElementById('live');
  if (!st.hand) {
    live.className = '';
    live.innerHTML = 'Camera does NOT see your hand' +
      '<small>Raise your whole hand in front of the camera (palm facing it, about 40-60 cm away), ' +
      'with light on your hand. ' + (HOWTO[st.mode] || '') + '</small>';
  } else {
    const up = (st.fingers || []).map((f, i) => f ? FINGER_NAMES[i] : null).filter(Boolean);
    const g = GESTURES.find(x => x.pose === st.pose);
    let extra = '';
    if (st.pose === 'point') extra = st.dwell > 0 ? `  - holding still ${Math.round(st.dwell * 100)}%` : '  - keep still to click';
    if (g && g.event && st.hold === g.event) extra = `  - holding ${Math.round(st.progress * 100)}%`;
    live.className = 'ok';
    live.innerHTML = `Hand seen: ${g ? g.emo + ' ' + g.name : 'shape not recognised'}${extra}` +
      `<small>fingers up: ${up.length ? up.join(', ') : 'none'}. ${HOWTO[st.mode] || ''}</small>`;
  }
  document.getElementById('modename').textContent = st.mode || '';
  document.getElementById('status').textContent = st.hand ? 'hand detected' : 'show your hand to the camera';
  const acts = st.actions || {};
  for (const g of GESTURES) {
    let what = g.fixed;
    if (g.event) what = acts[g.event] ? acts[g.event].toUpperCase() : 'not used on this screen';
    g.el.querySelector('.what').textContent = what;
    g.el.classList.toggle('unused', !!g.event && !acts[g.event]);
    g.el.classList.toggle('now', st.hand && st.pose === g.pose);
    const p = (g.event && st.hold === g.event) ? st.progress : 0;
    g.el.querySelector('.bar').style.width = (p * 100) + '%';
  }
}
setInterval(() => fetch('/state').then(r => r.json()).then(render).catch(() => {}), 150);

const view = document.getElementById('view'), file = document.getElementById('file');
function post(url, body) {
  return fetch(url, {method: 'POST', headers: {'Content-Type': 'application/json'},
                     body: JSON.stringify(body)}).then(r => r.json()).catch(() => ({}));
}
function pick() { file.value = ''; file.click(); }
function key(k) { post('/event', {type: 'key', key: k}); }
view.addEventListener('click', e => {
  const r = view.getBoundingClientRect();
  post('/event', {type: 'click', x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height})
    .then(res => { if (res.upload) pick(); });      // the Upload tile was clicked
});
view.addEventListener('wheel', e => { e.preventDefault(); post('/event', {type: 'wheel', d: e.deltaY < 0 ? 1 : -1}); },
                      {passive: false});
document.addEventListener('keydown', e => {
  if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown',' '].includes(e.key)) e.preventDefault();
  if (e.key.length === 1 || e.key.startsWith('Arrow') || ['Escape','Backspace'].includes(e.key)) key(e.key);
});
file.addEventListener('change', () => {
  const f = file.files[0]; if (!f) return;
  fetch('/upload', {method: 'POST', headers: {'X-Filename': encodeURIComponent(f.name)}, body: f})
    .then(r => r.json()).then(res => { if (!res.ok) alert(res.error || 'Upload failed'); });
});
// reconnect the video if the stream drops
view.addEventListener('error', () => setTimeout(() => view.src = '/stream?' + Date.now(), 1000));
</script>
</body></html>
"""


class WebUI:
    def __init__(self, upload_dir, port=8000, quality=80):
        self.upload_dir = upload_dir
        self.quality = quality
        self.events = queue.Queue()          # ("key", name) / ("click", (x, y)) / ("param", d) / ("open", path)
        self.jpeg = None
        self.frame_id = 0
        self.cond = threading.Condition()
        self.clients = 0
        self.ask_upload = False              # set by the app when the Upload tile is chosen
        self.state = {}                      # current gesture / mode (for the guide cards)
        self.server = None
        for p in range(port, port + 20):     # first free port
            try:
                self.server = ThreadingHTTPServer(("127.0.0.1", p), self._handler())
                self.port = p
                break
            except OSError:
                continue
        if self.server is None:
            raise RuntimeError("no free port for the web server")
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    # called by the app every frame
    def publish(self, canvas):
        if self.clients == 0 and self.jpeg is not None:
            return                           # nobody watching: skip encoding
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.frame_id += 1
                self.cond.notify_all()

    def get_events(self):
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def close(self):
        if self.server:
            self.server.shutdown()

    # ------------------------------------------------------------ handler
    def _handler(web):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):       # keep the terminal quiet
                pass

            def _json(self, obj, code=200):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/" or self.path.startswith("/index"):
                    data = PAGE.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif self.path.startswith("/stream"):
                    self._stream()
                elif self.path.startswith("/state"):
                    self._json(web.state)
                else:
                    self.send_error(404)

            def _stream(self):
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                web.clients += 1
                last = -1
                try:
                    while True:
                        with web.cond:
                            web.cond.wait_for(lambda: web.frame_id != last, timeout=2.0)
                            jpg, last = web.jpeg, web.frame_id
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (ConnectionError, OSError):
                    pass
                finally:
                    web.clients -= 1

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n) if n else b""
                if self.path == "/event":
                    try:
                        ev = json.loads(body or b"{}")
                    except ValueError:
                        return self._json({"ok": False}, 400)
                    t = ev.get("type")
                    if t == "key":
                        web.events.put(("key", str(ev.get("key", ""))))
                    elif t == "wheel":
                        web.events.put(("param", 1 if ev.get("d", 0) > 0 else -1))
                    elif t == "click":
                        web.ask_upload = False
                        web.events.put(("click", (float(ev.get("x", 0)), float(ev.get("y", 0)))))
                        # wait briefly: if the click hit the Upload tile, tell the page
                        # to open the file picker (allowed right after a user click)
                        end = time.time() + 0.5
                        while time.time() < end and not web.ask_upload:
                            time.sleep(0.02)
                        if web.ask_upload:
                            web.ask_upload = False
                            return self._json({"ok": True, "upload": True})
                    return self._json({"ok": True})
                if self.path == "/upload":
                    return self._upload(body)
                self.send_error(404)

            def _upload(self, body):
                from urllib.parse import unquote
                import numpy as np
                name = os.path.basename(unquote(self.headers.get("X-Filename", "upload.png"))) or "upload.png"
                img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if body else None
                if img is None:
                    return self._json({"ok": False, "error": "That file is not a readable image."})
                os.makedirs(web.upload_dir, exist_ok=True)
                base, ext = os.path.splitext(name)
                path = os.path.join(web.upload_dir, name)
                if os.path.exists(path):
                    path = os.path.join(web.upload_dir, f"{base}_{time.strftime('%Y%m%d_%H%M%S')}{ext}")
                with open(path, "wb") as f:
                    f.write(body)
                web.events.put(("open", path))
                return self._json({"ok": True, "path": path})
        return Handler
