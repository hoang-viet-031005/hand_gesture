"""
Demo realtime Model 1 trên Google Colab: webcam (trình duyệt) -> MediaPipe HandLandmarker -> MLP.
Trong notebook:
    from src.demo.demo_webcam_model1 import run_webcam_demo
    run_webcam_demo()
"""
import os, json, time, urllib.request
from base64 import b64decode, b64encode
from collections import deque

import numpy as np
import cv2
import torch

from src.utils.config import load_config, resolve
from src.preprocessing.normalize_landmarks import normalize_landmarks, fix_aspect
from src.models.model1_mlp.model import load_checkpoint

# ======================= JavaScript chạy trong trình duyệt (bật webcam, hiển thị overlay) =======================
JS_CODE = """
var video = null, wrapper = null, overlay = null, statusEl = null, captureCanvas = null;
var mediaStream = null, shutdown = false, boxW = 640, boxH = 480;
async function startStream(mirror, reqW, reqH) {
  if (wrapper !== null) { return 'already'; }
  shutdown = false;
  wrapper = document.createElement('div');
  document.body.appendChild(wrapper);
  var stopBtn = document.createElement('button');
  stopBtn.textContent = '⏹ Dừng camera';
  stopBtn.style.cssText = 'padding:6px 14px;font-size:14px;cursor:pointer;margin-bottom:6px;';
  stopBtn.onclick = function() { shutdown = true; };
  wrapper.appendChild(stopBtn);
  statusEl = document.createElement('div');
  statusEl.style.cssText = 'font:14px monospace;margin:4px 0;';
  statusEl.textContent = 'Đang khởi động camera...';
  wrapper.appendChild(statusEl);
  video = document.createElement('video');
  video.setAttribute('playsinline', '');
  video.muted = true;
  mediaStream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user', width: reqW, height: reqH}});
  video.srcObject = mediaStream;
  await video.play();
  boxW = video.videoWidth; boxH = video.videoHeight;
  var holder = document.createElement('div');
  holder.style.cssText = 'position:relative;width:' + boxW + 'px;height:' + boxH + 'px;max-width:100%;';
  video.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;' + (mirror ? 'transform:scaleX(-1);' : '');
  overlay = document.createElement('img');
  overlay.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;';
  holder.appendChild(video); holder.appendChild(overlay);
  wrapper.appendChild(holder);
  captureCanvas = document.createElement('canvas');
  captureCanvas.width = boxW; captureCanvas.height = boxH;
  google.colab.output.setIframeHeight(document.documentElement.scrollHeight + 20, true);
  return boxW + 'x' + boxH;
}
function stopStream() {
  if (mediaStream) { mediaStream.getTracks().forEach(function(t) { t.stop(); }); }
  if (wrapper) { wrapper.remove(); }
  video = wrapper = overlay = statusEl = captureCanvas = mediaStream = null;
  shutdown = false;
}
function streamFrame(overlayUrl, statusText, mirror) {
  if (shutdown) { stopStream(); return Promise.resolve(''); }
  if (overlay && overlayUrl) { overlay.src = overlayUrl; }
  if (statusEl && statusText) { statusEl.textContent = statusText; }
  return new Promise(function(resolve) {
    window.requestAnimationFrame(function() {
      var ctx = captureCanvas.getContext('2d');
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      if (mirror) { ctx.translate(boxW, 0); ctx.scale(-1, 1); }
      ctx.drawImage(video, 0, 0, boxW, boxH);
      resolve(captureCanvas.toDataURL('image/jpeg', 0.7));
    });
  });
}
"""

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/1/hand_landmarker.task")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def js_to_image(js_reply):
    """data-URL JPEG (từ JS) -> ảnh OpenCV BGR."""
    arr = np.frombuffer(b64decode(js_reply.split(",")[1]), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def overlay_to_data_url(overlay_bgra):
    ok, png = cv2.imencode(".png", overlay_bgra)
    return "data:image/png;base64," + b64encode(png.tobytes()).decode("utf-8")


def draw_text_box(img, text, org, color_bg, scale=1.0, thickness=2):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - base - 6), (x + tw + 10, y + 4), color_bg, -1)
    cv2.putText(img, text, (x + 5, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255, 255), thickness, cv2.LINE_AA)


class Model1Predictor:
    """Landmarks (21,3) của MediaPipe -> xác suất 28 lớp. Dùng đúng chuẩn hóa như lúc train."""

    def __init__(self, checkpoint_path, tta_mirror=False):
        self.model, ckpt = load_checkpoint(checkpoint_path, "cpu")
        self.classes = ckpt["classes"]
        self.mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
        self.scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
        self.tta_mirror = tta_mirror

    def predict_proba(self, pts_xyz, img_w, img_h):
        pts = fix_aspect(pts_xyz, img_w, img_h)
        vec = normalize_landmarks(pts.reshape(1, 63))     # (1) chuẩn hóa hình học
        vec_s = (vec - self.mean) / self.scale            # (2) StandardScaler đã lưu
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec_s.astype(np.float32)))
            probs = torch.softmax(logits, dim=1)[0].numpy()

        if self.tta_mirror:
            from src.preprocessing.augment import mirror_hand
            vec_m = mirror_hand(vec)
            vec_m_s = (vec_m - self.mean) / self.scale
            with torch.no_grad():
                logits_m = self.model(torch.from_numpy(vec_m_s.astype(np.float32)))
                probs_m = torch.softmax(logits_m, dim=1)[0].numpy()
            probs = 0.5 * (probs + probs_m)

        return probs


def build_landmarker(model_path):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.exists(model_path):
        print("⬇️ Đang tải mô hình MediaPipe HandLandmarker...")
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
    return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5))


class FrameProcessor:
    def __init__(self, predictor, landmarker, conf_threshold, smooth_frames):
        self.pred, self.lm = predictor, landmarker
        self.thr = conf_threshold
        self.hist = deque(maxlen=smooth_frames)
        self.last_ts = 0

    def __call__(self, frame_bgr):
        """Trả về (overlay_bgra trong suốt, status_text)."""
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.time() * 1000), self.last_ts + 1); self.last_ts = ts
        res = self.lm.detect_for_video(mp_image, ts)
        if not res.hand_landmarks:
            self.hist.clear()
            return overlay, "Không thấy bàn tay"

        pts = np.array([[p.x, p.y, p.z] for p in res.hand_landmarks[0]], dtype=np.float32)
        self.hist.append(self.pred.predict_proba(pts, w, h))
        probs = np.mean(self.hist, axis=0)
        k = int(np.argmax(probs)); conf = float(probs[k])
        ok = conf >= self.thr
        letter = self.pred.classes[k] if ok else "?"
        color = (0, 200, 0, 255) if ok else (0, 140, 255, 255)

        px = (pts[:, 0] * w).astype(int); py = (pts[:, 1] * h).astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(overlay, (px[a], py[a]), (px[b], py[b]), (255, 255, 0, 255), 2, cv2.LINE_AA)
        for i in range(21):
            cv2.circle(overlay, (px[i], py[i]), 4, (0, 0, 255, 255), -1, cv2.LINE_AA)
        pad = 20
        x1, y1 = max(px.min() - pad, 0), max(py.min() - pad, 0)
        x2, y2 = min(px.max() + pad, w - 1), min(py.max() + pad, h - 1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        ty = y1 - 6 if y1 > 45 else y2 + 40
        draw_text_box(overlay, f"{letter}  {conf*100:.0f}%", (x1, ty), color, 1.1, 2)
        return overlay, f"Dự đoán: {letter} ({conf*100:.0f}%)"


def run_webcam_demo(mirror=None, max_seconds=None, tta_mirror=None):
    """Mở webcam trong Colab và nhận diện realtime. Hỗ trợ cả hai tay trái và phải.
    Tham số mặc định lấy từ configs/config.yaml."""
    from IPython.display import display, Javascript
    from google.colab.output import eval_js

    cfg = load_config(); dm = cfg["demo"]
    mirror = dm["mirror"] if mirror is None else mirror
    max_seconds = dm["max_seconds"] if max_seconds is None else max_seconds
    tta_mirror = dm.get("tta_mirror", False) if tta_mirror is None else tta_mirror

    predictor = Model1Predictor(resolve(cfg["model1_mlp"]["checkpoint"]), tta_mirror=tta_mirror)
    print("✅ Đã nạp Model 1. Các lớp:", predictor.classes)
    proc = FrameProcessor(predictor, build_landmarker(str(resolve(dm["hand_model_path"]))),
                          dm["conf_threshold"], dm["smooth_frames"])

    display(Javascript(JS_CODE))
    size = eval_js(f"startStream({str(mirror).lower()}, {dm['cam_width']}, {dm['cam_height']})")
    print("📷 Camera đã bật:", size, "- bấm nút ⏹ trên video để dừng.")

    overlay_url, status = "", "Đang chạy..."
    t0, n, fps = time.time(), 0, 0.0
    try:
        while True:
            reply = eval_js(f"streamFrame({json.dumps(overlay_url)}, {json.dumps(status)}, {str(mirror).lower()})")
            if not reply:
                break
            overlay, msg = proc(js_to_image(reply))
            overlay_url = overlay_to_data_url(overlay)
            n += 1
            if n % 5 == 0:
                fps = n / (time.time() - t0)
            status = f"{msg}   |   ~{fps:.1f} FPS"
            if max_seconds and (time.time() - t0) > max_seconds:
                print("⏱ Hết thời gian cho phép, tự động dừng."); break
    except Exception as e:
        print("⚠️ Dừng do lỗi:", repr(e))
    finally:
        eval_js("stopStream()")
        print(f"Đã dừng. Tổng {n} khung hình, ~{fps:.1f} FPS.")
