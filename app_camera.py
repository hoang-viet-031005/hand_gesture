#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
ỨNG DỤNG CAMERA TEST NHẬN DIỆN CHỮ CÁI BẰNG CỬ CHỈ TAY (ASL SIGN LANGUAGE)
Mô hình: MediaPipe HandLandmarker + PyTorch MLP (Model 1)
Chức năng:
  - Mở camera toàn màn hình thời gian thực
  - Nhận diện trực tiếp chữ cái đang làm (A-Z, space, del)
  - Vẽ khung xương bàn tay (Skeleton) & Hộp bao quanh (Bounding box)
  - Hiển thị chữ cái nhận diện và độ tin cậy trực tiếp trên bàn tay và góc màn hình
  - Bảng Top 3 dự đoán chính xác nhất
  - Phím tắt: [M] Lật gương camera | [H] Ẩn/Hiện khung xương | [Q]/[ESC] Thoát
=============================================================================
"""

import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font/mã hóa cp1252
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tắt log rác từ TensorFlow / OneDNN
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import torch

# Thiết lập đường dẫn thư mục gốc project
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import các module từ source code đã huấn luyện
try:
    from src.models.model1_mlp.model import LandmarkMLP, load_checkpoint
    from src.preprocessing.normalize_landmarks import normalize_landmarks, fix_aspect
    from src.utils.config import load_config, resolve
except ImportError:
    pass

# URL tải mô hình MediaPipe HandLandmarker nếu chưa có
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Các cặp liên kết khớp xương tay của MediaPipe (21 điểm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),        # Ngón trỏ
    (5, 9), (9, 10), (10, 11), (11, 12),   # Ngón giữa
    (9, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (13, 17), (17, 18), (18, 19), (19, 20),# Ngón út
    (0, 17)                                # Lòng bàn tay
]


class ModelPredictor:
    """Nạp trọng số checkpoint Model 1 (MLP) và chuẩn hóa landmarks theo đúng lúc huấn luyện."""

    def __init__(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {checkpoint_path}")

        print(f"📦 Đang nạp mô hình từ: {checkpoint_path}...")
        self.model, ckpt = load_checkpoint(str(checkpoint_path), device="cpu")
        self.classes = ckpt["classes"]
        self.mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
        self.scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
        print(f"✅ Đã nạp thành công! 28 nhãn nhận diện: {self.classes}")

    def predict(self, pts_xyz: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """
        Input: pts_xyz shape (21, 3) tọa độ MediaPipe
        Output: mảng xác suất cho 28 lớp
        """
        pts = fix_aspect(pts_xyz, img_w, img_h)
        vec = normalize_landmarks(pts.reshape(1, 63))
        vec = (vec - self.mean) / self.scale
        with torch.no_grad():
            logits = self.model(torch.from_numpy(vec.astype(np.float32)))
            probs = torch.softmax(logits, dim=1)[0].numpy()
        return probs


def build_hand_landmarker(model_path: str):
    """Khởi tạo MediaPipe HandLandmarker."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_p = Path(model_path)
    if not model_p.exists():
        alt_p = CURRENT_DIR / model_path
        if alt_p.exists():
            model_p = alt_p
        else:
            print(f"⬇️ Không tìm thấy {model_p.name}, đang tự động tải...")
            urllib.request.urlretrieve(HAND_MODEL_URL, str(model_p))
            print("✅ Tải xong!")

    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_p)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


class CameraTestApp:
    def __init__(
        self,
        camera_id: int = 0,
        conf_threshold: float = 0.55,
        smooth_frames: int = 4,
        mirror: bool = True
    ):
        self.camera_id = camera_id
        self.conf_threshold = conf_threshold
        self.smooth_frames = smooth_frames
        self.mirror = mirror                # Lật gương camera (mặc định BẬT)
        self.show_skeleton = True           # Bật/tắt khung xương

        # Đường dẫn file checkpoint & hand landmarker
        ckpt_path = CURRENT_DIR / "checkpoints" / "model1_mlp_best.pth"
        if not ckpt_path.exists():
            ckpt_path = Path("checkpoints/model1_mlp_best.pth")

        task_path = CURRENT_DIR / "hand_landmarker.task"
        if not task_path.exists():
            task_path = Path("hand_landmarker.task")

        # Khởi tạo mô hình
        self.predictor = ModelPredictor(ckpt_path)
        self.landmarker = build_hand_landmarker(str(task_path))

        # Bộ nhớ đệm làm mượt kết quả
        self.prob_history = deque(maxlen=self.smooth_frames)
        self.last_timestamp_ms = 0

    def process_frame(self, frame_bgr: np.ndarray):
        """Xử lý từng khung hình: nhận diện bàn tay và hiển thị kết quả trực tiếp."""
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]

        # 1. Lật gương nếu chế độ mirror đang bật
        if self.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)

        # 2. Chuyển sang RGB để đưa vào MediaPipe
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp tăng dần
        now_ms = max(int(time.time() * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = now_ms

        res = self.landmarker.detect_for_video(mp_image, now_ms)

        detected_letter = "?"
        confidence = 0.0
        top3 = []
        hand_box = None

        if res.hand_landmarks and len(res.hand_landmarks) > 0:
            hand_lms = res.hand_landmarks[0]
            pts = np.array([[p.x, p.y, p.z] for p in hand_lms], dtype=np.float32)

            # Dự đoán xác suất 28 lớp
            raw_probs = self.predictor.predict(pts, w, h)
            self.prob_history.append(raw_probs)
            probs = np.mean(self.prob_history, axis=0)

            # Lấy top 3
            top3_idx = np.argsort(probs)[::-1][:3]
            top3 = [(self.predictor.classes[i], float(probs[i])) for i in top3_idx]

            best_idx = top3_idx[0]
            conf = float(probs[best_idx])
            confidence = conf

            if conf >= self.conf_threshold:
                detected_letter = self.predictor.classes[best_idx]
            else:
                detected_letter = "?"

            # Tọa độ pixel để vẽ
            px = np.clip((pts[:, 0] * w).astype(int), 0, w - 1)
            py = np.clip((pts[:, 1] * h).astype(int), 0, h - 1)

            # Bounding box của bàn tay
            pad = 25
            x1 = max(int(np.min(px)) - pad, 0)
            y1 = max(int(np.min(py)) - pad, 0)
            x2 = min(int(np.max(px)) + pad, w - 1)
            y2 = min(int(np.max(py)) + pad, h - 1)
            hand_box = (x1, y1, x2, y2)

            # Vẽ khung xương tay (Skeleton)
            if self.show_skeleton:
                # Các đoạn nối xương tay
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame_bgr, (px[a], py[a]), (px[b], py[b]), (0, 255, 255), 2, cv2.LINE_AA)
                # Điểm khớp nối
                for i in range(21):
                    pt_color = (0, 165, 255) if i in [4, 8, 12, 16, 20] else (0, 255, 0)
                    cv2.circle(frame_bgr, (px[i], py[i]), 5, pt_color, -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, (px[i], py[i]), 6, (255, 255, 255), 1, cv2.LINE_AA)

        else:
            self.prob_history.clear()

        # 3. Vẽ Bounding box và nhãn chữ nổi bật trên bàn tay
        if hand_box is not None:
            x1, y1, x2, y2 = hand_box
            is_valid = detected_letter != "?"
            box_color = (0, 255, 0) if is_valid else (0, 140, 255)

            # Vẽ viền 4 góc
            self._draw_corner_box(frame_bgr, x1, y1, x2, y2, box_color, thickness=2, corner_len=25)

            # Badge chữ ngay trên đầu bàn tay
            badge_text = f" {detected_letter}  {confidence*100:.0f}% "
            badge_y = max(y1 - 10, 35)
            self._draw_badge(frame_bgr, badge_text, (x1, badge_y), box_color)

        # 4. Vẽ bảng thông tin gọn gàng ở góc màn hình (không che camera)
        self._draw_compact_hud(frame_bgr, detected_letter, confidence, top3)

        return frame_bgr

    def _draw_corner_box(self, img, x1, y1, x2, y2, color, thickness=2, corner_len=25):
        """Vẽ viền góc phong cách camera hiện đại."""
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        # Góc trên - trái
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        # Góc trên - phải
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)
        # Góc dưới - trái
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness + 1)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness + 1)
        # Góc dưới - phải
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness + 1)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness + 1)

    def _draw_badge(self, img, text, pos, bg_color):
        """Vẽ badge chữ nổi bật phía trên bàn tay."""
        x, y = pos
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)
        cv2.rectangle(img, (x, y - th - 8), (x + tw + 10, y + 4), bg_color, -1)
        cv2.putText(img, text, (x + 5, y - 4), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)

    def _draw_compact_hud(self, img, letter, conf, top3):
        """Vẽ bảng thông tin nhỏ gọn ở góc trên cùng, hoàn toàn giải phóng không gian camera."""
        h, w = img.shape[:2]

        # --- BẢNG GÓC TRÊN TRÁI: KẾT QUẢ CHỮ ĐANG LÀM ---
        overlay = img.copy()
        cv2.rectangle(overlay, (15, 15), (250, 105), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        cv2.rectangle(img, (15, 15), (250, 105), (60, 60, 70), 1)

        cv2.putText(img, "CHU CAI DANG NHAN DIEN", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        char_str = letter if letter != "?" else "--"
        char_color = (0, 255, 0) if letter != "?" else (130, 130, 130)
        cv2.putText(img, char_str, (25, 90), cv2.FONT_HERSHEY_DUPLEX, 1.8, char_color, 3, cv2.LINE_AA)

        conf_str = f"Tin cay: {conf*100:.0f}%" if letter != "?" else "Chua ro"
        cv2.putText(img, conf_str, (120, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        # --- BẢNG GÓC TRÊN PHẢI: TOP 3 DỰ ĐOÁN ---
        if top3:
            overlay2 = img.copy()
            p2_w = 200
            p2_x = w - p2_w - 15
            cv2.rectangle(overlay2, (p2_x, 15), (w - 15, 105), (18, 18, 22), -1)
            cv2.addWeighted(overlay2, 0.75, img, 0.25, 0, img)
            cv2.rectangle(img, (p2_x, 15), (w - 15, 105), (60, 60, 70), 1)

            cv2.putText(img, "TOP 3 XAC SUAT", (p2_x + 10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            for i, (c, p) in enumerate(top3):
                row_y = 58 + i * 20
                c_color = (0, 255, 0) if i == 0 else (200, 200, 200)
                cv2.putText(img, f"{i+1}. {c}: {p*100:.0f}%", (p2_x + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, c_color, 1, cv2.LINE_AA)

        # --- DÒNG HƯỚNG DẪN DƯỚI ĐÁY MÀN HÌNH (GỌN GÀNG, KHÔNG CHE KHUNG HÌNH) ---
        guide_str = "[M] Lat guong (Mirror)  |  [H] Khung xuong (Skeleton)  |  [Q/ESC] Thoat"
        cv2.putText(img, guide_str, (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def run(self):
        """Khởi động camera và vòng lặp nhận diện."""
        print(f"\n🎥 Đang mở Camera ID {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)

        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"❌ LỖI: Không tìm thấy camera! Vui lòng kiểm tra webcam hoặc chạy: python app_camera.py --cam 1")
            return

        # Độ phân giải
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        window_name = "Camera Test Nhan Dien Cu Chi Tay (Model 1 - ASL)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 720)

        print("\n" + "=" * 60)
        print("🚀 CAMERA TEST NHẬN DIỆN CỬ CHỈ TAY ĐANG CHẠY!")
        print("=" * 60)
        print("👉 Hướng dẫn:")
        print("   • Giơ bàn tay vào khung hình để camera bắt điểm nhận diện.")
        print("   • Chữ cái nhận diện và độ tin cậy hiển thị trực tiếp trên bàn tay.")
        print("   • [M]    : Bật/Tắt lật gương camera (Mirror).")
        print("   • [H]    : Ẩn/Hiện khung xương khớp tay.")
        print("   • [Q]/ESC: Thoát camera.")
        print("=" * 60 + "\n")

        fps_calc = 0.0
        frame_count = 0
        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                output_frame = self.process_frame(frame)

                # Tính FPS
                frame_count += 1
                if frame_count % 10 == 0:
                    fps_calc = frame_count / (time.time() - t_start)

                # Hiển thị FPS
                cv2.putText(output_frame, f"FPS: {fps_calc:.1f}", (output_frame.shape[1] - 85, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow(window_name, output_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n🛑 Đã tắt camera.")
                    break
                elif key in [ord('m'), ord('M')]:
                    self.mirror = not self.mirror
                    print(f"🪞 Đã chuyển Lật gương: {'BẬT' if self.mirror else 'TẮT'}")
                elif key in [ord('h'), ord('H')]:
                    self.show_skeleton = not self.show_skeleton
                    print(f"🦴 Khung xương tay: {'HIỆN' if self.show_skeleton else 'ẨN'}")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                self.landmarker.close()
            except Exception:
                pass


def main():
    import argparse
    parser = argparse.ArgumentParser(description="App Camera Test Nhan Dien Cu Chi Tay")
    parser.add_argument("--cam", type=int, default=0, help="ID Camera (mặc định: 0)")
    parser.add_argument("--conf", type=float, default=0.55, help="Ngưỡng tin cậy (mặc định: 0.55)")
    parser.add_argument("--no-mirror", action="store_true", help="Tắt lật gương")
    args = parser.parse_args()

    app = CameraTestApp(
        camera_id=args.cam,
        conf_threshold=args.conf,
        mirror=not args.no_mirror
    )
    app.run()


# Tương thích ngược
CameraApp = CameraTestApp


if __name__ == "__main__":
    main()
