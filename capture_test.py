#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
CÔNG CỤ CHỤP ẢNH & TEST 10 TẤM / 1 CHỮ CÁI TẬP TRUNG VÀO BÀN TAY
- Cắt (crop) cận cảnh chỉ tập trung vào bàn tay kèm khung xương và nhãn nhận diện
- Lưu ảnh trực tiếp vào: T:\\anhtest\\<CHU_CAI>\\
=============================================================================
"""

import os
import sys
import time
import csv
from pathlib import Path
from collections import deque

# Bật hỗ trợ Unicode / UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tắt log rác TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import torch

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from app_camera import ModelPredictor, build_hand_landmarker, HAND_CONNECTIONS

# Thư mục lưu trữ chính trên ổ T
OUTPUT_BASE = Path("T:/anhtest")
if not OUTPUT_BASE.parent.exists():
    OUTPUT_BASE = CURRENT_DIR / "anhtest"

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Danh sách 28 nhãn
CLASSES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'del', 'space'
]

IMAGES_PER_CLASS = 10


def play_click():
    """Phát âm thanh chụp ảnh nhẹ."""
    try:
        import winsound
        winsound.Beep(1200, 70)
    except Exception:
        pass


class HandCaptureTestApp:
    def __init__(self, camera_id: int = 0, mirror: bool = True):
        self.camera_id = camera_id
        self.mirror = mirror
        self.show_skeleton = True

        # Nạp mô hình
        ckpt_path = CURRENT_DIR / "checkpoints" / "model1_mlp_best.pth"
        task_path = CURRENT_DIR / "hand_landmarker.task"
        self.predictor = ModelPredictor(ckpt_path)
        self.landmarker = build_hand_landmarker(str(task_path))

        self.prob_history = deque(maxlen=4)
        self.last_ts = 0

        # Quản lý lớp chụp
        self.current_idx = 0
        self.auto_capture = False       # Chế độ tự động chụp
        self.last_capture_time = 0.0
        self.flash_until = 0.0
        self.auto_delay = 0.9           # Giãn cách giữa 2 lần chụp (giây)
        self.last_saved_thumb = None    # Thumbnail ảnh vừa chụp để hiển thị
        self.thumb_until = 0.0

        # Tạo sẵn các thư mục
        for c in CLASSES:
            (OUTPUT_BASE / c).mkdir(parents=True, exist_ok=True)

        # File CSV ghi nhận kết quả test
        self.csv_file = OUTPUT_BASE / "ket_qua_test.csv"
        if not self.csv_file.exists():
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Target_Class", "Image_Path", "Predicted_Class", "Confidence", "Is_Correct"])

        # Tìm chữ cái đầu tiên chưa đủ 10 ảnh
        self._find_first_incomplete()

    def _find_first_incomplete(self):
        """Tự động chuyển tới chữ cái đầu tiên chưa đủ 10 ảnh."""
        for i, c in enumerate(CLASSES):
            if self._count_photos(c) < IMAGES_PER_CLASS:
                self.current_idx = i
                return
        self.current_idx = 0

    def _count_photos(self, class_name: str) -> int:
        """Đếm số ảnh .jpg đã chụp của chữ cái."""
        folder = OUTPUT_BASE / class_name
        if not folder.exists():
            return 0
        return len(list(folder.glob("*.jpg")))

    def capture_hand_image(
        self,
        base_frame_bgr: np.ndarray,
        px: np.ndarray,
        py: np.ndarray,
        box: tuple,
        pred_class: str,
        conf: float
    ):
        """
        Tạo ảnh cận cảnh tập trung vào bàn tay sạch (Chỉ có bàn tay, không vẽ khung xương & nhãn):
        1. Cắt trực tiếp từ khung hình camera gốc (base_frame_bgr).
        2. Thêm lề pad = 20px vừa vặn quanh bàn tay.
        3. Lưu vào T:\\anhtest\\<TARGET>\\<TARGET>_Tien_XX.jpg
        """
        target = CLASSES[self.current_idx]
        folder = OUTPUT_BASE / target
        current_count = self._count_photos(target)

        if current_count >= IMAGES_PER_CLASS:
            print(f"✅ Chữ '{target}' đã đủ {IMAGES_PER_CLASS} ảnh!")
            return False

        h, w = base_frame_bgr.shape[:2]
        x1, y1, x2, y2 = box

        # Cắt cận cảnh bàn tay hoàn toàn sạch từ camera gốc
        pad = 20
        crop_x1 = max(0, x1 - pad)
        crop_y1 = max(0, y1 - pad)
        crop_x2 = min(w, x2 + pad)
        crop_y2 = min(h, y2 + pad)

        clean_cropped = base_frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

        if clean_cropped.size == 0:
            print("⚠️ Lỗi vùng cắt không hợp lệ!")
            return False

        # 6. Lưu file vào ổ T với định dạng tên mới: <TARGET>_Tien_XX.jpg
        file_num = current_count + 1
        filename = f"{target}_Tien_{file_num:02d}.jpg"
        save_path = folder / filename

        cv2.imwrite(str(save_path), clean_cropped, [cv2.IMWRITE_JPEG_QUALITY, 96])

        # Ghi log kết quả
        is_correct = (pred_class == target)
        with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                target,
                str(save_path),
                pred_class,
                f"{conf*100:.1f}%",
                "DUNG" if is_correct else "SAI"
            ])

        # Âm thanh & hiệu ứng
        play_click()
        self.flash_until = time.time() + 0.15
        self.last_capture_time = time.time()

        # Lưu thumbnail nhỏ để hiển thị xem trước trên camera
        th_h, th_w = clean_cropped.shape[:2]
        thumb_scale = 120.0 / max(th_h, th_w)
        self.last_saved_thumb = cv2.resize(clean_cropped, (int(th_w * thumb_scale), int(th_h * thumb_scale)))
        self.thumb_until = time.time() + 1.2

        print(f"📸 [ĐÃ LƯU ẢNH TAY SẠCH] Lưu: {save_path.name} | Mục tiêu: [{target}] | Đoán: [{pred_class}] ({conf*100:.0f}%) -> {'✅ ĐÚNG' if is_correct else '❌ SAI'}")

        # Tự động chuyển chữ tiếp theo khi đủ 10 tấm
        if file_num >= IMAGES_PER_CLASS:
            print(f"🎉 HOÀN THÀNH CHỮ [{target}]! Đã chụp đủ {IMAGES_PER_CLASS} ảnh cận cảnh.")
            if self.current_idx < len(CLASSES) - 1:
                self.current_idx += 1
                next_target = CLASSES[self.current_idx]
                print(f"👉 Chuyển sang chữ tiếp theo: [{next_target}]")
            else:
                print("🏆 ĐÃ CHỤP XONG TẤT CẢ 28 CHỮ CÁI!")

        return True

    def run(self):
        print(f"\n🎥 Đang mở camera ID {self.camera_id}...")
        cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not cap.isOpened():
            cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            print(f"❌ Không mở được camera ID {self.camera_id}!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        win_name = "Chup Anh Can Canh Tay (10 Tam / 1 Chu Cai -> T:\\anhtest)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1000, 750)

        print("\n" + "=" * 65)
        print("🚀 CÔNG CỤ CHỤP ẢNH TẬP TRUNG BÀN TAY (CROP FOCUS)")
        print("=" * 65)
        print(f"📁 Thư mục lưu ảnh: {OUTPUT_BASE}")
        print("👉 Hướng dẫn phím tắt:")
        print("   • [SPACE]    : Bấm để chụp 1 ảnh cận cảnh tay ngay lập tức.")
        print("   • [A]        : Bật/Tắt tự động chụp (Auto Capture mỗi 0.9s).")
        print("   • [N]        : Chuyển sang chữ cái kế tiếp.")
        print("   • [P]        : Quay lại chữ cái trước đó.")
        print("   • [R]        : Xóa chụp lại từ đầu chữ cái này.")
        print("   • [M]        : Lật gương camera (Mirror).")
        print("   • [H]        : Ẩn/Hiện khung xương tay.")
        print("   • [Q] / ESC  : Thoát.")
        print("=" * 65 + "\n")

        import mediapipe as mp

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                clean_frame = frame.copy()
                h, w = frame.shape[:2]

                # MediaPipe inference
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now_ms = max(int(time.time() * 1000), self.last_ts + 1)
                self.last_ts = now_ms

                res = self.landmarker.detect_for_video(mp_img, now_ms)

                detected_letter = "?"
                conf = 0.0
                hand_found = False
                hand_box = None
                px, py = None, None

                if res.hand_landmarks and len(res.hand_landmarks) > 0:
                    hand_found = True
                    hand_lms = res.hand_landmarks[0]
                    pts = np.array([[p.x, p.y, p.z] for p in hand_lms], dtype=np.float32)

                    raw_probs = self.predictor.predict(pts, w, h)
                    self.prob_history.append(raw_probs)
                    probs = np.mean(self.prob_history, axis=0)

                    best_k = int(np.argmax(probs))
                    conf = float(probs[best_k])
                    detected_letter = self.predictor.classes[best_k]

                    # Tọa độ pixel
                    px = np.clip((pts[:, 0] * w).astype(int), 0, w - 1)
                    py = np.clip((pts[:, 1] * h).astype(int), 0, h - 1)

                    pad = 22
                    x1 = max(int(np.min(px)) - pad, 0)
                    y1 = max(int(np.min(py)) - pad, 0)
                    x2 = min(int(np.max(px)) + pad, w - 1)
                    y2 = min(int(np.max(py)) + pad, h - 1)
                    hand_box = (x1, y1, x2, y2)

                    # Vẽ khung xương tay trên màn hình xem trước
                    if self.show_skeleton:
                        for a, b in HAND_CONNECTIONS:
                            cv2.line(frame, (px[a], py[a]), (px[b], py[b]), (0, 255, 255), 2, cv2.LINE_AA)
                        for i in range(21):
                            cv2.circle(frame, (px[i], py[i]), 5, (0, 255, 0), -1, cv2.LINE_AA)
                else:
                    self.prob_history.clear()

                target = CLASSES[self.current_idx]
                photo_count = self._count_photos(target)

                # Vẽ Bounding Box và huy hiệu nhận diện
                if hand_box is not None:
                    x1, y1, x2, y2 = hand_box
                    box_color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    badge_str = f" {detected_letter} ({conf*100:.0f}%) "
                    (tw, th), base = cv2.getTextSize(badge_str, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
                    badge_h = th + base + 10
                    badge_w = max(tw + 10, x2 - x1)
                    badge_y1 = max(y1 - badge_h, 0)
                    badge_y2 = y1
                    badge_x1 = x1
                    badge_x2 = min(x1 + badge_w, w)
                    cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), box_color, -1)
                    cv2.putText(frame, badge_str, (badge_x1 + 4, badge_y2 - base - 3), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

                # Tự động chụp nếu bật chế độ Auto
                cur_t = time.time()
                if self.auto_capture and hand_found and photo_count < IMAGES_PER_CLASS:
                    if cur_t - self.last_capture_time >= self.auto_delay:
                        self.capture_hand_image(clean_frame, px, py, hand_box, detected_letter, conf)

                # Vẽ bảng HUD phía trên
                self._draw_capture_hud(frame, target, photo_count, detected_letter, conf)

                # Hiển thị thumbnail ảnh vừa chụp ở góc dưới bên phải
                if self.last_saved_thumb is not None and cur_t < self.thumb_until:
                    th_h, th_w = self.last_saved_thumb.shape[:2]
                    tx1 = w - th_w - 20
                    ty1 = h - th_h - 45
                    frame[ty1:ty1 + th_h, tx1:tx1 + th_w] = self.last_saved_thumb
                    cv2.rectangle(frame, (tx1, ty1), (tx1 + th_w, ty1 + th_h), (0, 255, 0), 2)
                    cv2.putText(frame, "ANH DA LUU", (tx1, ty1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                # Chớp trắng khi chụp
                if cur_t < self.flash_until:
                    white_flash = np.full_like(frame, 255)
                    cv2.addWeighted(white_flash, 0.45, frame, 0.55, 0, frame)

                cv2.imshow(win_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("\n🛑 Đã thoát công cụ chụp.")
                    break
                elif key == 32:  # Space
                    if hand_found and hand_box is not None:
                        self.capture_hand_image(clean_frame, px, py, hand_box, detected_letter, conf)
                    else:
                        print("⚠️ Chưa phát hiện bàn tay! Hãy giơ tay vào khung hình để chụp.")
                elif key in [ord('a'), ord('A')]:
                    self.auto_capture = not self.auto_capture
                    print(f"⚡ Tự động chụp: {'BẬT' if self.auto_capture else 'TẮT'}")
                elif key in [ord('n'), ord('N')]:
                    if self.current_idx < len(CLASSES) - 1:
                        self.current_idx += 1
                        print(f"👉 Chuyển sang chữ: [{CLASSES[self.current_idx]}]")
                elif key in [ord('p'), ord('P')]:
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        print(f"👈 Quay lại chữ: [{CLASSES[self.current_idx]}]")
                elif key in [ord('r'), ord('R')]:
                    self._reset_current_class()
                elif key in [ord('m'), ord('M')]:
                    self.mirror = not self.mirror
                    print(f"🪞 Lật gương: {'BẬT' if self.mirror else 'TẮT'}")
                elif key in [ord('h'), ord('H')]:
                    self.show_skeleton = not self.show_skeleton

        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                self.landmarker.close()
            except Exception:
                pass
            self._print_final_summary()

    def _reset_current_class(self):
        """Xóa toàn bộ ảnh đã chụp của chữ cái hiện tại để chụp lại."""
        target = CLASSES[self.current_idx]
        folder = OUTPUT_BASE / target
        files = list(folder.glob("*.jpg"))
        for f in files:
            try:
                f.unlink()
            except Exception:
                pass
        print(f"🗑 Đã xóa {len(files)} ảnh của chữ [{target}]. Bắt đầu chụp lại!")

    def _draw_capture_hud(self, img, target: str, count: int, pred: str, conf: float):
        """Vẽ thanh thông tin mục tiêu chụp và tiến độ."""
        h, w = img.shape[:2]

        bar_h = 95
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.line(img, (0, bar_h), (w, bar_h), (0, 200, 255), 2)

        # Chữ mục tiêu
        cv2.putText(img, f"MUC TIEU: [ {target} ]", (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)

        # Tiến độ
        pct_color = (0, 255, 0) if count >= IMAGES_PER_CLASS else (255, 255, 255)
        status_txt = f"{count} / {IMAGES_PER_CLASS} ANH" if count < IMAGES_PER_CLASS else f"{count} / {IMAGES_PER_CLASS} [DA XONG]"
        cv2.putText(img, status_txt, (280, 42), cv2.FONT_HERSHEY_DUPLEX, 0.9, pct_color, 2, cv2.LINE_AA)

        # Thanh tiến trình
        pb_x, pb_y, pb_w, pb_h = 280, 56, 220, 14
        cv2.rectangle(img, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (50, 50, 60), -1)
        fill_w = int(pb_w * (count / IMAGES_PER_CLASS))
        cv2.rectangle(img, (pb_x, pb_y), (pb_x + fill_w, pb_y + pb_h), (0, 255, 120), -1)
        cv2.rectangle(img, (pb_x, pb_y), (pb_x + pb_w, pb_y + pb_h), (120, 120, 130), 1)

        # Dự đoán hiện tại
        match = (pred == target)
        match_color = (0, 255, 0) if match else (0, 140, 255)
        pred_txt = f"Du doan: {pred} ({conf*100:.0f}%)" if pred != "?" else "Chua thay tay"
        cv2.putText(img, pred_txt, (530, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, match_color, 2, cv2.LINE_AA)

        # Trạng thái Auto
        mode_txt = "AUTO: [BAT]" if self.auto_capture else "AUTO: [TAT - Bam Space de chup]"
        mode_color = (0, 255, 0) if self.auto_capture else (180, 180, 180)
        cv2.putText(img, mode_txt, (530, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 1, cv2.LINE_AA)

        # Đường dẫn thư mục
        cv2.putText(img, f"Thu muc: T:\\anhtest\\{target}\\", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)

        # Dòng hướng dẫn ở đáy
        guide = "[Space] Chup can canh | [A] Tu dong | [N] Ke tiep | [P] Quay lai | [R] Chup lai | [M] Guong | [Q] Thoat"
        cv2.putText(img, guide, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _print_final_summary(self):
        """In tổng kết."""
        print("\n" + "=" * 65)
        print(f"📊 BÁO CÁO TỔNG KẾT BỘ ẢNH TEST TẠI: {OUTPUT_BASE}")
        print("=" * 65)
        total_photos = 0
        for c in CLASSES:
            cnt = self._count_photos(c)
            total_photos += cnt
            bar = "█" * cnt + "-" * (IMAGES_PER_CLASS - cnt)
            print(f"  • Chữ [{c:^5}]: {cnt:2d}/{IMAGES_PER_CLASS} ảnh  [{bar}]")
        print("-" * 65)
        print(f"👉 Tổng số ảnh đã thu thập: {total_photos} / {len(CLASSES) * IMAGES_PER_CLASS} ảnh")
        print(f"👉 File nhật ký chi tiết: {self.csv_file}")
        print("=" * 65 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chụp ảnh cận cảnh tay 10 tấm / 1 chữ cái lưu vào ổ T")
    parser.add_argument("--cam", type=int, default=0, help="ID camera")
    parser.add_argument("--no-mirror", action="store_true", help="Tắt lật gương")
    args = parser.parse_args()

    app = HandCaptureTestApp(camera_id=args.cam, mirror=not args.no_mirror)
    app.run()


if __name__ == "__main__":
    main()
