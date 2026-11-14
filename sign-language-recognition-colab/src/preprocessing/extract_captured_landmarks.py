#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trích xuất 21 landmarks bàn tay (63 tọa độ 3D: r0..r62) từ dữ liệu tự chụp
(data/raw/custom_captured) bằng MediaPipe Tasks HandLandmarker.

Tính năng:
- Quét đệ quy toàn bộ ảnh trong các thư mục {Member}_dataset.
- Trích xuất toàn khung hình (không crop).
- Áp dụng Adaptive Border Padding (50px constant black, 50px replicate, 80px replicate)
  nếu ảnh chụp quá sát cạnh viền hoặc khuất cổ tay.
- Ánh xạ ngược tọa độ (x, y, z) từ ảnh đệm về đúng tỷ lệ ảnh gốc.
- Xuất file: data/landmarks/landmarks_captured.csv (r0..r62, label, source_file, member).
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import cv2
import csv
import glob
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


def build_extractor(model_path: str):
    """Khởi tạo HandLandmarker ở chế độ IMAGE."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_with_padding(landmarker, img_bgr):
    """
    Trích xuất landmark với cơ chế Adaptive Border Padding:
    1. Thử trực tiếp trên ảnh gốc.
    2. Nếu không thấy tay, thử đệm viền đen 50px.
    3. Nếu vẫn không thấy, thử đệm viền replicate 50px rồi 80px.
    Ánh xạ ngược tọa độ chuẩn hóa về hệ quy chiếu ảnh gốc.
    """
    h, w = img_bgr.shape[:2]

    # Chiến lược padding: (pad_top, pad_bottom, pad_left, pad_right, border_type)
    strategies = [
        (0, 0, 0, 0, None),
        (50, 50, 50, 50, cv2.BORDER_CONSTANT),
        (50, 50, 50, 50, cv2.BORDER_REPLICATE),
        (80, 80, 80, 80, cv2.BORDER_REPLICATE),
    ]

    for pt, pb, pl, pr, b_type in strategies:
        if pt == 0:
            padded_bgr = img_bgr
        else:
            padded_bgr = cv2.copyMakeBorder(img_bgr, pt, pb, pl, pr, b_type, value=[0, 0, 0])

        rgb = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = landmarker.detect(mp_img)

        if res.hand_landmarks and len(res.hand_landmarks) > 0:
            lms = res.hand_landmarks[0]
            # Ánh xạ ngược tọa độ
            pad_h, pad_w = padded_bgr.shape[:2]
            mapped_pts = []
            for p in lms:
                # Pixel trên ảnh padded
                px = p.x * pad_w
                py = p.y * pad_h
                # Ánh xạ về ảnh gốc
                orig_x = (px - pl) / float(w)
                orig_y = (py - pt) / float(h)
                orig_z = p.z * (pad_w / float(w))  # Chuẩn hóa scale z
                mapped_pts.extend([orig_x, orig_y, orig_z])
            return mapped_pts

    return None


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_raw = project_root / "data" / "raw" / "custom_captured"
    out_csv = project_root / "data" / "landmarks" / "landmarks_captured.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    model_task = project_root / "hand_landmarker.task"
    if not model_task.exists():
        model_task = project_root / "src" / "models" / "hand_landmarker.task"

    if not model_task.exists():
        print(f"❌ Không tìm thấy hand_landmarker.task tại {model_task}")
        return

    landmarker = build_extractor(str(model_task))

    # Tìm tất cả file ảnh trong custom_captured
    image_paths = sorted(list(data_raw.glob("*/*/*.jpg")) + list(data_raw.glob("*/*/*.png")))
    print(f"Tìm thấy tổng cộng: {len(image_paths)} ảnh tự chụp.")

    headers = [f"r{i}" for i in range(63)] + ["label", "source_file", "member"]

    success_count = 0
    fail_count = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for idx, img_path in enumerate(image_paths, 1):
            fname = img_path.name
            # Quy ước tên file: {Label}_{Member}_{Index}.jpg
            parts = fname.rsplit(".", 1)[0].split("_")
            if len(parts) >= 3:
                label = parts[0]
                member = parts[1]
            else:
                label = img_path.parent.name
                member = img_path.parent.parent.name.replace("_dataset", "")

            img = cv2.imread(str(img_path))
            if img is None:
                fail_count += 1
                continue

            pts = detect_with_padding(landmarker, img)
            if pts is not None:
                rel_path = img_path.relative_to(project_root).as_posix()
                writer.writerow(pts + [label, rel_path, member])
                success_count += 1
            else:
                fail_count += 1

            if idx % 200 == 0 or idx == len(image_paths):
                print(f"  [{idx}/{len(image_paths)}] Thành công: {success_count} | Thất bại: {fail_count}")

    print("\n" + "=" * 60)
    print(f"HOÀN TẤT TRÍCH XUẤT LANDMARK TỰ CHỤP:")
    print(f"  • Tổng số ảnh quét: {len(image_paths)}")
    print(f"  • Trích xuất thành công: {success_count} ({success_count/len(image_paths)*100:.1f}%)")
    print(f"  • Thất bại (không tìm thấy tay): {fail_count}")
    print(f"  • File kết quả: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
