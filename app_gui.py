#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
GIAO DIỆN ĐỒ HỌA (GUI) CAMERA NHẬN DIỆN CHỮ CÁI BẰNG CỬ CHỈ TAY
Mô hình: MediaPipe HandLandmarker + PyTorch MLP (Model 1 - ASL)
Sử dụng: Tkinter + OpenCV + Pillow
=============================================================================
"""

import os
import sys
import time
import threading
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
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Thiết lập đường dẫn thư mục gốc project
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Import module từ app_camera
from app_camera import CameraApp, speak_async


class AppGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Hệ Thống Nhận Diện Chữ Cái Bằng Cử Chỉ Tay (Model 1 - ASL)")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 680)

        # Cài đặt giao diện tối (Dark Mode)
        self.bg_color = "#1E1E24"
        self.card_color = "#2B2B36"
        self.accent_color = "#00D26A"
        self.text_color = "#FFFFFF"
        self.sub_text_color = "#A0A0B0"

        self.root.configure(bg=self.bg_color)

        # Style cho ttk widgets
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=14, troughcolor="#3A3A4A", background=self.accent_color)
        style.configure("Hold.Horizontal.TProgressbar", thickness=14, troughcolor="#3A3A4A", background="#FFB300")

        # Khởi tạo lõi xử lý Camera & Model
        self.camera_core = CameraApp(
            camera_id=0,
            conf_threshold=0.60,
            hold_time=0.8,
            mirror=True
        )

        # Trạng thái camera
        self.cap = None
        self.is_running = True
        self.current_cam_id = 0
        self.last_frame_time = time.time()
        self.fps = 0.0

        # Xây dựng giao diện
        self._build_ui()

        # Bật camera
        self._start_camera(self.current_cam_id)

        # Lắng nghe phím tắt trên toàn cửa sổ
        self.root.bind("<Key>", self._on_key_press)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Bắt đầu vòng lặp cập nhật hình ảnh
        self.update_video_loop()

    def _build_ui(self):
        """Tạo layout giao diện gồm Panel Camera (Trái) và Panel Điều khiển (Phải)."""
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # ==================== CỘT TRÁI: CAMERA ====================
        left_frame = tk.Frame(main_frame, bg=self.card_color, relief=tk.FLAT, bd=0)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Header camera
        cam_header = tk.Frame(left_frame, bg=self.card_color)
        cam_header.pack(fill=tk.X, padx=12, pady=10)

        lbl_cam_title = tk.Label(
            cam_header,
            text="📷 Khung Hình Camera Thời Gian Thực",
            font=("Segoe UI", 13, "bold"),
            fg=self.text_color,
            bg=self.card_color
        )
        lbl_cam_title.pack(side=tk.LEFT)

        self.lbl_fps = tk.Label(
            cam_header,
            text="FPS: 0.0",
            font=("Segoe UI", 10, "bold"),
            fg=self.accent_color,
            bg=self.card_color
        )
        self.lbl_fps.pack(side=tk.RIGHT)

        # Khung chứa video canvas
        self.canvas_cam = tk.Canvas(left_frame, bg="#0D0D11", highlightthickness=0)
        self.canvas_cam.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        # Thanh trạng thái dưới camera
        cam_footer = tk.Frame(left_frame, bg=self.card_color)
        cam_footer.pack(fill=tk.X, padx=12, pady=(0, 10))

        self.var_mirror = tk.BooleanVar(value=self.camera_core.mirror)
        chk_mirror = tk.Checkbutton(
            cam_footer,
            text="Lật gương (Mirror - [M])",
            variable=self.var_mirror,
            command=self._toggle_mirror,
            bg=self.card_color,
            fg=self.text_color,
            selectcolor="#3A3A4A",
            activebackground=self.card_color,
            activeforeground=self.text_color,
            font=("Segoe UI", 9)
        )
        chk_mirror.pack(side=tk.LEFT, padx=(0, 10))

        self.var_skeleton = tk.BooleanVar(value=self.camera_core.show_skeleton)
        chk_skeleton = tk.Checkbutton(
            cam_footer,
            text="Khung xương tay (Skeleton - [H])",
            variable=self.var_skeleton,
            command=self._toggle_skeleton,
            bg=self.card_color,
            fg=self.text_color,
            selectcolor="#3A3A4A",
            activebackground=self.card_color,
            activeforeground=self.text_color,
            font=("Segoe UI", 9)
        )
        chk_skeleton.pack(side=tk.LEFT, padx=10)

        # Chọn Camera ID
        lbl_sel_cam = tk.Label(cam_footer, text="Cổng Cam:", bg=self.card_color, fg=self.sub_text_color, font=("Segoe UI", 9))
        lbl_sel_cam.pack(side=tk.LEFT, padx=(15, 5))
        self.combo_cam = ttk.Combobox(cam_footer, values=[0, 1, 2, 3], width=3, state="readonly")
        self.combo_cam.current(0)
        self.combo_cam.pack(side=tk.LEFT)
        self.combo_cam.bind("<<ComboboxSelected>>", self._change_camera)

        # ==================== CỘT PHẢI: KẾT QUẢ & ĐIỀU KHIỂN ====================
        right_frame = tk.Frame(main_frame, bg=self.card_color, width=380)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_frame.pack_propagate(False)

        # --- Card 1: Chữ cái đang nhận diện ---
        card_pred = tk.LabelFrame(
            right_frame,
            text="  KẾT QUẢ NHẬN DIỆN  ",
            font=("Segoe UI", 10, "bold"),
            fg="#00E5FF",
            bg=self.card_color,
            bd=1,
            relief=tk.SOLID
        )
        card_pred.pack(fill=tk.X, padx=14, pady=(12, 8))

        pred_inner = tk.Frame(card_pred, bg=self.card_color)
        pred_inner.pack(fill=tk.X, padx=12, pady=10)

        # Huy hiệu chữ cái to
        self.lbl_big_letter = tk.Label(
            pred_inner,
            text="--",
            font=("Segoe UI", 48, "bold"),
            fg=self.accent_color,
            bg="#1E1E24",
            width=3,
            relief=tk.GROOVE,
            bd=2
        )
        self.lbl_big_letter.pack(side=tk.LEFT, padx=(0, 15))

        # Cột độ tin cậy và tiến độ giữ
        bars_col = tk.Frame(pred_inner, bg=self.card_color)
        bars_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_conf = tk.Label(bars_col, text="Độ tin cậy: 0%", font=("Segoe UI", 9, "bold"), fg=self.text_color, bg=self.card_color)
        self.lbl_conf.pack(anchor="w")

        self.bar_conf = ttk.Progressbar(bars_col, orient="horizontal", mode="determinate", style="TProgressbar")
        self.bar_conf.pack(fill=tk.X, pady=(2, 6))

        self.lbl_hold = tk.Label(bars_col, text="Tiến độ gõ (Hold): 0%", font=("Segoe UI", 9), fg=self.sub_text_color, bg=self.card_color)
        self.lbl_hold.pack(anchor="w")

        self.bar_hold = ttk.Progressbar(bars_col, orient="horizontal", mode="determinate", style="Hold.Horizontal.TProgressbar")
        self.bar_hold.pack(fill=tk.X, pady=(2, 0))

        # Top 3
        self.lbl_top3 = tk.Label(
            card_pred,
            text="Top 3: 1. -- (0%)  |  2. -- (0%)  |  3. -- (0%)",
            font=("Segoe UI", 9),
            fg="#FFD54F",
            bg=self.card_color
        )
        self.lbl_top3.pack(anchor="w", padx=12, pady=(0, 8))

        # --- Card 2: Đoạn văn bản đã gõ (Sentence Builder) ---
        card_text = tk.LabelFrame(
            right_frame,
            text="  VĂN BẢN ĐÃ GHÉP (SENTENCE)  ",
            font=("Segoe UI", 10, "bold"),
            fg="#00E5FF",
            bg=self.card_color,
            bd=1,
            relief=tk.SOLID
        )
        card_text.pack(fill=tk.BOTH, expand=True, padx=14, pady=8)

        self.txt_display = tk.Text(
            card_text,
            height=6,
            font=("Consolas", 14, "bold"),
            bg="#15151B",
            fg="#FFFFFF",
            insertbackground="#00FF88",
            wrap=tk.WORD,
            bd=0,
            padx=10,
            pady=10
        )
        self.txt_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Các nút chức năng dưới ô văn bản
        btn_row1 = tk.Frame(card_text, bg=self.card_color)
        btn_row1.pack(fill=tk.X, padx=10, pady=(0, 8))

        btn_speak = tk.Button(
            btn_row1,
            text="🔊 Đọc văn bản [V]",
            command=self._btn_speak,
            bg="#0288D1",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2"
        )
        btn_speak.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        btn_copy = tk.Button(
            btn_row1,
            text="📋 Sao chép",
            command=self._btn_copy,
            bg="#5C6BC0",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2"
        )
        btn_copy.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        btn_del = tk.Button(
            btn_row1,
            text="⌫ Xóa chữ",
            command=self._btn_backspace,
            bg="#D32F2F",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2"
        )
        btn_del.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        btn_clear = tk.Button(
            btn_row1,
            text="🗑 Xóa hết [C]",
            command=self._btn_clear,
            bg="#424242",
            fg="white",
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=8,
            pady=4,
            cursor="hand2"
        )
        btn_clear.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        # --- Card 3: Tinh chỉnh tham số ---
        card_settings = tk.LabelFrame(
            right_frame,
            text="  CÀI ĐẶT THAM SỐ  ",
            font=("Segoe UI", 10, "bold"),
            fg="#00E5FF",
            bg=self.card_color,
            bd=1,
            relief=tk.SOLID
        )
        card_settings.pack(fill=tk.X, padx=14, pady=(8, 12))

        # Slider Độ tin cậy
        lbl_conf_s = tk.Label(card_settings, text="Ngưỡng tin cậy tối thiểu (%):", bg=self.card_color, fg=self.sub_text_color, font=("Segoe UI", 9))
        lbl_conf_s.pack(anchor="w", padx=12, pady=(6, 0))
        self.slider_conf = tk.Scale(
            card_settings,
            from_=40,
            to=95,
            orient=tk.HORIZONTAL,
            bg=self.card_color,
            fg=self.text_color,
            troughcolor="#3A3A4A",
            highlightthickness=0,
            command=self._on_conf_slider
        )
        self.slider_conf.set(int(self.camera_core.conf_threshold * 100))
        self.slider_conf.pack(fill=tk.X, padx=12, pady=(0, 6))

        # Slider Thời gian giữ cử chỉ (Hold Time)
        lbl_hold_s = tk.Label(card_settings, text="Thời gian giữ cử chỉ để chốt chữ (giây):", bg=self.card_color, fg=self.sub_text_color, font=("Segoe UI", 9))
        lbl_hold_s.pack(anchor="w", padx=12, pady=(0, 0))
        self.slider_hold = tk.Scale(
            card_settings,
            from_=0.4,
            to=2.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            bg=self.card_color,
            fg=self.text_color,
            troughcolor="#3A3A4A",
            highlightthickness=0,
            command=self._on_hold_slider
        )
        self.slider_hold.set(self.camera_core.hold_time)
        self.slider_hold.pack(fill=tk.X, padx=12, pady=(0, 8))

    def _start_camera(self, cam_id: int):
        """Khởi động camera theo ID."""
        if self.cap is not None:
            self.cap.release()

        print(f"🎥 Đang kết nối tới Camera {cam_id}...")
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cam_id)

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.current_cam_id = cam_id
        else:
            messagebox.showwarning(
                "Lỗi Camera",
                f"Không thể mở Camera ID {cam_id}!\nVui lòng thử chọn cổng khác trong danh sách."
            )

    def _change_camera(self, event=None):
        new_id = int(self.combo_cam.get())
        self._start_camera(new_id)

    def _toggle_mirror(self):
        self.camera_core.mirror = self.var_mirror.get()

    def _toggle_skeleton(self):
        self.camera_core.show_skeleton = self.var_skeleton.get()

    def _on_conf_slider(self, val):
        self.camera_core.conf_threshold = float(val) / 100.0

    def _on_hold_slider(self, val):
        self.camera_core.hold_time = float(val)

    def _btn_speak(self):
        text = self.camera_core.typed_text
        if text.strip():
            speak_async(text)
        else:
            messagebox.showinfo("Thông báo", "Chưa có từ hoặc câu nào để phát âm!")

    def _btn_copy(self):
        text = self.camera_core.typed_text
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Thành công", "Đã sao chép văn bản vào Clipboard!")

    def _btn_backspace(self):
        self.camera_core._commit_character("del")
        self._update_text_display()

    def _btn_clear(self):
        self.camera_core.typed_text = ""
        self._update_text_display()

    def _on_key_press(self, event):
        """Xử lý phím tắt khi người dùng bấm phím."""
        k = event.keysym.lower()
        if k == "space":
            self.camera_core._commit_character("space")
            self._update_text_display()
        elif k == "backspace":
            self.camera_core._commit_character("del")
            self._update_text_display()
        elif k == "c":
            self._btn_clear()
        elif k == "m":
            self.var_mirror.set(not self.var_mirror.get())
            self._toggle_mirror()
        elif k == "h":
            self.var_skeleton.set(not self.var_skeleton.get())
            self._toggle_skeleton()
        elif k == "v":
            self._btn_speak()
        elif k in ["escape", "q"]:
            self._on_close()

    def _update_text_display(self):
        """Đồng bộ văn bản hiển thị."""
        current = self.camera_core.typed_text
        self.txt_display.delete("1.0", tk.END)
        self.txt_display.insert(tk.END, current)
        self.txt_display.see(tk.END)

    def update_video_loop(self):
        """Vòng lặp đọc frame từ OpenCV và cập nhật lên canvas Tkinter."""
        if not self.is_running:
            return

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Xử lý frame qua pipeline model
                processed_frame = self.camera_core.process_frame(frame)

                # Tính FPS
                now = time.time()
                dt = now - self.last_frame_time
                self.last_frame_time = now
                if dt > 0:
                    self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
                    self.lbl_fps.config(text=f"FPS: {self.fps:.1f}")

                # Cập nhật thông số sang UI bên phải
                # 1. Ký tự & độ tin cậy
                detected = self.camera_core.candidate_char
                if detected:
                    self.lbl_big_letter.config(text=detected, fg=self.accent_color)
                else:
                    self.lbl_big_letter.config(text="--", fg=self.sub_text_color)

                # Lấy kết quả từ lịch sử prob
                if len(self.camera_core.prob_history) > 0:
                    probs = np.mean(self.camera_core.prob_history, axis=0)
                    top3_idx = np.argsort(probs)[::-1][:3]
                    best_c = self.camera_core.predictor.classes[top3_idx[0]]
                    best_p = float(probs[top3_idx[0]])

                    self.bar_conf["value"] = int(best_p * 100)
                    self.lbl_conf.config(text=f"Độ tin cậy: {best_p*100:.0f}% ({best_c})")

                    top3_str = (
                        f"Top 3: 1. {self.camera_core.predictor.classes[top3_idx[0]]} ({probs[top3_idx[0]]*100:.0f}%)  |  "
                        f"2. {self.camera_core.predictor.classes[top3_idx[1]]} ({probs[top3_idx[1]]*100:.0f}%)  |  "
                        f"3. {self.camera_core.predictor.classes[top3_idx[2]]} ({probs[top3_idx[2]]*100:.0f}%)"
                    )
                    self.lbl_top3.config(text=top3_str)
                else:
                    self.bar_conf["value"] = 0
                    self.lbl_conf.config(text="Độ tin cậy: 0%")
                    self.lbl_top3.config(text="Top 3: 1. -- (0%)  |  2. -- (0%)  |  3. -- (0%)")

                # Tiến độ giữ chữ (Hold progress)
                cur_t = time.time()
                if self.camera_core.candidate_char and self.camera_core.candidate_start_time > 0:
                    elapsed = cur_t - self.camera_core.candidate_start_time
                    progress = min(1.0, elapsed / self.camera_core.hold_time)
                    pct = int(progress * 100)
                    self.bar_hold["value"] = pct
                    self.lbl_hold.config(text=f"Tiến độ gõ '{self.camera_core.candidate_char}': {pct}%")
                else:
                    self.bar_hold["value"] = 0
                    self.lbl_hold.config(text="Tiến độ gõ (Hold): 0%")

                # Đồng bộ văn bản
                self._update_text_display()

                # Hiển thị frame lên canvas Tkinter
                # Resize cho khớp với kích thước canvas
                canv_w = self.canvas_cam.winfo_width()
                canv_h = self.canvas_cam.winfo_height()

                if canv_w > 100 and canv_h > 100:
                    # Giữ tỉ lệ khung hình
                    h, w = processed_frame.shape[:2]
                    scale = min(canv_w / w, canv_h / h)
                    nw, nh = int(w * scale), int(h * scale)
                    resized = cv2.resize(processed_frame, (nw, nh), interpolation=cv2.INTER_AREA)

                    # Đổi màu BGR -> RGB
                    rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(rgb_frame)
                    img_tk = ImageTk.PhotoImage(image=img_pil)

                    # Vẽ ở giữa canvas
                    x_pos = (canv_w - nw) // 2
                    y_pos = (canv_h - nh) // 2
                    self.canvas_cam.create_image(x_pos, y_pos, anchor=tk.NW, image=img_tk)
                    self.canvas_cam.image = img_tk

        # Gọi lại sau 25ms (~40 FPS)
        self.root.after(25, self.update_video_loop)

    def _on_close(self):
        """Dọn dẹp và đóng ứng dụng."""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        try:
            self.camera_core.landmarker.close()
        except Exception:
            pass
        self.root.destroy()
        print("\n👋 Đã tắt giao diện ứng dụng.")


def main():
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
