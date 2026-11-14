# 🤟 HƯỚNG DẪN SỬ DỤNG APP CAMERA NHẬN DIỆN CỬ CHỈ TAY (ASL)

Ứng dụng camera nhận diện chữ cái tiếng Anh / bảng chữ cái ký hiệu ngón tay (American Sign Language - ASL) theo thời gian thực dựa trên mô hình **MediaPipe HandLandmarker + PyTorch MLP (Model 1)** đã được huấn luyện.

---

## 📁 1. Vị trí mã nguồn và các file trong máy tính của bạn
Toàn bộ mã nguồn đã được giải nén và cấu hình sẵn tại:
`C:\Users\acer\.gemini\antigravity\scratch\hand_gesture_app\sign-language-recognition\sign-language-recognition`

Các file quan trọng mới được tạo:
* 📄 [`app_camera.py`](file:///C:/Users/acer/.gemini/antigravity/scratch/hand_gesture_app/sign-language-recognition/sign-language-recognition/app_camera.py): Ứng dụng Camera tốc độ cao với giao diện HUD hiện đại, vẽ khung xương tay, thanh tiến độ ghép chữ và bảng Top 3 dự đoán.
* 🖥️ [`app_gui.py`](file:///C:/Users/acer/.gemini/antigravity/scratch/hand_gesture_app/sign-language-recognition/sign-language-recognition/app_gui.py): Giao diện đồ họa Desktop (Dark Mode) có ô hiển thị văn bản, nút sao chép, nút đọc phát âm tiếng Anh, thanh trượt chỉnh độ nhạy và chọn cổng camera.
* ⚡ [`run_camera.bat`](file:///C:/Users/acer/.gemini/antigravity/scratch/hand_gesture_app/sign-language-recognition/sign-language-recognition/run_camera.bat): File khởi động nhanh Camera chỉ với 1 cú nhấp đúp chuột.
* ⚡ [`run_gui.bat`](file:///C:/Users/acer/.gemini/antigravity/scratch/hand_gesture_app/sign-language-recognition/sign-language-recognition/run_gui.bat): File khởi động nhanh Giao diện đồ họa bằng 1 cú nhấp đúp chuột.
* 📦 `checkpoints/model1_mlp_best.pth`: Trọng số mô hình MLP đã train đạt độ chính xác **99.18%**.
* 🤖 `hand_landmarker.task`: File mô hình trích xuất 21 điểm khớp xương tay của Google MediaPipe.

---

## 🚀 2. Cách chạy ứng dụng

### Cách 1: Nhấp đúp chuột (Nhanh nhất)
1. Mở thư mục:
   `C:\Users\acer\.gemini\antigravity\scratch\hand_gesture_app\sign-language-recognition\sign-language-recognition`
2. Nhấp đúp vào file:
   * **`run_camera.bat`** để mở Camera chế độ toàn màn hình HUD.
   * Hoặc **`run_gui.bat`** để mở Giao diện Desktop có nút bấm điều khiển.

---

### Cách 2: Chạy từ Terminal / PowerShell
Mở PowerShell hoặc Command Prompt và chạy các lệnh sau:

```powershell
# Chuyển vào thư mục dự án
cd "C:\Users\acer\.gemini\antigravity\scratch\hand_gesture_app\sign-language-recognition\sign-language-recognition"

# Chạy bản Camera HUD
python app_camera.py

# Hoặc chạy bản Giao diện Desktop GUI
python app_gui.py
```

#### Tùy chỉnh tham số khi chạy dòng lệnh:
```powershell
# Nếu bạn dùng Webcam rời (cổng 1 hoặc 2):
python app_camera.py --cam 1

# Thay đổi ngưỡng độ tin cậy (mặc định 0.60):
python app_camera.py --conf 0.70

# Thay đổi thời gian giữ cử chỉ để gõ chữ (mặc định 0.8 giây):
python app_camera.py --hold 1.0

# Tắt chế độ lật gương (nếu dùng tay phải thật nhưng bị ngược):
python app_camera.py --no-mirror
```

---

## ⌨️ 3. Phím tắt và thao tác khi dùng Camera

| Phím tắt | Chức năng |
| :--- | :--- |
| **`M`** | Bật / Tắt chế độ lật gương camera (**Mirror**). Nếu nhận diện chưa chuẩn do ngược hướng tay, hãy bấm phím này. |
| **`H`** | Ẩn / Hiện khung xương khớp ngón tay (**Skeleton**). |
| **`Space`** | Gõ dấu cách thủ công vào đoạn văn bản. |
| **`Backspace`** | Xóa 1 chữ cái cuối cùng vừa gõ. |
| **`C`** | Xóa sạch toàn bộ đoạn văn bản đang hiển thị. |
| **`V`** | **Phát âm từ vựng** qua loa máy tính (Text-to-Speech tự động). |
| **`S`** | Lưu đoạn văn bản đã ghép ra file `van_ban_da_go.txt`. |
| **`Q`** hoặc **`ESC`** | Thoát ứng dụng. |

---

## ✋ 4. Cơ chế ghép từ & ghép câu (Hold-to-Type)
* Khi bạn giơ bàn tay, mô hình nhận diện cử chỉ theo thời gian thực (30-60 FPS).
* Khi bạn giữ nguyên một cử chỉ (ví dụ: chữ `H`), thanh **Tiến độ giữ (Hold: 0% ➔ 100%)** sẽ chạy đầy trong khoảng **0.8 giây**.
* Ngay khi thanh đầy:
  * Nếu là chữ cái **`A` - `Z`**: Tự động gõ chữ đó vào câu.
  * Nếu là cử chỉ **`space`**: Tự động thêm khoảng trắng.
  * Nếu là cử chỉ **`del`**: Tự động xóa ký tự trước đó.
* Sau khi gõ, hệ thống nghỉ 0.4 giây trước khi nhận tiếp, tránh tình trạng bị gõ lặp chữ liên tục.

---

## 🛠️ 5. Xử lý sự cố thường gặp (Troubleshooting)

1. **Camera không mở được hoặc màn hình đen:**
   * Kiểm tra xem camera có đang bị ứng dụng khác (Zoom, Teams, trình duyệt) chiếm dụng hay không.
   * Thử đổi sang cổng camera khác: `python app_camera.py --cam 1` hoặc chọn `Cổng Cam: 1` trên giao diện GUI.
   * Đảm bảo Windows đã cấp quyền Camera: Vào *Windows Settings ➔ Privacy & Security ➔ Camera ➔ Cho phép ứng dụng truy cập camera*.

2. **Mô hình nhận nhầm giữa tay trái và tay phải:**
   * Nhấn phím **`M`** để lật ngược hình ảnh camera. Khi lật gương, cử chỉ tay sẽ khớp với chiều dữ liệu đã huấn luyện.
