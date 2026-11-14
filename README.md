# sign-language-recognition — Model 1 (MediaPipe landmarks + MLP)

Hệ thống nhận diện chữ cái ngôn ngữ ký hiệu ASL theo thời gian thực từ camera:
Ảnh bàn tay → MediaPipe Hands (21 điểm × xyz = 63 số) → Chuẩn hóa hình học → Data Augmentation (hỗ trợ cả hai tay trái và phải) → StandardScaler → MLP → 28 lớp (A–Z, `del`, `space`).

## Cấu trúc
```
sign-language-recognition/
├── data/landmarks/            train_landmarks.csv · val_landmarks.csv · test_landmarks.csv
├── src/
│   ├── preprocessing/
│   │   ├── normalize_landmarks.py   # chuẩn hóa theo cổ tay (0,0,0) + scale, nạp CSV
│   │   └── augment.py               # data augmentation: lật gương, xoay, jitter, nhiễu
│   ├── models/model1_mlp/
│   │   ├── model.py                 # kiến trúc LandmarkMLP (63 -> 128 -> 64 -> 32 -> 28)
│   │   ├── train.py                 # huấn luyện với nối đôi lật gương + augment on-the-fly
│   │   └── evaluate.py              # đánh giá tách biệt 3 số: test gốc, test lật, test gộp
│   ├── demo/demo_webcam_model1.py   # demo webcam trong Colab (nhận diện cả hai tay)
│   └── utils/                       config.py · confusion_matrix.py
├── tests/
│   └── test_augment.py              # unit tests cho module augmentation
├── checkpoints/
│   ├── model1_mlp_best.pth          # checkpoint mới nhất (hỗ trợ cả hai tay)
│   └── model1_mlp_best_old.pth      # checkpoint cũ sao lưu
├── notebooks/run_model1_colab.ipynb # notebook chạy trên Colab
├── reports/                         confusion_matrices/ · figures/ · model1_mlp_classification_report.txt
├── configs/config.yaml              # cấu hình dữ liệu, hyperparams, augment, demo
└── requirements.txt
```

## Chạy trên Colab
1. Upload file `sign-language-recognition.zip` lên Colab, hoặc mở `notebooks/run_model1_colab.ipynb`.
2. Chạy lần lượt các cell. Bản chất chỉ gồm các lệnh cơ bản từ thư mục gốc project:
```python
%cd /content/sign-language-recognition
!pip install -q mediapipe pyyaml
!python -m src.models.model1_mlp.train       # train với data augmentation
!python -m src.models.model1_mlp.evaluate    # đánh giá 3 con số (gốc, lật gương, gộp)
from src.demo.demo_webcam_model1 import run_webcam_demo; run_webcam_demo()   # demo webcam
```

## Ghi chú quan trọng
- **Hỗ trợ cả hai tay (tay trái và tay phải):** Nhờ kỹ thuật Data Augmentation (lật gương $x \to -x$ nhân đôi tập train và áp dụng xoay/co giãn/nhiễu on-the-fly), mô hình hiện tại nhận diện chính xác cả tay trái lẫn tay phải với độ chính xác ~98.85% trên cả hai tay mà không cần phải bật chế độ lật gương (`mirror=False`).
- **Thứ tự chuẩn hóa (đồng bộ train ↔ demo):** `fix_aspect` → `normalize_landmarks` → `StandardScaler` (mean/scale lưu trong checkpoint) → `MLP`. Quá trình data augmentation luôn được thực hiện TRƯỚC `StandardScaler`.
- **Kiểm thử augmentation:** Chạy `python tests/test_augment.py` để xác minh các phép biến đổi hình học bảo toàn gốc tọa độ cổ tay (0,0,0) và tính đối xứng.
