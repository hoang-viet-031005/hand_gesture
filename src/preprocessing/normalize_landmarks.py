"""Chuẩn hóa landmarks + nạp dữ liệu CSV (DÙNG CHUNG cho train / evaluate / demo webcam)."""
import numpy as np
import pandas as pd

FEATURE_COLS = [f"r{i}" for i in range(63)]       # x0,y0,z0, x1,y1,z1, ..., x20,y20,z20


def normalize_landmarks(X, method="max_abs"):
    """
    X: (N, 63) hoặc (63,) tọa độ thô. Trả về (N, 63) đã chuẩn hóa.
    
    Phương pháp hỗ trợ:
    1. 'max_abs' (Mặc định cho model1_mlp_best):
       - Trừ cổ tay (landmark 0).
       - Chia max |tọa độ| -> bất biến theo vị trí và kích thước bàn tay.
    2. 'palm_scale' (Palm-Scale Invariant Normalization):
       - Trừ cổ tay (landmark 0): P'_i = P_i - P_0.
       - Chia khoảng cách Euclid 3D giữa Cổ tay (joint 0) và khớp gốc ngón giữa (joint 9):
         P''_i = P'_i / ||P_9 - P_0||_2.
    """
    X = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3)
    # 1. Dịch gốc tọa độ về cổ tay (joint 0)
    X = X - X[:, 0:1, :]

    if method == "palm_scale":
        # Khoảng cách Euclid giữa joint 0 (đã = 0) và joint 9 (khớp gốc ngón giữa)
        # norm = sqrt(x9^2 + y9^2 + z9^2)
        scale = np.linalg.norm(X[:, 9:10, :], axis=2, keepdims=True)
        scale[scale < 1e-6] = 1.0
        return (X / scale).reshape(-1, 63)
    else:
        # Chuẩn hóa max |giá trị| (mặc định tương thích model1_mlp_best)
        scale = np.max(np.abs(X), axis=(1, 2), keepdims=True)
        scale[scale < 1e-6] = 1.0
        return (X / scale).reshape(-1, 63)


def normalize_landmarks_palm_scale(X):
    """Tiện ích gọi trực tiếp Palm-Scale Invariant Normalization."""
    return normalize_landmarks(X, method="palm_scale")


def fix_aspect(pts_xyz, img_w, img_h):
    """
    (21, 3) MediaPipe (x theo chiều rộng, y theo chiều cao, đều ~[0,1]) -> quy về cùng thang
    (cạnh dài nhất) để bàn tay không bị méo với khung hình chữ nhật (webcam 640x480).
    Ảnh huấn luyện vuông nên bước này không đổi gì lúc train.
    """
    pts = np.asarray(pts_xyz, dtype=np.float32).copy()
    S = float(max(img_w, img_h))
    pts[:, 0] *= img_w / S
    pts[:, 1] *= img_h / S
    pts[:, 2] *= img_w / S
    return pts


def load_split(csv_path, label_col="label"):
    """Đọc 1 file CSV -> (X_chuẩn_hóa_hình_học (N,63) float32, nhãn chữ (N,))."""
    df = pd.read_csv(csv_path)
    return normalize_landmarks(df[FEATURE_COLS].values), df[label_col].values
