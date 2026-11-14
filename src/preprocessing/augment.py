"""
Module data augmentation cho tọa độ landmarks (21 điểm x 3 = 63 feature).
Nhận đầu vào X shape (N, 63) hoặc (63,) ĐÃ qua normalize_landmarks (cổ tay tại gốc 0,0,0, |max| <= 1).
Trả về (N, 63) float32 cùng thang đo chuẩn hóa hình học.
"""
import numpy as np


def _get_rng(rng=None):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    return rng


def mirror_hand(X):
    """
    Lật gương bàn tay qua mặt phẳng yz (đổi dấu tọa độ x):
        x' = -x ; y' = y ; z' = z
    Áp dụng cho cả 21 điểm landmarks, giữ nguyên thứ tự các điểm.
    Biến bàn tay phải thành bàn tay trái và ngược lại.
    """
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    X_arr[:, :, 0] *= -1.0
    return X_arr.reshape(-1, 63)


def rotate_xy(X, max_deg=15.0, rng=None):
    """
    Xoay ngẫu nhiên mỗi mẫu trong mặt phẳng xy quanh cổ tay (gốc tọa độ 0,0,0).
    Góc xoay theta ~ U(-max_deg, +max_deg) (đơn vị độ).
    """
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    N = X_arr.shape[0]

    # Sinh góc xoay ngẫu nhiên cho từng mẫu
    angles_deg = rng.uniform(-max_deg, max_deg, size=(N, 1)).astype(np.float32)
    angles_rad = np.deg2rad(angles_deg)
    cos_a = np.cos(angles_rad)   # (N, 1)
    sin_a = np.sin(angles_rad)   # (N, 1)

    x = X_arr[:, :, 0]           # (N, 21)
    y = X_arr[:, :, 1]           # (N, 21)

    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    X_arr[:, :, 0] = x_new
    X_arr[:, :, 1] = y_new
    # Cổ tay (điểm 0) luôn tại gốc tọa độ (0, 0, 0)
    X_arr[:, 0, :] = 0.0

    return X_arr.reshape(-1, 63)


def scale_jitter(X, low=0.9, high=1.1, rng=None):
    """
    Nhân tỉ lệ ngẫu nhiên theo trục x, y nhẹ (~ [low, high])
    để mô phỏng các kích thước bàn tay to/nhỏ và góc nhìn khác nhau.
    """
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    N = X_arr.shape[0]

    # Tỉ lệ ngẫu nhiên cho trục x và trục y riêng biệt cho mỗi mẫu
    sx = rng.uniform(low, high, size=(N, 1)).astype(np.float32)
    sy = rng.uniform(low, high, size=(N, 1)).astype(np.float32)

    X_arr[:, :, 0] *= sx
    X_arr[:, :, 1] *= sy
    X_arr[:, 0, :] = 0.0

    return X_arr.reshape(-1, 63)


def gaussian_noise(X, sigma=0.01, rng=None):
    """
    Thêm nhiễu Gauss nhỏ (sigma ~ 0.01 - 0.02) vào tọa độ landmarks.
    Cổ tay (điểm 0) không thêm nhiễu (luôn giữ 0,0,0).
    """
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()

    noise = rng.normal(0.0, sigma, size=X_arr.shape).astype(np.float32)
    noise[:, 0, :] = 0.0  # Giữ nguyên cổ tay tại (0,0,0)

    X_arr += noise
    X_arr[:, 0, :] = 0.0

    return X_arr.reshape(-1, 63)


def augment_batch(X, cfg=None, rng=None):
    """
    Ghép các phép biến đổi ngẫu nhiên (rotate, scale, noise) theo cấu hình cfg.
    Luôn gọi lại bước chia scale (|max|) sau khi biến đổi để dữ liệu vẫn nằm
    trong cùng thang đo với normalize_landmarks.
    Tọa độ cổ tay luôn = (0, 0, 0).
    """
    if cfg is None:
        cfg = {}

    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    N = X_arr.shape[0]

    prob = float(cfg.get("prob", 0.5))
    rotate_deg = float(cfg.get("rotate_deg", 15.0))
    scale_range = cfg.get("scale_range", [0.9, 1.1])
    noise_sigma = float(cfg.get("noise_sigma", 0.01))

    # 1. Xoay ngẫu nhiên
    if rotate_deg > 0:
        mask_rot = rng.uniform(0.0, 1.0, size=N) < prob
        if np.any(mask_rot):
            rotated = rotate_xy(X_arr[mask_rot], max_deg=rotate_deg, rng=rng)
            X_arr[mask_rot] = rotated.reshape(-1, 21, 3)

    # 2. Scale jitter ngẫu nhiên
    if scale_range and len(scale_range) == 2:
        mask_scale = rng.uniform(0.0, 1.0, size=N) < prob
        if np.any(mask_scale):
            scaled = scale_jitter(X_arr[mask_scale], low=scale_range[0], high=scale_range[1], rng=rng)
            X_arr[mask_scale] = scaled.reshape(-1, 21, 3)

    # 3. Nhiễu Gauss ngẫu nhiên
    if noise_sigma > 0:
        mask_noise = rng.uniform(0.0, 1.0, size=N) < prob
        if np.any(mask_noise):
            noisy = gaussian_noise(X_arr[mask_noise], sigma=noise_sigma, rng=rng)
            X_arr[mask_noise] = noisy.reshape(-1, 21, 3)

    # 4. Đảm bảo cổ tay tại (0,0,0) và chuẩn hóa lại thang đo (|max|)
    X_arr[:, 0, :] = 0.0
    scale = np.max(np.abs(X_arr), axis=(1, 2), keepdims=True)
    scale[scale < 1e-6] = 1.0
    X_norm = (X_arr / scale).reshape(-1, 63).astype(np.float32)

    return X_norm
