#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Data Augmentation cho tọa độ landmarks 21 khớp bàn tay (63 features).
Hỗ trợ:
- Lật gương bàn tay ngang (Horizontal Mirror Flip: x -> -x).
- Xoay ngẫu nhiên mặt phẳng XY quanh cổ tay ([-12°, +12°]).
- Co giãn tỷ lệ đồng nhất ([0.93, 1.07]).
- Gaussian Jitter (sigma = 0.008).
- Joint Dropout (p = 0.03): mô phỏng khớp bị che khuất hoặc MediaPipe mất dấu.
- Tái chuẩn hóa đảm bảo cổ tay luôn (0, 0, 0).
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np


def _get_rng(rng=None):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(rng)
    return rng


def mirror_hand(X):
    """Lật gương bàn tay theo trục ngang (x' = -x; y' = y; z' = z)."""
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    X_arr[:, :, 0] *= -1.0
    return X_arr.reshape(-1, 63)


def rotate_xy(X, max_deg=12.0, rng=None):
    """Xoay ngẫu nhiên mỗi mẫu trong mặt phẳng XY quanh cổ tay [-max_deg, +max_deg]."""
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    N = X_arr.shape[0]

    angles_deg = rng.uniform(-max_deg, max_deg, size=(N, 1)).astype(np.float32)
    angles_rad = np.deg2rad(angles_deg)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    x = X_arr[:, :, 0]
    y = X_arr[:, :, 1]

    X_arr[:, :, 0] = x * cos_a - y * sin_a
    X_arr[:, :, 1] = x * sin_a + y * cos_a
    X_arr[:, 0, :] = 0.0  # Cổ tay luôn tại gốc (0, 0, 0)

    return X_arr.reshape(-1, 63)


def scale_jitter(X, low=0.93, high=1.07, rng=None):
    """Co giãn tỷ lệ ngẫu nhiên [low, high]."""
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    N = X_arr.shape[0]

    s = rng.uniform(low, high, size=(N, 1, 1)).astype(np.float32)
    X_arr *= s
    X_arr[:, 0, :] = 0.0

    return X_arr.reshape(-1, 63)


def gaussian_jitter(X, sigma=0.008, rng=None):
    """Cộng nhiễu Gauss nhỏ (sigma = 0.008) vào tọa độ landmarks."""
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()

    noise = rng.normal(0.0, sigma, size=X_arr.shape).astype(np.float32)
    noise[:, 0, :] = 0.0  # Cổ tay không nhiễu
    X_arr += noise
    X_arr[:, 0, :] = 0.0

    return X_arr.reshape(-1, 63)


def joint_dropout(X, p=0.03, rng=None):
    """Mô phỏng khớp bị che khuất: ngẫu nhiên đưa tọa độ khớp về 0 với xác suất p."""
    rng = _get_rng(rng)
    X_arr = np.asarray(X, dtype=np.float32).reshape(-1, 21, 3).copy()
    N = X_arr.shape[0]

    # Không áp dụng cho cổ tay (joint 0)
    drop_mask = rng.uniform(0.0, 1.0, size=(N, 21, 1)) < p
    drop_mask[:, 0, :] = False  # Luôn giữ cổ tay

    X_arr[drop_mask.repeat(3, axis=2)] = 0.0
    return X_arr.reshape(-1, 63)


def augment_sample(x_63, mirror_prob=0.5, rotate_deg=12.0, scale_range=(0.93, 1.07),
                   sigma=0.008, dropout_p=0.03, rng=None):
    """Áp dụng toàn bộ augmentation cho 1 mẫu landmarks (63,)."""
    rng = _get_rng(rng)
    X = np.asarray(x_63, dtype=np.float32).reshape(1, 63)

    # 1. Lật gương ngang
    if rng.uniform(0.0, 1.0) < mirror_prob:
        X = mirror_hand(X)

    # 2. Xoay XY
    if rotate_deg > 0:
        X = rotate_xy(X, max_deg=rotate_deg, rng=rng)

    # 3. Co giãn
    if scale_range:
        X = scale_jitter(X, low=scale_range[0], high=scale_range[1], rng=rng)

    # 4. Nhiễu Gauss
    if sigma > 0:
        X = gaussian_jitter(X, sigma=sigma, rng=rng)

    # 5. Joint Dropout
    if dropout_p > 0:
        X = joint_dropout(X, p=dropout_p, rng=rng)

    # Tái chuẩn hóa
    X_arr = X.reshape(1, 21, 3)
    X_arr[:, 0, :] = 0.0
    scale = np.max(np.abs(X_arr), axis=(1, 2), keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (X_arr / scale).reshape(63).astype(np.float32)
