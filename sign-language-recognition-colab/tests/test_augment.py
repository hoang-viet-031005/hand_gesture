"""
Unit tests cho module src/preprocessing/augment.py.
Kiểm tra 5 tiêu chí:
  1. mirror_hand(mirror_hand(X)) == X.
  2. Cổ tay (landmark 0) luôn là (0,0,0) sau mọi phép augment.
  3. Shape (N,63), dtype float32, không có NaN/inf, |max| <= 1 (sau chuẩn hóa lại).
  4. Cùng seed -> cùng kết quả (tính tái lập).
  5. Phép lật chỉ đổi dấu cột x (r0, r3, ..., r60), các cột y, z giữ nguyên tuyệt đối.
"""
import sys
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import numpy as np

# Thêm thư mục gốc vào sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.augment import (
    mirror_hand,
    rotate_xy,
    scale_jitter,
    gaussian_noise,
    augment_batch,
)
from src.preprocessing.normalize_landmarks import normalize_landmarks


def generate_dummy_landmarks(N=20, seed=42):
    """Tạo N mẫu landmarks ngẫu nhiên và chuẩn hóa qua normalize_landmarks."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.1, 0.9, size=(N, 63)).astype(np.float32)
    return normalize_landmarks(raw)


def test_mirror_double_flip():
    """1. mirror_hand lật 2 lần phải bằng bản gốc."""
    X = generate_dummy_landmarks(N=25, seed=10)
    X_double = mirror_hand(mirror_hand(X))
    np.testing.assert_allclose(X_double, X, atol=1e-6, err_msg="Lật 2 lần không bằng bản gốc!")
    print("✅ Test 1 PASS: mirror_hand(mirror_hand(X)) == X")


def test_wrist_zero_origin():
    """2. Cổ tay (landmark 0: r0, r1, r2) phải luôn là (0, 0, 0) sau mọi phép augment."""
    X = generate_dummy_landmarks(N=30, seed=20)
    rng = np.random.default_rng(123)

    funcs = [
        ("mirror_hand", lambda data: mirror_hand(data)),
        ("rotate_xy", lambda data: rotate_xy(data, max_deg=30.0, rng=rng)),
        ("scale_jitter", lambda data: scale_jitter(data, low=0.7, high=1.3, rng=rng)),
        ("gaussian_noise", lambda data: gaussian_noise(data, sigma=0.05, rng=rng)),
        ("augment_batch", lambda data: augment_batch(data, {"prob": 1.0, "rotate_deg": 25.0, "noise_sigma": 0.03}, rng=rng)),
    ]

    for name, fn in funcs:
        out = fn(X)
        wrist = out.reshape(-1, 21, 3)[:, 0, :]  # (N, 3)
        np.testing.assert_allclose(
            wrist, 0.0, atol=1e-7,
            err_msg=f"Cổ tay không bằng (0,0,0) sau phép {name}!"
        )
    print("✅ Test 2 PASS: Cổ tay luôn là (0,0,0) sau mọi phép augment")


def test_shape_dtype_nan_scale():
    """3. Shape (N,63), dtype float32, không có NaN/inf, |max| <= 1 (sau chuẩn hóa lại)."""
    N = 35
    X = generate_dummy_landmarks(N=N, seed=30)
    cfg = {"prob": 0.8, "rotate_deg": 20.0, "scale_range": [0.85, 1.15], "noise_sigma": 0.02}

    out = augment_batch(X, cfg=cfg, rng=42)

    assert out.shape == (N, 63), f"Sai shape: {out.shape} != ({N}, 63)"
    assert out.dtype == np.float32, f"Sai dtype: {out.dtype} != float32"
    assert not np.isnan(out).any(), "Kết quả chứa giá trị NaN!"
    assert not np.isinf(out).any(), "Kết quả chứa giá trị Inf!"

    max_vals = np.max(np.abs(out.reshape(N, 21, 3)), axis=(1, 2))
    assert np.all(max_vals <= 1.0 + 1e-6), f"Có giá trị vượt quá 1.0: {max_vals.max()}"
    print("✅ Test 3 PASS: Shape (N,63), dtype float32, no NaN/inf, |max| <= 1.0")


def test_reproducibility_same_seed():
    """4. Cùng seed -> cùng kết quả."""
    X = generate_dummy_landmarks(N=20, seed=40)
    cfg = {"prob": 0.7, "rotate_deg": 15.0, "scale_range": [0.9, 1.1], "noise_sigma": 0.01}

    out1 = augment_batch(X, cfg=cfg, rng=999)
    out2 = augment_batch(X, cfg=cfg, rng=999)

    np.testing.assert_allclose(out1, out2, atol=1e-7, err_msg="Cùng seed nhưng kết quả khác nhau!")
    print("✅ Test 4 PASS: Cùng seed sinh ra kết quả hoàn toàn trùng khớp")


def test_mirror_only_inverts_x():
    """5. Phép lật chỉ đổi dấu cột x (r0, r3, ..., r60), các cột y,z giữ nguyên."""
    X = generate_dummy_landmarks(N=15, seed=50)
    X_mirr = mirror_hand(X)

    X_3d = X.reshape(-1, 21, 3)
    M_3d = X_mirr.reshape(-1, 21, 3)

    # Cột x phải đổi dấu: x' == -x
    np.testing.assert_allclose(M_3d[:, :, 0], -X_3d[:, :, 0], atol=1e-6, err_msg="Cột x không đổi dấu đúng!")

    # Cột y và z phải giữ nguyên tuyệt đối: y' == y, z' == z
    np.testing.assert_allclose(M_3d[:, :, 1], X_3d[:, :, 1], atol=1e-6, err_msg="Cột y bị thay đổi!")
    np.testing.assert_allclose(M_3d[:, :, 2], X_3d[:, :, 2], atol=1e-6, err_msg="Cột z bị thay đổi!")
    print("✅ Test 5 PASS: Phép lật chỉ đổi dấu cột x, cột y và z giữ nguyên 100%")


def run_all_tests():
    print("=" * 60)
    print("Đang chạy kiểm thử bộ test_augment.py...")
    print("=" * 60)
    test_mirror_double_flip()
    test_wrist_zero_origin()
    test_shape_dtype_nan_scale()
    test_reproducibility_same_seed()
    test_mirror_only_inverts_x()
    print("=" * 60)
    print("🎉 TẤT CẢ 5 BÀI TEST ĐỀU ĐẠT CHUẨN!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
