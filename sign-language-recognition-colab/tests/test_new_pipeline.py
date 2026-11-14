import numpy as np
from src.preprocessing.augmentation import (
    mirror_hand, rotate_xy, scale_jitter, gaussian_jitter, joint_dropout, augment_sample
)
from src.preprocessing.normalize_landmarks import normalize_landmarks, normalize_landmarks_palm_scale

def test_new_features():
    rng = np.random.default_rng(42)
    dummy_pts = rng.normal(0, 1, size=(10, 63)).astype(np.float32)

    # 1. Test Palm Scale Normalization
    norm_palm = normalize_landmarks_palm_scale(dummy_pts)
    assert norm_palm.shape == (10, 63)
    assert not np.isnan(norm_palm).any()
    pts_3d = norm_palm.reshape(10, 21, 3)
    # wrist (0) must be 0
    np.testing.assert_allclose(pts_3d[:, 0, :], 0.0, atol=1e-6)
    # norm(P9 - P0) should be 1.0
    p9_norm = np.linalg.norm(pts_3d[:, 9, :], axis=1)
    np.testing.assert_allclose(p9_norm, 1.0, atol=1e-5)
    print("✅ Palm Scale Normalization test PASS!")

    # 2. Test Joint Dropout
    dropped = joint_dropout(norm_palm, p=0.1, rng=rng)
    assert dropped.shape == (10, 63)
    # Wrist must still be 0
    np.testing.assert_allclose(dropped.reshape(10, 21, 3)[:, 0, :], 0.0, atol=1e-6)
    print("✅ Joint Dropout test PASS!")

    # 3. Test Gaussian Jitter
    jittered = gaussian_jitter(norm_palm, sigma=0.008, rng=rng)
    assert jittered.shape == (10, 63)
    np.testing.assert_allclose(jittered.reshape(10, 21, 3)[:, 0, :], 0.0, atol=1e-6)
    print("✅ Gaussian Jitter test PASS!")

    # 4. Test Augment Sample
    one_sample = norm_palm[0]
    aug = augment_sample(one_sample, rng=rng)
    assert aug.shape == (63,)
    assert not np.isnan(aug).any()
    assert np.max(np.abs(aug)) <= 1.0001
    print("✅ Augment Sample test PASS!")

    print("\n🎉 ALL NEW TESTS PASS!")

if __name__ == "__main__":
    test_new_features()
