"""
Huấn luyện Model 1 (MLP) với Data Augmentation (hỗ trợ cả hai tay trái và phải).
Chạy từ thư mục gốc project:
    python -m src.models.model1_mlp.train
Đầu ra:
    - checkpoints/model1_mlp_best.pth
    - checkpoints/model1_mlp_best_old.pth (bản sao lưu cũ)
    - reports/figures/model1_mlp_training_curves.png
    - reports/confusion_matrices/model1_mlp_confusion_matrix.png
    - reports/confusion_matrices/model1_mlp_confusion_matrix_mirrored.png
    - reports/model1_mlp_classification_report.txt
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import time, random, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

from src.utils.config import load_config, resolve
from src.preprocessing.normalize_landmarks import load_split
from src.preprocessing.augment import mirror_hand, augment_batch
from src.models.model1_mlp.model import LandmarkMLP
from src.models.model1_mlp.evaluate import evaluate


class LandmarkDataset(Dataset):
    """Dataset chứa tọa độ hình học (N, 63) đã chuẩn hóa theo cổ tay."""
    def __init__(self, X_geom, y):
        self.X = np.asarray(X_geom, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AugmentedCollate:
    """
    Collate function thực hiện:
    1. Augmentation ngẫu nhiên on-the-fly (rotate/scale/noise) trên batch trước StandardScaler.
    2. Áp dụng StandardScaler (mean / scale) trước khi đưa vào mạng MLP.
    """
    def __init__(self, scaler_mean, scaler_scale, aug_cfg=None, seed=42):
        self.mean = np.array(scaler_mean, dtype=np.float32)
        self.scale = np.array(scaler_scale, dtype=np.float32)
        self.aug_cfg = aug_cfg
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch):
        xs, ys = zip(*batch)
        X_arr = np.array(xs, dtype=np.float32)
        y_arr = torch.tensor(ys, dtype=torch.long)

        # Augmentation on-the-fly trước StandardScaler
        if self.aug_cfg and self.aug_cfg.get("enabled", True):
            X_arr = augment_batch(X_arr, cfg=self.aug_cfg, rng=self.rng)

        # Áp dụng StandardScaler
        X_norm = (X_arr - self.mean) / self.scale
        return torch.from_numpy(X_norm.astype(np.float32)), y_arr


def main():
    cfg = load_config(); d, m = cfg["data"], cfg["model1_mlp"]
    aug_cfg = cfg.get("augment", {})
    aug_enabled = aug_cfg.get("enabled", True)

    seed = cfg["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch:", torch.__version__, "| Thiết bị:", device)

    # ---- 1. Đọc dữ liệu ban đầu (chuẩn hóa hình học qua normalize_landmarks) ----
    Xtr, ytr_s = load_split(resolve(d["train_csv"]), d["label_col"])
    Xva, yva_s = load_split(resolve(d["val_csv"]),   d["label_col"])
    print(f"Kích thước ban đầu train / val: {Xtr.shape}, {Xva.shape}")

    le = LabelEncoder().fit(ytr_s)
    classes = list(le.classes_); n_cls = len(classes)
    ytr, yva = le.transform(ytr_s), le.transform(yva_s)
    print(f"Số lớp: {n_cls} ->", classes)

    # ---- 2. Data Augmentation: LẬT GƯƠNG (BƯỚC 1 - BẮT BUỘC) ----
    # Nối đôi tập train: bản gốc (tay phải) + bản lật gương (tay trái)
    if aug_enabled and aug_cfg.get("mirror", True):
        print("⚡ Đang áp dụng augmentation: Lật gương tập train (x2 dữ liệu để học cả hai tay)...")
        Xtr_mirr = mirror_hand(Xtr)
        Xtr = np.vstack([Xtr, Xtr_mirr])
        ytr = np.concatenate([ytr, ytr])
        print(f"👉 Kích thước train sau khi nối đôi: {Xtr.shape} (nhãn giữ nguyên)")

        # Tập val phản ánh cả hai tay để early stopping và ReduceLROnPlateau khách quan
        Xva_mirr = mirror_hand(Xva)
        Xva = np.vstack([Xva, Xva_mirr])
        yva = np.concatenate([yva, yva])
        print(f"👉 Kích thước val sau khi gộp hai tay: {Xva.shape}")
    else:
        print("ℹ️ Augmentation mirror bị tắt hoặc không được bật.")

    # ---- 3. StandardScaler (fit CHỈ trên train đã gồm bản lật) ----
    scaler = StandardScaler()
    scaler.fit(Xtr)
    scaler_mean = scaler.mean_.astype(np.float32)
    scaler_scale = scaler.scale_.astype(np.float32)

    # Tập val chuẩn hóa cố định qua scaler
    Xva_scaled = scaler.transform(Xva).astype(np.float32)

    # ---- 4. Data Augmentation: ON-THE-FLY (BƯỚC 2 - KHUYẾN NGHỊ) ----
    # Áp dụng rotate, scale jitter, gaussian noise ngẫu nhiên theo batch cho tập train
    train_dataset = LandmarkDataset(Xtr, ytr)
    collate_fn = AugmentedCollate(scaler_mean, scaler_scale, aug_cfg=aug_cfg if aug_enabled else None, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=m["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xva_scaled), torch.from_numpy(yva).long()),
        batch_size=m["batch_size"],
        shuffle=False
    )

    # ---- 5. Mô hình / loss / optimizer ----
    model = LandmarkMLP(63, tuple(m["hidden"]), n_cls, m["dropout"]).to(device)
    print(model); print("Tổng số tham số:", sum(p.numel() for p in model.parameters()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    def run_epoch(loader_, train):
        model.train(train)
        tot, correct, n = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for xb, yb in loader_:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb); loss = criterion(logits, yb)
                if train:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                tot += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item(); n += xb.size(0)
        return tot / n, correct / n

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss, best_state, wait = float("inf"), None, 0
    t0 = time.time()

    print("\n🚀 Bắt đầu huấn luyện mô hình MLP với Data Augmentation...")
    for epoch in range(1, m["epochs"] + 1):
        tl, ta = run_epoch(train_loader, True)
        vl, va = run_epoch(val_loader, False)
        scheduler.step(vl)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if vl < best_loss - 1e-4:
            best_loss, wait = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{m['epochs']} | train loss {tl:.4f} acc {ta:.4f} "
                  f"| val loss {vl:.4f} acc {va:.4f} | lr {optimizer.param_groups[0]['lr']:.1e}")
        if wait >= m["patience"]:
            print(f"⏹ Early stopping tại epoch {epoch}."); break

    model.load_state_dict(best_state)
    print(f"Hoàn tất sau {time.time()-t0:.1f}s | val loss tốt nhất: {best_loss:.4f}")

    # ---- 6. Đồ thị Loss / Accuracy ----
    curves_path = resolve(m["training_curves"])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for a, (k1, k2, t) in zip(ax, [("train_loss", "val_loss", "Loss"), ("train_acc", "val_acc", "Accuracy")]):
        a.plot(hist[k1], label="Train"); a.plot(hist[k2], label="Val")
        a.set_title(t); a.set_xlabel("Epoch"); a.legend(); a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(curves_path, dpi=120); plt.close()
    print(f"Đã lưu đồ thị huấn luyện: {curves_path}")

    # ---- 7. Lưu checkpoint (sao lưu bản cũ trước) ----
    ckpt_path = resolve(m["checkpoint"])
    old_ckpt_path = ckpt_path.parent / (ckpt_path.stem + "_old" + ckpt_path.suffix)
    if ckpt_path.exists() and not old_ckpt_path.exists():
        shutil.copyfile(ckpt_path, old_ckpt_path)
        print(f"📦 Đã sao lưu checkpoint cũ sang: {old_ckpt_path}")

    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "config": {"in_dim": 63, "hidden": list(m["hidden"]), "num_classes": n_cls, "dropout": m["dropout"]},
        "classes": classes,
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
    }, ckpt_path)
    print(f"✅ Đã lưu mô hình mới: {ckpt_path}")

    # ---- 8. Đánh giá luôn trên tập test ----
    evaluate()


if __name__ == "__main__":
    main()
