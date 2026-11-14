"""
Đánh giá Model 1 trên tập test bằng checkpoint đã lưu. Chạy từ thư mục gốc project:
    python -m src.models.model1_mlp.evaluate
Báo cáo TÁCH BIỆT 3 con số trên tập test:
  (a) test gốc (tay phải)
  (b) test lật gương (mô phỏng tay trái)
  (c) gộp cả hai (hai tay)
"""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch

from src.utils.config import load_config, resolve
from src.utils.confusion_matrix import plot_confusion_matrix, top_confusions
from src.preprocessing.normalize_landmarks import load_split
from src.preprocessing.augment import mirror_hand
from src.models.model1_mlp.model import load_checkpoint


def _predict_batches(model, X_scaled, device, batch_size=1024):
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), batch_size):
            batch = torch.from_numpy(X_scaled[i:i + batch_size]).to(device)
            preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate():
    cfg = load_config(); d, m = cfg["data"], cfg["model1_mlp"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(resolve(m["checkpoint"]), device)
    classes = ckpt["classes"]
    mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
    scale = np.array(ckpt["scaler_scale"], dtype=np.float32)

    # 1. Đọc dữ liệu test gốc
    X_orig, y_str = load_split(resolve(d["test_csv"]), d["label_col"])
    
    # Lọc các nhãn nếu checkpoint hiện tại chưa hỗ trợ (ví dụ checkpoint 28 lớp gặp nhãn 'nothing')
    valid_mask = np.isin(y_str, classes)
    if not np.all(valid_mask):
        skipped = set(y_str[~valid_mask])
        print(f"ℹ️ Checkpoint hiện tại ({len(classes)} lớp) không có nhãn: {skipped}. Đang tạm lọc {np.sum(~valid_mask)} mẫu này để đánh giá.")
        X_orig = X_orig[valid_mask]
        y_str = y_str[valid_mask]

    y_true = np.array([classes.index(s) for s in y_str])

    # 2. Tạo bản test lật gương (mô phỏng tay trái)
    X_mirr = mirror_hand(X_orig)

    # 3. Chuẩn hóa qua đúng StandardScaler đã fit lúc train
    X_orig_s = ((X_orig - mean) / scale).astype(np.float32)
    X_mirr_s = ((X_mirr - mean) / scale).astype(np.float32)

    # 4. (a) Test gốc (tay phải)
    y_pred_orig = _predict_batches(model, X_orig_s, device)
    acc_orig = accuracy_score(y_true, y_pred_orig) * 100

    # 5. (b) Test lật gương (tay trái)
    y_pred_mirr = _predict_batches(model, X_mirr_s, device)
    acc_mirr = accuracy_score(y_true, y_pred_mirr) * 100

    # 6. (c) Gộp cả hai (hai tay)
    X_comb_s = np.vstack([X_orig_s, X_mirr_s])
    y_comb_true = np.concatenate([y_true, y_true])
    y_comb_pred = np.concatenate([y_pred_orig, y_pred_mirr])
    acc_comb = accuracy_score(y_comb_true, y_comb_pred) * 100

    print("=" * 65)
    print("🎯 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (3 CON SỐ TÁCH BIỆT):")
    print("=" * 65)
    print(f"  (a) Test gốc (tay phải):          {acc_orig:.2f}%")
    print(f"  (b) Test lật gương (mô phỏng tay trái): {acc_mirr:.2f}%")
    print(f"  (c) Gộp cả hai (hai tay):         {acc_comb:.2f}%")
    print("=" * 65)

    # Đánh giá phân rã theo thành viên nếu có cột 'member'
    import pandas as pd
    df_raw_test = pd.read_csv(resolve(d["test_csv"]))
    if len(df_raw_test) != len(y_true):
        df_raw_test = df_raw_test[valid_mask].reset_index(drop=True)
    member_acc_text = ""
    if "member" in df_raw_test.columns:
        df_raw_test["member"] = df_raw_test["member"].fillna("ASL_Raw").astype(str)
        print("\n📈 ĐỘ CHÍNH XÁC THEO TỪNG THÀNH VIÊN TRÊN TẬP TEST:")
        member_acc_text = "\nĐộ chính xác theo từng thành viên (Test gốc):\n"
        for mem in df_raw_test["member"].unique():
            mask = (df_raw_test["member"] == mem).values
            m_acc = accuracy_score(y_true[mask], y_pred_orig[mask]) * 100
            line = f"  • {mem:10s} ({np.sum(mask):5d} mẫu): {m_acc:.2f}%\n"
            member_acc_text += line
            print(line.strip())

    # Báo cáo chi tiết cho tập gộp (c)
    report_c = classification_report(
        y_comb_true, y_comb_pred,
        labels=range(len(classes)),
        target_names=classes,
        digits=3
    )
    print(f"\n📊 Báo cáo phân loại chi tiết (Tập test gộp hai tay):\n{report_c}")

    # Confusion matrix cho (c) gộp cả hai
    cm_c_path = resolve(m["confusion_matrix"])
    cm_c = plot_confusion_matrix(y_comb_true, y_comb_pred, classes, cm_c_path, title="Confusion Matrix - Test gộp (hai tay)")

    # Confusion matrix cho (b) lật gương
    cm_b_path = resolve(m.get("confusion_matrix_mirrored", "reports/confusion_matrices/model1_mlp_confusion_matrix_mirrored.png"))
    plot_confusion_matrix(y_true, y_pred_mirr, classes, cm_b_path, title="Confusion Matrix - Test lật gương (tay trái)")

    print("Top 5 cặp bị nhầm nhiều nhất (tập gộp thực tế -> dự đoán):")
    for a, b, n in top_confusions(cm_c, classes):
        print(f"  {a} -> {b}: {n} mẫu")

    # Lưu báo cáo vào text file
    summary_report = (
        f"KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST:\n"
        f"  (a) Test gốc (tay phải):          {acc_orig:.2f}%\n"
        f"  (b) Test lật gương (tay trái):   {acc_mirr:.2f}%\n"
        f"  (c) Gộp cả hai (hai tay):         {acc_comb:.2f}%\n\n"
        f"{member_acc_text}\n"
        f"Classification Report (Tập test gộp hai tay):\n{report_c}\n"
    )
    report_file = resolve(m["report_txt"])
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(summary_report)

    print(f"\nĐã lưu confusion matrix (gộp): {cm_c_path}")
    print(f"Đã lưu confusion matrix (lật gương): {cm_b_path}")
    print(f"Đã lưu báo cáo chi tiết: {report_file}")

    return acc_comb


if __name__ == "__main__":
    evaluate()
