"""Vẽ & lưu confusion matrix, liệt kê các cặp nhãn bị nhầm nhiều nhất."""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # không cần màn hình; trong notebook dùng plt.imshow/Image để hiển thị file đã lưu
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title="Confusion Matrix - tập test"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names, cbar=False, annot_kws={"size": 7})
    plt.xlabel("Dự đoán"); plt.ylabel("Thực tế"); plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=120); plt.close()
    return cm


def top_confusions(cm, class_names, k=5):
    """Trả về list (thực_tế, dự_đoán, số_mẫu) của k cặp bị nhầm nhiều nhất."""
    off = cm.copy(); np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:k], off.shape))[0]
    return [(class_names[i], class_names[j], int(off[i, j])) for i, j in idx if off[i, j] > 0]
