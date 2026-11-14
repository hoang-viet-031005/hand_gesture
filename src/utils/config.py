"""Đọc configs/config.yaml và quy đổi đường dẫn tương đối -> tuyệt đối (tính từ thư mục gốc project)."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]          # .../sign-language-recognition
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel_path):
    """'data/landmarks/x.csv' -> Path tuyệt đối; tự tạo thư mục cha nếu chưa có."""
    p = Path(rel_path)
    p = p if p.is_absolute() else ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
