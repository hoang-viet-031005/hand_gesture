"""Kiến trúc MLP: 63 -> 128 -> 64 -> 32 -> num_classes (Linear -> BatchNorm -> ReLU -> Dropout)."""
import torch
import torch.nn as nn


class LandmarkMLP(nn.Module):
    def __init__(self, in_dim=63, hidden=(128, 64, 32), num_classes=28, dropout=0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))     # logits (Softmax áp dụng khi dự đoán)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def load_checkpoint(path, device="cpu"):
    """Nạp checkpoint -> (model ở chế độ eval, ckpt dict gồm classes / scaler_mean / scaler_scale)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = LandmarkMLP(cfg["in_dim"], tuple(cfg["hidden"]), cfg["num_classes"], cfg["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
