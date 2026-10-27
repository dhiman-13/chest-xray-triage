"""Grad-CAM (Selvaraju et al., 2017) for timm CNNs, used for report figures.

The serving API uses the cheaper CAM variant baked into the ONNX graph (see export.py);
for DenseNet/EfficientNet heads (global-average-pool + linear) the two coincide up to scale.
"""

from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cxr.labels import LABELS, pretty
from cxr.preprocess import IMAGENET_MEAN, IMAGENET_STD


def gradcam(net: torch.nn.Module, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """x: (1, 3, H, W). Returns a heatmap in [0, 1] of shape (H, W)."""
    net.eval()
    with torch.enable_grad():
        feats = net.forward_features(x)
        feats.retain_grad()
        logits = net.forward_head(feats)
        net.zero_grad()
        logits[0, class_idx].backward()
        weights = feats.grad.mean(dim=(2, 3), keepdim=True)  # importance of each channel
        cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def denormalize(x: torch.Tensor) -> np.ndarray:
    img = x[0, 0].cpu().numpy() * IMAGENET_STD[0, 0, 0] + IMAGENET_MEAN[0, 0, 0]
    return np.clip(img, 0, 1)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    base = np.repeat(gray[..., None], 3, axis=2)
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


def save_gradcam_grid(net, samples: list[tuple[torch.Tensor, np.ndarray, np.ndarray]], path) -> None:
    """samples: list of (x, y_true, probs). Shows the highest-probability true finding per image."""
    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7), constrained_layout=True)
    axes = np.array(axes).reshape(2, n)
    device = next(net.parameters()).device
    for j, (x, y, p) in enumerate(samples):
        positives = np.flatnonzero(y)
        k = int(positives[np.argmax(p[positives])]) if len(positives) else int(np.argmax(p))
        cam = gradcam(net, x.to(device), k)
        gray = denormalize(x)
        axes[0, j].imshow(gray, cmap="gray")
        axes[0, j].set_title("truth: " + ", ".join(pretty(LABELS[i]) for i in positives)[:40], fontsize=8)
        axes[1, j].imshow(overlay(gray, cam))
        axes[1, j].set_title(f"{pretty(LABELS[k])}  p={p[k]:.2f}", fontsize=9)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle("Grad-CAM on test images", fontsize=12)
    fig.savefig(path, dpi=130)
    plt.close(fig)
