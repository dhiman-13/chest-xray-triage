"""LightningModule wrapping a timm CNN backbone for 14-label chest X-ray classification."""

from __future__ import annotations

import lightning as L
import numpy as np
import timm
import torch

from cxr.labels import LABELS, NUM_CLASSES
from cxr.metrics import per_class_auroc
from cxr.models.losses import build_loss


def create_backbone(backbone: str, pretrained: bool, drop_rate: float = 0.0) -> torch.nn.Module:
    # ImageNet weights transfer surprisingly well to X-rays: low-level edge/texture filters are
    # generic, and fine-tuning adapts the deeper layers. This is the standard CheXNet recipe.
    return timm.create_model(backbone, pretrained=pretrained, num_classes=NUM_CLASSES, drop_rate=drop_rate)


class CXRClassifier(L.LightningModule):
    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        img_size: int = 224,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        epochs: int = 12,
        warmup_epochs: int = 1,
        loss: str = "bce",
        focal_gamma: float = 2.0,
        pos_weight: list[float] | None = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = create_backbone(backbone, pretrained, drop_rate)
        self.criterion = build_loss(loss, pos_weight, focal_gamma)
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val/loss", self.criterion(logits, y), on_epoch=True, prog_bar=True)
        self._val_logits.append(logits.float().cpu())
        self._val_targets.append(y.cpu())

    def on_validation_epoch_end(self):
        if not self._val_logits:
            return
        logits, targets = torch.cat(self._val_logits), torch.cat(self._val_targets)
        if self.trainer.world_size > 1:  # DDP: gather every GPU's share of the validation set
            logits = self.all_gather(logits.to(self.device)).flatten(0, 1).cpu()
            targets = self.all_gather(targets.to(self.device)).flatten(0, 1).cpu()
        p = torch.sigmoid(logits).numpy()
        y = targets.numpy()
        self._val_logits.clear()
        self._val_targets.clear()
        auc = per_class_auroc(y, p)
        macro = float(np.nanmean(auc)) if np.isfinite(auc).any() else 0.5
        self.log("val/auroc_macro", macro, prog_bar=True)
        for label, value in zip(LABELS, auc, strict=True):
            if np.isfinite(value):
                self.log(f"val_auroc/{label}", float(value))

    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        warmup = max(int(hp.warmup_epochs), 0)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(hp.epochs - warmup, 1))
        if warmup == 0:
            sched = cosine
        else:
            # Warm-up protects the pretrained weights from large early updates of the random head.
            linear = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=warmup)
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [linear, cosine], milestones=[warmup])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
