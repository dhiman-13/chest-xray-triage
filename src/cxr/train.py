"""Train a chest X-ray classifier. Every run is tracked in MLflow.

    python -m cxr.train                                   # DenseNet-121 on NIH
    python -m cxr.train model=efficientnet_b3 train.batch_size=32
    python -m cxr.train data=synthetic model=tiny_test train.epochs=1   # smoke test
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from cxr.data.dataset import CXRDataModule
from cxr.models.classifier import CXRClassifier


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    L.seed_everything(cfg.seed, workers=True)
    out_dir = Path(cfg.paths.output_dir)
    ckpt_dir = out_dir / "checkpoints" / cfg.model.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dm = CXRDataModule(cfg.data.dir, cfg.model.img_size, cfg.train.batch_size, cfg.train.num_workers)
    dm.setup()
    pos_weight = dm.pos_weight(cfg.train.pos_weight_cap) if cfg.train.use_pos_weight else None

    model = CXRClassifier(
        backbone=cfg.model.backbone,
        pretrained=cfg.model.pretrained,
        drop_rate=cfg.model.drop_rate,
        img_size=cfg.model.img_size,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        epochs=cfg.train.epochs,
        warmup_epochs=cfg.train.warmup_epochs,
        loss=cfg.train.loss,
        focal_gamma=cfg.train.focal_gamma,
        pos_weight=pos_weight,
    )

    logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=cfg.mlflow.run_name,
        artifact_location=cfg.mlflow.artifact_location,
        tags={"model": cfg.model.name, "dataset": cfg.data.name, "stage": "train"},
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    logger.log_hyperparams(_flatten(resolved))
    logger.experiment.log_dict(logger.run_id, resolved, "config.json")

    precision = cfg.train.precision
    if precision == "auto":
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"

    checkpoint = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val/auroc_macro",
        mode="max",
        save_last=True,
        enable_version_counter=False,
    )
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="auto",
        precision=precision,
        logger=logger,
        callbacks=[
            checkpoint,
            EarlyStopping("val/auroc_macro", mode="max", patience=cfg.train.early_stopping_patience),
            LearningRateMonitor("epoch"),
        ],
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        limit_train_batches=cfg.train.limit_train_batches,
        limit_val_batches=cfg.train.limit_val_batches,
        log_every_n_steps=20,
        deterministic=False,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.train.resume_from)

    best = float(checkpoint.best_model_score) if checkpoint.best_model_score is not None else float("nan")
    run_info = {
        "run_id": logger.run_id,
        "tracking_uri": cfg.mlflow.tracking_uri,
        "model": cfg.model.name,
        "best_ckpt": str(checkpoint.best_model_path),
        "best_val_auroc_macro": best,
        "data_dir": cfg.data.dir,
    }
    if trainer.is_global_zero:  # DDP: only the main process records the run
        (out_dir / "last_run.json").write_text(json.dumps(run_info, indent=2))
        logger.experiment.log_metric(logger.run_id, "best_val_auroc_macro", best)
        print(json.dumps(run_info, indent=2))
    return best


if __name__ == "__main__":
    main()
