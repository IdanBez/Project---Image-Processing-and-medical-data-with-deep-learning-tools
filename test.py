import pyrootutils
import copy
import os
from typing import List, Tuple
import hydra
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.transforms.functional as TF
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.logger import Logger
from src import utils

torch.set_float32_matmul_precision("medium")

log = utils.get_pylogger(__name__)


def save_masks(output_dir, predictions, is_evaluation=False):
    preds, filenames, heights, widths, datasets = [], [], [], [], []
    
    for batch in predictions:
        preds.extend(list(batch["preds"]))
        filenames.extend(list(batch["mask_names"]))
        heights.extend(list(batch["heights"]))
        widths.extend(list(batch["widths"]))
        if "dataset" in batch:
            datasets.extend(list(batch["dataset"]))

    log.info(f"Saving masks to directory {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if not is_evaluation:
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))

    if preds[0].shape[0] > 1 and len(datasets) > 0:
        save_multiclass_masks(output_dir, preds, filenames, heights, widths, datasets)
    else:
        save_binary_masks(output_dir, preds, filenames, heights, widths)


def save_multiclass_masks(output_dir, preds, filenames, heights, widths, datasets):
    for pred, filename, h, w, dataset in zip(preds, filenames, heights, widths, datasets):
        for cls, p in zip(utils.CLASS_NAMES, pred):
            path = os.path.join(output_dir, dataset, cls, filename)
            save_single_mask(path, p.unsqueeze(0), h, w)


def save_binary_masks(output_dir, preds, filenames, heights, widths):
    for pred, filename, h, w in zip(preds, filenames, heights, widths):
        path = os.path.join(output_dir, filename)
        save_single_mask(path, pred, h, w)


def save_single_mask(path, pred, height, width):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    resized = TF.resize(
        pred.double(),
        size=[height, width],
        interpolation=TF.InterpolationMode.NEAREST_EXACT,
    )
    torchvision.utils.save_image(resized, path)


def setup_pipeline(cfg):
    if cfg.use_ckpt and not cfg.ckpt_path:
        raise ValueError("Checkpoint path required when use_ckpt=True")

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    loggers = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers)

    return datamodule, model, loggers, trainer


@utils.task_wrapper
def run_evaluation(cfg: DictConfig) -> Tuple[dict, dict]:
    datamodule, model, loggers, trainer = setup_pipeline(cfg)

    components = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(components)

    if cfg.get("task_name") == "eval":
        run_eval_task(cfg, trainer, model, datamodule)
    elif cfg.get("task_name") == "pred":
        run_pred_task(cfg, trainer, model, datamodule)
    else:
        raise ValueError(f"Task must be 'eval' or 'pred', got {cfg.get('task_name')}")

    return trainer.callback_metrics, components


def run_eval_task(cfg, trainer, model, datamodule):
    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    if cfg.get("output_masks_dir"):
        log.info("Generating evaluation masks")
        predictions = trainer.predict(
            model=model,
            dataloaders=datamodule,
            ckpt_path=cfg.ckpt_path,
        )
        save_masks(cfg.output_masks_dir, predictions, is_evaluation=True)


def run_pred_task(cfg, trainer, model, datamodule):
    if not cfg.get("output_masks_dir"):
        raise ValueError("output_masks_dir required for prediction task")

    log.info("Generating prediction masks")
    predictions = trainer.predict(
        model=model,
        dataloaders=datamodule,
        ckpt_path=cfg.ckpt_path,
    )
    save_masks(cfg.output_masks_dir, predictions)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
