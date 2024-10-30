import os
from operator import itemgetter
import torch
from torch import Tensor


accelerator = "gpu"
devices = [0]
precision = "16-mixed"
debugger = False

models_configs = {
    "biomed_clipseg_d": {"batch_size": 128, "lr": 0.02},
    "cris": {"batch_size": 8, "lr": 0.00002},
}

dataset_prompts = {
    "ChestXray": ["p4"],
    "AbdomenCT-1K": ["p4"],
    "kvasir_polyp": ["p4"],
    "bkai_polyp": ["p4"],
    "clinicdb_polyp": ["p4"],
    "isic": ["p4"],
    "busi": ["p4"],
    "colondb_polyp":["p4"],
    "cvc300_polyp":["p4"],
    "etis_polyp": ["p4"],
    "ChestXRay_Pneumothorax": ["p1"]
 }

models = [
    "cris",
    "biomed_clipseg_d"
 ]
freeze_encoder = True


for model in models:
    # Model specific cfgs
    cfg = models_configs[model]
    batch_size, lr = itemgetter("batch_size", "lr")(cfg)

    for dataset, prompts in dataset_prompts.items():
        for p in prompts:
            command = f"python src/train.py \
                experiment={model}.yaml \
                experiment_name={model}_ft_{dataset}_{p} \
                datamodule=img_txt_mask/{dataset}.yaml \
                datamodule.batch_size={batch_size} \
                model.optimizer.lr={lr} \
                trainer.accelerator={accelerator} \
                trainer.precision={precision} \
                trainer.devices={devices} \
                prompt_type={p} \
                logger=wandb.yaml \
                tags='[{model}, {dataset}, finetune, {p}]' \
                output_masks_dir=output_masks/{model}/ft/{dataset}/{p}"

            if debugger:
                command = f"{command} debug=default"
            # Log command in terminal
            print(f"RUNNING COMMAND \n{command}")
            if os.system(command=command) != 0:
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()
