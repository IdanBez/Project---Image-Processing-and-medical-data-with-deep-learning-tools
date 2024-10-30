import os
from operator import itemgetter


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
task_name = "eval"
# CUSTOM CONFIG -- end:


for model in models:
    # Model specific cfgs
    cfg = models_configs[model]
    batch_size, lr = itemgetter("batch_size", "lr")(cfg)

    for dataset, prompts in dataset_prompts.items():
        for p in prompts:
            experiment_name = f"{model}_zss_{dataset}_{p}"
            command = f"python /home/eeproj6/Yasmin/medvlsm-main/src/eval.py \
                experiment={model}.yaml \
                experiment_name={experiment_name} \
                datamodule=img_txt_mask/{dataset}.yaml \
                prompt_type={p} \
                datamodule.batch_size={batch_size} \
                trainer.accelerator={accelerator} \
                trainer.devices={devices} \
                use_ckpt=False \
                output_masks_dir=output_masks/{model}/zss/{dataset}/{p} \
                task_name={task_name}"
            # Log command in terminal
            print(command)

            # Run the command
            if os.system(command=command) != 0:
                print(f"!!! ERROR - COMMAND FAILED!!! \n{command}")
                exit()

