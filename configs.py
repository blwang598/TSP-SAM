from box import Box
from configs.base_config import base_config


config = {
"eval_interval": 100,
    "ema_rate": 0.999,
    "csv_keys": ["Name", "Prompt", "Mean IoU", "Mean F1",
                 "epoch", "Acc_bl", "Avg_recall_bl", "Acc_fh",
                 "Avg_recall_fh", "Acc_sd", "Avg_recall_sd"],
    "opt": {
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [5000, 10000],
        "warmup_steps": 0,
        "train_steps": 15000
    },
    "model": {
        "type": "vit_t_batched_mgda",
        "checkpoint": "/data2/wangbilin/backbone-pretrain/",
        "ckpt": "./save/best_iou.ckpt",
        "freeze": {
            "image_encoder": False,
            "prompt_encoder": False,
            "mask_decoder": False,
        },
    },
    "datasets": {
        "WLI": {
            "train_txt": "./train.txt",
            "val_txt": "./val.txt"
        },
    },
    "gpu_ids": "0,1",
    "batch_size": 8,
    "val_batchsize": 2,
    "num_workers": 4,
    "num_epochs": 50,
    "max_nums": 50,
    "num_points": 3,
    "dataset": "WLI",
    "visual": False,
    "load_type": "soft",
    "prompt": "box",
    "out_dir": "output/WLI/",
    "name": "base",
    "corrupt": None,
}
