import os
import torch
import lightning as L
from box import Box
from lightning.fabric.loggers import TensorBoardLogger
from configs.config import cfg

from model import Model
from utils.eval_utils_batch import validate
from datasets.WLI import load_datasets_soft

def main(cfg: Box) -> None:
    fabric = L.Fabric(accelerator="auto",
                      devices=[1],
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)],
                      precision='16',
                      )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    model = Model(cfg)
    model.setup()  # lora_vit

    train_data, val_data = load_datasets_soft(cfg, model.model.image_encoder.img_size)

    val_data = fabric._setup_dataloader(val_data)
    model = fabric.setup(model)

    full_checkpoint = fabric.load(cfg.model.ckpt)
    model.load_state_dict(full_checkpoint["model"])

    validate(fabric, cfg, model, val_data, name=cfg.name, step=0)

    del model, train_data, val_data


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids

    main(cfg)
    torch.cuda.empty_cache()
