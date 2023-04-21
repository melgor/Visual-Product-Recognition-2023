import argparse
import os
import sys
from datetime import datetime

import yaml
from loguru import logger

import torch
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from visual_search.dataset.augmentations import get_image_transforms
from visual_search.dataset.datamodule import Product10kDataModule
from visual_search.model.classification_model import ImageClassification



def main(args: argparse.Namespace) -> None:
    with open(args.cfg) as f:
        config = yaml.safe_load(f)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('medium')

    today = datetime.now()
    name_experiments = f"{args.name}_{today.strftime('%Y_%m_%d_%h_%H_%M_%s')}"
    os.makedirs(os.path.join(config["outdir"], name_experiments))

    train_transforms, valid_transform = get_image_transforms(config["dataset"]["input_size"])
    dataset_module = Product10kDataModule(config["dataset"],
                                          train_data_transform=train_transforms,
                                          valid_data_transform=valid_transform)
    dataset_module.setup()
    num_train_steps_in_epoch = len(dataset_module.train_dataloader())
    model = ImageClassification(config["dataset"]["num_of_classes"], config["train"], channel_last=config["channel_last"],
                                num_train_steps=num_train_steps_in_epoch, class_value_counts=dataset_module.train_dataset.value_counts)

    if config['train']['pretrain_model'] != "":
        print(f"Load pretrain weights: {config['train']['pretrain_model']}")
        model_weights = torch.load(config['train']['pretrain_model'])["state_dict"]
        logger.info(model.load_state_dict(model_weights, strict=True))

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="valid_mAP",
        min_delta=0.005,
        patience=30,
        verbose=True,
        mode="max",
    )
    model_checkpoints = ModelCheckpoint(
        dirpath=os.path.join(config["outdir"], name_experiments, "weights"),
        monitor="valid_mAP",
        save_top_k=2,
        mode="max",
        filename='epoch{epoch:02d}-valid_mAP{valid_mAP:.2f}',
        save_weights_only=True,
    )

    # loggers
    tensorboard_logger = TensorBoardLogger(config["outdir"], name_experiments, version="tb")
    loggers = [tensorboard_logger]

    # training
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=config["train"]["n_epoch"],
        callbacks=[lr_monitor, early_stop_callback, model_checkpoints],
        logger=loggers,
        precision=config["precision"],
        benchmark=True,
        deterministic=False,
        default_root_dir=config["default_root_dir"],
        log_every_n_steps=50,
        val_check_interval=0.25,
        accumulate_grad_batches=1,
        gradient_clip_val=5.0
    )

    logger.info("Start training")
    trainer.fit(model, dataset_module)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Path to config file.')
    parser.add_argument('--name', type=str, required=True, help='name of experiment')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
