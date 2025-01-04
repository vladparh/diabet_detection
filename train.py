import hydra
import pandas as pd
import pytorch_lightning as pl
from dvc.api import DVCFileSystem
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from diabet_detector.data_preprocessing import prepare_data_for_training
from diabet_detector.logger import get_logger
from diabet_detector.model import SimpleClassifier
from diabet_detector.trainer import DiabetDetector


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig):
    fs = DVCFileSystem()
    with fs.open(config["data_loading"]["train_data_path"]) as file:
        data = pd.read_csv(file)
    train_dataset, val_dataset = prepare_data_for_training(
        data=data,
        path_to_save_scaler=config["data_loading"]["scaler_path"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["train_batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["val_batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )

    model = SimpleClassifier(p_dropout=config["model"]["p_dropout"])
    module = DiabetDetector(model, None, lr=config["training"]["lr"])

    logger = get_logger(config["logging"])
    logger.log_hyperparams(
        {"p_dropout": config["model"]["p_dropout"], "lr": config["training"]["lr"]}
    )

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=config["model"]["save_dir"],
        filename="model_{val_AP:.2f}",
        monitor="val_AP",
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=config["training"]["num_epochs"],
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        logger=logger,
        callbacks=[model_checkpoint],
    )

    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
