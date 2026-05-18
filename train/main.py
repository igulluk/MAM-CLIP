import os
import torch
import lightning.pytorch as pl
import hydra

from omegaconf import DictConfig
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from files.model import lightningModel
from loaders import get_loaders

os.environ["TOKENIZERS_PARALLELISM"] = "True"
pl.seed_everything(42)
torch.set_float32_matmul_precision("high")


def train(cfg):
    """Run vision-language pretraining with the Hydra ``cfg``.

    Sets up the data loaders, the Lightning model, gradient accumulation
    (to reach ``effective_batch_size``), the W&B/TensorBoard logger and a
    checkpoint callback that saves on the best ``valid_loss``.
    """
    accumulation = (
        cfg.data.effective_batch_size // cfg.data.batch_size
    ) // cfg.trainparams.n_gpu

    train_loader, valid_loader = get_loaders(cfg)

    model = lightningModel(cfg)
    checkpoint_dirpath = os.path.join(
        cfg.trainparams.checkpoint_dir_path, cfg.trainparams.experiment_name
    )
    os.makedirs(checkpoint_dirpath, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        save_top_k=-1,
        monitor="valid_loss",
        mode="min",
        filename=cfg.trainparams.experiment_name + "{epoch}-{valid_loss:.2f}",
    )

    if cfg.trainparams.logger == "tensorboard":
        logger = pl_loggers.TensorBoardLogger(
            save_dir=cfg.trainparams.lightning_log_dir_path,
            version=cfg.trainparams.experiment_name,
        )

    if cfg.trainparams.logger == "wandb":
        logger = WandbLogger(
            log_model="False",
            project=cfg.trainparams.project_name,
            name=cfg.trainparams.experiment_name,
        )

    trainer = pl.Trainer(
        devices=cfg.trainparams.n_gpu,
        num_nodes=1,
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=cfg.trainparams.max_epochs,
        num_sanity_val_steps=0,
        callbacks=[
            checkpoint_callback,
        ],
        logger=logger,
        precision="16-mixed",
        accumulate_grad_batches=accumulation,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    trainer.logger._version = cfg.trainparams.experiment_name


@hydra.main(version_base="1.1", config_path=".", config_name="cfg.yaml")
def main(cfg: DictConfig):
    return train(cfg)


if __name__ == "__main__":
    main()
