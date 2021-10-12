import pytorch_lightning as pl
from pl_examples import _DATASETS_PATH
from pytorch_lightning.loggers import WandbLogger

from reid.callbacks import TranslationVisualization_WanDB
from reid.datamodules.market1501 import PairedMarket1501DataModule
from reid.models.mymodel import MyModel


def main(args=None):
    datamodule = PairedMarket1501DataModule(
        _DATASETS_PATH, num_workers=2, batch_size=8, shuffle=True, drop_last=True
    )

    model = MyModel(datamodule.dims, st=False)

    wandb_logger = WandbLogger(project="st-reid-gan")

    callbacks = [
        TranslationVisualization_WanDB(),
    ]

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=-1 if datamodule.num_workers > 0 else None,
        progress_bar_refresh_rate=1,
        max_epochs=323,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
