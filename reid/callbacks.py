import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from wandb.sdk.wandb_run import Run

from .models.mymodel import MyModel


class TranslationVisualization_WanDB(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_epoch_end(self, trainer: Trainer, pl_module: MyModel) -> None:
        run: Run = trainer.logger.experiment
        dataloader = trainer.train_dataloader

        for imgs, lbls in dataloader:
            img_anc, img_pos, img_neg = [tensor.to(pl_module.device) for tensor in imgs]
            lbl_anc, lbl_pos, lbl_neg = [tensor.to(pl_module.device) for tensor in lbls]

            cont_anc, sty_anc = pl_module.encoder(img_anc)
            cont_pos, sty_pos = pl_module.encoder(img_pos)
            cont_neg, sty_neg = pl_module.encoder(img_neg)

            img_anc_anc = pl_module.decoder(cont_anc, sty_anc)
            img_anc_pos = pl_module.decoder(cont_anc, sty_pos)
            img_anc_neg = pl_module.decoder(cont_anc, sty_neg)
            img_rnd_anc = pl_module.decoder(torch.randn_like(cont_anc), sty_anc)

            images = [
                wandb.Image(img_anc, caption="Anchor"),
                wandb.Image(img_pos, caption="Positive"),
                wandb.Image(img_neg, caption="Negative"),
                wandb.Image(img_anc_anc, caption="Anc-Anc"),
                wandb.Image(img_anc_pos, caption="Anc-Pos"),
                wandb.Image(img_anc_neg, caption="Anc-Neg"),
                wandb.Image(img_rnd_anc, caption="RND-ANC"),
                wandb.Image(img_anc_neg - img_anc_anc, caption="AN - AA"),
            ]
            break
        run.log({"examples": images}, step=trainer.global_step)
