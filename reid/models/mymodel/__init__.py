from typing import Tuple
from torchmetrics.classification.accuracy import Accuracy
import pytorch_lightning as pl
import torch

from .components import Classifier
from ..munit.components import Encoder, Decoder
from ..cyclegan.components import Discriminator


class MyModel(pl.LightningModule):
    class __HPARAMS:
        lr: float
        beta1: float
        beta2: float
        weight_decay: float
        lambda_adv: float
        lambda_id: float
        lambda_img_recon: float
        lambda_code_recon: float

    hparams: __HPARAMS

    def __init__(
        self,
        img_shape: Tuple[int, int, int] = None,
        lr=0.0001,
        beta1=0,
        beta2=0.999,
        weight_decay=0.0005,
        lambda_adv: float = 1,
        lambda_id: float = 0.5,
        lambda_img_recon: float = 5,
        lambda_code_recon: float = 5,
        use_stn = False,
        *args: any,
        **kwargs: any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.criterion_bce = torch.nn.BCELoss()
        self.criterion_l1 = torch.nn.L1Loss()
        self.criterion_ce = torch.nn.CrossEntropyLoss()

        self.metric_accuracy = Accuracy()

        self.encoder = Encoder(style_dim=256, use_stn=use_stn)
        self.decoder = Decoder(style_dim=256)
        self.classifier = Classifier(in_features=256)
        self.discriminator = Discriminator(img_shape)
        self.automatic_optimization = False

    def configure_optimizers(self):
        hparams = self.hparams
        opt_gen = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.classifier.parameters()),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
        )
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
            weight_decay=hparams.weight_decay,
        )
        return [opt_gen, opt_disc]

    def training_step(self, batch, batch_idx):
        opt_gen, opt_disc = self.optimizers()
        imgs, lbls = batch
        img_anc, img_pos, img_neg = imgs
        lbl_anc, lbl_pos, lbl_neg = lbls

        cont_anc, sty_anc = self.encoder(img_anc)
        cont_pos, sty_pos = self.encoder(img_pos)
        cont_neg, sty_neg = self.encoder(img_neg)

        img_anc_anc = self.decoder(cont_anc, sty_anc)
        img_anc_pos = self.decoder(cont_anc, sty_pos)
        img_anc_neg = self.decoder(cont_anc, sty_neg)
        img_neg_anc = self.decoder(cont_neg, sty_anc)

        ## self-identity generation
        lbl_anc_hat = self.classifier(sty_anc)
        lbl_pos_hat = self.classifier(sty_pos)
        lbl_neg_hat = self.classifier(sty_neg)
        loss_img_recon = (
            self.criterion_l1(img_anc_anc, img_anc)
            + self.criterion_l1(img_anc_pos, img_anc)
        ) / 2
        loss_id = (
            self.criterion_ce(lbl_anc_hat, lbl_anc)
            + self.criterion_ce(lbl_pos_hat, lbl_pos)
            + self.criterion_ce(lbl_neg_hat, lbl_neg)
        ) / 3
        loss_self_identity = (loss_img_recon * self.hparams.lambda_img_recon) + (
            loss_id
        )

        accuracy = (
            self.metric_accuracy(lbl_anc_hat, lbl_anc)
            + self.metric_accuracy(lbl_pos_hat, lbl_pos)
            + self.metric_accuracy(lbl_neg_hat, lbl_neg)
        ) / 3

        self.log(f"train/self-identity/accuracy", accuracy)
        self.log(f"train/self-identity/img_recon", loss_img_recon)
        self.log(f"train/self-identity/id", loss_id)

        ## cross-identity generation
        cont_anc_re, sty_neg_re = self.encoder(img_anc_neg)
        cont_neg_re, sty_anc_re = self.encoder(img_neg_anc)
        lbl_anc_re_hat = self.classifier(sty_anc_re)
        lbl_neg_re_hat = self.classifier(sty_neg_re)
        loss_code_recon = (
            self.criterion_l1(sty_anc_re, sty_anc)
            + self.criterion_l1(sty_neg_re, sty_neg)
            + self.criterion_l1(cont_neg_re, cont_neg)
            + self.criterion_l1(cont_anc_re, cont_anc)
        ) / 4
        loss_id = (
            self.criterion_ce(lbl_anc_re_hat, lbl_anc)
            + self.criterion_ce(lbl_neg_re_hat, lbl_neg)
        ) / 2

        accuracy = (
            self.metric_accuracy(lbl_anc_re_hat, lbl_anc)
            + self.metric_accuracy(lbl_neg_re_hat, lbl_neg)
        ) / 2

        loss_cross_identity = (loss_code_recon * self.hparams.lambda_code_recon) + (
            loss_id * self.hparams.lambda_id
        )
        self.log(f"train/cross-identity/accuracy", accuracy)
        self.log(f"train/cross-identity/code_recon", loss_code_recon)
        self.log(f"train/cross-identity/id", loss_id)

        ## discrimination(generator)
        d_anc_anc = self.discriminator(img_anc_anc)
        d_anc_pos = self.discriminator(img_anc_pos)
        d_neg_anc = self.discriminator(img_neg_anc)
        loss_adv = (
            (
                self.criterion_bce(d_anc_anc, torch.ones_like(d_anc_anc))
                + self.criterion_bce(d_anc_pos, torch.ones_like(d_anc_pos))
                + self.criterion_bce(d_neg_anc, torch.ones_like(d_neg_anc))
            )
            / 3
        ) * self.hparams.lambda_adv
        self.log(f"train/discrimination(generator)", loss_adv)

        loss = loss_self_identity + loss_cross_identity + loss_adv

        opt_gen.zero_grad()
        self.manual_backward(loss)
        opt_gen.step()

        ## discrimination(discriminator)
        d_anc_anc = self.discriminator(img_anc_anc.detach())
        d_anc_pos = self.discriminator(img_anc_pos.detach())
        d_neg_anc = self.discriminator(img_neg_anc.detach())
        d_anc = self.discriminator(img_anc)
        d_pos = self.discriminator(img_pos)
        d_neg = self.discriminator(img_neg)
        loss = (
            self.criterion_bce(d_anc_anc, torch.zeros_like(d_anc_anc))
            + self.criterion_bce(d_anc_pos, torch.zeros_like(d_anc_pos))
            + self.criterion_bce(d_neg_anc, torch.zeros_like(d_neg_anc))
            + self.criterion_bce(d_anc, torch.ones_like(d_anc))
            + self.criterion_bce(d_pos, torch.ones_like(d_pos))
            + self.criterion_bce(d_neg, torch.ones_like(d_neg))
        ) / 6
        self.log(f"train/discrimination(discriminator)", loss)

        opt_disc.zero_grad()
        self.manual_backward(loss)
        opt_disc.step()
