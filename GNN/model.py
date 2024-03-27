from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.cli import instantiate_class

from dig.ggraph.method.JTVAE import fast_jtnn

from .registry import MODEL_REGISTRY
from utils import get_logger

logger = get_logger(__name__)


def get_model(args, vocab):
    if args.name in MODEL_REGISTRY:
        model_args = MODEL_REGISTRY[args.name].copy()
        train_args = {k:model_args[k] for k in ["warmup", "beta", "max_beta", "step_beta", "kl_anneal_iter"]}
        model_args = {k:model_args[k] for k in ["hidden_size", "latent_size", "depthT", "depthG"]}
        model_args.update(vocab=dict(vocab=vocab))
        model = fast_jtnn.JTNNVAE(**model_args)
    else:
        raise ValueError(f'{args.name} is not a valid model; must be one of {MODEL_REGISTRY.keys()}')
    return model, train_args


class Model(LightningModule):
    def __init__(
        self,
        name: str
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        datamodule = self.trainer.datamodule
        self.vocab = fast_jtnn.Vocab(datamodule.vocab)
        self.model, self.train_args = get_model(self.hparams, self.vocab)

        self.warmup = self.train_args['warmup']
        self.beta = self.train_args['beta']
        self.max_beta = self.train_args['max_beta']
        self.step_beta = self.train_args['step_beta']
        self.kl_annealing_iter = self.train_args['kl_annealing_iter']

        self.global_step_ = 0
        print(f'# params: {sum(p.numel() for p in self.model.parameters())}')

        
    # def setup(self, stage):

    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx: int):
        if self.global_step_ % self.kl_anneal_iter == 0 and self.global_step_ >= self.warmup:
            self.beta = min(self.max_beta, self.beta + self.step_beta)
        loss, kl_div, wacc, tacc, sacc = self.model(batch, self.beta)
        wacc *= 100
        tacc *= 100
        sacc *= 100
        log = dict(beta=self.beta, kl_div=kl_div, loss=loss, wacc=wacc, tacc=tacc, sacc=sacc)
        for k, v in log.items():
            self.log(f'train/{k}', v)
        del log['beta']
        return log

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        for key in outputs[0].keys():
            loss_vec = torch.stack([outputs[i][key] for i in range(len(outputs))])
            mean, std = loss_vec.mean(), loss_vec.std()
            self.log(f"train/{key}_mean", mean)
            self.log(f"train/{key}_std", std)


    # def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):

    # def validation_epoch_end(self, outputs: List[Any]):

    # def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):

    # def test_epoch_end(self, outputs: List[Any]):