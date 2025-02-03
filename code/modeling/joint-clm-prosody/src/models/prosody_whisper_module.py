import os
from typing import Any, Dict, List, Tuple, DefaultDict

import torch
from lightning import LightningModule
from torch import nn, Tensor
from torchmetrics import MinMetric, MaxMetric
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup

from src.models.components.careful_whisper import (
    CarefulWhisperConfig,
    CarefulWhisper
)

from src.utils.torch_utils import (
    get_shifted_labels,
    clm_loss,
    calculate_accuracy,
)
from src.utils.torch_metrics import (
    MaskedMeanAbsoluteError, MaskedR2Score, MaskedPearsonCorrCoef,
    MeanMetric, MaskedAccuracy
)

class CarefulWhisperModule(LightningModule):
    def __init__(
            self,
            config: CarefulWhisperConfig,
            optimizer: torch.optim.Optimizer = AdamW,
            scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
        ):
            super().__init__()

            self.save_hyperparameters(logger=False)

            self.model = CarefulWhisper(self.hparams.config)

            self.loss_fn = clm_loss

            # Add prosody information
            self.use_prosody_embeddings = use_prosody_embeddings
            self.prosody_embed = nn.Linear(num_labels, self.model.config.hidden_size, bias=False)
            self.model._init_weights(self.prosody_embed)

            self._init_metrics()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        # forward pass
        logits = self.model(
            x=batch['text_tokens'],
            xa=batch['audio_inputs'],
            mask=batch['text_attention_mask'],
            context_mask=batch['text_attention_mask']
        )
        return logits
    
    def step(self, batch: Dict[str, torch.Tensor]):
        
        logits = self.forward(batch)

        # Calculate loss between labels and logits
        loss = self.loss_fn(labels=batch['text_tokens'], logits=logits, mask=batch['text_attention_mask'])
        preds = F.softmax(logits, dim=-1).argmax(-1)

        return {
            'loss': loss,
            'logits': logits,
            'preds': preds,
        }
    
    def _shared_step(self, batch: Dict[str, torch.tensor], stage: str):
        
        outputs = self.step(batch)

        metrics = self._calculate_metrics(outputs, batch, stage)
        self._log_metrics(metrics, stage)

        return {
            "loss": outputs["loss"],
            "preds": outputs["preds"],
            "targets": batch["text_tokens"],
            "attention_mask": batch["text_attention_mask"],
        }

    def training_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        return self._shared_step(batch, 'val')

    def test_step(self, batch: Dict[str, torch.tensor], batch_idx: int):
        return self._shared_step(batch, 'test')
    
    def _init_metrics(self):
        """Initialize all metrics used in the model."""
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        metric_classes = {
            'loss': MeanMetric,
            'perplexity': MeanMetric,
            'accuracy': MeanMetric,
        }

        for name, metric_class in metric_classes.items():
            for prefix in ['train', 'val', 'test']:
                setattr(self, f'{prefix}_{name}', metric_class())

        # for logging best so far validation metrics
        best_metrics = ['val_loss', 'val_accuracy', 'val_perplexity']

        for metric in best_metrics:
            setattr(self, f'{metric}_best', MinMetric() if 'loss' in metric or 'perplexity' in metric else MaxMetric())

    def _calculate_metrics(self, outputs, batch, stage):
        
        # perplexity is the exponentiated NLL of a token given a sequence
        perplexity = torch.exp(outputs['loss'].mean())

        # get shifted logits and labels for accuracy
        labels, logits, mask = get_shifted_labels(batch["text_tokens"], outputs["logits"], batch["text_attention_mask"])

        accuracy = calculate_accuracy(labels, logits, mask)

        metrics = {
            'loss': outputs['loss'],
            'perplexity': perplexity,
            'accuracy': accuracy,
        }

        return metrics

    def _log_metrics(self, metrics, stage):
        for name, value in metrics.items():
            metric = getattr(self, f'{stage}_{name}')
            if isinstance(value, tuple):
                metric(*value)
            else:
                metric(value)
            self.log(f"{stage}/{name}", metric, on_step=stage=='train', on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self):
        self._shared_epoch_end('train')

    def on_validation_epoch_end(self):
        self._shared_epoch_end('val')

    def on_test_epoch_end(self):
        self._shared_epoch_end('test')

    def _shared_epoch_end(self, stage):

        metrics = ['loss', 'perplexity', 'accuracy',]

        if stage == 'val':
            for metric in metrics:
                value = getattr(self, f'val_{metric}').compute()
                best_metric = getattr(self, f'val_{metric}_best')
                best_metric(value)
                self.log(f"val/{metric}_best", best_metric.compute(), prog_bar=True)

        print (f'Finished epoch {self.current_epoch}')

    @property
    def total_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if (
            isinstance(self.trainer.limit_train_batches, int)
            and self.trainer.limit_train_batches != 0
        ):
            dataset_size = self.trainer.limit_train_batches
        elif isinstance(self.trainer.limit_train_batches, float):
            # limit_train_batches is a percentage of batches
            dataset_size = len(self.trainer.datamodule.train_dataloader())
            dataset_size = int(dataset_size * self.trainer.limit_train_batches)
        else:
            dataset_size = len(self.trainer.datamodule.train_dataloader())

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_batch_size = self.trainer.accumulate_grad_batches * num_devices
        max_estimated_steps = (
            dataset_size // effective_batch_size
        ) * self.trainer.max_epochs

        if self.trainer.max_steps and 0 < self.trainer.max_steps < max_estimated_steps:
            return self.trainer.max_steps
        return max_estimated_steps

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}