import os
from typing import Any, Dict, List, Tuple, DefaultDict

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import einops

from lightning import LightningModule
from torchmetrics import MinMetric, MaxMetric
from transformers import AdamW, get_linear_schedule_with_warmup

from src.models.components.token_fusion import (
    TokenFusionMLP,
)

from src.utils.torch_metrics import (
    MeanMetric, 
)

class TokenFusionModule(LightningModule):
    def __init__(
        self, 
        input_dim1: int, 
        input_dim2: int, 
        input_name1: str,
        input_name2: str,
        hidden_dim: int = 1024, 
        loss_fn: str = 'MSE',
        optimizer: torch.optim.Optimizer = AdamW,
        scheduler: torch.optim.lr_scheduler = get_linear_schedule_with_warmup,
    ):
        """
        Lightning module for TokenFusionMLP autoencoder
        
        Args:
        - input_dim1 (int): Dimension of the first input vector
        - input_dim2 (int): Dimension of the second input vector
        - hidden_dim (int): Dimension of the compressed representation
        """
        super().__init__()
        
        # Save hyperparameters for logging and checkpoint
        self.save_hyperparameters()
        
        # Initialize the TokenFusionMLP model
        self.model = TokenFusionMLP(
            input_dim1=input_dim1, 
            input_dim2=input_dim2, 
            hidden_dim=hidden_dim
        )

        self._init_metrics()
    
    def forward(self, vec1, vec2):
        """
        Forward pass of the autoencoder
        
        Args:
        - vec1 (torch.Tensor): First input vector
        - vec2 (torch.Tensor): Second input vector
        
        Returns:
        - reconstructed_vec1 (torch.Tensor): Reconstructed first vector
        - reconstructed_vec2 (torch.Tensor): Reconstructed second vector
        - compressed_representation (torch.Tensor): Compressed joint representation
        """
        return self.model(vec1, vec2)

    def step(self, batch: Dict[str, torch.Tensor]):

        # BATCH_SIZE = 32

        # vectors = [batch[x] for x in [self.hparams.input_name1, self.hparams.input_name2]]
        # vectors = [einops.rearrange(vec, 'b n d -> (b n) d') for vec in vectors]

        # # Generate random indices without replacement
        # random_indices = torch.randperm(vectors[0].shape[0])[:BATCH_SIZE]
        # vec1, vec2 = [vec[random_indices, ...] for vec in vectors]

        # reconstructed_vec1, reconstructed_vec2, compressed = self.forward(vec1=vec1, vec2=vec2)
        vec1 = batch[self.hparams.input_name1]
        vec2 = batch[self.hparams.input_name2]
        
        reconstructed_vec1, reconstructed_vec2, compressed = self.forward(vec1=vec1, vec2=vec2)

        # Compute reconstruction losses
        loss_vec1 = F.mse_loss(reconstructed_vec1, vec1)
        loss_vec2 = F.mse_loss(reconstructed_vec2, vec2)

        # Get similarity to original vectors
        sim_vec1 = F.cosine_similarity(vec1, compressed, dim=-1).mean(0)
        sim_vec2 = F.cosine_similarity(vec2, compressed, dim=-1).mean(0)

        # Total loss
        # Reonstruct them while keeping them as similar as possible
        # to the original representational space
        if self.hparams.loss_fn == 'MSE':
            total_loss = loss_vec1 + loss_vec2
        elif self.hparams.loss_fn == 'representation_loss':
            total_loss = (loss_vec1 - sim_vec1) + (loss_vec2 - sim_vec2)
        elif self.hparams.loss_fn == 'orthogonality_loss':
            total_loss = (loss_vec1 + sim_vec1) + (loss_vec2 + sim_vec2)

        return {
            "loss": total_loss,
            f"loss_{self.hparams.input_name1}": loss_vec1,
            f"loss_{self.hparams.input_name2}": loss_vec2,
            f"similarity_{self.hparams.input_name1}": sim_vec1,
            f"similarity_{self.hparams.input_name2}": sim_vec2,
        }

    def _shared_step(self, batch: Dict[str, torch.tensor], stage: str):
        
        outputs = self.step(batch)
        self._log_metrics(outputs, stage)

        return {
            "loss": outputs["loss"],
            f"loss_{self.hparams.input_name1}": outputs[f"loss_{self.hparams.input_name1}"],
            f"loss_{self.hparams.input_name2}": outputs[f"loss_{self.hparams.input_name2}"],
            f"similarity_{self.hparams.input_name1}": outputs[f"similarity_{self.hparams.input_name1}"],
            f"similarity_{self.hparams.input_name2}": outputs[f"similarity_{self.hparams.input_name2}"],
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
            "loss": MeanMetric,
            f"loss_{self.hparams.input_name1}": MeanMetric,
            f"loss_{self.hparams.input_name2}": MeanMetric,
            f"similarity_{self.hparams.input_name1}": MeanMetric,
            f"similarity_{self.hparams.input_name2}": MeanMetric,
        }

        for name, metric_class in metric_classes.items():
            for prefix in ['train', 'val', 'test']:
                setattr(self, f'{prefix}_{name}', metric_class())

        # for logging best so far validation metrics
        best_metrics = [
            'val_loss', 
            f"val_loss_{self.hparams.input_name1}", 
            f"val_loss_{self.hparams.input_name2}", 
            f"val_similarity_{self.hparams.input_name1}", 
            f"val_similarity_{self.hparams.input_name2}"
        ]

        for metric in best_metrics:
            setattr(self, f'{metric}_best', MinMetric() if 'loss' in metric else MaxMetric())
    
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

        metrics = [
            'loss', 
            f"loss_{self.hparams.input_name1}", 
            f"loss_{self.hparams.input_name2}",
            f"similarity_{self.hparams.input_name1}", 
            f"similarity_{self.hparams.input_name2}"
        ]

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